from typing import Any, Dict, Optional
import json
import os
import sqlite3
import traceback
from datetime import datetime
import re

# Agent imports (use the confirmed path for text2sql)
try:
    from agents.agentic_tools.text2sql import generate_sql_and_execute
except Exception as e:
    def generate_sql_and_execute(*args, **kwargs):
        return {"sql": None, "dataframe": None, "csv_path": None, "error": True, "error_message": f"text2sql not available: {e}"}

try:
    from agents.insightAgent import generate_insights
except Exception:
    def generate_insights(*args, **kwargs):
        return {"summary": None, "analysis": {}, "dataframe_path": None, "error": True, "error_message": "insightAgent not available"}

try:
    from agents.visualizerAgent import visualize_from_csv_path
except Exception:
    def visualize_from_csv_path(*args, **kwargs):
        return {"error": True, "error_message": "visualizerAgent not available", "chart_path": None, "caption": None}

# Knowledge agent: prefer canonical import, otherwise load from uploaded path
KNOWLEDGE_MODULE_PATH = os.path.join("/mnt/data", "knowledge.py")
try:
    from agents.knowledge import get_knowledge_summary, get_finops_tip
except Exception:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("agents.knowledge", KNOWLEDGE_MODULE_PATH)
        knowledge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(knowledge)
        get_knowledge_summary = getattr(knowledge, "get_knowledge_summary")
        get_finops_tip = getattr(knowledge, "get_finops_tip")
    except Exception:
        def get_knowledge_summary(*args, **kwargs):
            return "Knowledge agent not available"
        def get_finops_tip(*args, **kwargs):
            return "FinOps tip agent not available"


# ------------------ Memory persistence helpers (JSON + SQLite) ------------------
MEMORY_JSON = os.getenv("FINOPS_MEMORY_JSON", "memory.json")
MEMORY_DB = os.getenv("FINOPS_MEMORY_DB", "finops_memory.db")

def _ensure_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
    finally:
        conn.close()

def load_memory() -> Dict[str, Any]:
    # Load from JSON first (if exists), then overlay SQLite keys
    mem = {}
    if os.path.exists(MEMORY_JSON):
        try:
            with open(MEMORY_JSON, "r", encoding="utf-8") as f:
                mem = json.load(f)
        except Exception:
            mem = {}
    # overlay sqlite
    _ensure_memory_db()
    try:
        conn = sqlite3.connect(MEMORY_DB)
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM memory")
        rows = cur.fetchall()
        for k, v in rows:
            try:
                mem[k] = json.loads(v)
            except Exception:
                mem[k] = v
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    # ensure basic structure
    mem.setdefault("context", "")
    mem.setdefault("entities", {})
    mem.setdefault("preferences", {})
    mem.setdefault("history", [])
    mem.setdefault("last_query", None)
    return mem

def save_memory(mem: Dict[str, Any]):
    # Save JSON (human-readable)
    try:
        with open(MEMORY_JSON, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2)
    except Exception:
        pass
    # Save to sqlite (atomic upserts)
    try:
        _ensure_memory_db()
        conn = sqlite3.connect(MEMORY_DB)
        cur = conn.cursor()
        for k, v in mem.items():
            # skip very large or unserializable items
            try:
                sval = json.dumps(v)
            except Exception:
                sval = str(v)
            cur.execute("INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (k, sval))
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ------------------------ LangGraph primitives & nodes -------------------------
class Node:
    def __init__(self, name: str):
        self.name = name

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        st = state.copy()
        st.setdefault("memory", load_memory())
        st.setdefault("error", False)
        st.setdefault("error_message", None)
        for node in self.nodes:
            if st.get("error"):
                break
            try:
                res = node.run(st)
                if res:
                    st.update(res)
            except Exception as exc:
                st["error"] = True
                st["error_message"] = f"Node {node.name} crashed: {exc}\n" + traceback.format_exc()
                break
        # persist memory at end
        try:
            if "memory" in st and isinstance(st["memory"], dict):
                save_memory(st["memory"])
        except Exception:
            pass
        return st

# ------------------------------ Node implementations --------------------------
class InputNode(Node):
    def __init__(self):
        super().__init__("InputNode")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("original_query") or state.get("query") or ""
        return {"original_query": q.strip()}

class Text2SQLNode(Node):
    def __init__(self):
        super().__init__("Text2SQLNode")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("original_query", "")
        csv_path = state.get("csv_path")
        db_path = state.get("db_path")
        table_name = state.get("table_name")
        model = state.get("t2s_model")
        result = generate_sql_and_execute(user_query=q, csv_path=csv_path, db_path=db_path, table_name=table_name, model=model)
        out = {
            "sql": result.get("sql"),
            "dataframe": result.get("dataframe"),
            "csv_result_path": result.get("csv_path"),
            "text2sql_error": result.get("error", False),
            "text2sql_error_message": result.get("error_message")
        }
        if out.get("text2sql_error"):
            out["error"] = True
            out["error_message"] = out.get("text2sql_error_message")
        return out

class InsightNode(Node):
    def __init__(self):
        super().__init__("InsightNode")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("original_query", "")
        df = state.get("dataframe")
        csv = state.get("csv_result_path")
        res = generate_insights(user_query=q, csv_path=csv, df=df, schema_context=None)
        out = {
            "insight_summary": res.get("summary"),
            "insight_analysis": res.get("analysis"),
            "insight_df_path": res.get("dataframe_path"),
            "insight_error": res.get("error", False),
            "insight_error_message": res.get("error_message")
        }
        if out.get("insight_error"):
            out["error"] = True
            out["error_message"] = out.get("insight_error_message")
        return out

class VisualizationNode(Node):
    def __init__(self):
        super().__init__("VisualizationNode")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("original_query", "")
        csv = state.get("csv_result_path")
        if not csv:
            return {}
        if not any(w in q.lower() for w in ["plot", "chart", "graph", "visualize", "show"]):
            return {}
        res = visualize_from_csv_path(csv, q)
        out = {
            "chart_path": res.get("chart_path"),
            "chart_caption": res.get("caption"),
            "viz_error": res.get("error", False),
            "viz_error_message": res.get("error_message")
        }
        if out.get("viz_error"):
            out["error"] = True
            out["error_message"] = out.get("viz_error_message")
        return out

class KnowledgeNode(Node):
    def __init__(self):
        super().__init__("KnowledgeNode")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("original_query", "")
        mem = state.get("memory", {}) or {}
        entities = mem.get("entities", {})
        history = state.get("conversation_history")
        try:
            summary = get_knowledge_summary(query=q, memory_context=mem.get("context", ""), remembered_entities=entities, conversation_history=history)
        except Exception as e:
            summary = f"Knowledge call failed: {e}"
        return {"knowledge_summary": summary}

class MemoryNode(Node):
    def __init__(self):
        super().__init__("MemoryNode")

    def _extract_entities_from_query(self, q: str):
        # naive extraction: look for known tokens or quoted strings
        entities = {}
        # service names often capitalized words; try to extract phrases after 'service' or 'ServiceName'
        m = re.findall(r"ServiceName\s*[:=]?\s*([A-Za-z0-9 _-]+)", q, flags=re.I)
        if m:
            entities["ServiceName"] = m[0].strip()
        # look for 'last month', 'yesterday', '2025-10'
        if re.search(r"last month|yesterday|today|this month|last week", q, flags=re.I):
            entities.setdefault("timeframe", []).append(re.search(r"(last month|this month|last week|yesterday|today)", q, flags=re.I).group(0))
        # look for cost column mention
        if re.search(r"billedcost|billed cost|effectivecost|effective cost|listcost", q, flags=re.I):
            entities.setdefault("cost_terms", []).append(q)
        return entities

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mem = state.get("memory", {}) or {}
        q = state.get("original_query", "")
        insight = state.get("insight_analysis") or {}
        knowledge = state.get("knowledge_summary")
        df = state.get("dataframe")

        # update last query and push to history
        mem["last_query"] = q
        hist_item = {"ts": datetime.utcnow().isoformat() + "Z", "query": q}
        mem.setdefault("history", [])
        mem["history"].append(hist_item)
        # cap history
        if len(mem["history"]) > 200:
            mem["history"] = mem["history"][-200:]

        # collect entities from query
        q_entities = self._extract_entities_from_query(q)
        ent = mem.setdefault("entities", {})
        for k, v in q_entities.items():
            ent[k] = v

        # try to infer service names from insight analysis
        try:
            top_services = insight.get("top_services") if isinstance(insight, dict) else None
            if top_services and isinstance(top_services, list):
                # store the top service names
                names = [row.get("ServiceName") or row.get("service") for row in top_services if isinstance(row, dict)]
                names = [n for n in names if n]
                if names:
                    ent.setdefault("recent_top_services", [])
                    for n in names:
                        if n not in ent["recent_top_services"]:
                            ent["recent_top_services"].append(n)
                    # keep small list
                    ent["recent_top_services"] = ent["recent_top_services"][-20:]
        except Exception:
            pass

        # remember preferred cost column if present in query
        if re.search(r"billedcost|effectivecost|listcost|contractedcost", q, flags=re.I):
            pref = re.search(r"billedcost|effectivecost|listcost|contractedcost", q, flags=re.I).group(0)
            mem.setdefault("preferences", {})
            mem["preferences"]["cost_column"] = pref

        # persist partial insight into memory context string for knowledge usage
        try:
            mem["context"] = (mem.get("context", "") + "\n" + (knowledge or ""))[:2000]
        except Exception:
            pass

        return {"memory": mem}

class ResponseBuilderNode(Node):
    def __init__(self):
        super().__init__("ResponseBuilderNode")

    def _df_preview(self, df):
        try:
            if df is None:
                return None
            s = df.head(20).to_csv(index=False)
            return s
        except Exception:
            return None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pieces = []
        if state.get("insight_summary"):
            pieces.append(state.get("insight_summary"))
        if state.get("knowledge_summary"):
            pieces.append("Knowledge:\n" + state.get("knowledge_summary"))
        if state.get("sql"):
            pieces.append("Executed SQL:\n" + (state.get("sql") or ""))
        df_preview = self._df_preview(state.get("dataframe"))
        if df_preview:
            pieces.append("SQL Result Preview:\n" + df_preview)
        if state.get("chart_path"):
            pieces.append(f"Visualization generated at: {state.get('chart_path')}\nCaption: {state.get('chart_caption')}")

        # personalization using memory
        mem = state.get("memory", {}) or {}
        if mem.get("preferences"):
            try:
                pref = mem["preferences"].get("cost_column")
                if pref:
                    pieces.append(f"User preference: cost column = {pref}")
            except Exception:
                pass

        response_text = "\n\n".join(pieces) if pieces else None
        return {"response": response_text}

class ErrorHandlerNode(Node):
    def __init__(self):
        super().__init__("ErrorHandlerNode")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("error"):
            em = state.get("error_message") or "An unexpected error occurred."
            hint = "Please check server logs for details."
            return {"response": f"Error: {em}\n{hint}", "error": True}
        return {}

# --------------------------- Public entrypoint --------------------------------

def run_supervisor(
    state: Dict[str, Any],
    csv_path: Optional[str] = None,
    db_path: Optional[str] = None,
    table_name: str = "finops_data",
    conversation_history: Optional[list] = None,
    t2s_model: Optional[str] = None
) -> Dict[str, Any]:
    g = Graph()
    g.add_node(InputNode())
    g.add_node(Text2SQLNode())
    g.add_node(InsightNode())
    g.add_node(VisualizationNode())
    g.add_node(KnowledgeNode())
    g.add_node(MemoryNode())
    g.add_node(ResponseBuilderNode())
    g.add_node(ErrorHandlerNode())

    st = {
        "original_query": state.get("original_query") or state.get("query") or "",
        "csv_path": csv_path or state.get("csv_path") or os.getenv("FINOPS_CSV_PATH", "data/data.csv"),
        "db_path": db_path or state.get("db_path") or os.getenv("FINOPS_SQLITE_DB", "finops.db"),
        "table_name": table_name,
        "conversation_history": conversation_history or state.get("conversation_history") or [],
        "t2s_model": t2s_model or os.getenv("FINOPS_T2S_MODEL", "gpt-4o-mini")
    }

    result = g.run(st)

    out = {
        "response": result.get("response"),
        "chart_path": result.get("chart_path"),
        "error": result.get("error", False),
        "error_message": result.get("error_message"),
        "sql": result.get("sql"),
        "insight": result.get("insight_summary"),
        "knowledge": result.get("knowledge_summary"),
        "analysis": result.get("insight_analysis"),
        "csv_path": result.get("csv_result_path") or result.get("insight_df_path"),
        "memory": result.get("memory")
    }
    return out

# ------------------------------- CLI quick test --------------------------------
if __name__ == "__main__":
    s = {"original_query": "Show total EffectiveCost by ServiceName for last month"}
    out = run_supervisor(s)
    print("RESPONSE:\n", out.get("response"))
    if out.get("error"):
        print("ERROR MESSAGE:\n", out.get("error_message"))
