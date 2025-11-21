import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import json
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from integrations.main import process_query

# ----------------------------------------------------
# ALWAYS USE THE DEFAULT CSV
# ----------------------------------------------------
CSV_FILE = os.path.join("data", "data.csv")
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing default CSV: {CSV_FILE}")

# ----------------------------------------------------
# DATABASE INITIALIZATION
# ----------------------------------------------------
DB_PATH = "finops_memory.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_activity 
        ON sessions(last_activity)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_session 
        ON conversation_history(session_id, timestamp)
    """)

    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


init_database()

# ----------------------------------------------------
# FASTAPI CONFIG
# ----------------------------------------------------
app_sqlite = FastAPI(title="FinOps AI API (Auto CSV Mode)")

app_sqlite.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# MODELS
# ----------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    session_id: str
    query: str
    # csv_path REMOVED – API always uses default CSV


class QueryResponse(BaseModel):
    session_id: str
    response: str
    chart_path: Optional[str] = None
    turn_number: int
    intent: Optional[str] = None
    subagent: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    last_activity: str


# ----------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------
@app_sqlite.post("/session/create")
def create_session_sqlite():
    session_id = str(uuid.uuid4())

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, created_at, last_activity, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            json.dumps({})  
        ))

    return {"session_id": session_id, "message": "Session created"}


# ----------------------------------------------------
# MAIN QUERY ENDPOINT (NO CSV UPLOAD)
# ----------------------------------------------------
@app_sqlite.post("/session/{session_id}/query", response_model=QueryResponse)
def query_sqlite(session_id: str, request: QueryRequest):

    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session = cursor.fetchone()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # the CSV path is ALWAYS our default file
        csv_path = CSV_FILE

        # Load conversation history
        cursor.execute("""
            SELECT role, content, timestamp, metadata 
            FROM conversation_history 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        history_rows = cursor.fetchall()

        conversation_history = [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
            for row in history_rows
        ]

        # Core processing
        result = process_query(
            user_query=request.query,
            csv_path=csv_path,
            conversation_history=conversation_history,
            session_id=session_id
        )

        # Save to history
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO conversation_history (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, "user", request.query, now, json.dumps({})))

        cursor.execute("""
            INSERT INTO conversation_history (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            "assistant",
            result.get("response", ""),
            now,
            json.dumps({
                "intent": result.get("intent"),
                "subagent": result.get("subagent"),
                "chart_path": result.get("chart_path")
            })
        ))

        cursor.execute("""
            UPDATE sessions SET last_activity = ?
            WHERE session_id = ?
        """, (now, session_id))

        turn_number = len(conversation_history) // 2 + 1

        return QueryResponse(
            session_id=session_id,
            response=result.get("response", ""),
            chart_path=result.get("chart_path"),
            turn_number=turn_number,
            intent=result.get("intent"),
            subagent=result.get("subagent")
        )


# ----------------------------------------------------
# HISTORY ENDPOINT
# ----------------------------------------------------
@app_sqlite.get("/session/{session_id}/history")
def get_history_sqlite(session_id: str, limit: int = 100):

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, timestamp, metadata 
            FROM conversation_history 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))

        history = [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
            for row in cursor.fetchall()
        ]

    return {"session_id": session_id, "history": list(reversed(history))}


# ----------------------------------------------------
# DELETE SESSION
# ----------------------------------------------------
@app_sqlite.delete("/session/{session_id}")
def delete_session_sqlite(session_id: str):

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully"}
