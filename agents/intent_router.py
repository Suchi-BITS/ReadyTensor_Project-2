# agents/intent_router.py
import os
import re
import sys

# Add parent directory to path for standalone testing
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.prompt_loader import load_prompt_from_hub

# -------------------------------------------------------------------
# Advanced intent router for multi-agent FinOps system
# -------------------------------------------------------------------

def classify_intent(user_query: str):
    """
    Classify user query into high-level intent and subagent type.
    Returns a dict with intent, category, subagent, confidence.
    
    This router supports a sophisticated multi-agent system with:
    - DataFetcher: SQL queries, entity extraction, specific data retrieval
    - InsightsAgent: Trend analysis, visualizations, pattern detection
    - SmallTalk: Greetings and casual conversation
    """

    query = user_query.lower().strip()
    
    # ===================================================================
    # SMALL TALK - Highest Priority (but VERY specific)
    # ===================================================================
    small_talk_patterns = [
        r"^(hello|hi|hey)[\s!?.]*$",
        r"^(thanks|thank you)[\s!?.]*$",
        r"^(how are you|good morning|good evening|good afternoon)[\s!?.]*$",
        r"^(bye|goodbye|see you)[\s!?.]*$"
    ]
    
    # Only match small talk if it's EXACTLY a greeting (not part of a longer query)
    if any(re.match(pattern, query) for pattern in small_talk_patterns):
        return {
            "intent": "small_talk",
            "category": "none",
            "subagent": "none",
            "confidence": 0.95,
        }
    
    # ===================================================================
    # DATA FETCHER - Specific queries requiring SQL/entity extraction
    # ===================================================================
    # These queries need:
    # 1. Entity extraction (services, resources, regions, etc.)
    # 2. Text-to-SQL conversion
    # 3. Specific data retrieval
    
    data_fetcher_indicators = {
        # Question words for specific data
        "what_questions": ["what is", "what are", "what was", "what were", "whats", "what's"],
        
        # Aggregation keywords
        "aggregations": ["total", "sum", "count", "average", "avg", "max", "min", "mean"],
        
        # Specific data requests
        "specificity": [
            "exact", "specific", "particular", "precise",
            "how much", "how many", "tell me", "show me",
            "find", "get", "fetch", "retrieve", "pull", "list"
        ],
        
        # Entity types (things that need extraction)
        "entities": [
            "service", "services", "resource", "resources",
            "region", "regions", "account", "accounts",
            "subscription", "subscriptions", "instance", "instances",
            "app", "apps", "application", "applications",
            "provider", "providers", "vendor", "vendors"
        ],
        
        # Top-N queries (classic SQL use case)
        "topn": ["top", "bottom", "highest", "lowest", "most", "least"],
        
        # Time-specific queries
        "time_specific": [
            "in january", "in february", "in march", "in april", "in may", 
            "in june", "in july", "in august", "in september", "in october",
            "in november", "in december", "in 2024", "in 2025", "last month",
            "this month", "last year", "this year", "last week", "this week"
        ],
        
        # Cost/billing specific
        "cost_billing": [
            "cost", "costs", "spend", "spending", "spent",
            "expense", "expenses", "bill", "billing", "billed",
            "charge", "charges", "price", "pricing", "amount"
        ],
        
        # Dataset/column queries (NEW)
        "dataset_queries": [
            "dataset", "data", "column", "columns", "field", "fields",
            "date", "dates", "period", "start", "end", "range",
            "chargeperiod", "billingperiod", "usageperiod",
            "in this dataset", "in the dataset", "in dataset",
            "what columns", "which columns", "available columns"
        ]
    }
    
    # Check for data fetcher indicators
    data_fetcher_score = 0
    matched_categories = []
    
    for category, keywords in data_fetcher_indicators.items():
        if any(keyword in query for keyword in keywords):
            data_fetcher_score += 1
            matched_categories.append(category)
    
    # ===================================================================
    # INSIGHTS AGENT - Analysis, trends, patterns, visualizations
    # ===================================================================
    # These queries need:
    # 1. Trend analysis over time
    # 2. Pattern detection
    # 3. Comparative analysis
    # 4. Visualizations
    
    insights_indicators = {
        # Trend keywords
        "trends": [
            "trend", "trends", "trending", "over time", "time series",
            "monthly", "daily", "weekly", "quarterly", "yearly",
            "growth", "decline", "increase", "decrease", "change", "changes"
        ],
        
        # Analysis keywords
        "analysis": [
            "analyze", "analysis", "breakdown", "distribution",
            "pattern", "patterns", "behavior", "variance",
            "compare", "comparison", "versus", "vs", "difference"
        ],
        
        # Visualization keywords (but not when asking for data)
        "visualization": [
            "plot", "chart", "graph", "visualize",
            "dashboard", "report"
        ],
        
        # Insight keywords
        "insights": [
            "insight", "insights", "driver", "drivers", "factor", "factors",
            "why", "reason", "reasons", "cause", "causes",
            "impact", "effect", "influence"
        ],
        
        # Comparative analysis
        "comparative": [
            "compare", "comparison", "versus", "vs", "between",
            "difference", "differ", "against"
        ]
    }
    
    # Check for insights indicators
    insights_score = 0
    insights_matched = []
    
    for category, keywords in insights_indicators.items():
        if any(keyword in query for keyword in keywords):
            insights_score += 1
            insights_matched.append(category)
    
    # ===================================================================
    # DECISION LOGIC
    # ===================================================================
    
    # If both scores are equal or both are high, use additional heuristics
    if data_fetcher_score > 0 and insights_score > 0:
        # Prioritize insights if visualization or trend keywords are present
        if any(word in query for word in ["trend", "plot", "chart", "analyze", "monthly", "over time"]):
            return {
                "intent": "finops_query",
                "category": "Insights & Analysis",
                "subagent": "insight_agent",
                "confidence": 0.85,
                "matched_indicators": insights_matched
            }
        # Prioritize data fetcher for specific entity queries or top-N
        elif any(word in query for word in ["top", "what is", "what are", "whats", "total", "in january", "in april", "specific", "dataset", "column", "field"]):
            return {
                "intent": "finops_query",
                "category": "Data Retrieval",
                "subagent": "data_fetcher",
                "confidence": 0.85,
                "matched_indicators": matched_categories
            }
    
    # Clear data fetcher intent
    if data_fetcher_score >= 2:
        return {
            "intent": "finops_query",
            "category": "Data Retrieval",
            "subagent": "data_fetcher",
            "confidence": min(0.9, 0.7 + (data_fetcher_score * 0.1)),
            "matched_indicators": matched_categories
        }
    
    # Clear insights intent
    if insights_score >= 2:
        return {
            "intent": "finops_query",
            "category": "Insights & Analysis",
            "subagent": "insight_agent",
            "confidence": min(0.9, 0.7 + (insights_score * 0.1)),
            "matched_indicators": insights_matched
        }
    
    # Single indicator matches - IMPORTANT: Still route to appropriate agent
    if data_fetcher_score >= 1:
        return {
            "intent": "finops_query",
            "category": "Data Retrieval",
            "subagent": "data_fetcher",
            "confidence": 0.7,
            "matched_indicators": matched_categories
        }
    
    if insights_score >= 1:
        return {
            "intent": "finops_query",
            "category": "Insights & Analysis",
            "subagent": "insight_agent",
            "confidence": 0.7,
            "matched_indicators": insights_matched
        }
    
    # Default: If nothing matches but it's a question, assume it's a finops query
    question_words = ["what", "how", "when", "where", "which", "who", "why", "show", "tell", "give", "list"]
    if any(query.startswith(word) or f" {word} " in query for word in question_words):
        return {
            "intent": "finops_query",
            "category": "Data Retrieval",
            "subagent": "data_fetcher",
            "confidence": 0.6,
            "matched_indicators": ["question_word_fallback"]
        }
    
    # True default: unrelated
    return {
        "intent": "unrelated",
        "category": "none",
        "subagent": "none",
        "confidence": 0.4,
        "matched_indicators": []
    }


# -------------------------------------------------------------------
# Optional: system prompt support (for debugging / expansion)
# -------------------------------------------------------------------

def get_intent_prompt() -> str:
    """
    Loads the intent router's instruction prompt from prompts/intent_router.txt.
    This is just for reference or debugging (not used in classification logic).
    """
    try:
        prompt_text = load_prompt_from_hub("intent_router")
        return prompt_text
    except Exception as e:
        return f"[Intent Router] Failed to load local prompt: {e}"


# -------------------------------------------------------------------
# Test script
# -------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        # Small talk
        "Hello there!",
        "Thanks for your help!",
        
        # Data Fetcher queries (should use entity extraction + SQL)
        "What is the total cost in April?",
        "What are the top 5 services by spend?",
        "Find the top 10 cost drivers",
        "How much did we spend on EC2 in January?",
        "What is the total production cost?",
        "Show me the billing for last month",
        "Get me the exact cost for us-east-1",
        
        # Column/dataset queries (NEW - should go to data_fetcher)
        "whats the chargeperiodstart and end date in this dataset",
        "what columns are in the dataset",
        "show me the date range",
        "what fields are available",
        
        # Insights Agent queries (should use trend analysis)
        "Show monthly cost trend",
        "Show me the monthly spend trend",
        "Analyze the cost pattern over time",
        "Compare storage cost between Jan and Feb",
        "Plot a trend of GCP cost growth",
        "What's driving the cost increase?",
        "Display the quarterly spending breakdown",
        
        # Ambiguous (should intelligently route)
        "total cost",  # Could be either, but likely data fetcher
        "cost trend",  # Likely insights
    ]

    print("Testing Advanced Intent Router\n" + "=" * 70)
    for q in test_queries:
        result = classify_intent(q)
        indicators = result.get('matched_indicators', [])
        print(f"\nQuery: '{q}'")
        print(f"→ Intent: {result['intent']}")
        print(f"→ Category: {result['category']}")
        print(f"→ Subagent: {result['subagent']}")
        print(f"→ Confidence: {result['confidence']}")
        if indicators:
            print(f"→ Matched: {', '.join(indicators)}")