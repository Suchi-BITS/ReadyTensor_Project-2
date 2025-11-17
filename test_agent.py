# test_intent.py
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.intent_router import classify_intent

test_queries = [
    "Hello there!",
    "Show me the monthly spend trend",
    "show monthly trend",
    "Fetch cost details for March",
    "Compare storage cost between Jan and Feb",
    "Thanks for your help!",
    "How much did we spend on AWS EC2 last month?",
    "Plot a trend of GCP cost growth"
]

print("🧭 Testing Intent Router\n" + "-" * 50)
for q in test_queries:
    result = classify_intent(q)
    print(f"Query: '{q}'")
    print(f"→ Intent: {result['intent']}, Subagent: {result['subagent']}, Confidence: {result['confidence']}\n")