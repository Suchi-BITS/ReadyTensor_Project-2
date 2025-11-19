import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))  # D:\A-Agent\integrations
PARENT = os.path.dirname(ROOT)  # D:\A-Agent

if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import streamlit as st
import os
from integrations.main import process_query
from datetime import datetime

# Streamlit App Setup
st.set_page_config(page_title="FinOps Agentic AI", layout="wide")

# Initialize Session State for Multi-Turn Conversation with Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "csv_path" not in st.session_state:
    st.session_state.csv_path = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.now()

# Header
st.title(" FinOps Agentic AI System")
st.markdown("Ask questions about cloud spend, trends, or usage. The system remembers your conversation context.")

# Sidebar: Data File Configuration and Memory Stats
with st.sidebar:
    st.header(" Data Configuration")
    
    # CSV file path input
    default_csv_path = st.text_input(
        "CSV File Path",
        value="data/sample_data.csv",
        help="Enter the path to your FinOps CSV file"
    )
    
    if st.button("Load Data File"):
        if os.path.exists(default_csv_path):
            st.session_state.csv_path = default_csv_path
            st.session_state.file_loaded = True
            st.success(f" File loaded: {default_csv_path}")
            
            # Add system message about file load
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Data file loaded successfully from `{default_csv_path}`. You can now ask me questions about your cloud spending data!",
                "chart_path": None,
                "timestamp": datetime.now().isoformat()
            })
        else:
            st.error(f" File not found: {default_csv_path}")
            st.session_state.file_loaded = False
    
    st.markdown("---")
    
    # File status
    if st.session_state.file_loaded:
        st.success(" Data Loaded")
        st.info(f"**File:** {os.path.basename(st.session_state.csv_path)}")
    else:
        st.warning(" No data loaded")
    
    st.markdown("---")
    
    # Memory and Conversation Stats
    st.header(" Memory Stats")
    
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    conversation_turns = user_messages
    
    st.metric("Total Messages", total_messages)
    st.metric("Conversation Turns", conversation_turns)
    st.metric("History Entries", len(st.session_state.conversation_history))
    
    # Session info
    session_duration = datetime.now() - st.session_state.session_start_time
    st.info(f"**Session Duration:** {session_duration.seconds // 60} minutes")
    st.caption(f"**Session ID:** {st.session_state.session_id[:8]}...")
    
    st.markdown("---")
    
    # Memory management options
    st.header(" Memory Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Clear Memory", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.session_id = str(__import__('uuid').uuid4())
            st.session_state.session_start_time = datetime.now()
            st.rerun()
    
    # Export conversation
    if st.button(" Export Conversation", use_container_width=True):
        if st.session_state.conversation_history:
            import json
            export_data = {
                "session_id": st.session_state.session_id,
                "start_time": st.session_state.session_start_time.isoformat(),
                "history": st.session_state.conversation_history
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"conversation_{st.session_state.session_id[:8]}.json",
                mime="application/json"
            )

# Main Chat Interface

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display chart if available
        if message.get("chart_path") and os.path.exists(message["chart_path"]):
            st.image(message["chart_path"], caption="Generated Visualization", use_column_width=True)

# Chat input
if prompt := st.chat_input("Ask a question about your cloud spending..."):
    
    # Check if data is loaded
    if not st.session_state.file_loaded:
        with st.chat_message("assistant"):
            st.warning(" Please load a data file first using the sidebar.")
        st.stop()
    
    # Add user message to chat display
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "chart_path": None,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query with conversation history
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your request with conversation context..."):
            
            # Pass conversation history to process_query
            result = process_query(
                user_query=prompt,
                csv_path=st.session_state.csv_path,
                conversation_history=st.session_state.conversation_history,
                session_id=st.session_state.session_id
            )
            
            # Extract response
            response_text = result.get("response", "")
            chart_path = result.get("chart_path")
            
            # Display response
            if response_text and str(response_text).strip():
                st.markdown(response_text)
            else:
                st.warning(" No response was generated. Please try rephrasing your question.")
                response_text = "No response generated."
            
            # Display chart if available
            if chart_path and os.path.exists(chart_path):
                st.image(chart_path, caption="Generated Visualization", use_column_width=True)
            
            # Add assistant response to chat display
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "chart_path": chart_path,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update conversation history (stored separately for memory)
            st.session_state.conversation_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "turn_number": len([m for m in st.session_state.messages if m["role"] == "user"])
                }
            })
            
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "intent": result.get("intent"),
                    "subagent": result.get("subagent"),
                    "chart_generated": chart_path is not None
                }
            })

# Footer with Quick Actions and Memory Context
if st.session_state.file_loaded:
    st.markdown("---")
    
    # Show recent context
    if len(st.session_state.conversation_history) > 0:
        with st.expander(" Recent Conversation Context"):
            recent_history = st.session_state.conversation_history[-6:]
            for entry in recent_history:
                role_emoji = "👤" if entry["role"] == "user" else "🤖"
                st.caption(f"{role_emoji} **{entry['role'].title()}:** {entry['content'][:100]}...")
    
    '''st.markdown("###  Suggested Questions")
    
    col1, col2, col3 = st.columns(3)
    
    suggested_queries = [
        "Show monthly cost trend",
        "What are the top spending services",
        "Analyze cost anomalies"
    ]
    
    # Contextual suggestions based on conversation
    if st.session_state.conversation_history:
        last_intent = None
        for entry in reversed(st.session_state.conversation_history):
            if entry.get("metadata", {}).get("intent"):
                last_intent = entry["metadata"]["intent"]
                break
        
        if last_intent == "finops_query":
            suggested_queries = [
                "Tell me more about that",
                "Show me a visualization",
                "What caused that trend"
            ]
    
    with col1:
        if st.button(suggested_queries[0]):
            st.rerun()
    
    with col2:
        if st.button(suggested_queries[1]):
            st.rerun()
    
    with col3:
        if st.button(suggested_queries[2]):
            st.rerun()'''

# Debug Panel
with st.expander(" Debug Information"):
    st.write("**Session State:**")
    st.write(f"- File loaded: {st.session_state.file_loaded}")
    st.write(f"- CSV path: {st.session_state.csv_path}")
    st.write(f"- Total messages: {len(st.session_state.messages)}")
    st.write(f"- Conversation history entries: {len(st.session_state.conversation_history)}")
    st.write(f"- Session ID: {st.session_state.session_id}")
    st.write(f"- Conversation turns: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")

st.markdown("---")
st.caption("Built with LangGraph + Groq + Streamlit | Memory-Enabled Conversational AI")