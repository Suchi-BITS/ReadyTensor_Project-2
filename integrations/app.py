import sys
import os
import warnings

# Suppress all torch warnings more comprehensively
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings if present

ROOT = os.path.dirname(os.path.abspath(__file__))  # D:\A-Agent\integrations
PARENT = os.path.dirname(ROOT)  # D:\A-Agent

if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import concurrent.futures
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from integrations.main import process_query
except Exception as e:
    logger.error(f"Failed to import process_query: {e}")
    st.error(f"Import Error: {e}")
    st.stop()


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
        value="data/data.csv",
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
            st.error(f"❌ File not found: {default_csv_path}")
            st.session_state.file_loaded = False
    
    st.markdown("---")
    
    # File status
    if st.session_state.file_loaded:
        st.success("Data Loaded")
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
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
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
            result = None
            try:
                logger.info(f"Processing query: {prompt}")
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        process_query,
                        prompt,
                        st.session_state.csv_path,
                        st.session_state.conversation_history,
                        st.session_state.session_id
                    )
                    result = future.result(timeout=65)
                
                logger.info(f"Query processed successfully. Result: {result}")

            except concurrent.futures.TimeoutError:
                logger.error("Request timed out")
                st.error(" Request timed out. Please try again with a simpler query.")
                result = {
                    "response": "The request took too long to process. Please try again.",
                    "error": True
                }

            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                logger.error(f"Error processing query: {error_msg}\n{error_trace}")
                
                # Show detailed error in expander
                st.error(" An error occurred while processing your request.")
                with st.expander(" Error Details"):
                    st.code(error_trace)
                    st.write("**Error Message:**", error_msg)
                
                result = {
                    "response": f"I encountered an error while processing your request. Please check that:\n1. Your CSV file is properly formatted\n2. The query is clear and specific\n3. Required dependencies are installed\n\nError: {error_msg}",
                    "error": True
                }
            
            # Extract response
            if result:
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
                        "chart_generated": chart_path is not None,
                        "error": result.get("error", False)
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
                content_preview = entry['content'][:100] + "..." if len(entry['content']) > 100 else entry['content']
                st.caption(f"{role_emoji} **{entry['role'].title()}:** {content_preview}")
    

# Debug Panel
with st.expander(" Debug Information"):
    st.write("**Session State:**")
    st.write(f"- File loaded: {st.session_state.file_loaded}")
    st.write(f"- CSV path: {st.session_state.csv_path}")
    st.write(f"- Total messages: {len(st.session_state.messages)}")
    st.write(f"- Conversation history entries: {len(st.session_state.conversation_history)}")
    st.write(f"- Session ID: {st.session_state.session_id}")
    st.write(f"- Conversation turns: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")
    
    st.write("\n**System Information:**")
    st.write(f"- Python version: {sys.version}")
    st.write(f"- Working directory: {os.getcwd()}")
    st.write(f"- sys.path includes parent: {PARENT in sys.path}")

st.markdown("---")
st.caption("Built with LangGraph + OpenAI + Streamlit | Memory-Enabled Conversational AI")
