# integrations/app.py
import streamlit as st
import os
from main import process_query

# --- Streamlit App Setup ---
st.set_page_config(page_title="FinOps Agentic AI", layout="wide")

# --- Initialize Session State for Multi-Turn Conversation ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "csv_path" not in st.session_state:
    st.session_state.csv_path = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# --- Header ---
st.title("💰 FinOps Agentic AI System")
st.markdown("Ask questions about cloud spend, trends, or usage. The system maintains conversation context.")

# --- Sidebar: Data File Configuration ---
with st.sidebar:
    st.header("📂 Data Configuration")
    
    # Option to specify CSV file path directly in code
    default_csv_path = st.text_input(
        "CSV File Path",
        value="data/sample_data.csv",
        help="Enter the path to your FinOps CSV file"
    )
    
    if st.button("Load Data File"):
        if os.path.exists(default_csv_path):
            st.session_state.csv_path = default_csv_path
            st.session_state.file_loaded = True
            st.success(f"✅ File loaded: {default_csv_path}")
            
            # Add system message about file load
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📁 Data file loaded successfully from `{default_csv_path}`. You can now ask me questions about your cloud spending data!",
                "chart_path": None
            })
        else:
            st.error(f"❌ File not found: {default_csv_path}")
            st.session_state.file_loaded = False
    
    st.markdown("---")
    
    # File status
    if st.session_state.file_loaded:
        st.success("📊 Data Loaded")
        st.info(f"**File:** {os.path.basename(st.session_state.csv_path)}")
    else:
        st.warning("⚠️ No data loaded")
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    # Show conversation stats
    if st.session_state.messages:
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Your Questions", user_msgs)

# --- Main Chat Interface ---

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
            st.warning("⚠️ Please load a data file first using the sidebar.")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "chart_path": None
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query and generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your request..."):
            # Process the query with the loaded CSV
            result = process_query(prompt, st.session_state.csv_path)
            
            # Extract response
            response_text = result.get("response", "")
            chart_path = result.get("chart_path")
            
            # Display response
            if response_text and str(response_text).strip():
                st.markdown(response_text)
            else:
                st.warning("⚠️ No response was generated. Please try rephrasing your question.")
            
            # Display chart if available
            if chart_path and os.path.exists(chart_path):
                st.image(chart_path, caption="Generated Visualization", use_column_width=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text if response_text else "Unable to generate response.",
                "chart_path": chart_path
            })




st.markdown("---")
st.caption("Built with 🧠 LangGraph + Groq + Streamlit | Multi-Turn Conversational AI")