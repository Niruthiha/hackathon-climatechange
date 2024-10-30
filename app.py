

import streamlit as st
import os
from pprint import pprint
import chromadb
import openai
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Climate Change Knowledge Assistant",
    page_icon="🍁",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #002D62;
    }
    .assistant-message {
        background-color: #e8f4f9;
        border-left: 4px solid #c41e3a;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #002D62;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
    }
    .header-container {
        padding: 1rem;
        background-color: white;
        color: #002D62;
        margin-bottom: 2rem;
        border-bottom: 3px solid #002D62;
    }
    .logo-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
    }
    .logo-item {
        text-align: center;
        flex: 1;
    }
    .logo-item img {
        max-height: 60px;
        object-fit: contain;
    }
    .header-text {
        color: #002D62;
        margin: 0;
        text-align: center;
    }
    .source-tag {
        background-color: #002D62;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8em;
        margin-right: 0.5rem;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        text-align: center;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
    .divider {
        height: 2px;
        background-color: #002D62;
        margin: 1rem 0;
    }
    .chat-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 0 1rem;
    }
    .example-button {
        margin: 0.25rem;
    }
    .chat-input {
        max-width: 800px;
        margin: 1rem auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ''

# Function to handle example question clicks
def set_example_question(question):
    st.session_state.current_question = question

# Header with logos and branding
st.markdown("""
    <div class="header-container">
        <div class="logo-container">
            <div class="logo-item">
                <img src="https://www.canada.ca/etc/designs/canada/wet-boew/assets/sig-blk-en.svg" alt="Government of Canada">
            </div>
            <div class="logo-item">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Intergovernmental_Panel_on_Climate_Change_Logo.svg/768px-Intergovernmental_Panel_on_Climate_Change_Logo.svg.png" alt="IPCC">
            </div>
            <div class="logo-item">
                <img src="https://umanitoba.ca/sites/default/files/styles/3x2_900w/public/2020-01/university-of-manitoba-logo_1.jpg?itok=FlKi__gl" alt="University of Manitoba">
            </div>
        </div>
        <div class="divider"></div>
        <h1 class="header-text">Climate Change Knowledge Assistant</h1>
        <p class="header-text" style="font-size: 0.9em;"> Data Sources Powered by Government of Canada, IPCC, and University of Manitoba</p>
    </div>
""", unsafe_allow_html=True)

# Introduction text in chat container
with st.container():
    st.markdown("""


Ask questions about:
- Climate science and impacts
- Canadian climate policies and programs
- Environmental regulations and initiatives
- Carbon pricing and economic impacts
- Sustainability and pollution prevention
    """)

# OpenAI API key check
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

openai.api_key = openai_api_key

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="db")
    chroma_collection = chroma_client.get_or_create_collection("ipcc")
except Exception as e:
    st.error(f"⚠️ Error connecting to the database: {str(e)}")
    st.stop()

def rag(query, n_results=5):
    """RAG function with multi-source context"""
    try:
        res = chroma_collection.query(query_texts=[query], n_results=n_results)
        docs = res["documents"][0]
        joined_information = '; '.join([f'{doc}' for doc in docs])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable climate policy advisor with expertise in Canadian environmental policies, "
                    "IPCC findings, and University of Manitoba research. Provide clear, accurate answers using the "
                    "provided information. Focus on Canadian context when relevant. Make complex topics accessible "
                    "while maintaining scientific accuracy."
                )
            },
            {"role": "user", "content": f"Question: {query}\nInformation: {joined_information}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message['content'], docs
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

# Chat interface in container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'>💭 You: {message['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div style='margin-bottom: 0.5rem;'>🍁 Assistant: {message['response']}</div>
                </div>
            """, unsafe_allow_html=True)
            if message.get("sources"):
                with st.expander("📚 View Source Documents"):
                    for idx, source in enumerate(message["sources"], 1):
                        source_type = "IPCC" if "IPCC" in source else "ECCC" if "ECCC" in source else "UManitoba" if "Manitoba" in source else "Research"
                        st.markdown(f"""
                            <div class='source-box'>
                                <span class='source-tag'>{source_type}</span>
                                {source}
                            </div>
                        """, unsafe_allow_html=True)

# Query input with examples
st.markdown("<div class='chat-input'>", unsafe_allow_html=True)

# If there's a current question in the session state, use it as the default value
user_query = st.text_input(
    label="Ask your question about climate change and Canadian environmental policies",
    help="Type your question or click an example below",
    placeholder="Example: What are Canada's climate targets?",
    value=st.session_state.current_question,
    key="user_input"
)

# Example questions as buttons
example_questions = [
    "What are Canada's main climate change initiatives?",
    "How does the carbon tax system work?",
    "What are the key findings of the IPCC 2023 report?",
    "What programs exist for preventing pollution?",
]

st.markdown("### 💡 Example Questions")
cols = st.columns(2)
for idx, question in enumerate(example_questions):
    if cols[idx % 2].button(question, key=f"example_{idx}", on_click=set_example_question, args=(question,)):
        pass

# Submit button logic
if st.button("Ask", type="primary", use_container_width=True):
    if not user_query:
        st.warning("Please enter a question!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("🤔 Analyzing sources..."):
            try:
                response, sources = rag(user_query)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "response": response,
                    "sources": sources
                })
                # Clear the current question after submission
                st.session_state.current_question = ''
                st.rerun()
            except Exception as e:
                st.error(f"Sorry, I encountered an error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <img src="https://www.canada.ca/etc/designs/canada/wet-boew/assets/wmms-blk.svg" alt="Government of Canada" style="width: 150px;">
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("""
    ### About This Assistant
    
    ClimateConnect is an innovative AI-powered assistant that brings together the most authoritative and up-to-date climate change information from trusted sources:
    
    1. **Government of Canada**
       - Environment and Climate Change Canada
       - Climate Action Plans
       - National Policies
    
    2. **IPCC Reports**
       - Latest synthesis findings
       - Global climate science
       - Impact assessments
    
    3. **University of Manitoba**
       - Research contributions
       - Climate science expertise
       - Regional impact studies
    
    #### 💡 Tips for Better Results
    - Be specific in your questions
    - Ask about Canadian context
    - Request clarification if needed
    - Explore different topics
    """)

# Enhanced footer with logos
st.markdown("---")
st.markdown("""
<div style="padding: 10px; font-size: 14px; color: #444;">
    <p><strong>Official Data Sources:</strong></p>
    <ul style="list-style-type: none; padding-left: 0;">
        <li>• Environment and Climate Change Canada (ECCC)</li>
        <li>• IPCC Climate Change 2023 Synthesis Report</li>
        <li>• University of Manitoba Climate Research</li>
    </ul>
    <p>For official guidance or emergency information, please consult Environment and Climate Change Canada or your local authorities.</p>
</div>
""", unsafe_allow_html=True)
