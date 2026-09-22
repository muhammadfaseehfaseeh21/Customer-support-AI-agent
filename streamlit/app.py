import os
import streamlit as st
from agent import run_support_agent

st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling: Vibrant Dark Gradient Background + Glassmorphic Cards
st.markdown(
    """
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #1a103c 40%, #2b0840 100%);
        color: #FFFFFF;
    }
    
    /* Title Styling */
    .main-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8E53, #FF0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        text-align: center;
        color: #B0B0D0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Input Field & Buttons */
    .stTextInput > div > div > input {
        background-color: #1A1A2E !important;
        color: #FFFFFF !important;
        border: 1px solid #FF4B4B !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #FF4B4B, #FF0080) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        transition: transform 0.2s ease;
    }

    .stButton > button:hover {
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>⚡ Customer Support AI Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Single-Agent CrewAI Assistant powered by vector chunking & <code>openai/gpt-oss-120b</code></p>", unsafe_allow_html=True)

# Sidebar for API Key Management (Secure & Secret)
with st.sidebar:
    st.header("🔑 API Credentials")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Secrets are safely loaded from st.secrets on Streamlit Cloud or manually entered here."
    )
    
    # Check Streamlit Secrets first
    api_key = st.secrets.get("OPENAI_API_KEY", api_key_input)
    
    if api_key:
        st.success("API Key configured!", icon="✅")
    else:
        st.warning("Please enter your API Key to proceed.", icon="⚠️")

    st.markdown("---")
    st.markdown("### 📋 Sample Queries")
    st.code("What is the status of CUST-1001?")
    st.code("Find order date and address for John Doe.")
    st.code("Where is Sarah's order being delivered?")

# Main Interface
user_query = st.text_input("How can I assist you with your order today?", placeholder="e.g. Check order status for CUST-1001...")

if st.button("Ask Assistant"):
    if not api_key:
        st.error("Please provide a valid OpenAI API key in the sidebar or Streamlit Secrets.")
    elif not user_query.strip():
        st.warning("Please type a valid query.")
    else:
        with st.spinner("Agent searching knowledge chunks and processing query..."):
            try:
                response = run_support_agent(user_query=user_query, api_key=api_key)
                st.markdown("### 💬 Support Response")
                st.info(response)
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
