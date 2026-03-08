import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

from utils.vector_store import get_chroma_client, get_or_create_collection
from components.video_ingestion import render_video_ingestion
from components.knowledge_base import render_knowledge_base
from components.chat_interface import render_chat_interface, render_clear_chat

st.set_page_config(
    page_title="YouTube Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎓 YouTube Tutor AI</h1>
    <p>Multi-Video Knowledge Base · Timestamp-Aware Answers · Topic Clustering</p>
</div>
""", unsafe_allow_html=True)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

@st.cache_resource
def init_db():
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return collection

try:
    collection = init_db()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=60)
    st.title("🎓 YouTube Tutor")
    st.markdown("---")

    tab_add, tab_kb = st.tabs(["➕ Add Videos", "📚 Knowledge Base"])

    with tab_add:
        render_video_ingestion(collection)

    with tab_kb:
        render_knowledge_base(collection)

    st.markdown("---")
    render_clear_chat()

    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. Paste YouTube URL in Add Videos tab
        2. Wait for ingestion to complete
        3. Ask questions in the chat
        4. Click timestamps to jump to exact video moment
        5. Add multiple videos for topic clustering
        """)

render_chat_interface(collection)