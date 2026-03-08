import streamlit as st
from utils.vector_store import query_collection, get_all_video_ids
from utils.qa_chain import answer_question


def render_chat_interface(collection):
    """Render the main chat interface."""
    st.subheader("💬 Ask Questions")

    # Check if any videos are stored
    video_ids = get_all_video_ids(collection)
    if not video_ids:
        st.warning("⚠️ No videos in the knowledge base yet. Please add YouTube videos first.")
        return

    st.caption(f"Answering from **{len(video_ids)} video(s)** in your knowledge base")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    _render_chat_history()

    # Chat input
    if question := st.chat_input("Ask anything about the videos..."):
        _handle_question(question, collection)


def _render_chat_history():
    """Display existing chat messages."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show source cards if available
            if message["role"] == "assistant" and "sources" in message:
                _render_source_cards(message["sources"])


def _handle_question(question: str, collection):
    """Process a user question and generate answer."""
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Add to history
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Retrieve relevant chunks
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                results = query_collection(collection, question, n_results=5)

                if not results:
                    answer = "I couldn't find relevant content in the videos. Try rephrasing your question."
                    sources = []
                else:
                    # Generate answer
                    answer = answer_question(
                        question=question,
                        retrieved_results=results,
                        chat_history=st.session_state.chat_history[:-1],  # exclude current question
                    )
                    sources = results

                st.markdown(answer)

                # Show source cards
                if sources:
                    _render_source_cards(sources)

                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"❌ Error generating answer: {e}"
                st.error(error_msg)


def _render_source_cards(sources: list[dict]):
    """Render timestamp source cards below an answer."""
    if not sources:
        return

    # Deduplicate by video_id + timestamp
    seen = set()
    unique_sources = []
    for s in sources:
        key = f"{s['video_id']}_{s['timestamp_str']}"
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    st.markdown("---")
    st.caption("📍 **Jump to Sources:**")

    cols = st.columns(min(len(unique_sources), 3))
    for i, source in enumerate(unique_sources[:3]):
        timestamp_url = f"{source['video_url']}&t={int(source['start_seconds'])}"
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 0.8em;
                    background: #1e1e1e;
                ">
                    🎬 <b>{source['video_title'][:40]}...</b><br>
                    ⏱️ <a href="{timestamp_url}" target="_blank">Watch at {source['timestamp_str']}</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_clear_chat():
    """Render a button to clear chat history."""
    if st.session_state.get("chat_history"):
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
