import streamlit as st
import traceback
from utils.youtube_loader import (
    extract_video_id,
    get_video_metadata,
    get_transcript_with_timestamps,
    build_chunks_with_timestamps,
)
from utils.vector_store import (
    add_video_chunks,
    is_video_already_stored,
)
from utils.qa_chain import generate_video_summary


def render_video_ingestion(collection):
    st.subheader("➕ Add YouTube Videos")

    with st.form("add_video_form", clear_on_submit=True):
        urls_input = st.text_area(
            "Paste YouTube URL(s) — one per line",
            placeholder="https://www.youtube.com/watch?v=...",
            height=120,
        )
        submitted = st.form_submit_button("📥 Ingest Video(s)", use_container_width=True)

    if submitted and urls_input.strip():
        urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
        _process_urls(urls, collection)
    elif submitted:
        st.warning("Please paste a YouTube URL first!")


def _process_urls(urls, collection):
    for idx, url in enumerate(urls):
        st.info(f"Processing {idx+1}/{len(urls)}: {url}")
        try:
            # Step 1: Extract video ID
            st.write("🔍 Extracting video ID...")
            video_id = extract_video_id(url)
            st.write(f"✅ Video ID: {video_id}")

            # Step 2: Check if already stored
            if is_video_already_stored(collection, video_id):
                st.warning(f"⚠️ Already ingested: {video_id} — skipping.")
                continue

            # Step 3: Fetch metadata
            st.write("📋 Fetching metadata...")
            metadata = get_video_metadata(video_id)
            video_title = metadata["title"]
            st.write(f"✅ Title: {video_title}")

            # Step 4: Fetch transcript
            st.write("📝 Fetching transcript...")
            segments = get_transcript_with_timestamps(video_id)
            st.write(f"✅ Got {len(segments)} transcript segments")

            # Step 5: Build chunks
            st.write("✂️ Building chunks...")
            chunks = build_chunks_with_timestamps(segments, chunk_size=5)
            st.write(f"✅ Created {len(chunks)} chunks")

            # Step 6: Store vectors
            st.write("💾 Storing in vector store...")
            add_video_chunks(collection, video_id, url, video_title, chunks)
            st.write("✅ Stored successfully!")

            # Step 7: Generate summary
            st.write("🤖 Generating summary...")
            summary = generate_video_summary(video_title, chunks)
            st.write("✅ Summary generated!")

            st.success(f"🎉 Successfully ingested: {video_title}")
            with st.expander(f"📋 Summary"):
                st.markdown(summary)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())

    st.rerun()