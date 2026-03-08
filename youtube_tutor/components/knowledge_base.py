import streamlit as st
from utils.vector_store import get_all_videos_info, delete_video
from utils.topic_clustering import cluster_videos
from utils.youtube_loader import get_video_metadata
import os


def render_knowledge_base(collection):
    """Render the knowledge base panel showing all stored videos with clusters."""
    st.subheader("📚 Knowledge Base")

    videos = get_all_videos_info(collection)

    if not videos:
        st.info("No videos ingested yet. Add some YouTube links to get started!")
        return

    st.caption(f"**{len(videos)} video(s)** in your knowledge base")

    # Enrich with descriptions for clustering (fetch from metadata if available)
    for v in videos:
        v["description"] = ""  # Description not stored; kept blank for clustering

    # Cluster if more than 1 video
    if len(videos) > 1:
        with st.spinner("Clustering videos by topic..."):
            try:
                videos = cluster_videos(videos)
                _render_clustered_view(videos, collection)
            except Exception as e:
                st.warning(f"Clustering unavailable: {e}. Showing flat list.")
                _render_flat_view(videos, collection)
    else:
        _render_flat_view(videos, collection)


def _render_clustered_view(videos: list[dict], collection):
    """Render videos grouped by cluster."""
    clusters = {}
    for v in videos:
        cid = v.get("cluster_id", 0)
        label = v.get("cluster_label", f"Topic {cid+1}")
        if cid not in clusters:
            clusters[cid] = {"label": label, "videos": []}
        clusters[cid]["videos"].append(v)

    for cid, cluster_data in sorted(clusters.items()):
        with st.expander(
            f"🗂️ **Topic Group {cid+1}:** {cluster_data['label']} — {len(cluster_data['videos'])} video(s)",
            expanded=True,
        ):
            for v in cluster_data["videos"]:
                _render_video_card(v, collection)


def _render_flat_view(videos: list[dict], collection):
    """Render videos in a simple list."""
    for v in videos:
        _render_video_card(v, collection)


def _render_video_card(v: dict, collection):
    """Render a single video card with delete option."""
    col1, col2 = st.columns([5, 1])
    with col1:
        url = v.get("video_url", "#")
        title = v.get("video_title", "Unknown")
        vid_id = v.get("video_id", "")
        st.markdown(f"🎬 **[{title}]({url})**")
        st.caption(f"ID: `{vid_id}`")
    with col2:
        if st.button("🗑️", key=f"del_{v['video_id']}", help="Remove this video"):
            delete_video(collection, v["video_id"])
            st.success(f"Removed: {v['video_title']}")
            st.rerun()
    st.divider()
