import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Load embedding model once globally ─────────────────────────────────────
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Storage paths ───────────────────────────────────────────────────────────
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore")
INDEX_PATH = os.path.join(BASE_PATH, "index.npy")
META_PATH  = os.path.join(BASE_PATH, "metadata.json")

os.makedirs(BASE_PATH, exist_ok=True)


def _load_store():
    """Load existing vectors and metadata from disk."""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        vectors = np.load(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return vectors, metadata
    return np.array([]), []


def _save_store(vectors, metadata):
    """Save vectors and metadata to disk."""
    np.save(INDEX_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def add_video_chunks(collection, video_id, video_url, video_title, chunks):
    """Embed and store chunks for a video."""
    vectors, metadata = _load_store()

    texts = [c["text"] for c in chunks]
    new_vectors = _model.encode(texts, show_progress_bar=False)

    new_meta = []
    for i, chunk in enumerate(chunks):
        new_meta.append({
            "video_id": video_id,
            "video_url": video_url,
            "video_title": video_title,
            "timestamp_str": chunk["timestamp_str"],
            "start_seconds": chunk["start_seconds"],
            "text": chunk["text"],
        })

    if len(vectors) == 0:
        vectors = new_vectors
    else:
        vectors = np.vstack([vectors, new_vectors])

    metadata.extend(new_meta)
    _save_store(vectors, metadata)


def is_video_already_stored(collection, video_id):
    """Check if a video is already stored."""
    _, metadata = _load_store()
    return any(m["video_id"] == video_id for m in metadata)


def query_collection(collection, query, n_results=5):
    """Find top matching chunks for a query."""
    vectors, metadata = _load_store()
    if len(vectors) == 0:
        return []

    query_vec = _model.encode([query])
    
    # Cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / (norms + 1e-10)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    scores = (normed @ query_norm.T).flatten()

    top_indices = np.argsort(scores)[::-1][:n_results]

    results = []
    for i in top_indices:
        m = metadata[i]
        results.append({
            "text": m["text"],
            "video_title": m.get("video_title", "Unknown"),
            "video_url": m.get("video_url", ""),
            "video_id": m.get("video_id", ""),
            "timestamp_str": m.get("timestamp_str", "00:00"),
            "start_seconds": m.get("start_seconds", 0),
            "distance": float(1 - scores[i]),
        })
    return results


def get_all_video_ids(collection):
    """Return all unique video IDs."""
    _, metadata = _load_store()
    return list({m["video_id"] for m in metadata if "video_id" in m})


def get_all_videos_info(collection):
    """Return info for all stored videos."""
    _, metadata = _load_store()
    seen = {}
    for m in metadata:
        vid = m.get("video_id")
        if vid and vid not in seen:
            seen[vid] = {
                "video_id": vid,
                "video_title": m.get("video_title", "Unknown"),
                "video_url": m.get("video_url", ""),
            }
    return list(seen.values())


def delete_video(collection, video_id):
    """Remove all chunks for a video."""
    vectors, metadata = _load_store()
    keep = [i for i, m in enumerate(metadata) if m["video_id"] != video_id]
    if keep:
        new_vectors = vectors[keep]
        new_meta = [metadata[i] for i in keep]
    else:
        new_vectors = np.array([])
        new_meta = []
    _save_store(new_vectors, new_meta)


def get_chroma_client():
    """Dummy function — kept for compatibility with app.py."""
    return None


def get_or_create_collection(client, collection_name="youtube_tutor"):
    """Dummy function — kept for compatibility with app.py."""
    return None