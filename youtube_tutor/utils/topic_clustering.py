import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# ✅ Load model once globally (downloads ~80MB first time)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    return _model.encode(text).tolist()

def get_optimal_cluster_count(embeddings, max_k=6):
    n = len(embeddings)
    if n <= 2:
        return 1
    best_k, best_score = 2, -1
    for k in range(2, min(max_k, n - 1) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def cluster_videos(videos_info):
    if len(videos_info) <= 1:
        for v in videos_info:
            v["cluster_id"] = 0
            v["cluster_label"] = "General"
        return videos_info

    texts = [f"{v.get('video_title','')} {v.get('description','')[:300]}" for v in videos_info]
    
    # ✅ Local embeddings — no API call needed
    embeddings = np.array([get_embedding(t) for t in texts])
    k = get_optimal_cluster_count(embeddings, max_k=min(6, len(videos_info)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_labels = {}
    for cid in range(k):
        indices = [i for i, l in enumerate(labels) if l == cid]
        center = kmeans.cluster_centers_[cid]
        dists = [np.linalg.norm(embeddings[i] - center) for i in indices]
        closest = indices[np.argmin(dists)]
        cluster_labels[cid] = " ".join(videos_info[closest]["video_title"].split()[:4]) + "..."

    for i, v in enumerate(videos_info):
        v["cluster_id"] = int(labels[i])
        v["cluster_label"] = cluster_labels[int(labels[i])]

    return videos_info