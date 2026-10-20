"""
Question Clusterer
==================
Clusters exam questions by semantic similarity using agglomerative clustering.
Persists clusters to data/index/exam_clusters.json.
KEY NAMES: "most_frequent" is used consistently here and in the UI.
"""
from __future__ import annotations

import json
import logging
import shutil
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# In-memory cache
_clusters: list[dict] = []
_loaded: bool = False


def cluster_questions(questions: list[dict]) -> list[dict]:
    """
    Cluster questions, persist results, update cache.
    Returns list of cluster dicts.
    """
    global _clusters

    if not questions:
        _clusters = []
        _persist([])
        return []

    if len(questions) == 1:
        _clusters = [_make_cluster(0, questions)]
        _persist(_clusters)
        return _clusters

    from core.retrieval.embedder import embed_texts
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances
    from core.config import settings

    texts = [q["question_text"] for q in questions]
    logger.info(f"Embedding {len(texts)} questions…")
    embeddings = embed_texts(texts)

    dist_matrix = cosine_distances(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=settings.cluster_distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    groups: dict[int, list[dict]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(questions[i])

    raw = [_make_cluster(cid, grp) for cid, grp in sorted(groups.items())]
    raw.sort(key=lambda c: c["frequency"], reverse=True)
    for i, c in enumerate(raw):
        c["cluster_id"] = i

    _clusters = raw
    _persist(_clusters)

    logger.info(
        f"Clustered {len(questions)} Qs into {len(_clusters)} clusters. "
        f"Top freq: {_clusters[0]['frequency'] if _clusters else 0}"
    )
    return _clusters


def _make_cluster(cluster_id: int, questions: list[dict]) -> dict:
    representative = max(questions, key=lambda q: len(q["question_text"]))
    years = sorted({q["year"] for q in questions if q.get("year")})
    papers = sorted({q["source_document"] for q in questions if q.get("source_document")})
    return {
        "cluster_id": cluster_id,
        "representative_question": representative["question_text"],
        "questions": questions,
        "frequency": len(set(q.get("source_document", "") for q in questions)),
        "total_occurrences": len(questions),
        "years": years,
        "papers": papers,
    }


def find_matching_cluster(query: str, threshold: float = 0.4) -> Optional[dict]:
    _ensure_loaded()
    if not _clusters:
        return None

    from core.retrieval.embedder import embed_query, embed_texts
    q_emb = embed_query(query)
    rep_texts = [c["representative_question"] for c in _clusters]
    rep_embs = embed_texts(rep_texts)

    sims = rep_embs @ q_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= threshold:
        result = dict(_clusters[best_idx])
        result["match_similarity"] = round(best_sim, 4)
        return result
    return None


def get_all_clusters() -> list[dict]:
    _ensure_loaded()
    return list(_clusters)


def get_frequency_summary() -> dict:
    """
    Returns summary dict. KEY: "most_frequent" (not "top_clusters").
    """
    _ensure_loaded()
    if not _clusters:
        return {
            "total_questions": 0,
            "total_clusters": 0,
            "most_frequent": [],
            "year_distribution": {},
        }

    total_q = sum(c["total_occurrences"] for c in _clusters)
    year_dist: dict[str, int] = {}
    for c in _clusters:
        for yr in c["years"]:
            year_dist[yr] = year_dist.get(yr, 0) + c["total_occurrences"]

    return {
        "total_questions": total_q,
        "total_clusters": len(_clusters),
        "most_frequent": _clusters[:10],           # standard key
        "top_clusters": _clusters[:10],            # backward-compatible key
        "year_distribution": dict(sorted(year_dist.items())),
    }


def load_clusters() -> None:
    """Load persisted clusters from disk on app startup."""
    global _clusters, _loaded
    from core.config import settings
    path = settings.clusters_path
    if path.exists():
        try:
            _clusters = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"Loaded {len(_clusters)} clusters from {path}")
        except Exception as e:
            logger.error(f"Failed to load clusters: {e}")
            _clusters = []
    else:
        _clusters = []
    _loaded = True


def _ensure_loaded():
    global _loaded
    if not _loaded:
        load_clusters()


def _persist(clusters: list[dict]) -> None:
    from core.config import settings
    path = settings.clusters_path
    try:
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(tmp), str(path))
        logger.debug(f"Clusters persisted: {len(clusters)} clusters → {path}")
    except Exception as e:
        logger.error(f"Failed to persist clusters: {e}")
