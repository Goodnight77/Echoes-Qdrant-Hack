"""Multi-vector search with Reciprocal Rank Fusion across visual / audio_transcript / ocr_text."""
from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient

from indexer import COLLECTION, embed_text, embed_text_clip

RRF_K = 60
PER_SPACE_LIMIT = 20
TOP_K = 12

# Per-space cosine floor: hits below this threshold are dropped before fusion.
# all-MiniLM-L6-v2 short text similarity floors around 0.3 for unrelated pairs;
# CLIP text↔image floor around 0.18.
SCORE_FLOOR = {"visual": 0.20, "audio_transcript": 0.30, "ocr_text": 0.30}

# Weighted RRF: voice memos live only in audio_transcript and need a small boost
# to compete in fusion against items present in both visual + ocr_text.
SPACE_WEIGHT = {"visual": 1.0, "audio_transcript": 1.6, "ocr_text": 1.0}


def _query_space(client: QdrantClient, using: str, vector: list[float], limit: int = PER_SPACE_LIMIT):
    res = client.query_points(
        collection_name=COLLECTION,
        using=using,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return res.points


def _best_video_moment(query_visual: list[float], keyframes: list[dict]) -> tuple[float, float] | None:
    if not keyframes:
        return None
    qv = np.array(query_visual, dtype=np.float32)
    qn = np.linalg.norm(qv) or 1.0
    best_ts = 0.0
    best_score = -1.0
    for kf in keyframes:
        v = np.array(kf["vec"], dtype=np.float32)
        vn = np.linalg.norm(v) or 1.0
        s = float(np.dot(qv, v) / (qn * vn))
        if s > best_score:
            best_score = s
            best_ts = float(kf["t"])
    return best_ts, best_score


def search(client: QdrantClient, query: str, top_k: int = TOP_K) -> list[dict]:
    q_clip = embed_text_clip(query)
    q_text = embed_text(query)

    visual_hits = _query_space(client, "visual", q_clip)
    audio_hits = _query_space(client, "audio_transcript", q_text)
    ocr_hits = _query_space(client, "ocr_text", q_text)

    hits_by_space = {
        "visual": visual_hits,
        "audio_transcript": audio_hits,
        "ocr_text": ocr_hits,
    }

    fused: dict[str, dict] = {}
    for space, hits in hits_by_space.items():
        floor = SCORE_FLOOR.get(space, 0.0)
        kept = [h for h in hits if float(h.score) >= floor]
        for rank, h in enumerate(kept):
            pid = str(h.id)
            entry = fused.setdefault(pid, {
                "id": pid,
                "score": 0.0,
                "matched_via": [],
                "payload": h.payload,
                "raw_scores": {},
            })
            entry["score"] += SPACE_WEIGHT.get(space, 1.0) / (RRF_K + rank + 1)
            entry["matched_via"].append(space)
            entry["raw_scores"][space] = float(h.score)

    ranked = sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:top_k]

    results = []
    for r in ranked:
        payload = r["payload"] or {}
        item = {
            "id": r["id"],
            "score": r["score"],
            "matched_via": sorted(set(r["matched_via"])),
            "type": payload.get("type"),
            "path": payload.get("path"),
            "timestamp": payload.get("timestamp"),
            "transcript": payload.get("transcript"),
            "ocr_text": payload.get("ocr_text"),
            "raw_scores": r["raw_scores"],
        }
        if payload.get("type") == "video":
            best = _best_video_moment(q_clip, payload.get("keyframes") or [])
            if best:
                item["best_moment_seconds"] = round(best[0], 2)
        results.append(item)
    return results


def recommend_more_like(client: QdrantClient, point_id: str, top_k: int = 12) -> list[dict]:
    point = client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id],
        with_vectors=True,
        with_payload=True,
    )
    if not point:
        return []
    p = point[0]
    vectors = p.vector or {}
    fused: dict[str, dict] = {}
    for space in ("visual", "audio_transcript", "ocr_text"):
        if space not in vectors:
            continue
        hits = _query_space(client, space, vectors[space], limit=PER_SPACE_LIMIT)
        for rank, h in enumerate(hits):
            if str(h.id) == str(point_id):
                continue
            pid = str(h.id)
            entry = fused.setdefault(pid, {"id": pid, "score": 0.0, "payload": h.payload})
            entry["score"] += 1.0 / (RRF_K + rank + 1)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:top_k]
