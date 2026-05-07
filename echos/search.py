"""Multi-vector search with cosine-weighted fusion across visual / audio_transcript / ocr_text.

Fusion strategy (cosine-primary, RRF-tiebreak):
- Each hit's score = weighted average of raw cosine similarities across the spaces
  it matched. Items carry only the spaces they were indexed with — a photo without
  a transcript isn't penalised for missing audio_transcript.
- Multi-modal items get a tiny RRF-style bonus (+0.02 per extra space) to break
  ties in favour of richer matches.
- Per-space cosine floors guard against noise; per-space weight lets voice memos
  compete fairly with photos (voice memos only exist in one space).
"""
from __future__ import annotations

import re

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition

from echos.faces_lib import list_known_labels, memories_for_label
from echos.indexer import COLLECTION, embed_text, embed_text_clip

PER_SPACE_LIMIT = 30
TOP_K = 12

# Per-space cosine floor: hits below this threshold are dropped before fusion.
# all-MiniLM-L6-v2 short text similarity floors around 0.3 for unrelated pairs;
# CLIP text↔image floor is lower — ~0.15 for edge-of-relevance.
SCORE_FLOOR = {"visual": 0.17, "audio_transcript": 0.25, "ocr_text": 0.25}

# Per-space weight: voice memos live only in audio_transcript and need a boost
# to compete in fusion against items present in both visual + ocr_text.
SPACE_WEIGHT = {"visual": 1.0, "audio_transcript": 1.8, "ocr_text": 1.0}

# Multi-modal bonus per extra matched space (tiny — only breaks ties).
MM_BONUS = 0.02

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{1,}")


def _query_space(
    client: QdrantClient,
    using: str,
    vector: list[float],
    limit: int = PER_SPACE_LIMIT,
    query_filter: Filter | None = None,
):
    res = client.query_points(
        collection_name=COLLECTION,
        using=using,
        query=vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
    return res.points


def _detect_label_filter(client: QdrantClient, query: str) -> tuple[Filter | None, list[str]]:
    """If query contains any known face-label token, return a filter restricting
    search to memory_ids carrying any of those labels (union). Otherwise (None, [])."""
    try:
        known = list_known_labels(client)
    except Exception:
        return None, []
    if not known:
        return None, []
    tokens = {t.lower() for t in _TOKEN_RE.findall(query)}
    matched = sorted(known & tokens)
    if not matched:
        return None, []
    mids: set[str] = set()
    for label in matched:
        for mid in memories_for_label(client, label):
            mids.add(mid)
    if not mids:
        return None, matched
    return Filter(must=[HasIdCondition(has_id=sorted(mids))]), matched


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


def search(
    client: QdrantClient,
    query: str,
    top_k: int = TOP_K,
    *,
    geo_lat: float | None = None,
    geo_lon: float | None = None,
    geo_radius_m: float | None = None,
) -> dict:
    q_clip = embed_text_clip(query)
    q_text = embed_text(query)

    label_filter, matched_labels = _detect_label_filter(client, query)
    # When face-filter narrows the universe, lift per-space limit so RRF still has
    # enough candidates to fuse across modalities.
    per_space_limit = PER_SPACE_LIMIT * 2 if label_filter else PER_SPACE_LIMIT

    visual_hits = _query_space(client, "visual", q_clip, limit=per_space_limit, query_filter=label_filter)
    audio_hits = _query_space(client, "audio_transcript", q_text, limit=per_space_limit, query_filter=label_filter)
    ocr_hits = _query_space(client, "ocr_text", q_text, limit=per_space_limit, query_filter=label_filter)

    hits_by_space = {
        "visual": visual_hits,
        "audio_transcript": audio_hits,
        "ocr_text": ocr_hits,
    }

    # Cosine-primary fusion: weighted average of raw cosine scores.
    # An item is scored only on the spaces it was indexed with — a photo
    # without transcript isn't penalised for missing audio_transcript.
    fused: dict[str, dict] = {}
    for space, hits in hits_by_space.items():
        floor = SCORE_FLOOR.get(space, 0.0)
        space_w = SPACE_WEIGHT.get(space, 1.0)
        for h in hits:
            cosine = float(h.score)
            if cosine < floor:
                continue
            pid = str(h.id)
            entry = fused.setdefault(pid, {
                "id": pid,
                "cosine_sum": 0.0,
                "cosine_weight": 0.0,
                "matched_via": [],
                "payload": h.payload,
                "raw_scores": {},
            })
            entry["cosine_sum"] += cosine * space_w
            entry["cosine_weight"] += space_w
            entry["matched_via"].append(space)
            entry["raw_scores"][space] = cosine

    for entry in fused.values():
        n_spaces = len(entry["matched_via"])
        base = entry["cosine_sum"] / entry["cosine_weight"] if entry["cosine_weight"] > 0 else 0.0
        # Tiny multi-modal bonus breaks ties toward richer matches.
        entry["score"] = base + MM_BONUS * (n_spaces - 1) if n_spaces > 1 else base

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
        if payload.get("lat") and payload.get("lon"):
            item["lat"] = payload["lat"]
            item["lon"] = payload["lon"]
        results.append(item)

    # Geo filter: post-filter by Haversine distance if geo params provided
    if geo_lat is not None and geo_lon is not None and geo_radius_m:
        import math as _math
        def _haversine_m(lat1, lon1, lat2, lon2):
            R = 6_371_000
            dlat = _math.radians(lat2 - lat1)
            dlon = _math.radians(lon2 - lon1)
            a = (_math.sin(dlat / 2) ** 2 +
                 _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2)) *
                 _math.sin(dlon / 2) ** 2)
            return R * 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1 - a))
        results = [
            r for r in results
            if r.get("lat") and r.get("lon") and
               _haversine_m(geo_lat, geo_lon, r["lat"], r["lon"]) <= geo_radius_m
        ]

    return {"results": results, "matched_labels": matched_labels}


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
        for h in hits:
            if str(h.id) == str(point_id):
                continue
            cosine = float(h.score)
            pid = str(h.id)
            entry = fused.setdefault(pid, {
                "id": pid,
                "cosine_sum": 0.0,
                "cosine_weight": 0.0,
                "payload": h.payload,
            })
            entry["cosine_sum"] += cosine
            entry["cosine_weight"] += 1.0
    for entry in fused.values():
        entry["score"] = entry["cosine_sum"] / entry["cosine_weight"] if entry["cosine_weight"] > 0 else 0.0
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:top_k]
