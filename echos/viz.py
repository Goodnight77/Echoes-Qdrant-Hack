"""3D vector galaxy projection: UMAP per named-vector space + k-NN edges."""
from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient

from echos.indexer import COLLECTION

VALID_SPACES = ("visual", "audio_transcript", "ocr_text")

_cache: dict[tuple, dict] = {}


def invalidate_cache() -> None:
    _cache.clear()


def _scroll_all_with_vectors(client: QdrantClient) -> list:
    out: list = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        out.extend(pts)
        if offset is None:
            break
    return out


def _knn_edges_local(matrix: np.ndarray, ids: list[str], k: int) -> list[list[str]]:
    """Cosine top-k neighbors per row (excluding self)."""
    if len(ids) <= 1:
        return [[] for _ in ids]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    eff_k = min(k, len(ids) - 1)
    top = np.argpartition(-sim, eff_k, axis=1)[:, :eff_k]
    out: list[list[str]] = []
    for i, row in enumerate(top):
        ordered = row[np.argsort(-sim[i, row])]
        out.append([ids[j] for j in ordered])
    return out


def projection(client: QdrantClient, space: str = "visual", neighbors: int = 3) -> dict:
    if space not in VALID_SPACES:
        raise ValueError(f"invalid space: {space}")

    points = _scroll_all_with_vectors(client)

    rows: list[tuple[str, np.ndarray, dict]] = []
    for p in points:
        vec_obj = p.vector or {}
        if not isinstance(vec_obj, dict):
            continue
        v = vec_obj.get(space)
        if v is None:
            continue
        rows.append((str(p.id), np.asarray(v, dtype=np.float32), p.payload or {}))

    n = len(rows)
    cache_key = (space, neighbors, n, tuple(r[0] for r in rows))
    if cache_key in _cache:
        return _cache[cache_key]

    if n == 0:
        result = {"space": space, "items": [], "total": 0}
        _cache[cache_key] = result
        return result

    matrix = np.stack([r[1] for r in rows])

    if n < 4:
        # UMAP fails on tiny n; lay out on a circle in xy with z=0.
        coords = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            ang = 2 * np.pi * i / max(n, 1)
            coords[i, 0] = np.cos(ang)
            coords[i, 1] = np.sin(ang)
    else:
        from umap import UMAP
        n_neighbors = max(2, min(15, n - 1))
        reducer = UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.05,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(matrix).astype(np.float32)

    # Center + scale to fit roughly [-5, 5] cube.
    coords -= coords.mean(axis=0, keepdims=True)
    span = float(np.max(np.abs(coords))) or 1.0
    coords *= 5.0 / span

    edges = _knn_edges_local(matrix, [r[0] for r in rows], neighbors)

    items = []
    for (pid, _vec, payload), xyz, nbrs in zip(rows, coords, edges):
        items.append({
            "id": pid,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
            "type": payload.get("type"),
            "timestamp": payload.get("timestamp"),
            "transcript": (payload.get("transcript") or "")[:140],
            "ocr_text": (payload.get("ocr_text") or "")[:140],
            "best_moment_seconds": payload.get("best_moment_seconds"),
            "neighbors": nbrs,
        })

    result = {"space": space, "items": items, "total": n}
    _cache[cache_key] = result
    return result
