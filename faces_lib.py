"""Face detection + ArcFace embedding + cluster auto-merge in a separate Qdrant collection.

Design
------
- One Qdrant point per detected face (NOT per memory). A photo with 3 faces -> 3 points.
- Single named vector `embedding` (512-d ArcFace).
- Payload: {memory_id, bbox, det_score, cluster_id, label?}.
- Cluster assignment: query existing faces collection for nearest neighbor.
  If cosine >= COS_THRESHOLD -> inherit that face's cluster_id (joins cluster).
  Else -> new cluster_id (uuid).
- Labels live as redundant payload on every face in the cluster. Labeling a cluster
  iterates faces with that cluster_id and sets `label`.

Safety
------
Face detection runs only on faces. A door does not produce a face crop, so it cannot
join a labeled person cluster even if the user mislabels later. False-positive
detections produce off-distribution embeddings (cosine ~0.05 to real faces) and fall
into orphan clusters by design.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

FACES_COLLECTION = "faces"
FACE_DIM = 512
# ArcFace + cosine-normalized vectors: same person ~0.5+, different people ~0.2.
# 0.45 is a comfortable threshold; raise to 0.55 for stricter clusters.
COS_THRESHOLD = 0.45
MIN_DET_SCORE = 0.5  # InsightFace detector confidence floor
MIN_FACE_PIXELS = 40  # bbox shorter side; below this, embedding is unreliable

_NS = uuid.UUID("9f6c7e2d-3b1f-4d8e-a4c2-7e9f0a1b2c3d")


class _FaceModel:
    _app = None

    @classmethod
    def app(cls):
        if cls._app is None:
            from insightface.app import FaceAnalysis
            # buffalo_s: small, fast, sufficient. Auto-downloads to ~/.insightface/models/.
            app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            cls._app = app
        return cls._app


def ensure_faces_collection(client: QdrantClient) -> None:
    if client.collection_exists(FACES_COLLECTION):
        return
    client.create_collection(
        collection_name=FACES_COLLECTION,
        vectors_config=VectorParams(size=FACE_DIM, distance=Distance.COSINE),
    )


def _to_rgb_array(image_or_path) -> np.ndarray:
    if isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("RGB")
    else:
        img = Image.open(image_or_path).convert("RGB")
    return np.array(img)


def detect_and_embed(image_or_path) -> list[dict]:
    """Returns list of {bbox: [x1,y1,x2,y2], embedding: list[float], det_score: float}."""
    arr = _to_rgb_array(image_or_path)
    # InsightFace expects BGR.
    bgr = arr[:, :, ::-1]
    faces = _FaceModel.app().get(bgr)
    out: list[dict] = []
    for f in faces:
        det_score = float(getattr(f, "det_score", 0.0))
        if det_score < MIN_DET_SCORE:
            continue
        bbox = [float(x) for x in f.bbox]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if min(w, h) < MIN_FACE_PIXELS:
            continue
        emb = np.asarray(f.normed_embedding, dtype=np.float32)
        out.append({
            "bbox": bbox,
            "det_score": det_score,
            "embedding": emb.tolist(),
        })
    return out


def _assign_cluster(client: QdrantClient, embedding: list[float]) -> tuple[str, str | None]:
    """Returns (cluster_id, inherited_label_or_None)."""
    res = client.query_points(
        collection_name=FACES_COLLECTION,
        query=embedding,
        limit=1,
        with_payload=True,
    )
    hits = res.points
    if hits and float(hits[0].score) >= COS_THRESHOLD:
        payload = hits[0].payload or {}
        return str(payload.get("cluster_id") or uuid.uuid4()), payload.get("label")
    return str(uuid.uuid4()), None


def index_faces_for_memory(
    client: QdrantClient,
    memory_id: str,
    image_or_path,
) -> int:
    """Detect, embed, cluster-assign, upsert. Returns face count."""
    ensure_faces_collection(client)
    detected = detect_and_embed(image_or_path)
    if not detected:
        return 0

    points: list[PointStruct] = []
    for d in detected:
        cluster_id, inherited_label = _assign_cluster(client, d["embedding"])
        face_id = str(uuid.uuid4())
        payload = {
            "memory_id": str(memory_id),
            "bbox": d["bbox"],
            "det_score": d["det_score"],
            "cluster_id": cluster_id,
        }
        if inherited_label:
            payload["label"] = inherited_label
        points.append(PointStruct(id=face_id, vector=d["embedding"], payload=payload))
    client.upsert(collection_name=FACES_COLLECTION, points=points)
    return len(points)


def list_clusters(client: QdrantClient) -> list[dict]:
    """Aggregate faces by cluster_id. Returns sorted by count desc."""
    ensure_faces_collection(client)
    by_cluster: dict[str, dict] = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            payload = p.payload or {}
            cid = str(payload.get("cluster_id") or "")
            if not cid:
                continue
            entry = by_cluster.setdefault(cid, {
                "cluster_id": cid,
                "label": payload.get("label"),
                "count": 0,
                "sample_face_id": str(p.id),
                "sample_score": float(payload.get("det_score") or 0.0),
                "memory_ids": set(),
            })
            entry["count"] += 1
            entry["memory_ids"].add(payload.get("memory_id"))
            # Pick highest detection-score face as the cluster avatar.
            if float(payload.get("det_score") or 0.0) > entry["sample_score"]:
                entry["sample_face_id"] = str(p.id)
                entry["sample_score"] = float(payload.get("det_score") or 0.0)
            if payload.get("label") and not entry["label"]:
                entry["label"] = payload.get("label")
        if offset is None:
            break
    out = []
    for v in by_cluster.values():
        out.append({
            "cluster_id": v["cluster_id"],
            "label": v["label"],
            "count": v["count"],
            "sample_face_id": v["sample_face_id"],
            "memory_count": len(v["memory_ids"]),
        })
    out.sort(key=lambda x: x["count"], reverse=True)
    return out


def _merge_label_clusters(client: QdrantClient, label: str) -> tuple[int, str | None]:
    """If multiple clusters share the same label (case-insensitive), merge into the
    largest. Returns (merged_face_count, canonical_cluster_id)."""
    target_norm = label.strip().lower()
    by_cluster: dict[str, list] = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            pl = p.payload or {}
            l = (pl.get("label") or "").strip().lower()
            if l != target_norm:
                continue
            cid = pl.get("cluster_id")
            if not cid:
                continue
            by_cluster.setdefault(str(cid), []).append(p.id)
        if offset is None:
            break

    if len(by_cluster) <= 1:
        canonical = next(iter(by_cluster), None)
        return 0, canonical

    canonical = max(by_cluster.items(), key=lambda kv: len(kv[1]))[0]
    merged = 0
    for cid, face_ids in by_cluster.items():
        if cid == canonical:
            continue
        client.set_payload(
            collection_name=FACES_COLLECTION,
            payload={"cluster_id": canonical, "label": label},
            points=face_ids,
        )
        merged += len(face_ids)
    return merged, canonical


def set_cluster_label(client: QdrantClient, cluster_id: str, label: str) -> dict:
    """Apply label to every face in cluster. If another cluster already carries the
    same label (case-insensitive), merge clusters into the largest.

    Returns {labeled, merged, canonical_cluster_id}.
    """
    ensure_faces_collection(client)
    flt = Filter(must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))])
    ids: list = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            scroll_filter=flt,
            limit=1024,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        ids.extend(p.id for p in pts)
        if offset is None:
            break
    if not ids:
        return {"labeled": 0, "merged": 0, "canonical_cluster_id": cluster_id}

    client.set_payload(
        collection_name=FACES_COLLECTION,
        payload={"label": label},
        points=ids,
    )
    merged, canonical = _merge_label_clusters(client, label)
    return {
        "labeled": len(ids),
        "merged": merged,
        "canonical_cluster_id": canonical or cluster_id,
    }


def consolidate_all_labels(client: QdrantClient) -> dict:
    """Scan every distinct label and merge duplicate-label clusters.

    Useful after restoring older labeled data, or to clean up clusters labeled
    before auto-merge was deployed. Returns per-label merge counts.
    """
    ensure_faces_collection(client)
    distinct_labels: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            l = ((p.payload or {}).get("label") or "").strip()
            if l:
                distinct_labels.add(l.lower())
        if offset is None:
            break

    report: dict[str, int] = {}
    total_merged = 0
    for norm in distinct_labels:
        merged, _canonical = _merge_label_clusters(client, norm)
        if merged:
            report[norm] = merged
            total_merged += merged
    return {"labels_processed": len(distinct_labels), "total_faces_merged": total_merged, "per_label": report}


def memories_for_label(client: QdrantClient, label: str) -> list[str]:
    """Distinct memory_ids whose faces carry this label."""
    ensure_faces_collection(client)
    flt = Filter(must=[FieldCondition(key="label", match=MatchValue(value=label))])
    seen: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            scroll_filter=flt,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            mid = (p.payload or {}).get("memory_id")
            if mid:
                seen.add(str(mid))
        if offset is None:
            break
    return sorted(seen)


def list_known_labels(client: QdrantClient) -> set[str]:
    """Distinct labels currently in the faces collection, normalized lowercase."""
    ensure_faces_collection(client)
    labels: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            l = ((p.payload or {}).get("label") or "").strip().lower()
            if l:
                labels.add(l)
        if offset is None:
            break
    return labels


def memories_for_cluster(client: QdrantClient, cluster_id: str) -> list[str]:
    ensure_faces_collection(client)
    flt = Filter(must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))])
    seen: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=FACES_COLLECTION,
            scroll_filter=flt,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            mid = (p.payload or {}).get("memory_id")
            if mid:
                seen.add(str(mid))
        if offset is None:
            break
    return sorted(seen)


def get_face(client: QdrantClient, face_id: str) -> dict | None:
    pts = client.retrieve(collection_name=FACES_COLLECTION, ids=[face_id], with_payload=True)
    if not pts:
        return None
    p = pts[0]
    return {"id": str(p.id), "payload": p.payload or {}}


def crop_face_jpeg(memory_path: str, bbox: list[float], expand: float = 0.25, size: int = 256) -> bytes:
    """Read source memory image, crop face bbox with padding, return JPEG bytes."""
    img = Image.open(memory_path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = w * expand
    pad_y = h * expand
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(W, int(x2 + pad_x))
    cy2 = min(H, int(y2 + pad_y))
    crop = img.crop((cx1, cy1, cx2, cy2))
    crop.thumbnail((size, size))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def remove_faces_for_memory(client: QdrantClient, memory_id: str) -> int:
    ensure_faces_collection(client)
    flt = Filter(must=[FieldCondition(key="memory_id", match=MatchValue(value=str(memory_id)))])
    pts, _ = client.scroll(
        collection_name=FACES_COLLECTION,
        scroll_filter=flt,
        limit=1024,
        with_payload=False,
        with_vectors=False,
    )
    if not pts:
        return 0
    ids = [p.id for p in pts]
    client.delete(collection_name=FACES_COLLECTION, points_selector=ids)
    return len(ids)


def scan_all_photos(client: QdrantClient) -> dict:
    """Re-run face detection across every photo/screenshot in `memories` collection."""
    from indexer import COLLECTION as MEM_COLLECTION
    ensure_faces_collection(client)
    # Wipe existing faces collection first to avoid stale clusters.
    client.delete_collection(FACES_COLLECTION)
    ensure_faces_collection(client)

    counts = {"photos_processed": 0, "faces_found": 0, "skipped": 0}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=MEM_COLLECTION,
            limit=128,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            payload = p.payload or {}
            ptype = payload.get("type")
            path = payload.get("path")
            if ptype not in ("photo", "screenshot") or not path:
                counts["skipped"] += 1
                continue
            if not Path(path).exists():
                counts["skipped"] += 1
                continue
            try:
                n = index_faces_for_memory(client, str(p.id), path)
                counts["photos_processed"] += 1
                counts["faces_found"] += n
            except Exception as e:
                print(f"[faces] {path}: {type(e).__name__}: {e}")
                counts["skipped"] += 1
        if offset is None:
            break
    return counts
