"""Échos backend — FastAPI app exposing multi-modal memory search."""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient

from indexer import (
    AUDIO_EXTS,
    COLLECTION,
    PHOTO_EXTS,
    SCREENSHOT_PREFIX,
    VIDEO_EXTS,
    ensure_collection,
    index_folder,
    index_path,
)
from search import recommend_more_like, search
from viz import VALID_SPACES, invalidate_cache as viz_invalidate, projection as viz_projection
from faces_lib import (
    consolidate_all_labels,
    crop_face_jpeg,
    ensure_faces_collection,
    get_face,
    list_clusters,
    memories_for_cluster,
    memories_for_label,
    remove_faces_for_memory,
    scan_all_photos,
    set_cluster_label,
)

DATA_DIR = Path(os.environ.get("ECHOS_DATA", "./qdrant_storage")).resolve()
MEMORIES_DIR = Path(os.environ.get("ECHOS_MEMORIES", "./memories")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

client = QdrantClient(path=str(DATA_DIR))
ensure_collection(client)
ensure_faces_collection(client)

app = FastAPI(title="Échos", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    idx = Path("static/index.html")
    if idx.exists():
        return FileResponse(idx)
    return {"name": "Échos", "endpoints": ["/search", "/index-folder", "/stats"]}


class SearchBody(BaseModel):
    query: str
    top_k: int | None = 12


@app.post("/search")
def search_endpoint(body: SearchBody):
    if not body.query.strip():
        raise HTTPException(400, "empty query")
    out = search(client, body.query, body.top_k or 12)
    return {"query": body.query, "results": out["results"], "matched_labels": out["matched_labels"]}


class IndexBody(BaseModel):
    folder: str | None = None
    reset: bool = False


@app.post("/index-folder")
def index_folder_endpoint(body: IndexBody):
    folder = Path(body.folder).resolve() if body.folder else MEMORIES_DIR
    if not folder.exists():
        raise HTTPException(404, f"folder not found: {folder}")
    if body.reset:
        client.delete_collection(COLLECTION)
        ensure_collection(client)
    t0 = time.time()
    counts = index_folder(client, folder)
    viz_invalidate()
    return {"folder": str(folder), "counts": counts, "took_seconds": round(time.time() - t0, 2)}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), kind: str | None = None):
    """Accept one or more files from a phone/browser, save into ./memories/, index immediately.

    `kind` (optional): "screenshot" forces screenshot indexing regardless of filename.
    Otherwise extension + filename prefix decide.
    """
    saved: list[dict] = []
    for f in files:
        name = f.filename or "upload"
        suffix = Path(name).suffix.lower()
        # Reject unsupported types upfront so /memories doesn't fill with junk.
        if suffix not in PHOTO_EXTS | VIDEO_EXTS | AUDIO_EXTS:
            saved.append({"name": name, "skipped": "unsupported extension"})
            continue
        # Avoid clobber: timestamp-prefix the name.
        ts = int(time.time() * 1000)
        prefix = SCREENSHOT_PREFIX if kind == "screenshot" else ""
        out = MEMORIES_DIR / f"{prefix}{ts}_{Path(name).name}"
        with out.open("wb") as fh:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        try:
            pid = index_path(client, out)
        except Exception as e:
            saved.append({"name": name, "error": f"{type(e).__name__}: {e}"})
            continue
        saved.append({"name": name, "saved_as": out.name, "point_id": pid})
    viz_invalidate()
    return {"uploaded": saved}


@app.get("/stats")
def stats():
    info = client.get_collection(COLLECTION)
    total = info.points_count or 0
    by_type: dict[str, int] = {}
    last_ts = 0.0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            t = (p.payload or {}).get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            ts = (p.payload or {}).get("timestamp") or 0.0
            if ts > last_ts:
                last_ts = ts
        if offset is None:
            break
    return {"total": total, "by_type": by_type, "last_indexed": last_ts}


def _get_point(point_id: str):
    pts = client.retrieve(collection_name=COLLECTION, ids=[point_id], with_payload=True)
    if not pts:
        raise HTTPException(404, "not found")
    return pts[0]


@app.get("/thumbnail/{point_id}")
def thumbnail(point_id: str):
    p = _get_point(point_id)
    payload = p.payload or {}
    path = payload.get("path")
    ptype = payload.get("type")
    if not path or not Path(path).exists():
        raise HTTPException(404, "media missing on disk")

    img: Image.Image | None = None
    if ptype in ("photo", "screenshot"):
        img = Image.open(path).convert("RGB")
    elif ptype == "video":
        import cv2
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
    if img is None:
        img = Image.new("RGB", (200, 200), color=(40, 40, 50))

    img.thumbnail((200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"point_id": point_id, "data_url": f"data:image/jpeg;base64,{b64}"}


@app.get("/media/{point_id}")
def media(point_id: str):
    p = _get_point(point_id)
    payload = p.payload or {}
    path = payload.get("path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "media missing on disk")
    suffix = Path(path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
    }
    return FileResponse(path, media_type=media_types.get(suffix, "application/octet-stream"))


@app.post("/debug/space-search")
def debug_space_search(body: dict):
    """Hit one named vector space directly."""
    from indexer import embed_text, embed_text_clip
    space = body.get("space", "audio_transcript")
    q = body["query"]
    vec = embed_text_clip(q) if space == "visual" else embed_text(q)
    res = client.query_points(
        collection_name=COLLECTION,
        using=space,
        query=vec,
        limit=10,
        with_payload=True,
    )
    return [{"id": str(h.id), "score": float(h.score),
             "type": (h.payload or {}).get("type"),
             "path": ((h.payload or {}).get("path") or "").split("\\")[-1].split("/")[-1],
             "transcript": ((h.payload or {}).get("transcript") or "")[:80]} for h in res.points]


@app.get("/debug/vectors")
def debug_vectors():
    """Dump per-point: type, payload keys, vector names."""
    out: list[dict] = []
    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=128,
        with_payload=True,
        with_vectors=True,
    )
    for p in points:
        v = p.vector or {}
        if isinstance(v, dict):
            vector_keys = sorted(v.keys())
        else:
            vector_keys = ["<unnamed>"]
        out.append({
            "id": str(p.id),
            "type": (p.payload or {}).get("type"),
            "path": ((p.payload or {}).get("path") or "").split("\\")[-1].split("/")[-1],
            "transcript_len": len((p.payload or {}).get("transcript") or ""),
            "ocr_len": len((p.payload or {}).get("ocr_text") or ""),
            "vector_keys": vector_keys,
        })
    return out


@app.get("/library")
def library(limit: int = 60, before: float | None = None):
    """Newest-first list of all memories. Cursor: `before` = timestamp upper bound (exclusive)."""
    all_points = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION,
            limit=512,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        all_points.extend(pts)
        if offset is None:
            break
    all_points.sort(key=lambda p: (p.payload or {}).get("timestamp") or 0.0, reverse=True)
    if before is not None:
        all_points = [p for p in all_points if ((p.payload or {}).get("timestamp") or 0.0) < before]
    page = all_points[:limit]
    items = []
    for p in page:
        pl = p.payload or {}
        items.append({
            "id": str(p.id),
            "type": pl.get("type"),
            "timestamp": pl.get("timestamp"),
            "transcript": (pl.get("transcript") or "")[:140],
            "ocr_text": (pl.get("ocr_text") or "")[:140],
            "best_moment_seconds": pl.get("best_moment_seconds"),
            "matched_via": [],
        })
    next_before = items[-1]["timestamp"] if len(page) == limit and len(all_points) > limit else None
    return {"items": items, "next_before": next_before, "total": len(all_points) if before is None else None}


@app.delete("/reset")
def reset():
    client.delete_collection(COLLECTION)
    ensure_collection(client)
    viz_invalidate()
    return {"ok": True}


@app.get("/viz/projection")
def viz_projection_endpoint(space: str = "visual", neighbors: int = 3):
    if space not in VALID_SPACES:
        raise HTTPException(400, f"space must be one of {VALID_SPACES}")
    return viz_projection(client, space=space, neighbors=neighbors)


@app.post("/resurface")
def resurface():
    """Pick most recent item, find similar items >=365 days older, return up to 3."""
    points, _ = client.scroll(collection_name=COLLECTION, limit=512, with_payload=True, with_vectors=False)
    if not points:
        return {"seed": None, "results": []}
    seed = max(points, key=lambda p: (p.payload or {}).get("timestamp") or 0.0)
    seed_id = str(seed.id)
    seed_ts = (seed.payload or {}).get("timestamp") or 0.0
    cutoff = seed_ts - 365 * 24 * 3600
    similar = recommend_more_like(client, seed_id, top_k=20)
    older = [s for s in similar if ((s["payload"] or {}).get("timestamp") or 0.0) <= cutoff][:3]
    return {"seed": {"id": seed_id, "payload": seed.payload}, "results": older}


class ThreadBody(BaseModel):
    query: str


@app.post("/thread")
def thread(body: ThreadBody):
    """Group results for a query in chronological order."""
    out = search(client, body.query, top_k=20)
    items = out["results"]
    items.sort(key=lambda r: r.get("timestamp") or 0.0)
    return {"query": body.query, "thread": items, "matched_labels": out["matched_labels"]}


class ForgetBody(BaseModel):
    point_id: str
    delete_file: bool = False


@app.post("/forget")
def forget(body: ForgetBody):
    p = _get_point(body.point_id)
    payload = p.payload or {}
    client.delete(collection_name=COLLECTION, points_selector=[body.point_id])
    remove_faces_for_memory(client, body.point_id)
    if body.delete_file and payload.get("path"):
        try:
            Path(payload["path"]).unlink(missing_ok=True)
        except OSError:
            pass
    viz_invalidate()
    return {"ok": True, "forgot": body.point_id, "file_deleted": body.delete_file}


@app.post("/faces/scan")
def faces_scan_endpoint():
    """Re-detect faces across every photo/screenshot. Wipes old face clusters first."""
    return scan_all_photos(client)


@app.get("/faces/clusters")
def faces_clusters_endpoint():
    return {"clusters": list_clusters(client)}


@app.post("/faces/consolidate")
def faces_consolidate_endpoint():
    """Merge any duplicate-label clusters across the whole collection."""
    return consolidate_all_labels(client)


class FaceLabelBody(BaseModel):
    cluster_id: str
    label: str


@app.post("/faces/label")
def faces_label_endpoint(body: FaceLabelBody):
    label = body.label.strip()
    if not label:
        raise HTTPException(400, "label is empty")
    result = set_cluster_label(client, body.cluster_id, label)
    return {
        "ok": True,
        "cluster_id": body.cluster_id,
        "canonical_cluster_id": result["canonical_cluster_id"],
        "label": label,
        "faces_updated": result["labeled"],
        "merged_faces": result["merged"],
    }


@app.get("/faces/avatar/{face_id}")
def faces_avatar_endpoint(face_id: str):
    from fastapi.responses import Response
    f = get_face(client, face_id)
    if not f:
        raise HTTPException(404, "face not found")
    payload = f["payload"]
    memory_id = payload.get("memory_id")
    if not memory_id:
        raise HTTPException(404, "no memory_id on face")
    mem = client.retrieve(collection_name=COLLECTION, ids=[memory_id], with_payload=True)
    if not mem:
        raise HTTPException(404, "memory not found")
    src_path = (mem[0].payload or {}).get("path")
    if not src_path or not Path(src_path).exists():
        raise HTTPException(404, "source image missing")
    jpeg = crop_face_jpeg(src_path, payload["bbox"])
    return Response(content=jpeg, media_type="image/jpeg")


def _resolve_memories(memory_ids: list[str]) -> list[dict]:
    if not memory_ids:
        return []
    pts = client.retrieve(collection_name=COLLECTION, ids=memory_ids, with_payload=True)
    items = []
    for p in pts:
        pl = p.payload or {}
        items.append({
            "id": str(p.id),
            "type": pl.get("type"),
            "timestamp": pl.get("timestamp"),
            "transcript": (pl.get("transcript") or "")[:140],
            "ocr_text": (pl.get("ocr_text") or "")[:140],
            "best_moment_seconds": pl.get("best_moment_seconds"),
            "matched_via": [],
        })
    items.sort(key=lambda x: x["timestamp"] or 0.0, reverse=True)
    return items


@app.get("/faces/by-cluster/{cluster_id}")
def faces_by_cluster_endpoint(cluster_id: str):
    return {"cluster_id": cluster_id, "items": _resolve_memories(memories_for_cluster(client, cluster_id))}


@app.get("/faces/by-label/{label}")
def faces_by_label_endpoint(label: str):
    return {"label": label, "items": _resolve_memories(memories_for_label(client, label))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
