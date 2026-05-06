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

DATA_DIR = Path(os.environ.get("ECHOS_DATA", "./qdrant_storage")).resolve()
MEMORIES_DIR = Path(os.environ.get("ECHOS_MEMORIES", "./memories")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

client = QdrantClient(path=str(DATA_DIR))
ensure_collection(client)

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
    return {"query": body.query, "results": search(client, body.query, body.top_k or 12)}


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


@app.delete("/reset")
def reset():
    client.delete_collection(COLLECTION)
    ensure_collection(client)
    return {"ok": True}


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
    res = search(client, body.query, top_k=20)
    res.sort(key=lambda r: r.get("timestamp") or 0.0)
    return {"query": body.query, "thread": res}


class ForgetBody(BaseModel):
    point_id: str
    delete_file: bool = False


@app.post("/forget")
def forget(body: ForgetBody):
    p = _get_point(body.point_id)
    payload = p.payload or {}
    client.delete(collection_name=COLLECTION, points_selector=[body.point_id])
    if body.delete_file and payload.get("path"):
        try:
            Path(payload["path"]).unlink(missing_ok=True)
        except OSError:
            pass
    return {"ok": True, "forgot": body.point_id, "file_deleted": body.delete_file}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
