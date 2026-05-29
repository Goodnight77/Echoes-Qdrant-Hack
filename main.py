"""Échos backend — FastAPI app exposing multi-modal memory search."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import time
from pathlib import Path

# Load .env early so any later import (e.g. groq_labels reading GROQ_API_KEY at
# module level) sees the keys. Quiet fallback if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Thumbnail cache: key = SHA1(path|mtime|size), value = base64 data URL.
# Caps at 500 entries; LRU-ish eviction on insert.
_thumbnail_cache: dict[str, str] = {}
_thumbnail_cache_max = 500

from echos.indexer import (
    AUDIO_EXTS,
    COLLECTION,
    PHOTO_EXTS,
    SCREENSHOT_PREFIX,
    VIDEO_EXTS,
    VOICE_COLLECTION,
    embed_text,
    ensure_collection,
    ensure_voice_collection,
    index_folder,
    index_path,
    request_cancel as indexer_request_cancel,
    reset_cancel as indexer_reset_cancel,
    transcribe_audio,
)
from echos.search import recommend_more_like, search
from echos.viz import VALID_SPACES, invalidate_cache as viz_invalidate, projection as viz_projection
from echos.museum import (
    cached_layout as museum_cached,
    invalidate_cache as museum_invalidate,
    layout as museum_layout,
    mark_dirty as museum_mark_dirty,
    status as museum_status,
)
import asyncio
import json as _json
from echos.faces_lib import (
    consolidate_all_labels,
    crop_face_jpeg,
    ensure_faces_collection,
    get_face,
    list_clusters,
    memories_for_cluster,
    memories_for_label,
    remove_faces_for_cluster,
    remove_faces_for_memory,
    scan_all_photos,
    set_cluster_label,
)

BASE_DIR = Path(__file__).resolve().parent  # root of this repo, survives cd-elsewhere launches
DATA_DIR = Path(os.environ.get("ECHOS_DATA", "./qdrant_storage")).resolve()
MEMORIES_DIR = Path(os.environ.get("ECHOS_MEMORIES", "./memories")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

client = QdrantClient(path=str(DATA_DIR))
ensure_collection(client)
ensure_faces_collection(client)
ensure_voice_collection(client)

# Qdrant Edge write buffer — fast ingest layer. On boot, replay any
# crash-leftover points into the main collection.
try:
    from echos.edge_buffer import EdgeBuffer
    _edge = EdgeBuffer(DATA_DIR)
    _stale_edge = _edge.count()
    if _stale_edge:
        replayed = _edge.drain_into(client, COLLECTION)
        if replayed:
            import logging
            logging.getLogger("uvicorn").info(
                f"edge_buffer: replayed {replayed} stale points into main collection"
            )
except Exception:
    _edge = None  # Edge is optional — app works without it

app = FastAPI(title="Échos", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _bump_threadpool():
    """Sync routes (/thumbnail, /media, /museum/layout when forced) all run on
    anyio's thread pool. The default tokens count is 40 — a museum boot fires
    100+ /thumbnail requests in parallel and the rebuild path competes with
    them. Lift the cap so neither side starves the other."""
    from anyio import to_thread
    try:
        to_thread.current_default_thread_limiter().total_tokens = 200
    except Exception:
        pass

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve React frontend (built output from echoes-front/dist).
REACT_DIST = BASE_DIR / "echoes-front" / "dist"
if REACT_DIST.exists():
    app.mount("/assets", StaticFiles(directory=REACT_DIST / "assets"), name="react-assets")


@app.get("/favicon.ico")
def favicon():
    logo = BASE_DIR / "static" / "echos-logo1.png"
    if logo.exists():
        return FileResponse(logo, media_type="image/png")
    raise HTTPException(404)


@app.get("/")
def root():
    idx = REACT_DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    # fallback: old vanilla index.html
    old = BASE_DIR / "static" / "index.html"
    if old.exists():
        return FileResponse(old)
    return {"name": "Échos", "endpoints": ["/search", "/index-folder", "/stats"]}


# SPA fallback: serve React index.html for client-side routes
# (/search, /people). API routes are matched first, so this doesn't
# shadow /search POST, /faces/*, etc.
@app.get("/search")
@app.get("/people")
def spa_fallback():
    idx = REACT_DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return {"ok": False, "detail": "React frontend not built"}
# the cached layout stale, so the UI can offer a "refresh" button instead of
# auto-jumping the player around mid-walk.
_museum_subscribers: set[asyncio.Queue] = set()


def _museum_broadcast(event: dict) -> None:
    payload = _json.dumps(event)
    for q in list(_museum_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def _notify_museum_dirty(reason: str) -> None:
    info = museum_mark_dirty()
    cached = museum_cached() or {}
    last_total = cached.get("stats", {}).get("total_memories", 0)
    try:
        live_total = client.get_collection(COLLECTION).points_count or 0
    except Exception:
        live_total = last_total
    _museum_broadcast({
        "type": "stale",
        "reason": reason,
        "version": info["version"],
        "last_total": last_total,
        "live_total": live_total,
    })


@app.get("/")
def root():
    idx = Path("static/index.html")
    if idx.exists():
        return FileResponse(idx)
    return {"name": "Échos", "endpoints": ["/search", "/index-folder", "/stats"]}


class SearchBody(BaseModel):
    query: str
    top_k: int | None = 12
    geo_lat: float | None = None
    geo_lon: float | None = None
    geo_radius_m: float | None = None


@app.post("/search")
def search_endpoint(body: SearchBody):
    if not body.query.strip():
        raise HTTPException(400, "empty query")
    out = search(
        client, body.query, body.top_k or 12,
        geo_lat=body.geo_lat,
        geo_lon=body.geo_lon,
        geo_radius_m=body.geo_radius_m,
    )
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
    indexer_reset_cancel()
    counts = index_folder(client, folder)
    viz_invalidate()
    _notify_museum_dirty("index-folder")
    resp: dict = {"folder": str(folder), "counts": counts, "took_seconds": round(time.time() - t0, 2)}
    if counts.get("cancelled"):
        resp["message"] = "indexing was cancelled — partial results saved"
    return resp


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
    _notify_museum_dirty("upload")
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

    # Cache key = SHA1 of source path, mtime, and file size so we never serve
    # a stale thumbnail after the source file changes or is replaced.
    try:
        st = Path(path).stat()
        cache_key = hashlib.sha1(
            f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()
        ).hexdigest()
    except OSError:
        cache_key = None

    if cache_key and cache_key in _thumbnail_cache:
        return {"point_id": point_id, "data_url": _thumbnail_cache[cache_key]}

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
    data_url = f"data:image/jpeg;base64,{b64}"

    if cache_key:
        if len(_thumbnail_cache) >= _thumbnail_cache_max:
            # LRU-ish: evict first key (dict insertion order preserves in 3.7+).
            _thumbnail_cache.pop(next(iter(_thumbnail_cache)))
        _thumbnail_cache[cache_key] = data_url

    return {"point_id": point_id, "data_url": data_url}


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


@app.post("/index-cancel")
def index_cancel():
    """Request cancellation of a running /index-folder operation."""
    indexer_request_cancel()
    return {"ok": True, "message": "cancel requested — index will stop after current item"}


@app.delete("/thumbnail-cache")
def thumbnail_cache_clear():
    n = len(_thumbnail_cache)
    _thumbnail_cache.clear()
    return {"ok": True, "cleared": n}


@app.delete("/reset")
def reset():
    client.delete_collection(COLLECTION)
    ensure_collection(client)
    viz_invalidate()
    museum_invalidate()
    _thumbnail_cache.clear()
    return {"ok": True}


@app.get("/viz/projection")
def viz_projection_endpoint(space: str = "visual", neighbors: int = 3):
    if space not in VALID_SPACES:
        raise HTTPException(400, f"space must be one of {VALID_SPACES}")
    return viz_projection(client, space=space, neighbors=neighbors)


@app.get("/museum/layout")
async def museum_layout_endpoint(min_cluster_size: int = 3, fresh: int = 0):
    """Default: serve the last good layout if one exists (fast, doesn't block
    the museum during a heavy upload). `?fresh=1` forces a rebuild.
    Async + offload to thread so a slow rebuild doesn't choke the FastAPI
    sync thread pool — important when /thumbnail spam from boot is in flight."""
    if not fresh:
        cached = museum_cached()
        if cached is not None:
            return cached
    return await asyncio.to_thread(museum_layout, client, min_cluster_size)


@app.get("/museum/status")
def museum_status_endpoint():
    return museum_status()


@app.get("/museum/events")
async def museum_events():
    """Server-Sent Events: pushes a `stale` event whenever uploads / deletions
    leave the cached layout out of date. Client uses this to surface a manual
    refresh button (we don't auto-redraw the user's view mid-walk)."""
    from fastapi.responses import StreamingResponse

    async def gen():
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        _museum_subscribers.add(q)
        try:
            # Initial hello so the client knows the channel is live.
            yield f"event: ready\ndata: {_json.dumps(museum_status())}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"event: stale\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Comment frame to keep proxies / browsers from closing the
                    # connection on idle.
                    yield ": keepalive\n\n"
        finally:
            _museum_subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.get("/museum")
def museum_page():
    page = BASE_DIR / "static" / "museum.html"
    if not page.exists():
        raise HTTPException(404, "museum.html missing")
    return FileResponse(page)


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
    _notify_museum_dirty("forget")
    return {"ok": True, "forgot": body.point_id, "file_deleted": body.delete_file}


@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    """Accept a browser-recorded audio blob, transcribe with local Whisper (tiny).
    Fully offline — no network call. The model is already loaded by indexer.

    Browser MediaRecorder typically sends audio/webm with opus codec. Whisper
    uses ffmpeg/torchaudio under the hood — both handle webm fine. On Windows
    the tempfile must be closed before Whisper opens it for reading."""
    import tempfile
    # Write to a temp file, close it, then transcribe — NamedTemporaryFile
    # keeps the handle open by default on Windows, which blocks Whisper.
    suffix = Path(audio.filename or "recording").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = tmp.name
    finally:
        tmp.close()  # close before Whisper reads on Windows

    try:
        text = transcribe_audio(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    return {"text": text, "model": "whisper-tiny", "offline": True}


class AskBody(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask")
def ask_endpoint(body: AskBody):
    """Voice/conversational assistant endpoint.

    1. Embeds question, searches memories + past voice context.
    2. Calls local LLM (LM Studio) to synthesize a natural answer from results.
    3. Stores Q&A pair in voice_memories for future conversation context.

    If the LLM is unreachable, falls back to a structured list of top results."""
    from search import search
    from echos.llm_client import ask_llm

    q = body.question.strip()
    if not q:
        raise HTTPException(400, "empty question")

    # 1) Search main memories
    main = search(client, q, body.top_k)
    results = main["results"]

    # 2) Search past voice memories for conversation context
    past_context: list[dict] = []
    try:
        q_vec = embed_text(q)
        voice_hits = client.query_points(
            collection_name=VOICE_COLLECTION,
            using="text",
            query=q_vec,
            limit=3,
            with_payload=True,
        )
        for pt in voice_hits.points:
            pl = pt.payload or {}
            past_context.append({
                "question": pl.get("question"),
                "answer": pl.get("answer"),
                "timestamp": pl.get("timestamp"),
            })
    except Exception:
        past_context = []

    # 3) Synthesize answer with local LLM
    answer: str | None = None
    # Build a compact summary of search results — keep lines short to save tokens
    result_lines: list[str] = []
    for r in results[:4]:
        parts: list[str] = [f"[{r.get('type', 'unknown')}]"]
        ts = r.get("timestamp")
        if ts:
            import datetime as _dt
            parts.append(_dt.datetime.fromtimestamp(ts).strftime("%b %d"))
        t = (r.get("transcript") or "").strip()
        if t:
            parts.append(f'"{t[:80]}"')
        o = (r.get("ocr_text") or "").strip()
        if o:
            parts.append(f'ocr:"{o[:80]}"')
        result_lines.append(" · ".join(parts))

    # Don't include past context in the LLM prompt — it confuses the model.
    # Past context is still returned in the API response for the frontend.
    prompt = (
        f'User asked: "{q}"\n\n'
        f'Search results:\n'
        + "\n".join(f"  {i + 1}. {line}" for i, line in enumerate(result_lines))
    )

    llm_used = False
    try:
        raw = ask_llm(
            "You are Échos, a memory search assistant. Describe what the user's "
            "memories contain in 1-2 short sentences. Mention dates and content "
            "from the results. If nothing matches, say so directly. Never mention "
            "scores, technical details, or emojis. Be brief.",
            prompt,
            max_tokens=40,
            temperature=0.1,
        )
        if raw and raw.strip():
            answer = raw.strip()
            llm_used = True
    except Exception:
        answer = None

    # Fallback: structured list if LLM unavailable
    if not answer:
        if not results:
            answer = "No memories matched that question."
        else:
            parts = []
            for i, r in enumerate(results[:3]):
                t = r.get("type", "unknown")
                ts = r.get("timestamp")
                ago = ""
                if ts:
                    d = (_time.time() - ts) / 86400
                    ago = "today" if d < 1 else "yesterday" if d < 2 else f"{int(d)} days ago"
                parts.append(f"#{i + 1}: {t} from {ago}")
            answer = f"Found {len(results)} memories. " + ". ".join(parts) + "."

    # 4) Store Q&A in voice-memories
    import uuid as _uuid
    import time as _time
    from qdrant_client.models import PointStruct
    qa_id = str(_uuid.uuid4())
    qa_vec = embed_text(q)
    client.upsert(
        collection_name=VOICE_COLLECTION,
        points=[PointStruct(
            id=qa_id,
            vector={"text": qa_vec},
            payload={
                "question": q,
                "answer": answer,
                "result_ids": [r["id"] for r in results],
                "timestamp": _time.time(),
            },
        )],
    )

    return {
        "question": q,
        "qa_id": qa_id,
        "answer": answer,
        "results": results,
        "matched_labels": main["matched_labels"],
        "past_context": past_context,
        "llm_used": llm_used,
    }


@app.post("/ask/{qa_id}/answer")
def ask_store_answer(qa_id: str, body: dict):
    """Store the LLM-synthesized answer for a previous /ask call.
    Body: {"answer": "Last June 14th you were at..."}"""
    answer = (body.get("answer") or "").strip()
    if not answer:
        raise HTTPException(400, "empty answer")
    try:
        client.set_payload(
            collection_name=VOICE_COLLECTION,
            payload={"answer": answer},
            points=[qa_id],
        )
    except Exception:
        raise HTTPException(404, f"qa_id {qa_id} not found")
    return {"ok": True, "qa_id": qa_id, "answer": answer}


@app.post("/faces/scan")
def faces_scan_endpoint():
    """Re-detect faces across every photo/screenshot. Wipes old face clusters first."""
    return scan_all_photos(client)


@app.get("/faces/clusters")
def faces_clusters_endpoint():
    return {"clusters": list_clusters(client)}


@app.delete("/faces/cluster/{cluster_id}")
def faces_delete_cluster_endpoint(cluster_id: str):
    """Delete all faces in a cluster. Removes the person from the people view."""
    removed = remove_faces_for_cluster(client, cluster_id)
    return {"ok": True, "cluster_id": cluster_id, "faces_removed": removed}


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
