"""Indexer: embed photos / screenshots / voice memos / videos into Qdrant named vectors."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

COLLECTION = "memories"

VISUAL_DIM = 512
TEXT_DIM = 384

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
SCREENSHOT_PREFIX = "screenshot_"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

OCR_MIN_CHARS = 20
OCR_MIN_WORDS = 4


def _ocr_meaningful(text: str) -> bool:
    if not text or len(text) < OCR_MIN_CHARS:
        return False
    alpha_words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 3]
    return len(alpha_words) >= OCR_MIN_WORDS


class _Models:
    """Lazy singleton holding the heavy ML models. Load only what is asked for."""

    _clip = None
    _minilm = None
    _whisper = None
    _ocr = None

    @classmethod
    def clip(cls):
        if cls._clip is None:
            from sentence_transformers import SentenceTransformer
            cls._clip = SentenceTransformer("clip-ViT-B-32")
        return cls._clip

    @classmethod
    def minilm(cls):
        if cls._minilm is None:
            from sentence_transformers import SentenceTransformer
            cls._minilm = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._minilm

    @classmethod
    def whisper(cls):
        if cls._whisper is None:
            import whisper
            cls._whisper = whisper.load_model("tiny")
        return cls._whisper

    @classmethod
    def ocr(cls):
        if cls._ocr is None:
            import easyocr
            last_err = None
            for _ in range(3):
                try:
                    cls._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
                    break
                except Exception as e:
                    last_err = e
                    cls._ocr = None
            if cls._ocr is None:
                raise RuntimeError(f"easyocr init failed after retries: {last_err}")
        return cls._ocr


def embed_image(path_or_img) -> list[float]:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img).convert("RGB")
    vec = _Models.clip().encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32).tolist()


def embed_text_clip(text: str) -> list[float]:
    vec = _Models.clip().encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32).tolist()


def embed_text(text: str) -> list[float]:
    vec = _Models.minilm().encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32).tolist()


def transcribe_audio(path: str | Path) -> str:
    result = _Models.whisper().transcribe(str(path), fp16=False)
    return (result.get("text") or "").strip()


def ocr_image(path_or_img) -> str:
    try:
        if isinstance(path_or_img, Image.Image):
            arr = np.array(path_or_img.convert("RGB"))
        else:
            arr = np.array(Image.open(path_or_img).convert("RGB"))
        lines = _Models.ocr().readtext(arr, detail=0, paragraph=True)
        return " ".join(lines).strip()
    except Exception as e:
        print(f"[ocr_image] skipped due to {type(e).__name__}: {e}")
        return ""


def extract_keyframes(video_path: str | Path, every_seconds: float = 2.0) -> list[tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * every_seconds), 1)
    frames: list[tuple[float, Image.Image]] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            ts = i / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((ts, Image.fromarray(rgb)))
        i += 1
    cap.release()
    return frames or [(0.0, Image.new("RGB", (224, 224)))]


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "visual": VectorParams(size=VISUAL_DIM, distance=Distance.COSINE),
            "audio_transcript": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
            "ocr_text": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
        },
    )


_NS = uuid.UUID("4e9f5e2c-1a4b-4f5d-9e0b-1a5d3c7e9f01")


def _new_id(path: str | Path | None = None) -> str:
    """Deterministic UUID5 from file path so re-indexing is idempotent."""
    if path is None:
        return str(uuid.uuid4())
    return str(uuid.uuid5(_NS, str(Path(path).resolve())))


def _mtime(path: str | Path) -> float:
    return os.path.getmtime(path)


def index_photo(client: QdrantClient, path: str | Path) -> str:
    visual = embed_image(path)
    ocr = ocr_image(path)
    vectors: dict[str, list[float]] = {"visual": visual}
    if _ocr_meaningful(ocr):
        vectors["ocr_text"] = embed_text(ocr)
    pid = _new_id(path)
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=pid,
            vector=vectors,
            payload={
                "type": "photo",
                "path": str(path),
                "timestamp": _mtime(path),
                "ocr_text": ocr,
            },
        )],
    )
    return pid


def index_screenshot(client: QdrantClient, path: str | Path) -> str:
    visual = embed_image(path)
    ocr = ocr_image(path)
    vectors: dict[str, list[float]] = {"visual": visual}
    if _ocr_meaningful(ocr):
        vectors["ocr_text"] = embed_text(ocr)
    pid = _new_id(path)
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=pid,
            vector=vectors,
            payload={
                "type": "screenshot",
                "path": str(path),
                "timestamp": _mtime(path),
                "ocr_text": ocr,
            },
        )],
    )
    return pid


def index_voice_memo(client: QdrantClient, path: str | Path) -> str:
    transcript = transcribe_audio(path) or "(silent)"
    pid = _new_id(path)
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=pid,
            vector={"audio_transcript": embed_text(transcript)},
            payload={
                "type": "voice_memo",
                "path": str(path),
                "timestamp": _mtime(path),
                "transcript": transcript,
            },
        )],
    )
    return pid


def index_video(client: QdrantClient, path: str | Path) -> str:
    frames = extract_keyframes(path)
    keyframe_vecs: list[tuple[float, list[float]]] = []
    visual_acc = np.zeros(VISUAL_DIM, dtype=np.float32)
    for ts, img in frames:
        v = np.array(embed_image(img), dtype=np.float32)
        keyframe_vecs.append((ts, v.tolist()))
        visual_acc += v
    mean_visual = (visual_acc / max(len(frames), 1))
    norm = np.linalg.norm(mean_visual)
    if norm > 0:
        mean_visual = mean_visual / norm

    transcript = ""
    try:
        transcript = transcribe_audio(path)
    except Exception:
        transcript = ""

    vectors: dict[str, list[float]] = {"visual": mean_visual.astype(np.float32).tolist()}
    if transcript:
        vectors["audio_transcript"] = embed_text(transcript)

    pid = _new_id(path)
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=pid,
            vector=vectors,
            payload={
                "type": "video",
                "path": str(path),
                "timestamp": _mtime(path),
                "transcript": transcript,
                "keyframes": [{"t": ts, "vec": v} for ts, v in keyframe_vecs],
            },
        )],
    )
    return pid


def index_path(client: QdrantClient, path: str | Path) -> str | None:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in VIDEO_EXTS:
        return index_video(client, p)
    if ext in AUDIO_EXTS:
        return index_voice_memo(client, p)
    if ext in PHOTO_EXTS:
        if p.name.lower().startswith(SCREENSHOT_PREFIX):
            return index_screenshot(client, p)
        return index_photo(client, p)
    return None


def index_folder(client: QdrantClient, folder: str | Path) -> dict[str, int]:
    folder = Path(folder)
    counts = {"photo": 0, "screenshot": 0, "voice_memo": 0, "video": 0, "skipped": 0}
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            index_video(client, p); counts["video"] += 1
        elif ext in AUDIO_EXTS:
            index_voice_memo(client, p); counts["voice_memo"] += 1
        elif ext in PHOTO_EXTS:
            if p.name.lower().startswith(SCREENSHOT_PREFIX):
                index_screenshot(client, p); counts["screenshot"] += 1
            else:
                index_photo(client, p); counts["photo"] += 1
        else:
            counts["skipped"] += 1
    return counts
