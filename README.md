# Échos — search every memory on your phone, in one place

> Last month I had a conversation about a kitchen renovation. He gave me a price, a timeline, and a list of materials. I recorded it. Two weeks later I was at the store trying to remember if he said oak or walnut. I knew the answer was on my phone. I never found it.
>
> **Échos** finds it. One search box across photos, videos, screenshots, and voice memos — running entirely on your machine.

---

## Features

- **Multi-modal search** — one query searches photos, videos, voice memos, and screenshots simultaneously
- **3D Memory Museum** — walk through your memories arranged in themed rooms (HDBSCAN clustering + LLM-named rooms)
- **Face detection & clustering** — people are automatically grouped and labelable
- **Voice assistant** — push-to-talk in the museum, local Whisper STT + Qwen LLM synthesis via LM Studio, fully offline TTS
- **Library browse** — scroll every memory chronologically
- **People view** — browse by face clusters, rename or delete people
- **Museum search kiosk** — search from inside the 3D museum, teleport to results
- **Direct upload** — upload from any device on local WiFi, parallel progress
- **Qdrant Edge buffer** — crash-safe write layer for fast ingest
- **Nothing leaves your device** — embedded Qdrant, local models, no cloud

---

## Architecture

```
┌────────────┐  WiFi  ┌─────────────────────────────────────────┐
│  Browser / │ ─────► │  FastAPI :8000                          │
│  iPhone    │        │   /search   /library   /upload          │
└────────────┘        │   /faces/*  /museum/*  /ask             │
                      │   /transcribe   /thread   /forget       │
                      │                                         │
                      │  ┌────────────────────────────────────┐ │
                      │  │  Qdrant (embedded)                 │ │
                      │  │  ┌─────────┐  ┌────────────────┐  │ │
                      │  │  │ memories│  │ faces          │  │ │
                      │  │  │ visual  │  │ embedding (512) │  │ │
                      │  │  │ audio   │  └────────────────┘  │ │
                      │  │  │ ocr     │  ┌────────────────┐  │ │
                      │  │  └─────────┘  │ voice_memories │  │ │
                      │  │               │ text (384)     │  │ │
                      │  │               └────────────────┘  │ │
                      │  └────────────────────────────────────┘ │
                      │  ┌────────────────────────────────────┐ │
                      │  │  Qdrant Edge buffer (crash-safe)   │ │
                      │  └────────────────────────────────────┘ │
                      │                                         │
                      │  Models (all local):                     │
                      │  CLIP · Whisper · MiniLM · EasyOCR       │
                      │  ArcFace · Qwen 2.5 (LM Studio)          │
                      └─────────────────────────────────────────┘

React Frontend (echoes-front/):
  Vite · React 19 · TypeScript · TailwindCSS · React Query · wouter
```

---

## Qdrant Collections

| Collection | Vector spaces | Purpose |
|-----------|--------------|---------|
| `memories` | `visual` (512d CLIP), `audio_transcript` (384d MiniLM), `ocr_text` (384d MiniLM) | All indexed memories |
| `faces` | `embedding` (512d ArcFace) | Face detection + clustering |
| `voice_memories` | `text` (384d MiniLM) | Past Q&A pairs for conversational context |

A query fans out into all spaces in parallel and results merge via cosine-weighted fusion across spaces.

---

## Setup

Requires Python 3.11+, [`uv`](https://docs.astral.sh/uv/), `ffmpeg` on PATH.

```powershell
uv venv --python 3.11
uv sync
uv run python main.py
```

Open `http://localhost:8000` — the React app loads automatically.

### Optional: LM Studio for voice + better cluster names

1. Install [LM Studio](https://lmstudio.ai/)
2. Download `Qwen 2.5 3B Instruct` (Q4_K_M) — fits in 4GB VRAM
3. Start the local server (port 1234)
4. Voice assistant + museum room naming now use local LLM instead of Groq

### Optional: Groq API for faster cluster labeling

Set `GROQ_API_KEY` in `.env`. Local LLM is tried first; Groq is the fallback.

---

## Run on iPhone over WiFi

1. PC and iPhone on same WiFi
2. Find PC IP: `ipconfig` → IPv4 Address
3. Open `http://<PC_IP>:8000` in Safari
4. Add to Home Screen for app-like experience

---

## Frontend (echoes-front/)

```powershell
cd echoes-front
npm install
npm run build       # production build → dist/
npm run dev         # dev server with HMR on :5173
```

The Python backend serves the built React app at `/`. Museum is at `/museum`.

---

## Endpoints

### Search & Browse
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/search` | Multi-space search |
| GET | `/library?limit=&before=` | Paginated all memories |
| POST | `/thread` | Chronological search results |
| POST | `/resurface` | Find old memories similar to recent |

### Indexing
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload` | Upload files (multipart, parallel) |
| POST | `/index-folder` | Scan a folder, dispatch by type |
| POST | `/index-cancel` | Cancel running index operation |
| DELETE | `/reset` | Wipe and recreate collections |
| DELETE | `/thumbnail-cache` | Clear thumbnail cache |

### Media
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/thumbnail/{id}` | 200px JPEG data URL |
| GET | `/media/{id}` | Raw file stream |
| GET | `/stats` | Total count + by-type breakdown |
| POST | `/forget` | Delete point + optionally delete file |

### Museum
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/museum` | 3D museum page |
| GET | `/museum/layout?fresh=1` | Museum layout data (cached, force-rebuild with fresh) |
| GET | `/museum/events` | SSE stream for layout staleness |
| GET | `/viz/projection?space=&neighbors=` | 3D galaxy projection data |

### Faces & People
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/faces/clusters` | List face clusters |
| GET | `/faces/by-cluster/{id}` | Memories for a cluster |
| GET | `/faces/by-label/{label}` | Memories for a label |
| GET | `/faces/avatar/{id}` | Cropped face JPEG |
| POST | `/faces/label` | Name a cluster |
| POST | `/faces/scan` | Re-detect faces across all photos |
| POST | `/faces/consolidate` | Merge duplicate-label clusters |
| DELETE | `/faces/cluster/{id}` | Delete a face cluster |

### Voice Assistant
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/transcribe` | Audio blob → text (Whisper tiny, offline) |
| POST | `/ask` | Question → LLM-synthesized answer (LM Studio / Groq) |
| POST | `/ask/{id}/answer` | Store synthesized answer |

---

## Supported file types

| Modality | Extensions | Indexed via |
|----------|-----------|-------------|
| Photo | `.jpg .jpeg .png .webp .bmp .heic` | CLIP visual + EasyOCR |
| Screenshot | same, filename `screenshot_*` | CLIP visual + EasyOCR (mandatory) |
| Voice memo | `.m4a .mp3 .wav .ogg .flac` | Whisper tiny → MiniLM |
| Video | `.mp4 .mov .mkv .webm .avi` | CLIP keyframe mean + Whisper |
| HEIC/HEIF | iPhone default | Supported via `pillow-heif` |

---

## Caveats

- **MiniLM short-text retrieval** — voice memo discrimination improved by per-space cosine floor (0.30) and weight bump (1.6×) for `audio_transcript`.
- **Sentence-transformers** pinned to 3.3.1 (5.x has broken pooling for all-MiniLM-L6-v2).
- **Cold start** downloads ~1.5 GB of model weights. Run once with internet, then `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for fully offline.
- **EXIF dates** extracted where available; filename dates used as fallback (screenshots, camera roll naming). Files without either use file mtime (upload time).

---

## Project name

**Échos** (French: "echoes") — memories that come back to you.
