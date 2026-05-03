# Échos — Search every memory on your phone, in one place

> Last month I had a conversation with a contractor about a kitchen renovation. He gave me a price, a timeline, and a list of materials. I recorded it. Two weeks later I was at the store trying to remember if he said oak or walnut. I knew the answer was on my phone. I never found it.
>
> **Échos** finds it. One search box across photos, videos, screenshots, and voice memos — running entirely on your machine. The model that knows everything about your life is the one you own.

Built for the [Qdrant 2026 *Think Outside the Bot* Hackathon](https://try.qdrant.tech/hackathon-vsd).

---

## Why Qdrant — named vectors as the technical pitch

A single Qdrant collection, **`memories`**, with three named vector spaces:

| Space             | Dim | Encoder                    | What it indexes                              |
|-------------------|----:|----------------------------|----------------------------------------------|
| `visual`          | 512 | `clip-ViT-B-32`            | photos, screenshots, video keyframes (mean) |
| `audio_transcript`| 384 | `all-MiniLM-L6-v2`         | voice memo + video transcripts (Whisper)    |
| `ocr_text`        | 384 | `all-MiniLM-L6-v2`         | OCR-extracted text from images / shots      |

A query fans out into all three spaces in parallel and the results merge via **weighted Reciprocal Rank Fusion** (RRF, k=60). Every hit reports which modalities matched — `matched_via: ["visual", "ocr_text"]` — so the UI can prove the match was multi-modal.

```
                 ┌──────────────────┐
        query →  │  CLIP text  (512)│ → visual space        ┐
                 │  MiniLM     (384)│ → audio_transcript   ─┼─ RRF → top 12
                 │             (384)│ → ocr_text            ┘
                 └──────────────────┘
```

This is the *whole* technical bet: a single collection where each item lives in only the modalities it has, and a single query lights up the right ones. No keyword search. No re-ranker. No LLM in the loop.

---

## Why Edge — nothing leaves your device

- Qdrant runs **embedded** in-process via `qdrant-client` against a local file store — no Qdrant Cloud.
- CLIP, Whisper (`tiny`), MiniLM, and EasyOCR all run **locally on CPU**.
- The FastAPI server binds `0.0.0.0:8000` for the iPhone client to reach over local WiFi — and never beyond.
- "Forget this" deletes the point + the underlying file. Real privacy means you can delete a memory.

---

## Architecture

```
┌────────────┐  WiFi  ┌────────────────────────────────┐
│  iPhone /  │ ─────► │  FastAPI :8000                 │
│  Browser   │        │   /search   /index-folder      │
└────────────┘        │   /stats    /thumbnail/{id}    │
                      │   /media/{id}  /resurface      │
                      │   /thread   /forget            │
                      │                                │
                      │  ┌───────────────────────────┐ │
                      │  │  Qdrant (embedded)        │ │
                      │  │  collection: memories     │ │
                      │  │  named vectors: visual,   │ │
                      │  │  audio_transcript, ocr    │ │
                      │  └───────────────────────────┘ │
                      │  CLIP · Whisper · MiniLM · OCR │
                      └────────────────────────────────┘
```

---

## Setup + run

Requires Python 3.11, [`uv`](https://docs.astral.sh/uv/), `ffmpeg` on PATH. Tesseract is optional — falls back to EasyOCR.

```powershell
uv venv --python 3.11
uv sync
uv run python generate_demo.py     # writes 26 Maya-arc items to ./memories/
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

In another shell, populate the index and try a query:

```powershell
curl -X POST http://127.0.0.1:8000/index-folder -H "Content-Type: application/json" -d "{}"
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d '{\"query\":\"dog at the beach\"}'
```

Or just open <http://127.0.0.1:8000/> in a browser.

### Run on iPhone (Safari, no Expo yet)

The web UI is responsive + dark mode → works on iPhone Safari today. Native Expo app is a later session.

1. **PC and iPhone on the same WiFi.** If conference WiFi is unreliable, enable iPhone hotspot and connect the PC to it.
2. **Find PC IP** — PowerShell: `ipconfig` → look for "IPv4 Address" under your active WiFi adapter (e.g. `192.168.1.42`).
3. **Open port 8000 in Windows Firewall** — Defender Firewall → Advanced Settings → Inbound Rules → New Rule → Port → TCP 8000 → Allow.
4. **Smoke test** — on iPhone Safari, open `http://<PC_IP>:8000/stats`. Should return JSON `{"total":26,...}`. If it doesn't: firewall is blocking, or FastAPI is bound to `127.0.0.1` instead of `0.0.0.0`.
5. **Open the app** — `http://<PC_IP>:8000/`. Type a query. Tap a card → modal opens.
6. **Add to Home Screen** (Safari → Share → Add to Home Screen) for an app-like icon. Removes Safari chrome on launch.

iOS blocks autoplay — tap the play button on the video / audio player manually inside the modal.

### Index your real iPhone gallery

The demo data is fine; **real data wins the demo.** Drop your own files into `./memories/` on the PC, then re-index.

**Three ways to get iPhone media onto the PC:**

1. **USB cable + File Explorer** (fastest, lossless): plug iPhone into PC, "Trust this computer" on iPhone → in File Explorer find the iPhone under "This PC" → `Internal Storage / DCIM / 100APPLE` → drag photos & videos into `C:\...\image-memo-qdrant\memories\`.
2. **iCloud / Google Photos web** (no cable): on iPhone upload the photos you want → on PC open the web app → download → drop into `memories/`.
3. **Email or AirDrop-to-Windows alternatives** (small batch): email yourself the files from iPhone Photos / Voice Memos.

**Voice memos:** iPhone Voice Memos app → tap memo → Share → Save to Files → transfer via any method above. They land as `.m4a` (the indexer handles `.m4a` natively).

**Screenshots:** filename must start with `screenshot_` for the indexer to treat it as a screenshot (OCR mandatory). iPhone screenshots come in as `IMG_NNNN.PNG` — rename to `screenshot_<anything>.png` first, or just let them index as photos (CLIP will still match them).

After dropping files into `memories/`, re-index:

```powershell
curl -X POST http://127.0.0.1:8000/index-folder -H "Content-Type: application/json" -d "{}"
curl http://127.0.0.1:8000/stats
```

`/index-folder` is **idempotent** — point IDs are derived from the file path, so re-running it doesn't create dupes. New files get added; existing files get re-embedded only if you `DELETE /reset` first.

**Hard reset** (wipe collection + start clean):
```powershell
curl -X DELETE http://127.0.0.1:8000/reset
curl -X POST http://127.0.0.1:8000/index-folder -H "Content-Type: application/json" -d "{}"
```

Cold-start on real photos is roughly 3–5 seconds per image (CLIP + EasyOCR) and 2–4 seconds per voice memo (Whisper tiny). Plan ~1 min per 20 items on CPU.

### Supported file types

| Modality   | Extensions                                  | Indexed via                              |
|------------|---------------------------------------------|------------------------------------------|
| Photo      | `.jpg .jpeg .png .webp .bmp`                | CLIP visual + EasyOCR (if text present)  |
| Screenshot | same as Photo, filename `screenshot_*`      | CLIP visual + EasyOCR (mandatory)        |
| Voice memo | `.m4a .mp3 .wav .ogg .flac`                 | Whisper tiny → MiniLM embed of transcript |
| Video      | `.mp4 .mov .mkv .webm .avi`                 | mean of CLIP keyframe embeds + Whisper   |

Anything else in the folder is silently skipped.

---

## Demo queries to try

| Query                          | What surfaces                                                  |
|--------------------------------|----------------------------------------------------------------|
| `dog at the beach`             | the dog photo **and** the dog beach video (best moment 0:10)   |
| `Sarah wedding venue`          | airbnb screenshot + Sarah voice memo + wedding venue photo     |
| `Python error`                 | the KeyError traceback screenshot                              |
| `sunset`                       | the sunset photo                                               |
| `race condition bug`           | whiteboard photo + the auth-middleware voice memo              |
| `ceramic mug gift`             | the mom's birthday voice memo                                  |

Each one hits a different mix of modalities — that's the point.

---

## The Maya arc (demo data)

`generate_demo.py` writes 26 items telling 60 days of one fictional life:
- 13 photos · 5 screenshots · 5 voice memos · 3 short videos
- Three threads woven through: planning her sister's **wedding**, debugging a production **bug**, hunting an **apartment** in Park Slope
- A few incidental memories (sunset, dog, coffee) as life texture

Searching `venue` should pull 5 related items across 3 modalities, in chronological order — that's `/thread` doing its job.

---

## Endpoints

| Method | Path                  | Purpose                                                    |
|--------|-----------------------|------------------------------------------------------------|
| POST   | `/search`             | named-vector RRF over query                                |
| POST   | `/index-folder`       | scan a folder, dispatch by extension, upsert points        |
| GET    | `/stats`              | total · by-type breakdown · last indexed timestamp         |
| GET    | `/thumbnail/{id}`     | base64 200px thumbnail                                     |
| GET    | `/media/{id}`         | streams the underlying file                                |
| DELETE | `/reset`              | drop + recreate the collection                             |
| POST   | `/resurface`          | "Echoes": find old memories similar to today's most recent |
| POST   | `/thread`             | one query → chronological grouping across modalities       |
| POST   | `/forget`             | drop a point (optionally delete the file)                  |

---

## Caveats

- **MiniLM short-text retrieval** is mediocre — voice memo discrimination is improved by a per-space cosine floor (0.30) and a small RRF weight bump (1.6×) for `audio_transcript`. Documented in `search.py`.
- **EasyOCR** is the OCR engine (tesseract binary not assumed). Init does a one-time download of `english_g2.pth`.
- **`sentence-transformers` 5.x has a broken pooling path** for `all-MiniLM-L6-v2` that collapses unrelated short texts to cosine ≈ 0.95. Pinned to `sentence-transformers==3.3.1` + `transformers==4.46.0` in `pyproject.toml` as the working baseline.
- Cold start downloads ~1.5 GB of model weights (CLIP, Whisper tiny, MiniLM, EasyOCR). Run once with internet, then `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for fully offline operation.

---

## What's next

This is the **backend** session. The multi-session plan:
1. ✅ Backend on PC — all 6 demo queries pass via curl. *(this commit)*
2. ⏭ Expo / React Native iPhone client — `expo-av` video/audio, reanimated transitions, haptics, settings screen for the PC IP.
3. ⏭ Real personal data dropped into `./memories/` for the demo video.
4. ⏭ 3-minute demo video, contractor-story opener, privacy-cut close.

Built for **Qdrant 2026 Think Outside the Bot Hackathon**.
