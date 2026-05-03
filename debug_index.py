"""Inspect what's actually stored in Qdrant."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import urllib.request, json

# Inspect via direct Qdrant via main module - but that conflicts with running server.
# Instead: hit /search but with empty term for each vector space directly is harder.
# Use /resurface to see something, plus print payloads via custom debug endpoint.
# Easiest: read scroll directly via a temp client pointed at server's data dir won't work either
# (lock held). So just curl every point's media path and infer state.

# But we have a trick: hit /search with each demo query and look at what audio_transcript
# returns when querying ONLY a known voice phrase.
queries = [
    "Brooklyn venue catering June 14",  # exact wording from voice memo 1
    "auth middleware line 47 race condition",  # voice memo 3
    "ceramic mug market mom",  # voice memo 4
]
for q in queries:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/search",
        data=json.dumps({"query": q, "top_k": 5}).encode(),
        headers={"content-type": "application/json"},
    )
    res = json.loads(urllib.request.urlopen(req).read())
    print(f"\nquery: {q!r}")
    for r in res["results"][:5]:
        path = (r.get("path") or "").replace("\\", "/").split("/")[-1]
        via = "+".join(r["matched_via"])
        raw = r.get("raw_scores", {})
        ts_score = raw.get("audio_transcript")
        print(f"  [{r['type']:10s}] {path:42s} via={via:30s} score={r['score']:.4f}  audio_score={ts_score}")
