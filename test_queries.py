"""Run the 6 hackathon demo queries and print top hits."""
import json
import urllib.request

QUERIES = [
    "dog at the beach",
    "Sarah wedding venue",
    "Python error",
    "sunset",
    "race condition bug",
    "ceramic mug gift",
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


for q in QUERIES:
    print(f"\n==== {q!r} ====")
    res = post("http://127.0.0.1:8000/search", {"query": q, "top_k": 5})
    for i, r in enumerate(res["results"][:5]):
        path = (r.get("path") or "").replace("\\", "/").split("/")[-1]
        via = "+".join(r["matched_via"])
        extra = ""
        if r.get("transcript"):
            extra = f"  | {r['transcript'][:60]!r}"
        elif r.get("ocr_text"):
            extra = f"  | OCR: {r['ocr_text'][:60]!r}"
        if r.get("best_moment_seconds") is not None:
            extra += f"  @{r['best_moment_seconds']}s"
        print(f"  {i+1}. [{r['type']:10s}] {path:42s} via={via:30s} score={r['score']:.4f}{extra}")
