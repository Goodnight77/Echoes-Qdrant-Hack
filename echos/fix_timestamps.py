"""One-off script: re-extract timestamps for all indexed memories using the
updated _photo_timestamp() which now falls back to filename-date parsing
before using mtime."""
from qdrant_client import QdrantClient
from pathlib import Path
from echos.indexer import _photo_timestamp, COLLECTION
import time as _t

client = QdrantClient(path="./qdrant_storage")

points, _ = client.scroll(
    collection_name=COLLECTION, limit=2000, with_payload=True, with_vectors=False,
)

updated = 0
unchanged = 0
missing = 0

for p in points:
    pl = p.payload or {}
    path = pl.get("path")
    if not path or not Path(path).exists():
        missing += 1
        continue

    old_ts = pl.get("timestamp", 0)
    new_ts = _photo_timestamp(path)

    if abs(old_ts - new_ts) > 1:
        client.set_payload(
            collection_name=COLLECTION,
            payload={"timestamp": new_ts},
            points=[p.id],
        )
        updated += 1
        if updated <= 15:
            old_d = _t.strftime("%Y-%m-%d", _t.localtime(old_ts))
            new_d = _t.strftime("%Y-%m-%d", _t.localtime(new_ts))
            print(f"FIXED: {Path(path).name}")
            print(f"  {old_d} -> {new_d}")
    else:
        unchanged += 1

print(f"\nUpdated: {updated}")
print(f"Unchanged: {unchanged}")
print(f"Missing files: {missing}")
print("Done.")
