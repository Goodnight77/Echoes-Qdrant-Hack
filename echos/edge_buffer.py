"""Qdrant Edge write buffer — crash-safe ingest layer on top of embedded Qdrant.

A mutable Edge shard (no HNSW index) accepts writes faster than the main
indexed collection. On startup we replay any pending Edge points into the
main Qdrant store, then clear the Edge shard. Makes bulk indexing feel
snappy while the main HNSW index builds in the background.

The Edge shard is a crash-safe buffer: if the process dies mid-index, the
points are still in the Edge store and will be replayed on next boot."""

from __future__ import annotations

from pathlib import Path

from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    PointStruct,
    ScrollRequest,
    UpsertOperation,
)


# Edge shard stores raw vectors + payload blobs until replayed into Qdrant.
# After replay the shard is deleted and recreated fresh.
EDGE_DIR_NAME = "edge_buffer"

# Minimal vector config — Edge is a write buffer, not queried for search.
# We store the same 3 named vectors as the main collection so replay is 1:1.
_EDGE_VECTORS = {
    "visual": EdgeVectorParams(512, Distance.Cosine),
    "audio_transcript": EdgeVectorParams(384, Distance.Cosine),
    "ocr_text": EdgeVectorParams(384, Distance.Cosine),
}


class EdgeBuffer:
    def __init__(self, data_dir: Path | str) -> None:
        self._dir = Path(data_dir) / EDGE_DIR_NAME
        self._dir.mkdir(parents=True, exist_ok=True)
        config = EdgeConfig(vectors=_EDGE_VECTORS, on_disk_payload=True)
        self._shard = EdgeShard.create(str(self._dir), config)

    def store(
        self,
        point_id: str,
        vector: dict[str, list[float]],
        payload: dict,
    ) -> None:
        """Write a single point into the Edge buffer. Fast — no index build."""
        pt = PointStruct(id=point_id, vector=vector, payload=payload)
        op = UpsertOperation(upsert=[pt])
        self._shard.update(op)

    def count(self) -> int:
        return self._shard.count(CountRequest(exact=True))

    def drain_into(
        self, client, collection_name: str, batch_size: int = 64
    ) -> int:
        """Replay every point in the Edge buffer into the main Qdrant collection,
        then clear the Edge shard. Returns total points replayed."""
        total = self.count()
        if total == 0:
            return 0

        # Scroll all points from the Edge shard
        points = self._shard.scroll(ScrollRequest())
        replayed = 0
        batch: list[dict] = []
        for pt in points:
            batch.append({
                "id": pt.id,
                "vector": pt.vector or {},
                "payload": pt.payload or {},
            })
            if len(batch) >= batch_size:
                client.upsert(collection_name=collection_name, points=batch)
                replayed += len(batch)
                batch = []

        if batch:
            client.upsert(collection_name=collection_name, points=batch)
            replayed += len(batch)

        # Clear the Edge shard — close, delete dir, recreate
        self._shard.close()
        import shutil
        shutil.rmtree(str(self._dir), ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)
        config = EdgeConfig(vectors=_EDGE_VECTORS, on_disk_payload=True)
        self._shard = EdgeShard.create(str(self._dir), config)
        return replayed

    def close(self) -> None:
        self._shard.close()
