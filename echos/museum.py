"""Museum layout: HDBSCAN clusters → wall panels, cosine-to-centroid → column,
timestamp → row. Returns 3D world coords every memory snaps to deterministically.

Why deterministic placement matters: judges see the same memory in the same slot
between demos, and a newly indexed memory has a predictable home (its cluster's
panel, position decided by cosine + recency) instead of jumping around the room.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from sklearn.cluster import HDBSCAN

from echos.indexer import COLLECTION
from echos.viz import _knn_edges_local, _scroll_all_with_vectors

# Room geometry (world units = metres-ish)
ROOM_WIDTH = 30.0
ROOM_HEIGHT = 8.0
ROOM_DEPTH = 30.0
WALL_INSET = 0.25  # paint this far inside the wall plane to avoid z-fighting

# Painting layout
PAINTING_W = 1.4
PAINTING_H = 1.0
PAINTING_GAP = 0.2
WALL_MARGIN = 1.0       # space at each wall corner with no paintings
FLOOR_CLEAR = 1.4       # bottom row's lower edge sits this high above floor
CEIL_CLEAR = 1.0        # top row's upper edge sits this far below ceiling

# Person identity is used as a SECONDARY ordering signal within a cluster, not
# as its own cluster. Two photos of the same labeled person should sit next to
# each other on the wall when they land in the same visual cluster — but the
# person never gets a dedicated room (visual themes always win at the cluster
# level).
def _person_of_map(client: QdrantClient) -> dict[str, str]:
    """memory_id -> person_label, for every memory carrying a labeled face.
    Empty dict if face module is unavailable or no labels exist."""
    try:
        from faces_lib import list_known_labels, memories_for_label
    except Exception:
        return {}
    try:
        labels = sorted(list_known_labels(client))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for label in labels:
        try:
            ids = memories_for_label(client, label)
        except Exception:
            continue
        for mid in ids:
            out.setdefault(mid, label)
    return out


# Clustering. min_cluster_size=6 instead of 3 so HDBSCAN groups more
# aggressively: fewer rooms, each with a clearer theme. Lone outliers get
# folded into the nearest cluster downstream rather than spawning their own
# room. Adjust upward if you still see too many rooms.
MIN_CLUSTER_SIZE = 6
MAX_PANELS = 8          # single-room cap — more panels than this gets noisy on one room

# Hub-and-spoke mode kicks in when there are 2+ clusters. A central octagonal
# hub sits at origin with up to 8 doors, one per cluster room. Each room is a
# square box positioned outside its hub door. Cluster names are mounted as
# lintels ABOVE the hub-side doors so the player reads them while still in the
# hub deciding which room to enter.
MULTI_ROOM_THRESHOLD = 2      # n_clusters >= this → switch to hub-spoke
HUB_APOTHEM = 8.5             # hub center → side midpoint distance
HUB_HEIGHT = 7.0
ROOM_DIM = 11.0               # square room footprint
ROOM_HEIGHT_MULTI = 6.0
CORRIDOR_GAP = 0.0            # 0 = room directly attached to hub side
DOORWAY_WIDTH = 3.0           # door width in both hub side and room front
DOORWAY_HEIGHT = 3.6          # door clearance
MULTI_PAINTING_W = 1.4
MULTI_PAINTING_H = 1.0
MULTI_FLOOR_CLEAR = 1.4
MULTI_CEIL_CLEAR = 0.8
MULTI_WALL_MARGIN = 0.7
MAX_ROOMS = 8                 # cap = number of octagon sides

# Cluster label vocabulary. Stoplist is intentionally broad so cluster names
# pick up content words ("birthday", "venue") rather than chatter ("really",
# "maybe"). Bigrams ("birthday party") win over single words when frequent
# enough — they read like room titles instead of tag clouds.
_LABEL_STOP = {
    # articles, conjunctions, prepositions
    "the","a","an","and","or","of","to","in","on","at","for","with","from","by",
    "into","onto","upon","about","over","under","above","below","between","through",
    "before","after","during","while","since","until","than","as","because","though",
    # pronouns
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your",
    "his","its","our","their","this","that","these","those","there","here","what",
    "when","where","which","who","whom","whose","how","why","myself","yourself",
    # be/have/do verbs
    "is","am","are","was","were","be","been","being","have","has","had","having",
    "do","did","does","doing","done","will","would","shall","should","can","could",
    "may","might","must","ought",
    # filler / chatter
    "just","really","actually","basically","like","very","quite","rather","kind",
    "sort","stuff","thing","things","something","anything","nothing","someone",
    "anyone","everyone","nobody","everybody","gonna","wanna","gotta","yeah",
    "nope","okay","sure","fine","well","maybe","probably","perhaps","right",
    "left","good","bad","new","old","still","already","also","even","much","many",
    "more","most","less","least","some","any","all","none","each","every","other",
    "another","both","either","neither","one","two","three","four","five","first",
    "last","next","previous","same","different","such","whole","whether",
    # time / location words too generic to label a room
    "now","today","tomorrow","yesterday","tonight","morning","afternoon","evening",
    "night","week","month","year","hour","minute","second","time","day","days",
    "weeks","months","years","here","there","place","over","under","above","below",
    # politeness / hedging
    "please","thanks","thank","sorry","welcome","hello","goodbye",
    # OCR junk often picked up
    "https","http","www","com","net","org","html","jpg","png","pdf","mp4","mp3",
    "screenshot","photo","video","image","file","files",
    # short verbs that survive 5-char filter ("going","doing","getting")
    "going","doing","getting","making","taking","coming","seeing","saying","asking",
    "trying","looking","thinking","feeling","working","running","walking","talking",
    "telling","calling","wanting","needing","using","letting","keeping","showing",
    "knowing","finding","giving",
}
# Min 5 chars — drops "the/and/got/did" without an explicit list, and most
# 4-letter words (just/like/well) are noise anyway.
_WORD_RE = re.compile(r"[a-zA-Z]{5,}")

# Stable distinct cluster colors
_PALETTE = ["#d4a574", "#8ab8e8", "#c79bd4", "#9bd4a8",
            "#e8a8a8", "#a8a8e8", "#e8d4a8", "#a8e8d4"]

_cache: dict[tuple, dict] = {}

# Sticky slot map: point_id -> (cluster_id, col, row). Survives cache wipes
# so a memory keeps the same wall position across rebuilds, as long as its
# cluster assignment hasn't changed. Without this, every upload re-runs
# HDBSCAN, centroids drift slightly, cosine ranks reshuffle, and paintings
# hop around for no real reason.
_slot_memory: dict[str, tuple[int, int, int]] = {}

# Layout version + dirty flag. Endpoint serves the last successfully built
# layout (cached) while an upload is being indexed, instead of forcing the
# client to wait. After upload, dirty=True bumps version; SSE notifies the
# client to offer a refresh.
_version = 0
_dirty = False
_last_good: dict | None = None


def invalidate_cache() -> None:
    """Hard wipe — used by /reset which clears the whole collection."""
    _cache.clear()
    _slot_memory.clear()
    global _last_good, _dirty, _version
    _last_good = None
    _dirty = False
    _version += 1


def mark_dirty(new_total: int | None = None) -> dict:
    """Soft invalidate: bump version, mark stale, but keep the last good
    layout available so the museum stays interactive while indexing runs."""
    global _dirty, _version
    _cache.clear()
    _dirty = True
    _version += 1
    return {"version": _version, "new_total": new_total}


def status() -> dict:
    return {"version": _version, "dirty": _dirty,
            "has_layout": _last_good is not None,
            "total_in_layout": (_last_good or {}).get("stats", {}).get("total_memories", 0)}


def cached_layout() -> dict | None:
    return _last_good


def _cluster_label(payloads: list[dict], top_k: int = 2) -> str | None:
    """Bigram-first labeling: a single phrase like 'birthday party' is more
    evocative than two separate top words. Falls back to top-K unigrams when
    no bigram is repeated."""
    unigrams: list[str] = []
    bigrams: list[str] = []
    for p in payloads:
        text = (p.get("transcript") or "") + " . " + (p.get("ocr_text") or "")
        # Sentence-ish split keeps bigrams from spanning unrelated clauses.
        for sentence in re.split(r"[.!?\n]", text):
            toks = [w for w in _WORD_RE.findall(sentence.lower()) if w not in _LABEL_STOP]
            unigrams.extend(toks)
            for i in range(len(toks) - 1):
                if toks[i] != toks[i + 1]:  # drop "really really" repeats
                    bigrams.append(f"{toks[i]} {toks[i + 1]}")
    if bigrams:
        top_bg, count_bg = Counter(bigrams).most_common(1)[0]
        # Require the bigram to repeat across the cluster — a one-off phrase
        # in a single transcript shouldn't title the whole room.
        if count_bg >= 2:
            return top_bg
    if not unigrams:
        return None
    common = Counter(unigrams).most_common(top_k)
    return " · ".join(w for w, _ in common[:top_k])


def _allocate_panels(n_panels: int) -> list[tuple[int, int, int]]:
    """Spread n_panels evenly across the 4 walls.
    Returns list of (wall_idx, slot_in_wall, total_slots_on_wall).
    Wall index: 0=north(-Z), 1=east(+X), 2=south(+Z), 3=west(-X)."""
    n_panels = max(1, min(n_panels, MAX_PANELS))
    base = n_panels // 4
    extra = n_panels % 4
    layout: list[tuple[int, int, int]] = []
    for w in range(4):
        slots = base + (1 if w < extra else 0)
        for s in range(slots):
            layout.append((w, s, slots))
    return layout


def _wall_geometry(wall_idx: int) -> dict:
    """For a given wall index, return origin, inward-normal, tangent, wall length, rotation_y.
    Rotation_y rotates a painting plane (default facing +Z) so its normal points into the room."""
    half_w = ROOM_WIDTH / 2 - WALL_INSET
    half_d = ROOM_DEPTH / 2 - WALL_INSET
    if wall_idx == 0:  # north wall, faces +Z (into room)
        return dict(origin=(0.0, 0.0, -half_d), normal=(0.0, 0.0, 1.0),
                    tangent=(1.0, 0.0, 0.0), wall_len=ROOM_WIDTH, rot_y=0.0)
    if wall_idx == 1:  # east wall, faces -X
        return dict(origin=(half_w, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                    tangent=(0.0, 0.0, 1.0), wall_len=ROOM_DEPTH, rot_y=-np.pi / 2)
    if wall_idx == 2:  # south wall, faces -Z
        return dict(origin=(0.0, 0.0, half_d), normal=(0.0, 0.0, -1.0),
                    tangent=(-1.0, 0.0, 0.0), wall_len=ROOM_WIDTH, rot_y=np.pi)
    # west wall, faces +X
    return dict(origin=(-half_w, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
                tangent=(0.0, 0.0, -1.0), wall_len=ROOM_DEPTH, rot_y=np.pi / 2)


def _panel_bounds(wall_len: float, slots_on_wall: int, slot_idx: int) -> tuple[float, float]:
    """Return (panel_start_local, panel_width). Local coord runs along the wall tangent,
    centered at 0."""
    inner = wall_len - 2 * WALL_MARGIN
    panel_w = inner / slots_on_wall
    start = -wall_len / 2 + WALL_MARGIN + slot_idx * panel_w
    return start, panel_w


def _grid_dims(n_items: int, panel_w: float, usable_h: float) -> tuple[int, int]:
    """Pick cols × rows so cells stay close to PAINTING_W : PAINTING_H aspect.
    Rows is at least 1; cols chosen to make grid roughly fill panel."""
    if n_items <= 0:
        return 1, 1
    target_aspect = (PAINTING_W + PAINTING_GAP) / (PAINTING_H + PAINTING_GAP)
    # Solve cols/rows ≈ panel_w / usable_h × target_aspect
    desired_ratio = (panel_w / max(usable_h, 0.1)) / target_aspect
    cols = max(1, int(round(np.sqrt(n_items * desired_ratio))))
    rows = int(np.ceil(n_items / cols))
    # Make sure paintings fit at advertised size; shrink grid if too cramped
    while cols > 1 and panel_w / cols < (PAINTING_W * 0.6 + PAINTING_GAP):
        cols -= 1
        rows = int(np.ceil(n_items / cols))
    return cols, rows


def _color_for(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


import math as _math
import os as _os
from pathlib import Path as _Path

from echos import groq_labels as _groq

# Octagon side selection order. Cardinals first (E, S, W, N) so for ≤4 clusters
# the layout reads as a clean cross around the hub. Ordinals fill in for 5-8.
# Indices map into the 8 octagon sides (each at angle i*π/4).
HUB_SIDE_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _hub_side_angle(slot_idx: int) -> float:
    """Outward-normal angle of the hub side assigned to the slot_idx-th room.
    slot_idx 0 → east, 1 → south, 2 → west, 3 → north, 4-7 → ordinals."""
    return HUB_SIDE_ORDER[slot_idx % 8] * _math.pi / 4


def _room_transform(theta: float) -> tuple[float, float, float]:
    """Return (cx, cz, alpha) for the room behind hub side at outward angle θ.
    α rotates the room around Y so its local +Z (back wall direction) aligns
    with the outward direction from the hub."""
    dist = HUB_APOTHEM + ROOM_DIM / 2 + CORRIDOR_GAP
    cx = dist * _math.cos(theta)
    cz = dist * _math.sin(theta)
    # Solve for α such that local +Z = (sin α, 0, cos α) maps to outward = (cos θ, 0, sin θ)
    alpha = _math.atan2(_math.cos(theta), _math.sin(theta))
    return cx, cz, alpha


def _local_to_world(lx: float, lz: float, cx: float, cz: float, alpha: float) -> tuple[float, float]:
    """Project a (lx, lz) room-local point into world coords given the room's
    center and rotation. Y is unchanged (always world Y)."""
    wx = cx + _math.cos(alpha) * lx + _math.sin(alpha) * lz
    wz = cz - _math.sin(alpha) * lx + _math.cos(alpha) * lz
    return wx, wz


def _split_for_three_walls(n_items: int) -> tuple[int, int, int]:
    """Split into (back, left, right) where back gets the largest share.
    Most-similar items go to the back (focal) wall — that's what the player
    sees first when entering the room."""
    if n_items <= 0:
        return 0, 0, 0
    if n_items <= 3:
        # Tiny clusters: everything on back wall, side walls empty.
        return n_items, 0, 0
    # Aim for back ~ half, sides ~ quarter each.
    back = max(1, n_items // 2)
    remainder = n_items - back
    left = remainder // 2 + (remainder % 2)
    right = remainder // 2
    return back, left, right


def _wall_grid(n_items: int, wall_len: float, usable_h: float) -> tuple[int, int]:
    if n_items <= 0:
        return 1, 1
    inner_w = wall_len - 2 * MULTI_WALL_MARGIN
    target_aspect = (MULTI_PAINTING_W + PAINTING_GAP) / (MULTI_PAINTING_H + PAINTING_GAP)
    desired_ratio = (inner_w / max(usable_h, 0.1)) / target_aspect
    cols = max(1, int(round(np.sqrt(n_items * desired_ratio))))
    rows = int(np.ceil(n_items / cols))
    while cols > 1 and inner_w / cols < (MULTI_PAINTING_W * 0.6 + PAINTING_GAP):
        cols -= 1
        rows = int(np.ceil(n_items / cols))
    return cols, rows


def _matched_via(payload: dict) -> list[str]:
    via: list[str] = []
    if payload.get("type") in ("photo", "screenshot", "video"):
        via.append("visual")
    if payload.get("transcript"):
        via.append("audio_transcript")
    if payload.get("ocr_text"):
        via.append("ocr_text")
    return via


def _build_hub_spoke_layout(rows, matrix, labels, unique_labels,
                              min_cluster_size, noise_count,
                              person_of: dict[str, str] | None = None):
    """Hub-and-spoke layout: octagonal hub at origin, one room per visual
    cluster, doors aligned, lintel mounted on hub interior above each door.
    `person_of` (optional): memory_id → person label. When supplied, memories
    sharing a label are placed adjacent on the wall within their cluster —
    same-person photos cluster on the wall without becoming their own room."""
    person_of = person_of or {}
    global _last_good, _dirty
    n_rooms = len(unique_labels)
    n = len(rows)

    rooms_out: list[dict] = []
    item_pos: dict[str, tuple[float, float, float]] = {}
    item_cluster: dict[str, int] = {}

    side_len = 2.0 * HUB_APOTHEM * _math.tan(_math.pi / 8)
    half_room = ROOM_DIM / 2

    # Hub side metadata for the frontend (which sides have doors, where
    # lintels go, what labels read).
    hub_sides: list[dict] = []

    # Pre-fetch LLM labels for every cluster in parallel. One round-trip per
    # cluster, capped at 4 in flight. None entries → fall back to bigram.
    label_inputs: list[tuple[list[str], list[dict], list[str | None]]] = []
    for cluster_lbl in unique_labels:
        mask = labels == cluster_lbl
        idxs = np.where(mask)[0]
        pids = [rows[i][0] for i in idxs]
        payloads = [rows[i][2] for i in idxs]
        filenames = [_Path(p.get("path") or "").name or None for p in payloads]
        label_inputs.append((pids, payloads, filenames))
    llm_labels = _groq.label_clusters_parallel(label_inputs, max_workers=4)

    for cluster_idx, cluster_lbl in enumerate(unique_labels):
        mask = labels == cluster_lbl
        cluster_indices = np.where(mask)[0]
        cluster_size = int(len(cluster_indices))
        if cluster_size == 0:
            continue
        cluster_lbl_int = int(cluster_lbl)

        cluster_vecs = matrix[cluster_indices]
        cluster_payloads = [rows[i][2] for i in cluster_indices]
        cluster_ids = [rows[i][0] for i in cluster_indices]

        centroid = cluster_vecs.mean(axis=0)
        cnorm = float(np.linalg.norm(centroid)) or 1.0
        cosines = (cluster_vecs @ centroid) / cnorm
        cohesion = float(np.mean(cosines))

        timestamps = np.array([(p.get("timestamp") or 0.0) for p in cluster_payloads])
        # Sort: same-person photos adjacent (primary), then most-central first
        # within each person-group (secondary), then chronological.
        # Sentinel '~' sorts after every letter so faceless photos go LAST,
        # leaving the focal back wall for the recognised faces of the cluster.
        person_keys = np.array(
            [person_of.get(cluster_ids[i], '~~~') for i in range(cluster_size)],
            dtype=object,
        )
        order = sorted(
            range(cluster_size),
            key=lambda i: (person_keys[i], -cosines[i], timestamps[i]),
        )

        # Pick hub side (cardinal-first), compute room transform.
        side_idx_in_octagon = HUB_SIDE_ORDER[cluster_idx % 8]
        theta = _hub_side_angle(cluster_idx)
        room_cx, room_cz, alpha = _room_transform(theta)

        n_back, n_left, n_right = _split_for_three_walls(cluster_size)
        is_person_room = False
        # Label priority: Groq → bigram → "Cluster #N"
        cluster_label_text = (
            llm_labels[cluster_idx]
            or _cluster_label(cluster_payloads)
            or f"Cluster #{cluster_idx + 1}"
        )
        cluster_color = _color_for(cluster_idx)

        usable_h = ROOM_HEIGHT_MULTI - MULTI_FLOOR_CLEAR - MULTI_CEIL_CLEAR

        # Wall placements: each wall sits in the room's local frame.
        # local axes: +Z = back (focal), +X = right, -X = left, -Z = front (door)
        wall_specs = [
            # (n_paintings, wall_axis, wall_pos_local, painting_normal_local, local_rotation_y, label)
            ("back",  n_back,  "z",  +half_room, ( 0, 0, -1), _math.pi),       # back wall, faces -Z
            ("left",  n_left,  "x",  -half_room, ( 1, 0,  0), _math.pi / 2),    # left wall, faces +X
            ("right", n_right, "x",  +half_room, (-1, 0,  0), -_math.pi / 2),   # right wall, faces -X
        ]

        memories: list[dict] = []
        slot_offset = 0

        for wall_name, wall_n, axis, wall_pos, normal, local_rot_y in wall_specs:
            if wall_n == 0:
                continue
            grid_cols, grid_rows = _wall_grid(wall_n, ROOM_DIM, usable_h)
            inner_w = ROOM_DIM - 2 * MULTI_WALL_MARGIN
            cell_w = inner_w / grid_cols
            cell_h = usable_h / grid_rows
            p_w = min(MULTI_PAINTING_W, cell_w - PAINTING_GAP)
            p_h = min(MULTI_PAINTING_H, cell_h - PAINTING_GAP)
            wall_start = -inner_w / 2  # local along the wall's tangent

            for slot in range(wall_n):
                col_in_wall = slot % grid_cols
                row_in_wall = slot // grid_cols
                idx_in_cluster = int(order[slot_offset + slot])

                # Tangent offset along the wall (left→right, increasing local axis).
                tan_offset = wall_start + (col_in_wall + 0.5) * cell_w
                # Vertical center of this row (row 0 at top → highest y).
                y = MULTI_FLOOR_CLEAR + (grid_rows - row_in_wall - 0.5) * cell_h

                if axis == "z":  # back wall: tangent runs along local X
                    lx = tan_offset
                    lz = wall_pos + normal[2] * 0.02   # 2 cm forward of wall plane
                else:            # side walls: tangent runs along local Z
                    lx = wall_pos + normal[0] * 0.02
                    lz = tan_offset

                wx, wz = _local_to_world(lx, lz, room_cx, room_cz, alpha)
                world_rot_y = alpha + local_rot_y

                payload = cluster_payloads[idx_in_cluster]
                pid = cluster_ids[idx_in_cluster]
                sim = float(cosines[idx_in_cluster])
                via = _matched_via(payload)

                memories.append({
                    "id": pid,
                    "wall": wall_name,
                    "col": col_in_wall,
                    "row": row_in_wall,
                    "x": float(wx),
                    "y": float(y),
                    "z": float(wz),
                    "width": float(p_w),
                    "height": float(p_h),
                    "rotation_y": float(world_rot_y),
                    "thumb_url": f"/thumbnail/{pid}",
                    "media_url": f"/media/{pid}",
                    "type": payload.get("type"),
                    "timestamp": payload.get("timestamp"),
                    "transcript": (payload.get("transcript") or "")[:140],
                    "ocr_text": (payload.get("ocr_text") or "")[:140],
                    "score_to_centroid": sim,
                    "match_caption": f"score {sim:.2f} · matched via {'+'.join(via) or '—'}",
                    "matched_via": via,
                })
                item_pos[pid] = (float(wx), float(y), float(wz))
                item_cluster[pid] = cluster_lbl_int
            slot_offset += wall_n

        # Lintel: textured plane mounted ON the hub-interior wall, just above
        # the doorway — like a real museum room sign. Rotation aligns the
        # plane with the wall so it stays stuck regardless of viewing angle.
        lintel_inset = 0.06   # 6cm in front of the wall plane to avoid z-fighting
        lintel_pos = (
            (HUB_APOTHEM - lintel_inset) * _math.cos(theta),
            DOORWAY_HEIGHT + 0.55,   # plane center ~55cm above the door top
            (HUB_APOTHEM - lintel_inset) * _math.sin(theta),
        )
        # Rotation that aligns plane normal with hub-inward direction.
        lintel_rot_y = _math.atan2(-_math.cos(theta), -_math.sin(theta))

        # Room collision walls in WORLD coords. Front wall has a door gap.
        room_walls_world: list[dict] = []
        # Back wall: from local (-half, *, +half) to (+half, *, +half)
        wx1, wz1 = _local_to_world(-half_room, +half_room, room_cx, room_cz, alpha)
        wx2, wz2 = _local_to_world(+half_room, +half_room, room_cx, room_cz, alpha)
        room_walls_world.append({"x1": wx1, "z1": wz1, "x2": wx2, "z2": wz2})
        # Left wall: (-half, *, -half) to (-half, *, +half)
        wx1, wz1 = _local_to_world(-half_room, -half_room, room_cx, room_cz, alpha)
        wx2, wz2 = _local_to_world(-half_room, +half_room, room_cx, room_cz, alpha)
        room_walls_world.append({"x1": wx1, "z1": wz1, "x2": wx2, "z2": wz2})
        # Right wall: (+half, *, -half) to (+half, *, +half)
        wx1, wz1 = _local_to_world(+half_room, -half_room, room_cx, room_cz, alpha)
        wx2, wz2 = _local_to_world(+half_room, +half_room, room_cx, room_cz, alpha)
        room_walls_world.append({"x1": wx1, "z1": wz1, "x2": wx2, "z2": wz2})
        # Front wall: split around the door
        # Door is centered on (0, *, -half) in local frame, width DOORWAY_WIDTH.
        half_door = DOORWAY_WIDTH / 2
        wx1, wz1 = _local_to_world(-half_room, -half_room, room_cx, room_cz, alpha)
        wx2, wz2 = _local_to_world(-half_door,  -half_room, room_cx, room_cz, alpha)
        room_walls_world.append({"x1": wx1, "z1": wz1, "x2": wx2, "z2": wz2})
        wx1, wz1 = _local_to_world( half_door,  -half_room, room_cx, room_cz, alpha)
        wx2, wz2 = _local_to_world( half_room, -half_room, room_cx, room_cz, alpha)
        room_walls_world.append({"x1": wx1, "z1": wz1, "x2": wx2, "z2": wz2})

        rooms_out.append({
            "id": f"room_{cluster_idx}",
            "cluster_id": cluster_lbl_int,
            "label": cluster_label_text,
            "color": cluster_color,
            "size": cluster_size,
            "cohesion": cohesion,
            "world": {
                "cx": float(room_cx), "cz": float(room_cz),
                "rotation_y": float(alpha),
                "width": ROOM_DIM, "depth": ROOM_DIM, "height": ROOM_HEIGHT_MULTI,
            },
            "hub_side": {
                "octagon_idx": int(side_idx_in_octagon),
                "theta": float(theta),
            },
            "is_person": is_person_room,
            "lintel": {"x": float(lintel_pos[0]), "y": float(lintel_pos[1]),
                       "z": float(lintel_pos[2]), "rotation_y": float(lintel_rot_y)},
            "walls": room_walls_world,
            "memories": memories,
        })

        # Build hub-side record for this cluster (room HAS a door here).
        hub_sides.append({
            "octagon_idx": int(side_idx_in_octagon),
            "theta": float(theta),
            "has_door": True,
            "room_id": f"room_{cluster_idx}",
            "cluster_id": cluster_lbl_int,
            "label": cluster_label_text,
            "color": cluster_color,
            "is_person": is_person_room,
        })

    # Fill remaining hub sides as solid (no door).
    used = {s["octagon_idx"] for s in hub_sides}
    for i in range(8):
        if i not in used:
            hub_sides.append({
                "octagon_idx": i,
                "theta": i * _math.pi / 4,
                "has_door": False,
            })
    hub_sides.sort(key=lambda s: s["octagon_idx"])

    nbrs = _knn_edges_local(matrix, [r[0] for r in rows], 3)
    edges: list[dict] = []
    for i, (pid, _, _) in enumerate(rows):
        if pid not in item_pos:
            continue
        src_cluster = item_cluster.get(pid)
        for nbr_id in nbrs[i]:
            if nbr_id not in item_pos:
                continue
            if item_cluster.get(nbr_id) == src_cluster:
                continue
            edges.append({"from": pid, "to": nbr_id})

    hub_meta = {
        "apothem": HUB_APOTHEM,
        "side_length": float(side_len),
        "height": HUB_HEIGHT,
        "sides": hub_sides,
        "doorway_width": DOORWAY_WIDTH,
        "doorway_height": DOORWAY_HEIGHT,
        "spawn": {"x": 0.0, "y": 1.7, "z": 0.0},
    }

    result = {
        "version": _version,
        "mode": "hub-spoke",
        "hub": hub_meta,
        "stats": {
            "total_memories": n,
            "n_clusters": len(unique_labels),
            "n_rooms": len(rooms_out),
            "noise_count": noise_count,
            "vector_dims": [512, 384, 384],
            "clustering": f"HDBSCAN min_cluster_size={min_cluster_size} (cosine via L2-normalized euclidean)",
        },
        "rooms": rooms_out,
        "edges": edges,
    }
    _last_good = result
    _dirty = False
    return result


def layout(client: QdrantClient, min_cluster_size: int = MIN_CLUSTER_SIZE) -> dict[str, Any]:
    """Build a deterministic museum layout. Cached until invalidate_cache() is called.
    Single-room mode (≤4 clusters) uses sticky slots so existing memories keep
    their wall positions across rebuilds. Multi-room mode (5+ clusters) tiles
    rooms in a grid with doorways in shared walls."""
    global _last_good, _dirty
    points = _scroll_all_with_vectors(client)

    rows: list[tuple[str, np.ndarray, dict]] = []
    for p in points:
        vec_obj = p.vector or {}
        if not isinstance(vec_obj, dict):
            continue
        v = vec_obj.get("visual")
        if v is None:
            continue
        rows.append((str(p.id), np.asarray(v, dtype=np.float32), p.payload or {}))

    n = len(rows)
    cache_key = (min_cluster_size, n, tuple(r[0] for r in rows))
    if cache_key in _cache:
        return _cache[cache_key]

    room = {"width": ROOM_WIDTH, "height": ROOM_HEIGHT, "depth": ROOM_DEPTH}

    if n == 0:
        result = {
            "version": _version,
            "mode": "single-room",
            "room": room,
            "stats": {"total_memories": 0, "n_clusters": 0, "noise_count": 0,
                      "vector_dims": [512, 384, 384],
                      "clustering": f"HDBSCAN min_cluster_size={min_cluster_size}"},
            "panels": [],
            "edges": [],
        }
        _cache[cache_key] = result
        _last_good = result
        _dirty = False
        return result

    matrix = np.stack([r[1] for r in rows])

    # Person labels are loaded once, used later as a SECONDARY ordering signal
    # so same-person photos sit adjacent on the same wall within their cluster.
    # Persons no longer create their own rooms — the user wants visual themes
    # to drive room assignment.
    person_of = _person_of_map(client)

    # Cluster on visual space. Vectors are L2-normalized at embed time, so
    # euclidean on unit vectors is monotonic with cosine — same neighborhoods.
    if n >= max(min_cluster_size * 2, 6):
        labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(matrix)
    else:
        labels = np.zeros(n, dtype=int)

    unique_labels = sorted(set(labels.tolist()) - {-1})
    noise_count = int(np.sum(labels == -1))

    # If HDBSCAN found nothing, treat the whole corpus as one cluster so the
    # museum still has at least one panel populated.
    if not unique_labels:
        unique_labels = [0]
        labels = np.zeros(n, dtype=int)
        noise_count = 0

    # Cap clusters to MAX_PANELS by size; the rest are merged into one
    # "miscellaneous" cluster so nothing is hidden from the viewer.
    if len(unique_labels) > MAX_PANELS:
        counts = Counter(labels.tolist())
        kept = [lbl for lbl, _ in counts.most_common(MAX_PANELS) if lbl != -1][:MAX_PANELS - 1]
        kept_set = set(kept)
        misc_label = max(unique_labels) + 1
        for i, lbl in enumerate(labels):
            if lbl not in kept_set and lbl != -1:
                labels[i] = misc_label
        unique_labels = sorted(kept_set | {misc_label})

    # ---- Multi-room branch ----
    # When clusters outnumber what one room can hold legibly, give each cluster
    # its own room and tile them in a grid. Doorways punch through shared
    # walls so the player can stroll between themes.
    if len(unique_labels) >= MULTI_ROOM_THRESHOLD:
        # Cap to MAX_ROOMS — overflow rooms are folded back into the smallest
        # populated room (rather than dropped) so every memory still has a wall.
        if len(unique_labels) > MAX_ROOMS:
            counts = Counter(labels.tolist())
            kept = [lbl for lbl, _ in counts.most_common(MAX_ROOMS) if lbl != -1][:MAX_ROOMS - 1]
            kept_set = set(kept)
            misc_label = max(unique_labels) + 1
            for i, lbl in enumerate(labels):
                if lbl not in kept_set and lbl != -1:
                    labels[i] = misc_label
            unique_labels = sorted(kept_set | {misc_label})
        # Promote noise into the largest cluster so multi-room never has empty rooms
        if noise_count > 0:
            counts = Counter(labels.tolist())
            biggest = max((lbl for lbl in counts if lbl != -1), key=lambda l: counts[l])
            for i, lbl in enumerate(labels):
                if lbl == -1:
                    labels[i] = biggest
        return _build_hub_spoke_layout(rows, matrix, labels, unique_labels,
                                        min_cluster_size, noise_count,
                                        person_of=person_of)

    # ---- Single-room fallback (n_clusters < MULTI_ROOM_THRESHOLD) ----
    # Promote noise points into nearest cluster — they still get a slot
    # (in the misc panel if one exists, else the largest cluster). Without
    # this step, judges only see the "clean" clusters and wonder where the
    # rest of their photos went.
    if noise_count > 0 and unique_labels:
        centroids = []
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.any():
                centroids.append((lbl, matrix[mask].mean(axis=0)))
        for i, lbl in enumerate(labels):
            if lbl == -1:
                # cosine similarity = dot since vectors are L2-normalized
                best = max(centroids, key=lambda c: float(np.dot(matrix[i], c[1])))
                labels[i] = best[0]

    panel_slots = _allocate_panels(len(unique_labels))

    panels: list[dict] = []
    item_pos: dict[str, tuple[float, float, float]] = {}
    item_cluster: dict[str, int] = {}  # for cross-cluster edge filter
    new_slot_memory: dict[str, tuple[int, int, int]] = {}

    # LLM labels (single-room path) — same fallback chain as hub-spoke.
    label_inputs_sr: list[tuple[list[str], list[dict], list[str | None]]] = []
    for cluster_lbl in unique_labels:
        idxs_lbl = np.where(labels == cluster_lbl)[0]
        pids = [rows[i][0] for i in idxs_lbl]
        payloads_lbl = [rows[i][2] for i in idxs_lbl]
        filenames_lbl = [_Path(p.get("path") or "").name or None for p in payloads_lbl]
        label_inputs_sr.append((pids, payloads_lbl, filenames_lbl))
    llm_labels_sr = _groq.label_clusters_parallel(label_inputs_sr, max_workers=4)

    for cluster_idx, cluster_lbl in enumerate(unique_labels):
        mask = labels == cluster_lbl
        cluster_indices = np.where(mask)[0]
        cluster_size = int(len(cluster_indices))
        if cluster_size == 0:
            continue

        cluster_vecs = matrix[cluster_indices]
        cluster_payloads = [rows[i][2] for i in cluster_indices]
        cluster_ids = [rows[i][0] for i in cluster_indices]
        cluster_lbl_int = int(cluster_lbl)

        # Centroid + cosine each-to-centroid (unit vectors so dot = cosine)
        centroid = cluster_vecs.mean(axis=0)
        cnorm = float(np.linalg.norm(centroid)) or 1.0
        cosines = (cluster_vecs @ centroid) / cnorm
        cohesion = float(np.mean(cosines))

        # Sort: most central first, ties broken by oldest first so newer memories
        # land on outer columns (visually "fresh" at the edges of the theme).
        timestamps = np.array([(p.get("timestamp") or 0.0) for p in cluster_payloads])
        order = np.lexsort((timestamps, -cosines))

        wall_idx, slot_in_wall, slots_on_wall = panel_slots[cluster_idx]
        wg = _wall_geometry(wall_idx)
        panel_start_local, panel_w = _panel_bounds(wg["wall_len"], slots_on_wall, slot_in_wall)

        usable_h = ROOM_HEIGHT - FLOOR_CLEAR - CEIL_CLEAR
        cols, grows = _grid_dims(cluster_size, panel_w, usable_h)

        cell_w = panel_w / cols
        cell_h = usable_h / grows
        # Cap painting size to keep a uniform look across panels
        p_w = min(PAINTING_W, cell_w - PAINTING_GAP)
        p_h = min(PAINTING_H, cell_h - PAINTING_GAP)

        cluster_label_text = (
            llm_labels_sr[cluster_idx]
            or _cluster_label(cluster_payloads)
            or f"Cluster #{cluster_idx + 1}"
        )
        cluster_color = _color_for(cluster_idx)

        # ---- Sticky slot assignment ----
        # Pass 1: any memory that already had a slot in THIS cluster, AND whose
        # slot still fits the new grid, keeps its (col, row).
        # Pass 2: every remaining memory, taken in similarity order, fills the
        # next free slot. Result: established paintings don't move on upload;
        # newcomers slot into the gaps.
        taken: set[tuple[int, int]] = set()
        slot_assignment: dict[int, tuple[int, int]] = {}  # idx_in_cluster -> (col, row)

        for idx_in_cluster, pid in enumerate(cluster_ids):
            prev = _slot_memory.get(pid)
            if prev is None:
                continue
            prev_cluster, prev_col, prev_row = prev
            if prev_cluster != cluster_lbl_int:
                continue
            if prev_col >= cols or prev_row >= grows:
                continue
            if (prev_col, prev_row) in taken:
                continue
            slot_assignment[idx_in_cluster] = (prev_col, prev_row)
            taken.add((prev_col, prev_row))

        free_slots: list[tuple[int, int]] = [
            (c, r) for r in range(grows) for c in range(cols) if (c, r) not in taken
        ]
        free_iter = iter(free_slots)
        for idx_in_cluster in order:
            if int(idx_in_cluster) in slot_assignment:
                continue
            try:
                slot_assignment[int(idx_in_cluster)] = next(free_iter)
            except StopIteration:
                # More memories than grid cells (shouldn't happen since
                # _grid_dims is sized to fit, but guard anyway by overflowing
                # into a synthetic outer column).
                spill = len(slot_assignment) - cols * grows
                slot_assignment[int(idx_in_cluster)] = (cols + spill, 0)

        memories: list[dict] = []
        # Iterate in similarity order so the JSON `memories` array reads
        # from-most-central-out, even though slot positions are sticky.
        for idx_in_cluster in order:
            col, row = slot_assignment[int(idx_in_cluster)]

            tangent_offset = panel_start_local + (col + 0.5) * cell_w
            y = FLOOR_CLEAR + (grows - row - 0.5) * cell_h  # row 0 at top

            # World coords = wall origin + tangent * offset, then nudge into room
            # by a hair to keep the painting plane in front of the wall.
            tx = wg["tangent"][0] * tangent_offset + wg["normal"][0] * 0.02
            tz = wg["tangent"][2] * tangent_offset + wg["normal"][2] * 0.02
            x = wg["origin"][0] + tx
            z = wg["origin"][2] + tz

            payload = cluster_payloads[idx_in_cluster]
            pid = cluster_ids[idx_in_cluster]
            sim = float(cosines[idx_in_cluster])
            via = _matched_via(payload)
            new_slot_memory[pid] = (cluster_lbl_int, int(col), int(row))
            item_cluster[pid] = cluster_lbl_int

            memories.append({
                "id": pid,
                "col": col,
                "row": row,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "width": float(p_w),
                "height": float(p_h),
                "rotation_y": float(wg["rot_y"]),
                "thumb_url": f"/thumbnail/{pid}",
                "media_url": f"/media/{pid}",
                "type": payload.get("type"),
                "timestamp": payload.get("timestamp"),
                "transcript": (payload.get("transcript") or "")[:140],
                "ocr_text": (payload.get("ocr_text") or "")[:140],
                "score_to_centroid": sim,
                "match_caption": f"score {sim:.2f} · matched via {'+'.join(via) or '—'}",
                "matched_via": via,
            })
            item_pos[pid] = (float(x), float(y), float(z))

        # Lintel sits above the panel center
        lintel_local = panel_start_local + panel_w / 2
        lintel_x = wg["origin"][0] + wg["tangent"][0] * lintel_local + wg["normal"][0] * 0.05
        lintel_z = wg["origin"][2] + wg["tangent"][2] * lintel_local + wg["normal"][2] * 0.05
        lintel_y = ROOM_HEIGHT - 0.4

        panels.append({
            "id": f"panel_{cluster_idx}",
            "cluster_id": int(cluster_lbl),
            "wall": ["north", "east", "south", "west"][wall_idx],
            "wall_idx": wall_idx,
            "label": cluster_label_text,
            "color": cluster_color,
            "size": cluster_size,
            "cohesion": cohesion,
            "grid": {"cols": cols, "rows": grows},
            "panel_start_local": float(panel_start_local),
            "panel_width": float(panel_w),
            "rotation_y": float(wg["rot_y"]),
            "lintel": {"x": float(lintel_x), "y": float(lintel_y), "z": float(lintel_z),
                       "rotation_y": float(wg["rot_y"])},
            "memories": memories,
        })

    # k-NN edges → ceiling threads. Only emit CROSS-cluster edges: same-cluster
    # neighbors already sit next to each other on the same wall, so threading
    # them adds visual noise without conveying information. Cross-cluster
    # threads are the actual story — "this beach photo's nearest neighbor in
    # vector space lives in the wedding room".
    nbrs = _knn_edges_local(matrix, [r[0] for r in rows], 3)
    edges: list[dict] = []
    for i, (pid, _, _) in enumerate(rows):
        if pid not in item_pos:
            continue
        src_cluster = item_cluster.get(pid)
        for nbr_id in nbrs[i]:
            if nbr_id not in item_pos:
                continue
            if item_cluster.get(nbr_id) == src_cluster:
                continue
            edges.append({"from": pid, "to": nbr_id})

    result = {
        "version": _version,
        "mode": "single-room",
        "room": room,
        "stats": {
            "total_memories": n,
            "n_clusters": len(unique_labels),
            "noise_count": noise_count,
            "vector_dims": [512, 384, 384],
            "clustering": f"HDBSCAN min_cluster_size={min_cluster_size} (cosine via L2-normalized euclidean)",
        },
        "panels": panels,
        "edges": edges,
    }
    _cache[cache_key] = result
    # Commit sticky slot map AFTER a successful build so a partial / failed
    # build never poisons the cache with half-assignments.
    _slot_memory.clear()
    _slot_memory.update(new_slot_memory)
    _last_good = result
    _dirty = False
    return result
