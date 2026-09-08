"""Index images: a CLIP embedding and the faces in each, matched to named people.

Pass 1 works on thumbnails (fast, one small download each); the source
rendition is recorded so a later pass on medium previews can redo the faces
of photos that deserve it. Seeds are the face crops iCloud keeps for each
named person: embedding those gives a labelled reference set, so a face in a
photo is assigned to a person when it is close enough to one of that
person's seeds. Progress is committed per batch and `index` resumes.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from .adapter import Adapter
from .cache import BudgetExceeded, Cache
from .catalog import Catalog, now
from .ml import Models, as_blob, decode_image, from_blob

Progress = Callable[[dict[str, Any]], None]
DEFAULT_THRESHOLD = 0.5      # cosine similarity; Immich's default recognition distance is 0.5 too


def _noop(_: dict[str, Any]) -> None:
    pass


class SeedIndex:
    """All seed embeddings in one matrix, for one dot product per face."""

    def __init__(self, seeds: list[dict[str, Any]]) -> None:
        self.person_ids = [s["person_id"] for s in seeds]
        self.matrix = np.stack([from_blob(s["embedding"]) for s in seeds]) if seeds else np.zeros((0, 512), np.float32)

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float | None]:
        if not len(self.person_ids):
            return None, None
        sims = self.matrix @ embedding
        best = int(np.argmax(sims))
        sim = float(sims[best])
        return (self.person_ids[best], sim) if sim >= threshold else (None, sim)


def seed_people(catalog: Catalog, adapter: Adapter, models: Models, *, named_only: bool = True,
                progress: Progress = _noop) -> dict[str, Any]:
    """Embed iCloud's face crops for named people into person_seeds."""
    crops = catalog.unseeded_crops(named_only=named_only)
    person_of = {c["id"]: c["person_id"] for c in crops}
    result = {"crops": len(crops), "seeded": 0, "no_face": 0, "failed": 0}
    for n, (crop_id, data) in enumerate(adapter.download_face_crops(list(person_of)), 1):
        crop = {"id": crop_id, "person_id": person_of[crop_id]}
        img = decode_image(data) if data else None
        if img is None:
            result["failed"] += 1
        else:
            faces = models.faces(img)
            if not faces:
                result["no_face"] += 1
            else:
                best = max(faces, key=lambda f: f["det"])
                catalog.put_seed(crop["id"], crop["person_id"], best["embedding"], best["det"], "icloud-face-crop")
                result["seeded"] += 1
        if n % 20 == 0:
            progress(result | {"done": n})
    return result


def index(catalog: Catalog, adapter: Adapter, cache: Cache, models: Models, *, limit: int | None = None,
          batch: int = 25, threshold: float = DEFAULT_THRESHOLD, source: str = "thumb",
          progress: Progress = _noop) -> dict[str, Any]:
    todo = catalog.unindexed(models.clip_name, models.face_name, limit=limit)
    result: dict[str, Any] = {"started": now(), "todo": len(todo), "done": 0, "faces": 0, "named": 0,
                              "fetched": 0, "failed": 0, "source": source}
    seeds = SeedIndex(catalog.seeds())
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def rendition(a: dict[str, Any]) -> str | None:
        """The rendition to analyse: the pass's own, else a medium preview, else a small
        original; None for what iCloud offers no usable image of (RAW files, typically)."""
        versions = a.get("versions") or {}
        if isinstance(versions, str):
            versions = json.loads(versions)
        if source in versions:
            return source
        if "medium" in versions:
            return "medium"
        original = versions.get("original") or {}
        if (original.get("bytes") or 0) <= 8 * 1024 * 1024 and "raw" not in (original.get("type") or ""):
            return "original"
        return None

    used: dict[str, str] = {}   # asset id -> the rendition it was analysed from

    def prepare(chunk: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        """Main thread: what the cache already has, and what must be downloaded."""
        images: dict[str, np.ndarray] = {}
        need = []
        for a in chunk:
            version = rendition(a)
            if version is None:
                continue
            used[a["id"]] = version
            path = cache.get(a["id"], version)
            if path is not None:
                img = decode_image(path.read_bytes())
                if img is not None:
                    images[a["id"]] = img
                    continue
            need.append(a)
        return images, need

    def download(need: list[dict[str, Any]]) -> list[tuple[str, bytes]]:
        """Worker thread: network only, no catalogue access; one batch per rendition."""
        out: list[tuple[str, bytes]] = []
        for version in sorted({used[a["id"]] for a in need}):
            items = [(a["id"], a["master_id"]) for a in need if used[a["id"]] == version]
            out += [(asset_id, data) for asset_id, data in adapter.download_many(items, version) if data is not None]
        return out

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)     # the next chunk downloads while this one is analysed
    # Only the chunk being analysed and the next one are held; keeping every staged
    # chunk's decoded images alive grew the worker by ~12 MB per chunk until the OOM killer took it.
    staged = prepare(chunks[0]) if chunks else None
    pending = pool.submit(download, staged[1]) if chunks else None
    for i, chunk in enumerate(chunks):
        images, _ = staged
        downloaded = pending.result()
        if i + 1 < len(chunks):
            staged = prepare(chunks[i + 1])
            pending = pool.submit(download, staged[1])
        for asset_id, data in downloaded:
            img = decode_image(data)
            if img is None:
                continue
            images[asset_id] = img
            result["fetched"] += 1
            try:
                cache.put(asset_id, used[asset_id], data, ".jpg")
            except BudgetExceeded:
                pass   # indexing does not need the file kept
        catalog.db.execute("BEGIN")
        for a in chunk:
            if a["id"] not in used:
                # nothing to analyse; recorded so the asset is not retried every run
                catalog.set_index_state(a["id"], models.clip_name, models.face_name, "none", 0)
                result["skipped"] = result.get("skipped", 0) + 1
                continue
            img = images.get(a["id"])
            if img is None:
                result["failed"] += 1
                continue
            catalog.put_clip(a["id"], as_blob(models.embed_image(img)))
            faces = models.faces(img)
            for f in faces:
                person, sim = seeds.match(from_blob(f["embedding"]), threshold)
                f["person_id"], f["similarity"] = person, sim
                result["named"] += bool(person)
            catalog.put_faces(a["id"], faces, used[a["id"]])
            catalog.set_index_state(a["id"], models.clip_name, models.face_name, used[a["id"]], len(faces))
            result["done"] += 1
            result["faces"] += len(faces)
        catalog.set_meta("index_progress", json.dumps(result))
        catalog.db.execute("COMMIT")
        progress(result)
    pool.shutdown(wait=True)
    result["finished"] = now()
    catalog.set_meta("last_index", json.dumps(result))
    catalog.set_meta("index_progress", None)
    return result


def rematch(catalog: Catalog, threshold: float = DEFAULT_THRESHOLD) -> dict[str, int]:
    """Re-run seed matching over every automatically assigned or unassigned face (after new seeds)."""
    seeds = SeedIndex(catalog.seeds())
    changed = 0
    rows = catalog.db.execute("SELECT id, embedding, person_id FROM faces WHERE assigned IS NULL OR assigned='auto'").fetchall()
    catalog.db.execute("BEGIN")
    for r in rows:
        person, sim = seeds.match(from_blob(r["embedding"]), threshold)
        if person != r["person_id"]:
            changed += 1
        catalog.assign_face(r["id"], person, sim, "auto")
    catalog.db.execute("COMMIT")
    return {"faces": len(rows), "changed": changed}
