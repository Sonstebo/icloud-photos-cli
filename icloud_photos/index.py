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
    result = {"crops": len(crops), "seeded": 0, "no_face": 0, "failed": 0}
    for n, crop in enumerate(crops, 1):
        data = adapter.download_face_crop(crop["id"])
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
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        images: dict[str, np.ndarray] = {}
        need = []
        for a in chunk:
            path = cache.get(a["id"], source)
            if path is not None:
                img = decode_image(path.read_bytes())
                if img is not None:
                    images[a["id"]] = img
                    continue
            need.append(a)
        for asset_id, data in adapter.download_many([(a["id"], a["master_id"]) for a in need], source):
            if data is None:
                continue
            result["fetched"] += 1
            img = decode_image(data)
            if img is None:
                continue
            images[asset_id] = img
            a = next(x for x in need if x["id"] == asset_id)
            ext = ".jpg"
            try:
                cache.put(asset_id, source, data, ext)
            except BudgetExceeded:
                pass   # indexing does not need the file kept
        catalog.db.execute("BEGIN")
        for a in chunk:
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
            catalog.put_faces(a["id"], faces, source)
            catalog.set_index_state(a["id"], models.clip_name, models.face_name, source, len(faces))
            result["done"] += 1
            result["faces"] += len(faces)
        catalog.set_meta("index_progress", json.dumps(result))
        catalog.db.execute("COMMIT")
        progress(result)
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
