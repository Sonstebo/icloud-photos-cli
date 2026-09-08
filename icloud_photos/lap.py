"""Export the catalogue into a lap library (https://github.com/julyx10/lap).

lap is a desktop photo manager that browses folders: a library is a folder
root, files are rows mirrored from the folder, thumbnails are blobs, CLIP
vectors live in a column of the file table, and images reach its window
through URL schemes keyed by file id. This writes the rows a folder scan
would have produced, so lap needs no change to browse our library:

- one album rooted at a tree of symlinks under our cache
  (`<root>/YYYY/MM/<asset id>__<stem>.jpg` -> the best cached JPEG rendition;
  the id is filename-safe as in the cache),
- `afiles` with our dates, dimensions, GPS, favourite and caption,
- `athumbs` from our thumbnails, scaled to lap's thumbnail size,
- `afiles.embeds` from our CLIP vectors (lap uses the same ViT-B/32 weights),
- `persons` and `faces` from ours (both 512-d; lap clusters by cosine),
- `acollections` from our collections.

A file whose rendition is not cached keeps its row and thumbnail but has a
dangling symlink until something fetches it; lap shows the thumbnail and
cannot open it. Re-running is idempotent: rows are matched by folder and name.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .cache import Cache, safe_name
from .catalog import Catalog

Progress = Callable[[dict[str, Any]], None]

ALBUM_NAME = "iCloud Photos"
LAP_TABLES = ("albums", "afolders", "afiles", "athumbs", "persons", "faces", "acollections", "acollections_files")
FORMAT_LABELS = {"JPEG": "JPG", "JPE": "JPG", "JFIF": "JPG", "TIF": "TIFF", "MPG": "MPEG", "M4V": "MP4"}


def default_library() -> Path | None:
    """lap's first library database, if lap has run on this machine."""
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "com.julyx10.lap/libraries"
    for name in ("default.db",):
        if (base / name).exists():
            return base / name
    found = sorted(base.glob("*.db")) if base.exists() else []
    return found[0] if found else None


def _noop(_: dict[str, Any]) -> None:
    pass


def _seconds(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso).timestamp())
    except ValueError:
        return None


def format_label(filename: str) -> str | None:
    ext = Path(filename).suffix.lstrip(".").upper()
    return FORMAT_LABELS.get(ext, ext) or None


def scaled_thumbnail(data: bytes, size: int) -> bytes:
    """A JPEG whose longer side is `size`, as lap stores its own thumbnails."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((size, size))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


class LapLibrary:
    """One lap library database plus the album tree on disk."""

    def __init__(self, db_path: Path, root: Path, thumb_size: int = 200) -> None:
        self.db_path = Path(db_path)
        self.root = Path(root)
        self.thumb_size = thumb_size
        # autocommit mode: the export runs as one explicit transaction (BEGIN/COMMIT in `export`)
        self.db = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=30000")
        have = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = [t for t in LAP_TABLES if t not in have]
        if missing:
            raise ValueError(f"{self.db_path} is not a lap library (missing tables {', '.join(missing)}); run lap once to create one")

    def close(self) -> None:
        self.db.close()

    # --- album and folders ----------------------------------------------------
    def album_id(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        row = self.db.execute("SELECT id FROM albums WHERE path=?", (str(self.root),)).fetchone()
        if row:
            return int(row["id"])
        now = int(time.time())
        cur = self.db.execute("INSERT INTO albums (name, path, created_at, modified_at, indexed) VALUES (?,?,?,?,1)",
                              (ALBUM_NAME, str(self.root), now, now))
        return int(cur.lastrowid)

    def folder_id(self, album_id: int, path: Path, has_subfolders: bool) -> int:
        row = self.db.execute("SELECT id FROM afolders WHERE album_id=? AND path=?", (album_id, str(path))).fetchone()
        if row:
            self.db.execute("UPDATE afolders SET has_subfolders=? WHERE id=?", (int(has_subfolders), row["id"]))
            return int(row["id"])
        path.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        cur = self.db.execute(
            "INSERT INTO afolders (album_id, name, path, created_at, modified_at, is_favorite, has_subfolders) VALUES (?,?,?,?,?,0,?)",
            (album_id, path.name, str(path), now, now, int(has_subfolders)))
        return int(cur.lastrowid)

    # --- files ----------------------------------------------------------------
    def upsert_file(self, folder_id: int, name: str, values: dict[str, Any]) -> int:
        row = self.db.execute("SELECT id FROM afiles WHERE folder_id=? AND name=?", (folder_id, name)).fetchone()
        cols = list(values)
        if row:
            self.db.execute(f"UPDATE afiles SET {', '.join(f'{c}=?' for c in cols)} WHERE id=?",
                            [values[c] for c in cols] + [row["id"]])
            return int(row["id"])
        cur = self.db.execute(f"INSERT INTO afiles (folder_id, name, {', '.join(cols)}) VALUES (?,?,{','.join('?' * len(cols))})",
                              [folder_id, name] + [values[c] for c in cols])
        return int(cur.lastrowid)

    def put_thumb(self, file_id: int, data: bytes) -> None:
        now = int(time.time())
        try:
            data = scaled_thumbnail(data, self.thumb_size)
        except Exception:  # noqa: BLE001 - keep whatever the cache holds rather than no thumbnail
            pass
        self.db.execute(
            "INSERT OR REPLACE INTO athumbs (file_id, error_code, thumb_data, thumb_key, thumb_mtime, thumb_size, updated_at) VALUES (?,0,?,NULL,?,?,?)",
            (file_id, data, now, self.thumb_size, now))

    # --- people and collections ------------------------------------------------
    def person_id(self, name: str) -> int:
        row = self.db.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
        if row:
            return int(row["id"])
        cur = self.db.execute("INSERT INTO persons (name, created_at) VALUES (?,?)", (name, int(time.time())))
        return int(cur.lastrowid)

    def replace_faces(self, file_id: int, faces: list[tuple[dict[str, float], bytes | None, int | None]]) -> None:
        self.db.execute("DELETE FROM faces WHERE file_id=?", (file_id,))
        now = int(time.time())
        self.db.executemany("INSERT INTO faces (file_id, bbox, embedding, person_id, created_at) VALUES (?,?,?,?,?)",
                            [(file_id, json.dumps(box), emb, pid, now) for box, emb, pid in faces])

    def cover_faces(self) -> None:
        """Every person without a cover gets its largest face; lap renders the thumbnail from it."""
        self.db.execute("""UPDATE persons SET cover_face_id = (
            SELECT f.id FROM faces f WHERE f.person_id = persons.id
            ORDER BY json_extract(f.bbox, '$.width') * json_extract(f.bbox, '$.height') DESC LIMIT 1)
            WHERE cover_face_id IS NULL OR cover_face_id NOT IN (SELECT id FROM faces)""")

    def replace_collection(self, name: str, file_ids: list[int]) -> int:
        now = int(time.time())
        row = self.db.execute("SELECT id FROM acollections WHERE name=?", (name,)).fetchone()
        if row:
            cid = int(row["id"])
            self.db.execute("UPDATE acollections SET updated_at=? WHERE id=?", (now, cid))
        else:
            cid = int(self.db.execute("INSERT INTO acollections (name, sort_order, created_at, updated_at) VALUES (?,0,?,?)",
                                      (name, now, now)).lastrowid)
        self.db.execute("DELETE FROM acollections_files WHERE collection_id=?", (cid,))
        self.db.executemany("INSERT INTO acollections_files (collection_id, file_id, added_at) VALUES (?,?,?)",
                            [(cid, fid, now + i) for i, fid in enumerate(file_ids)])
        return cid


def _link(path: Path, target: Path | None) -> None:
    """A symlink at `path` to `target`, or none when there is nothing to point at."""
    if path.is_symlink() or path.exists():
        if target is not None and path.is_symlink() and os.readlink(path) == str(target):
            return
        path.unlink()
    if target is not None:
        path.symlink_to(target)


def rendition_dims(asset: dict[str, Any], version: str) -> tuple[int | None, int | None]:
    versions = asset.get("versions")
    if isinstance(versions, str):
        try:
            versions = json.loads(versions)
        except ValueError:
            versions = None
    v = (versions or {}).get(version) or {}
    return v.get("width"), v.get("height")


def export(catalog: Catalog, cache: Cache, lap: LapLibrary, *, limit: int | None = None,
           progress: Progress = _noop) -> dict[str, Any]:
    result: dict[str, Any] = {"library": str(lap.db_path), "root": str(lap.root), "files": 0, "linked": 0,
                              "thumbs": 0, "embeddings": 0, "faces": 0, "people": 0, "collections": 0, "skipped": 0}
    album = lap.album_id()
    root_folder = lap.folder_id(album, lap.root, True)
    folders: dict[str, int] = {}
    file_ids: dict[str, int] = {}
    people = {p["id"]: p for p in catalog.people() if p.get("name")}
    lap_people: dict[str, int] = {}
    q = "SELECT * FROM assets WHERE hidden=0 AND missing_since IS NULL ORDER BY taken DESC" + (f" LIMIT {int(limit)}" if limit else "")
    assets = [dict(r) for r in catalog.db.execute(q)]
    lap.db.execute("BEGIN")
    try:
        for n, a in enumerate(assets, 1):
            taken = _seconds(a.get("taken")) or _seconds(a.get("added")) or 0
            when = datetime.fromtimestamp(taken) if taken else datetime(1970, 1, 1)
            ykey, mkey = f"{when.year:04d}", f"{when.year:04d}/{when.month:02d}"
            if ykey not in folders:
                folders[ykey] = lap.folder_id(album, lap.root / ykey, True)
            if mkey not in folders:
                folders[mkey] = lap.folder_id(album, lap.root / mkey, False)
            stem = Path(a["filename"] or a["id"]).stem
            is_image = a.get("kind") == "image"
            # the file the row stands for: the best cached JPEG rendition of an image, the original of a movie
            if is_image:
                target = cache.peek(a["id"], "medium") or cache.peek(a["id"], "thumb")
                linked_version = "medium" if target and target.parent.name == "medium" else "thumb"
                name = f"{safe_name(a['id'])}__{stem}.jpg"
            else:
                target = cache.peek(a["id"], "original")
                linked_version = "original"
                name = f"{safe_name(a['id'])}__{a['filename'] or stem}"
            values = {
                "size": int(a.get("bytes") or 0), "file_type": 1 if is_image else 2,
                "format_label": "JPG" if is_image else format_label(a.get("filename") or ""),
                "created_at": taken, "modified_at": taken, "taken_date": taken,
                "width": a.get("width"), "height": a.get("height"),
                "is_favorite": int(bool(a.get("favorite"))), "comments": a.get("caption"),
                "gps_latitude": a.get("latitude"), "gps_longitude": a.get("longitude"),
                "last_scan_time": int(time.time() * 1000),
            }
            clip = catalog.clip_of(a["id"]) if is_image else None
            if clip is not None:
                values["embeds"] = clip
                result["embeddings"] += 1
            faces = [dict(f) for f in catalog.db.execute("SELECT * FROM faces WHERE asset_id=?", (a["id"],))] if is_image else []
            values["has_faces"] = 1 if faces else (2 if catalog.db.execute(
                "SELECT 1 FROM index_state WHERE asset_id=?", (a["id"],)).fetchone() else 0)
            fid = lap.upsert_file(folders[mkey], name, values)
            file_ids[a["id"]] = fid
            result["files"] += 1
            _link(lap.root / mkey / name, target)
            if target is not None:
                result["linked"] += 1
            thumb = cache.peek(a["id"], "thumb")
            if thumb is not None:
                lap.put_thumb(fid, thumb.read_bytes())
                result["thumbs"] += 1
            if faces:
                # our boxes are in the pixels of the rendition the face pass saw; lap wants the linked file's
                sw, sh = rendition_dims(a, faces[0].get("source") or "thumb")
                lw, lh = rendition_dims(a, linked_version) if target is not None else (sw, sh)
                sx = (lw / sw) if sw and lw else 1.0
                sy = (lh / sh) if sh and lh else 1.0
                rows = []
                for f in faces:
                    x1, y1, x2, y2 = json.loads(f["box"]) if isinstance(f["box"], str) else f["box"]
                    box = {"x": x1 * sx, "y": y1 * sy, "width": (x2 - x1) * sx, "height": (y2 - y1) * sy,
                           "confidence": float(f.get("det") or 0)}
                    pid = None
                    if f.get("person_id") in people:
                        pname = people[f["person_id"]].get("display_name") or people[f["person_id"]]["name"]
                        if pname not in lap_people:
                            lap_people[pname] = lap.person_id(pname)
                        pid = lap_people[pname]
                    rows.append((box, f.get("embedding"), pid))
                lap.replace_faces(fid, rows)
                result["faces"] += len(rows)
            if n % 500 == 0:
                progress(dict(result, done=n, total=len(assets)))
        result["people"] = len(lap_people)
        lap.cover_faces()
        for c in catalog.db.execute("SELECT name FROM collections ORDER BY created"):
            ids = [file_ids[r["asset_id"]] for r in catalog.db.execute(
                "SELECT asset_id FROM collection_assets WHERE collection=? ORDER BY position", (c["name"],)) if r["asset_id"] in file_ids]
            lap.replace_collection(c["name"], ids)
            result["collections"] += 1
        lap.db.execute("UPDATE albums SET total=?, indexed=1, last_scan_time=?, modified_at=? WHERE id=?",
                       (result["files"], int(time.time() * 1000), int(time.time()), album))
        lap.db.execute("COMMIT")
    except BaseException:
        lap.db.execute("ROLLBACK")
        raise
    result["skipped"] = len(assets) - result["files"]
    return result
