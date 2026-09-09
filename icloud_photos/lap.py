"""Export the catalogue into a lap library (https://github.com/julyx10/lap).

lap is a desktop photo manager that browses folders: a library is a folder
root, files are rows mirrored from the folder, thumbnails are blobs, CLIP
vectors live in a column of the file table, and images reach its window
through URL schemes keyed by file id. This writes the rows a folder scan
would have produced, so lap needs no change to browse our library:

- one album rooted at a tree of symlinks under our cache
  (`<root>/YYYY/MM/<asset id>@<original name>` -> that rendition once it is cached;
  the id is filename-safe as in the cache, and `@` cannot occur in it),
- `afiles` with our dates, dimensions, GPS, favourite and caption,
- `athumbs` from our thumbnails (`--fetch-thumbs` downloads the missing ones,
  movies included), within lap's thumbnail size (512 by default,
  its gallery setting; a mismatch makes lap regenerate every thumbnail),
- `afiles.embeds` from our CLIP vectors (lap uses the same ViT-B/32 weights),
- `persons` and `faces` from ours (both 512-d; lap clusters by cosine),
- `acollections` from our collections.

A file whose rendition is not cached keeps its row and thumbnail but has a
dangling symlink until lap's fetch-on-open command (`photos lap-fetch`) gets
it. Re-running is idempotent: rows are matched by folder and name, and rows or
links the catalogue no longer has are removed.
"""

from __future__ import annotations

import fcntl
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
COMMIT_EVERY = 2000
LAP_TABLES = ("albums", "afolders", "afiles", "athumbs", "persons", "faces", "acollections", "acollections_files")
FORMAT_LABELS = {"JPEG": "JPG", "JPE": "JPG", "JFIF": "JPG", "TIF": "TIFF", "MPG": "MPEG", "M4V": "MP4"}
# The app's own file types (t_utils.rs::get_file_type): 1 image, 2 video, 3 RAW. A RAW
# labelled 1 is sent to the webview, which cannot decode it; it must be 3 so the app
# decodes it with LibRaw. This mirrors t_common.rs::RAW_IMGS.
RAW_SUFFIXES = {
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "raf", "rw2", "orf", "pef", "dng",
    "srw", "rwl", "mrw", "3fr", "mos", "iiq", "dcr", "kdc", "erf", "mef", "raw", "mdc",
}


def file_type_of(name: str, kind: str) -> int:
    if kind != "image":
        return 2
    return 3 if Path(name).suffix.lstrip(".").lower() in RAW_SUFFIXES else 1


# The GUI's data directory, newest identifier first; a build may use any of them.
APP_IDS = ("com.sonstebo.photos", "com.julyx10.lap")


def default_library() -> Path | None:
    """The GUI's library database, if it has run on this machine."""
    root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    for app_id in APP_IDS:
        base = (root / app_id / "libraries").resolve()
        if (base / "default.db").exists():
            return base / "default.db"
        found = sorted(base.glob("*.db")) if base.exists() else []
        if found:
            return found[0]
    return None


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
    """A JPEG whose longer side is at most `size`, as lap stores its own
    thumbnails; bytes already within the size are kept as they are."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if max(img.size) <= size and img.format == "JPEG":
        return data
    img = img.convert("RGB")
    img.thumbnail((size, size))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


class LapLibrary:
    """One lap library database plus the album tree on disk."""

    def __init__(self, db_path: Path, root: Path, thumb_size: int = 512) -> None:
        self.db_path = Path(db_path)
        self.root = Path(root)
        self.thumb_size = thumb_size
        # autocommit mode: the export runs as one explicit transaction (BEGIN/COMMIT in `export`)
        self.db = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=120000")   # the app may hold the write lock while it starts or stops
        self.db.execute("PRAGMA foreign_keys=ON")   # the app's tables cascade: deleting a file drops its thumbnail and faces
        self.db.execute("PRAGMA cache_size=-20000")  # 20 MB, not the default share of a 2 GB database
        # One export at a time: two would fight over the same write transaction.
        self._lock = (self.db_path.parent / f".{self.db_path.name}.export.lock").open("w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            self.db.close()
            raise ValueError(f"another export is already writing {self.db_path}") from None
        have = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = [t for t in LAP_TABLES if t not in have]
        if missing:
            raise ValueError(f"{self.db_path} is not a lap library (missing tables {', '.join(missing)}); run lap once to create one")

    def close(self) -> None:
        self.db.close()
        try:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
        except (OSError, ValueError):
            pass

    def app_is_running(self) -> bool:
        """Whether a GUI process holds this library open. Writing while it runs is
        safe (WAL, 30 s busy timeout) but its own writes can fail, so callers warn."""
        import subprocess
        try:
            out = subprocess.run(["pgrep", "-x", "Photos", "-x", "Lap"], capture_output=True, text=True, timeout=5)
            return out.returncode == 0 and bool(out.stdout.strip())
        except Exception:  # noqa: BLE001 - pgrep is optional
            return False

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

    def set_fetch_command(self, album_id: int, command: str | None) -> None:
        """The album's fetch-on-open command, when the app has that column (its migration 17)."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(albums)")}
        if "fetch_command" in cols:
            self.db.execute("UPDATE albums SET fetch_command=? WHERE id=?", (command, album_id))

    def set_cli(self, album_id: int, cli: str) -> bool:
        """The program that maintains this album, which the app asks for edits and
        versions (its migration 19). A path, never a shell line."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(albums)")}
        if "cli" not in cols:
            return False
        self.db.execute("UPDATE albums SET cli=? WHERE id=?", (cli, album_id))
        return True

    def set_managed(self, album_id: int) -> bool:
        """Mark the album as maintained here (the app's migration 18). Without it the
        app rescans the folder on start and deletes every row whose file is not on
        disk, which is all of them until a preview is opened."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(albums)")}
        if "managed" not in cols:
            return False
        self.db.execute("UPDATE albums SET managed=1 WHERE id=?", (album_id,))
        return True

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

    def put_thumb(self, file_id: int, data: bytes, source_mtime: int | None) -> None:
        """`source_mtime` is the linked file's modification time in seconds: lap
        treats a thumbnail whose recorded mtime differs from the file's as stale
        and regenerates it, which for 33k entries is what made it unresponsive."""
        now = int(time.time())
        try:
            data = scaled_thumbnail(data, self.thumb_size)
        except Exception:  # noqa: BLE001 - keep whatever the cache holds rather than no thumbnail
            pass
        self.db.execute(
            "INSERT OR REPLACE INTO athumbs (file_id, error_code, thumb_data, thumb_key, thumb_mtime, thumb_size, updated_at) VALUES (?,0,?,NULL,?,?,?)",
            (file_id, data, source_mtime if source_mtime is not None else now, self.thumb_size, now))

    def put_thumb_error(self, file_id: int) -> None:
        """Mark a file as having no thumbnail source; lap keeps error rows for missing files."""
        if self.db.execute("SELECT 1 FROM athumbs WHERE file_id=?", (file_id,)).fetchone():
            return
        now = int(time.time())
        self.db.execute("INSERT INTO athumbs (file_id, error_code, thumb_data, thumb_key, thumb_mtime, thumb_size, updated_at) VALUES (?,1,NULL,NULL,?,?,?)",
                        (file_id, now, self.thumb_size, now))

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


def asset_id_of(entry_name: str) -> str:
    """The (filename-safe) asset id an album entry name starts with."""
    return entry_name.split("@", 1)[0]


def version_for_entry(entry_name: str, asset: dict[str, Any], preferred: str = "original") -> str:
    """The rendition an entry stands for: the preferred one when the asset has it and
    its type matches the entry's extension, else whatever does."""
    versions = versions_of(asset)
    suffix = Path(entry_name).suffix.lower()
    order = [preferred, "original", "medium", "thumb"] if asset.get("kind") == "image" else [
        "original", "medium_image", "thumb_image"]
    for v in order:
        info = versions.get(v)
        if not info:
            continue
        name = info.get("filename") or ""
        if v == "medium" and suffix == ".jpg":
            return v
        if Path(name).suffix.lower() == suffix or v == preferred:
            return v
    return next((v for v in order if v in versions), "thumb")


def link(path: Path, target: Path | None) -> None:
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


def _fetch_thumb(a: dict[str, Any], adapter: Any, cache: Cache) -> Path | None:
    """Download something to make a thumbnail from: the thumbnail rendition (movies call
    it `thumb_image`), else the medium preview, which is what iCloud offers for HEIC and
    RAW imports that have no thumbnail at all."""
    versions = versions_of(a)
    version = "thumb" if a.get("kind") == "image" else "thumb_image"
    if version not in versions:
        version = "medium" if "medium" in versions else "medium_image"
    if version not in versions:
        return None
    try:
        data = adapter.download(a["id"], version, a.get("master_id"))
    except Exception:  # noqa: BLE001 - one thumbnail is not worth ending the export
        return None
    if not data:
        return None
    try:
        return cache.put(a["id"], version, data, ".jpg")
    except Exception:  # noqa: BLE001 - cache full: still usable for this run
        tmp = cache.root / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        p = tmp / f"{safe_name(a['id'])}.jpg"
        p.write_bytes(data)
        return p


def versions_of(a: dict[str, Any]) -> dict[str, Any]:
    v = a.get("versions") or {}
    return json.loads(v) if isinstance(v, str) else v


def export(catalog: Catalog, cache: Cache, lap: LapLibrary, *, limit: int | None = None,
           fetch_command: str | None = None, cli: str | None = None, fetch_thumbs: bool = False,
           adapter: Any = None, open_version: str = "original", progress: Progress = _noop) -> dict[str, Any]:
    result: dict[str, Any] = {"library": str(lap.db_path), "root": str(lap.root), "open_version": open_version, "files": 0, "linked": 0,
                              "thumbs": 0, "embeddings": 0, "faces": 0, "people": 0, "collections": 0, "skipped": 0}
    album = lap.album_id()
    lap.set_fetch_command(album, fetch_command)
    result["managed"] = lap.set_managed(album)
    if cli:
        lap.set_cli(album, cli)
    root_folder = lap.folder_id(album, lap.root, True)
    folders: dict[str, int] = {}
    file_ids: dict[str, int] = {}
    people = {p["id"]: p for p in catalog.people() if p.get("name")}
    lap_people: dict[str, int] = {}
    q = "SELECT * FROM assets WHERE hidden=0 AND missing_since IS NULL ORDER BY taken DESC" + (f" LIMIT {int(limit)}" if limit else "")
    assets = [dict(r) for r in catalog.db.execute(q)]
    run_started_ms = int(time.time() * 1000)
    seen: set[str] = set()
    lap.db.execute("BEGIN")
    try:
        for n, a in enumerate(assets, 1):
            if n % COMMIT_EVERY == 0:
                # A single transaction over every thumbnail blob costs gigabytes of page
                # cache on a library this size. Committing in batches keeps it flat; the
                # export is idempotent, so a partial pass is picked up by the next one.
                lap.db.execute("COMMIT")
                lap.db.execute("BEGIN")
            taken = _seconds(a.get("taken")) or _seconds(a.get("added")) or 0
            when = datetime.fromtimestamp(taken) if taken else datetime(1970, 1, 1)
            ykey, mkey = f"{when.year:04d}", f"{when.year:04d}/{when.month:02d}"
            if ykey not in folders:
                folders[ykey] = lap.folder_id(album, lap.root / ykey, True)
            if mkey not in folders:
                folders[mkey] = lap.folder_id(album, lap.root / mkey, False)
            filename = a["filename"] or f"{a['id']}.jpg"
            stem, suffix = Path(filename).stem, Path(filename).suffix
            is_image = a.get("kind") == "image"
            # The entry stands for one rendition and is named after it, because the app picks
            # its decoder from the extension: an original HEIC named .jpg would go to the
            # webview, which cannot read it. `open_version` says which rendition that is; the
            # entry dangles until the fetch-on-open command materialises it.
            linked_version = open_version if open_version in (a_versions := versions_of(a)) else (
                "medium" if "medium" in a_versions else "original")
            if linked_version == "medium":
                suffix = ".jpg"      # the medium rendition is always JPEG
            target = cache.peek(a["id"], linked_version)
            name = f"{safe_name(a['id'])}@{stem}{suffix}"
            values = {
                "size": int(a.get("bytes") or 0), "file_type": file_type_of(name, a.get("kind") or ""),
                "format_label": format_label(name),
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
            seen.add(str(lap.root / mkey / name))
            result["files"] += 1
            link(lap.root / mkey / name, target)
            if target is not None:
                result["linked"] += 1
            thumb = cache.peek(a["id"], "thumb")
            if thumb is None and fetch_thumbs and adapter is not None:
                thumb = _fetch_thumb(a, adapter, cache)
                if thumb is not None:
                    result["fetched"] = result.get("fetched", 0) + 1
            if thumb is not None:
                lap.put_thumb(fid, thumb.read_bytes(), int(target.stat().st_mtime) if target is not None else None)
                result["thumbs"] += 1
            elif target is None and not is_image:
                # A movie with nothing on disk: an error row stops the app spawning ffmpeg
                # for it on every start. An image is left without a row, so the app can
                # make its own thumbnail once the fetch-on-open command brings the file in.
                lap.put_thumb_error(fid)
            if faces:
                # our boxes are in the pixels of the rendition the face pass saw; lap wants the linked file's
                sw, sh = rendition_dims(a, faces[0].get("source") or "thumb")
                lw, lh = rendition_dims(a, linked_version)
                if not (lw and lh):
                    lw, lh = sw, sh
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
        # thumbnails and faces whose file row is gone (from before foreign keys were on)
        result["orphans"] = lap.db.execute("DELETE FROM athumbs WHERE file_id NOT IN (SELECT id FROM afiles)").rowcount
        result["orphans"] += lap.db.execute("DELETE FROM faces WHERE file_id NOT IN (SELECT id FROM afiles)").rowcount
        if limit is None:
            # rows and links for assets the catalogue no longer has (or renamed entries)
            stale = lap.db.execute(
                "SELECT a.id, b.path, a.name FROM afiles a JOIN afolders b ON a.folder_id=b.id "
                "WHERE b.album_id=? AND a.last_scan_time < ?", (album, run_started_ms)).fetchall()
            for r in stale:
                lap.db.execute("DELETE FROM afiles WHERE id=?", (r["id"],))
                link(Path(r["path"]) / r["name"], None)
            result["removed"] = len(stale)
            for p in lap.root.rglob("*"):
                if p.is_symlink() and str(p) not in seen:
                    p.unlink()
        lap.db.execute("UPDATE albums SET total=?, indexed=1, last_scan_time=?, modified_at=? WHERE id=?",
                       (result["files"], int(time.time() * 1000), int(time.time()), album))
        lap.db.execute("COMMIT")
    except BaseException:
        lap.db.execute("ROLLBACK")
        raise
    result["skipped"] = len(assets) - result["files"]
    return result
