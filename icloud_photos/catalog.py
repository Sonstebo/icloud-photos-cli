"""The durable local catalogue: SQLite, one file, no ORM.

Holds what is known about every asset, album membership, what the cache has
on disk, and the user's collections. Queries never touch the network.
Dates are stored as ISO 8601 UTC text ("2024-05-01T12:00:00+00:00") so
they sort correctly as strings.
"""
from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .adapter import AssetInfo

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    master_id TEXT,
    filename TEXT,
    kind TEXT,
    live INTEGER,
    taken TEXT,
    added TEXT,
    width INTEGER,
    height INTEGER,
    bytes INTEGER,
    favorite INTEGER,
    caption TEXT,
    latitude REAL,
    longitude REAL,
    hidden INTEGER,
    versions TEXT,
    fingerprint TEXT,
    first_seen TEXT,
    last_seen TEXT,
    missing_since TEXT
);
CREATE INDEX IF NOT EXISTS assets_taken ON assets (taken DESC, id);
CREATE INDEX IF NOT EXISTS assets_master ON assets (master_id);
CREATE TABLE IF NOT EXISTS albums (id TEXT PRIMARY KEY, name TEXT, fullname TEXT, synced TEXT);
CREATE TABLE IF NOT EXISTS album_assets (
    album_id TEXT, asset_id TEXT, PRIMARY KEY (album_id, asset_id));
CREATE TABLE IF NOT EXISTS cache (
    asset_id TEXT, version TEXT, path TEXT, bytes INTEGER,
    fetched TEXT, used TEXT, pinned INTEGER DEFAULT 0,
    PRIMARY KEY (asset_id, version));
CREATE TABLE IF NOT EXISTS records (
    name TEXT PRIMARY KEY, type TEXT, deleted INTEGER, modified TEXT, master_ref TEXT, json TEXT);
CREATE INDEX IF NOT EXISTS records_master_ref ON records (master_ref);
CREATE INDEX IF NOT EXISTS records_type ON records (type);
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY, name TEXT, display_name TEXT, verified INTEGER, kind INTEGER, modified TEXT, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS face_crops (
    id TEXT PRIMARY KEY, person_id TEXT, kind INTEGER, bytes INTEGER, modified TEXT, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS face_crops_person ON face_crops (person_id);
CREATE TABLE IF NOT EXISTS index_state (
    asset_id TEXT PRIMARY KEY, clip_model TEXT, face_model TEXT, source TEXT, indexed TEXT, faces INTEGER);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY, asset_id TEXT, box TEXT, det REAL, age INTEGER, gender INTEGER,
    embedding BLOB, source TEXT, person_id TEXT, similarity REAL, assigned TEXT);
CREATE INDEX IF NOT EXISTS faces_asset ON faces (asset_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces (person_id);
CREATE TABLE IF NOT EXISTS person_seeds (
    id TEXT PRIMARY KEY, person_id TEXT, embedding BLOB, det REAL, origin TEXT, created TEXT);
CREATE INDEX IF NOT EXISTS person_seeds_person ON person_seeds (person_id);
CREATE TABLE IF NOT EXISTS collections (name TEXT PRIMARY KEY, created TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS collection_assets (
    collection TEXT, asset_id TEXT, position INTEGER, note TEXT,
    PRIMARY KEY (collection, asset_id));
"""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value else None


def encode_cursor(taken: str | None, asset_id: str) -> str:
    return base64.urlsafe_b64encode(json.dumps([taken, asset_id]).encode()).decode()


def decode_cursor(cursor: str) -> tuple[str | None, str]:
    try:
        taken, asset_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return taken, asset_id
    except Exception as err:  # noqa: BLE001
        raise ValueError("bad cursor") from err


class Catalog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)  # autocommit; explicit BEGIN below
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        if self.get_meta("schema") is None:
            self.set_meta("schema", str(SCHEMA_VERSION))
        self.vec = self._load_vec()

    def _load_vec(self) -> bool:
        """sqlite-vec gives the catalogue nearest-neighbour search; without it, no semantic search."""
        try:
            import sqlite_vec
        except ImportError:
            return False
        try:
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError):
            return False
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip USING vec0(asset_id TEXT PRIMARY KEY, e float[512])")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_face USING vec0(face_id INTEGER PRIMARY KEY, e float[512])")
        return True

    def close(self) -> None:
        self.db.close()

    # --- meta -------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str | None) -> None:
        if value is None:
            self.db.execute("DELETE FROM meta WHERE key=?", (key,))
        else:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # --- assets -----------------------------------------------------------
    def upsert_asset(self, info: AssetInfo, seen: str | None = None) -> str:
        """Insert or update; returns 'new', 'changed' or 'same'."""
        seen = seen or now()
        fp = info.fingerprint()
        row = self.db.execute("SELECT fingerprint FROM assets WHERE id=?", (info.id,)).fetchone()
        if row is None:
            status = "new"
        elif row["fingerprint"] != fp:
            status = "changed"
        else:
            status = "same"
            self.db.execute("UPDATE assets SET last_seen=? WHERE id=?", (seen, info.id))
            return status
        self.db.execute(
            """INSERT INTO assets (id, master_id, filename, kind, live, taken, added, width, height,
                   bytes, favorite, caption, latitude, longitude, hidden, versions, fingerprint,
                   first_seen, last_seen, missing_since)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET master_id=excluded.master_id, filename=excluded.filename,
                   kind=excluded.kind, live=excluded.live, taken=excluded.taken, added=excluded.added,
                   width=excluded.width, height=excluded.height, bytes=excluded.bytes,
                   favorite=excluded.favorite, caption=excluded.caption, latitude=excluded.latitude,
                   longitude=excluded.longitude, hidden=excluded.hidden, versions=excluded.versions,
                   fingerprint=excluded.fingerprint, last_seen=excluded.last_seen,
                   missing_since=CASE WHEN excluded.missing_since IS NULL THEN NULL ELSE COALESCE(assets.missing_since, excluded.missing_since) END""",
            (info.id, info.master_id, info.filename, info.kind, int(info.live), iso(info.taken),
             iso(info.added), info.width, info.height, info.bytes, int(info.favorite), info.caption,
             info.latitude, info.longitude, int(info.hidden), json.dumps(info.versions), fp, seen, seen,
             seen if info.deleted else None),
        )
        return status

    # --- raw records --------------------------------------------------------
    def put_record(self, name: str, rtype: str | None, deleted: bool, modified: str | None,
                   master_ref: str | None, payload: dict[str, Any] | None) -> None:
        self.db.execute(
            """INSERT INTO records (name, type, deleted, modified, master_ref, json) VALUES (?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET type=COALESCE(excluded.type, records.type), deleted=excluded.deleted,
                   modified=COALESCE(excluded.modified, records.modified),
                   master_ref=COALESCE(excluded.master_ref, records.master_ref),
                   json=CASE WHEN excluded.json IS NULL THEN records.json ELSE excluded.json END""",
            (name, rtype, int(deleted), modified, master_ref, json.dumps(payload) if payload is not None else None))

    def get_record(self, name: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM records WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["json"] = json.loads(d["json"]) if d["json"] else None
        return d

    def assets_of_master(self, master_name: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM records WHERE master_ref=? AND type='CPLAsset' AND deleted=0", (master_name,))
        out = []
        for row in rows:
            d = dict(row); d["json"] = json.loads(d["json"]) if d["json"] else None
            out.append(d)
        return out

    def record_counts(self) -> dict[str, int]:
        return {r["type"] or "tombstone": r["n"] for r in self.db.execute(
            "SELECT type, COUNT(*) n FROM records WHERE deleted=0 GROUP BY type")}

    # --- album membership from container relations --------------------------
    def set_relation(self, container_id: str, item_id: str, deleted: bool) -> None:
        if deleted:
            self.db.execute("DELETE FROM album_assets WHERE album_id=? AND asset_id=?", (container_id, item_id))
        else:
            self.db.execute("INSERT OR IGNORE INTO album_assets (album_id, asset_id) VALUES (?, ?)",
                            (container_id, item_id))

    # --- people -------------------------------------------------------------
    def put_person(self, pid: str, name: str | None, display: str | None, verified: bool, kind: int | None,
                   modified: str | None, deleted: bool = False) -> None:
        self.db.execute(
            """INSERT INTO people (id, name, display_name, verified, kind, modified, deleted) VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, display_name=excluded.display_name,
                   verified=excluded.verified, kind=excluded.kind, modified=excluded.modified, deleted=excluded.deleted""",
            (pid, name, display, int(verified), kind, modified, int(deleted)))

    def put_face_crop(self, fid: str, person_id: str | None, kind: int | None, size: int | None,
                      modified: str | None, deleted: bool = False) -> None:
        self.db.execute(
            """INSERT INTO face_crops (id, person_id, kind, bytes, modified, deleted) VALUES (?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET person_id=excluded.person_id, kind=excluded.kind, bytes=excluded.bytes,
                   modified=excluded.modified, deleted=excluded.deleted""",
            (fid, person_id, kind, size, modified, int(deleted)))

    def mark_deleted(self, table: str, key: str) -> bool:
        assert table in ("people", "face_crops")
        return self.db.execute(f"UPDATE {table} SET deleted=1 WHERE id=?", (key,)).rowcount > 0

    def people(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM face_crops f WHERE f.person_id = p.id AND f.deleted=0) AS face_crops "
            "FROM people p WHERE p.deleted=0 ORDER BY (p.name IS NULL OR p.name = ''), face_crops DESC, p.name")]

    def mark_missing(self, asset_id: str, when: str | None = None) -> bool:
        cur = self.db.execute(
            "UPDATE assets SET missing_since=COALESCE(missing_since, ?) WHERE id=?", (when or now(), asset_id))
        return cur.rowcount > 0

    def asset_id_for_master(self, master_id: str) -> str | None:
        row = self.db.execute("SELECT id FROM assets WHERE master_id=?", (master_id,)).fetchone()
        return row["id"] if row else None

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        return self._asset_dict(row) if row else None

    def asset_ids(self) -> set[str]:
        return {r["id"] for r in self.db.execute("SELECT id FROM assets")}

    @staticmethod
    def _asset_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["live"] = bool(d["live"])
        d["favorite"] = bool(d["favorite"])
        d["hidden"] = bool(d["hidden"])
        d["versions"] = json.loads(d["versions"] or "{}")
        d.pop("fingerprint", None)
        d["missing"] = d.pop("missing_since") is not None
        return d

    def counts(self) -> dict[str, int]:
        q = self.db.execute
        return {
            "assets": q("SELECT COUNT(*) FROM assets").fetchone()[0],
            "images": q("SELECT COUNT(*) FROM assets WHERE kind='image'").fetchone()[0],
            "movies": q("SELECT COUNT(*) FROM assets WHERE kind='movie'").fetchone()[0],
            "missing": q("SELECT COUNT(*) FROM assets WHERE missing_since IS NOT NULL").fetchone()[0],
            "albums": q("SELECT COUNT(*) FROM albums").fetchone()[0],
            "collections": q("SELECT COUNT(*) FROM collections").fetchone()[0],
            "people": q("SELECT COUNT(*) FROM people WHERE deleted=0").fetchone()[0],
        }

    def search(
        self,
        *,
        text: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        favorite: bool | None = None,
        album: str | None = None,
        collection: str | None = None,
        person: str | None = None,
        ids: list[str] | None = None,
        located: bool | None = None,
        live: bool | None = None,
        include_missing: bool = False,
        include_hidden: bool = False,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Keyset-paged search, newest capture first. Returns (rows, next_cursor)."""
        where, args = [], []
        if not include_missing:
            where.append("a.missing_since IS NULL")
        if not include_hidden:
            where.append("a.hidden = 0")
        if text:
            where.append("(a.filename LIKE ? OR a.caption LIKE ?)")
            args += [f"%{text}%", f"%{text}%"]
        if since:
            where.append("a.taken >= ?"); args.append(since)
        if until:
            where.append("a.taken < ?"); args.append(until)
        if kind:
            where.append("a.kind = ?"); args.append(kind)
        if favorite is not None:
            where.append("a.favorite = ?"); args.append(int(favorite))
        if live is not None:
            where.append("a.live = ?"); args.append(int(live))
        if located is not None:
            where.append("a.latitude IS " + ("NOT NULL" if located else "NULL"))
        if album:
            where.append(
                "a.id IN (SELECT asset_id FROM album_assets aa JOIN albums al ON al.id = aa.album_id "
                "WHERE al.id = ? OR al.name = ? OR al.fullname = ?)")
            args += [album, album, album]
        if collection:
            where.append("a.id IN (SELECT asset_id FROM collection_assets WHERE collection = ?)")
            args.append(collection)
        if person:
            where.append("a.id IN (SELECT asset_id FROM faces WHERE person_id = ?)")
            args.append(person)
        if ids is not None:
            where.append(f"a.id IN ({','.join('?' * len(ids))})" if ids else "0")
            args += ids
        if cursor:
            taken, asset_id = decode_cursor(cursor)
            # rows sort by (taken DESC, id DESC); NULL taken sorts last in DESC order
            if taken is None:
                where.append("a.taken IS NULL AND a.id < ?"); args.append(asset_id)
            else:
                where.append("(a.taken < ? OR (a.taken = ? AND a.id < ?) OR a.taken IS NULL)")
                args += [taken, taken, asset_id]
        sql = "SELECT a.* FROM assets a"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY a.taken DESC, a.id DESC LIMIT ?"
        args.append(limit + 1)
        rows = [self._asset_dict(r) for r in self.db.execute(sql, args)]
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            last = rows[-1]
            next_cursor = encode_cursor(last["taken"], last["id"])
        return rows, next_cursor

    # --- albums -----------------------------------------------------------
    def replace_albums(self, albums: Iterable[tuple[str, str, str]]) -> None:
        albums = list(albums)
        self.db.execute("BEGIN")
        self.db.execute("DELETE FROM albums")
        self.db.executemany(
            "INSERT INTO albums (id, name, fullname) VALUES (?, ?, ?)", albums)
        self.db.execute("COMMIT")

    def replace_album_members(self, album_id: str, asset_ids: Iterable[str]) -> int:
        ids = list(asset_ids)
        self.db.execute("BEGIN")
        self.db.execute("DELETE FROM album_assets WHERE album_id=?", (album_id,))
        self.db.executemany(
            "INSERT OR IGNORE INTO album_assets (album_id, asset_id) VALUES (?, ?)",
            [(album_id, i) for i in ids])
        self.db.execute("UPDATE albums SET synced=? WHERE id=?", (now(), album_id))
        self.db.execute("COMMIT")
        return len(ids)

    def albums(self) -> list[dict[str, Any]]:
        return [dict(r) | {"count": c} for r, c in (
            (r, self.db.execute("SELECT COUNT(*) FROM album_assets WHERE album_id=?", (r["id"],)).fetchone()[0])
            for r in self.db.execute("SELECT * FROM albums ORDER BY fullname"))]

    def find_album(self, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM albums WHERE id=? OR name=? OR fullname=?", (key, key, key)).fetchone()
        return dict(row) if row else None

    # --- cache ------------------------------------------------------------
    def cache_get(self, asset_id: str, version: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM cache WHERE asset_id=? AND version=?", (asset_id, version)).fetchone()
        return dict(row) if row else None

    def cache_put(self, asset_id: str, version: str, path: str, size: int, pinned: bool = False) -> None:
        t = now()
        self.db.execute(
            """INSERT INTO cache (asset_id, version, path, bytes, fetched, used, pinned)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(asset_id, version) DO UPDATE SET path=excluded.path, bytes=excluded.bytes,
                   fetched=excluded.fetched, used=excluded.used, pinned=MAX(cache.pinned, excluded.pinned)""",
            (asset_id, version, path, size, t, t, int(pinned)))

    def cache_touch(self, asset_id: str, version: str) -> None:
        self.db.execute("UPDATE cache SET used=? WHERE asset_id=? AND version=?", (now(), asset_id, version))

    def cache_pin(self, asset_id: str, version: str | None, pinned: bool) -> int:
        if version:
            cur = self.db.execute(
                "UPDATE cache SET pinned=? WHERE asset_id=? AND version=?", (int(pinned), asset_id, version))
        else:
            cur = self.db.execute("UPDATE cache SET pinned=? WHERE asset_id=?", (int(pinned), asset_id))
        return cur.rowcount

    def cache_delete(self, asset_id: str, version: str) -> None:
        self.db.execute("DELETE FROM cache WHERE asset_id=? AND version=?", (asset_id, version))

    def cache_rows(self, *, asset_id: str | None = None, unpinned_only: bool = False,
                   oldest_first: bool = False) -> list[dict[str, Any]]:
        where, args = [], []
        if asset_id:
            where.append("asset_id=?"); args.append(asset_id)
        if unpinned_only:
            where.append("pinned=0")
        sql = "SELECT * FROM cache"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY used ASC" if oldest_first else " ORDER BY asset_id, version"
        return [dict(r) for r in self.db.execute(sql, args)]

    def cache_usage(self) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(bytes),0) b, COALESCE(SUM(CASE WHEN pinned THEN bytes ELSE 0 END),0) p "
            "FROM cache").fetchone()
        by_version = {r["version"]: {"files": r["n"], "bytes": r["b"]} for r in self.db.execute(
            "SELECT version, COUNT(*) n, COALESCE(SUM(bytes),0) b FROM cache GROUP BY version")}
        return {"files": row["n"], "bytes": row["b"], "pinned_bytes": row["p"], "by_version": by_version}

    # --- index: clip embeddings, faces, seeds --------------------------------
    def unindexed(self, clip_model: str, face_model: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Images not yet indexed with these models, newest first."""
        sql = ("SELECT a.* FROM assets a LEFT JOIN index_state s ON s.asset_id = a.id "
               "WHERE a.kind='image' AND a.missing_since IS NULL "
               "AND (s.asset_id IS NULL OR s.clip_model IS NOT ? OR s.face_model IS NOT ?) "
               "ORDER BY a.taken DESC, a.id DESC")
        args: list[Any] = [clip_model, face_model]
        if limit:
            sql += " LIMIT ?"; args.append(limit)
        return [self._asset_dict(r) for r in self.db.execute(sql, args)]

    def put_clip(self, asset_id: str, embedding: bytes) -> None:
        self.db.execute("DELETE FROM vec_clip WHERE asset_id=?", (asset_id,))
        self.db.execute("INSERT INTO vec_clip (asset_id, e) VALUES (?, ?)", (asset_id, embedding))

    def put_faces(self, asset_id: str, faces: list[dict[str, Any]], source: str) -> list[int]:
        """Replace the automatically found faces of an asset; user-assigned ones are kept."""
        for row in self.db.execute("SELECT id FROM faces WHERE asset_id=? AND (assigned IS NULL OR assigned='auto')", (asset_id,)):
            self.db.execute("DELETE FROM vec_face WHERE face_id=?", (row["id"],))
        self.db.execute("DELETE FROM faces WHERE asset_id=? AND (assigned IS NULL OR assigned='auto')", (asset_id,))
        ids = []
        for f in faces:
            cur = self.db.execute(
                "INSERT INTO faces (asset_id, box, det, age, gender, embedding, source, person_id, similarity, assigned) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (asset_id, json.dumps(f["box"]), f["det"], f.get("age"), f.get("gender"), f["embedding"], source,
                 f.get("person_id"), f.get("similarity"), "auto" if f.get("person_id") else None))
            ids.append(cur.lastrowid)
            self.db.execute("INSERT INTO vec_face (face_id, e) VALUES (?, ?)", (cur.lastrowid, f["embedding"]))
        return ids

    def set_index_state(self, asset_id: str, clip_model: str | None, face_model: str | None, source: str, faces: int) -> None:
        self.db.execute(
            """INSERT INTO index_state (asset_id, clip_model, face_model, source, indexed, faces) VALUES (?,?,?,?,?,?)
               ON CONFLICT(asset_id) DO UPDATE SET clip_model=COALESCE(excluded.clip_model, index_state.clip_model),
                   face_model=COALESCE(excluded.face_model, index_state.face_model), source=excluded.source,
                   indexed=excluded.indexed, faces=excluded.faces""",
            (asset_id, clip_model, face_model, source, now(), faces))

    def index_counts(self) -> dict[str, Any]:
        q = self.db.execute
        return {
            "indexed": q("SELECT COUNT(*) FROM index_state").fetchone()[0],
            "images": q("SELECT COUNT(*) FROM assets WHERE kind='image' AND missing_since IS NULL").fetchone()[0],
            "faces": q("SELECT COUNT(*) FROM faces").fetchone()[0],
            "faces_named": q("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL").fetchone()[0],
            "seeds": q("SELECT COUNT(*) FROM person_seeds").fetchone()[0],
            "seeded_people": q("SELECT COUNT(DISTINCT person_id) FROM person_seeds").fetchone()[0],
        }

    def put_seed(self, seed_id: str, person_id: str, embedding: bytes, det: float | None, origin: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO person_seeds (id, person_id, embedding, det, origin, created) VALUES (?,?,?,?,?,?)",
            (seed_id, person_id, embedding, det, origin, now()))

    def seeds(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT id, person_id, embedding, det, origin FROM person_seeds")]

    def unseeded_crops(self, named_only: bool = True) -> list[dict[str, Any]]:
        """Face crops of (named) people that have not been embedded as seeds yet."""
        sql = ("SELECT f.id, f.person_id, f.bytes FROM face_crops f JOIN people p ON p.id = f.person_id "
               "LEFT JOIN person_seeds s ON s.id = f.id WHERE f.deleted=0 AND p.deleted=0 AND s.id IS NULL")
        if named_only:
            sql += " AND p.name IS NOT NULL AND p.name <> ''"
        return [dict(r) for r in self.db.execute(sql)]

    def nearest_clip(self, embedding: bytes, limit: int) -> list[tuple[str, float]]:
        return [(r["asset_id"], r["distance"]) for r in self.db.execute(
            "SELECT asset_id, distance FROM vec_clip WHERE e MATCH ? ORDER BY distance LIMIT ?", (embedding, limit))]

    def clip_of(self, asset_id: str) -> bytes | None:
        row = self.db.execute("SELECT e FROM vec_clip WHERE asset_id=?", (asset_id,)).fetchone()
        return bytes(row["e"]) if row else None

    def faces_of(self, asset_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self.db.execute(
                "SELECT f.id, f.asset_id, f.box, f.det, f.age, f.gender, f.source, f.person_id, f.similarity, f.assigned, "
                "p.name AS person_name FROM faces f LEFT JOIN people p ON p.id = f.person_id WHERE f.asset_id=? ORDER BY f.id",
                (asset_id,)):
            d = dict(r); d["box"] = json.loads(d["box"]); out.append(d)
        return out

    def get_face(self, face_id: int) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM faces WHERE id=?", (face_id,)).fetchone()
        if row is None:
            return None
        d = dict(row); d["box"] = json.loads(d["box"]); return d

    def assign_face(self, face_id: int, person_id: str | None, similarity: float | None, how: str) -> None:
        self.db.execute("UPDATE faces SET person_id=?, similarity=?, assigned=? WHERE id=?",
                        (person_id, similarity, how if person_id else None, face_id))

    def find_person(self, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM people WHERE deleted=0 AND (id=? OR name=? COLLATE NOCASE OR display_name=? COLLATE NOCASE)",
            (key, key, key)).fetchone()
        return dict(row) if row else None

    def person_photo_counts(self) -> dict[str, int]:
        return {r["person_id"]: r["n"] for r in self.db.execute(
            "SELECT person_id, COUNT(DISTINCT asset_id) n FROM faces WHERE person_id IS NOT NULL GROUP BY person_id")}

    # --- collections ------------------------------------------------------
    def collection_create(self, name: str, note: str | None = None) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO collections (name, created, note) VALUES (?, ?, ?)", (name, now(), note))
        return cur.rowcount > 0

    def collection_delete(self, name: str) -> bool:
        self.db.execute("BEGIN")
        self.db.execute("DELETE FROM collection_assets WHERE collection=?", (name,))
        cur = self.db.execute("DELETE FROM collections WHERE name=?", (name,))
        self.db.execute("COMMIT")
        return cur.rowcount > 0

    def collection_exists(self, name: str) -> bool:
        return self.db.execute("SELECT 1 FROM collections WHERE name=?", (name,)).fetchone() is not None

    def collection_add(self, name: str, asset_ids: Iterable[str], note: str | None = None) -> int:
        pos = self.db.execute(
            "SELECT COALESCE(MAX(position), 0) FROM collection_assets WHERE collection=?", (name,)).fetchone()[0]
        added = 0
        for asset_id in asset_ids:
            pos += 1
            cur = self.db.execute(
                "INSERT OR IGNORE INTO collection_assets (collection, asset_id, position, note) VALUES (?,?,?,?)",
                (name, asset_id, pos, note))
            added += cur.rowcount
        return added

    def collection_remove(self, name: str, asset_ids: Iterable[str]) -> int:
        removed = 0
        for asset_id in asset_ids:
            removed += self.db.execute(
                "DELETE FROM collection_assets WHERE collection=? AND asset_id=?", (name, asset_id)).rowcount
        return removed

    def collections(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(
            "SELECT c.name, c.created, c.note, COUNT(ca.asset_id) AS count FROM collections c "
            "LEFT JOIN collection_assets ca ON ca.collection = c.name GROUP BY c.name ORDER BY c.name")]

    def collection_items(self, name: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT ca.position, ca.note, a.* FROM collection_assets ca JOIN assets a ON a.id = ca.asset_id "
            "WHERE ca.collection=? ORDER BY ca.position", (name,))
        out = []
        for r in rows:
            d = self._asset_dict(r)
            d["position"], d["note"] = r["position"], r["note"]
            out.append(d)
        return out
