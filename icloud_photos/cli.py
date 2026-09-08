"""The `photos` command.

Written for an assistant working through a shell as much as for a person:
every command takes --json, results are bounded and paged, errors are one
line on stderr with a non-zero exit, and --help is meant to be enough to
use the tool without reading anything else.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .adapter import Adapter, CloudError, ICloudAdapter, NotLoggedIn
from .cache import BudgetExceeded, Cache
from .catalog import Catalog
from .paths import DEFAULTS, Config, Paths
from .sync import sync as run_sync

EXT_BY_TYPE = {
    "public.jpeg": ".jpg", "public.heic": ".heic", "public.heif": ".heif", "public.png": ".png",
    "com.apple.quicktime-movie": ".mov", "public.mpeg-4": ".mp4", "com.apple.m4v-video": ".m4v",
    "com.adobe.raw-image": ".dng",
}
PREVIEW_SIZES = ("thumb", "medium")


class CliError(Exception):
    """Reported as one line on stderr (or a JSON object with --json); exits non-zero."""

    def __init__(self, code: str, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


class App:
    """Everything a command needs, opened lazily so `--help` costs nothing."""

    def __init__(self, json_mode: bool, adapter_factory: Callable[["App"], Adapter] | None = None) -> None:
        self.json = json_mode
        self.paths = Paths.discover().ensure()
        self.config = Config.load(self.paths.config_file)
        self._catalog: Catalog | None = None
        self._cache: Cache | None = None
        self._adapter: Adapter | None = None
        self._adapter_factory = adapter_factory or (
            lambda app: ICloudAdapter(app.paths.session_dir, app.config.values.get("username", "")))

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = Catalog(self.paths.catalog_db)
        return self._catalog

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            self._cache = Cache(self.paths.cache_dir, self.catalog, self.config.cache_budget_bytes)
        return self._cache

    @property
    def adapter(self) -> Adapter:
        if self._adapter is None:
            self._adapter = self._adapter_factory(self)
        return self._adapter

    def close(self) -> None:
        if self._catalog is not None:
            self._catalog.close()

    # --- output -----------------------------------------------------------
    def emit(self, payload: Any, text: Callable[[Any], str] | None = None) -> None:
        if self.json:
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(text(payload) if text else json.dumps(payload, indent=2, default=str))


def asset_line(a: dict[str, Any]) -> str:
    size = f"{a.get('width') or '?'}x{a.get('height') or '?'}"
    flags = "".join(c for c, on in (("*", a.get("favorite")), ("L", a.get("live")),
                                    ("!", a.get("missing")), ("h", a.get("hidden"))) if on)
    taken = (a.get("taken") or "-")[:19]
    line = f"{a['id']}  {taken}  {a['kind']:<5} {size:>10}  {a['filename']}"
    if flags:
        line += f"  [{flags}]"
    if a.get("caption"):
        line += f'  "{a["caption"]}"'
    return line


def human_bytes(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def parse_date(text: str | None, end: bool = False) -> str | None:
    """YYYY, YYYY-MM, YYYY-MM-DD or a full ISO timestamp -> ISO UTC. `end` rounds up."""
    if not text:
        return None
    try:
        parts = [int(p) for p in text[:10].split("-")] if len(text) <= 10 else None
        if parts is None:
            dt = datetime.fromisoformat(text)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        else:
            y, m, d = (parts + [1, 1])[:3]
            dt = datetime(y, m, d, tzinfo=timezone.utc)
            if end:
                if len(parts) == 1:
                    dt = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
                elif len(parts) == 2:
                    dt = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=timezone.utc)
                else:
                    dt = datetime.fromtimestamp(dt.timestamp() + 86400, timezone.utc)
    except ValueError as err:
        raise CliError("bad-date", f"cannot read date {text!r}: use YYYY, YYYY-MM, YYYY-MM-DD or ISO 8601") from err
    return dt.astimezone(timezone.utc).isoformat()


# --- commands ---------------------------------------------------------------

def cmd_login(app: App, args: argparse.Namespace) -> int:
    icloud = Path(sys.executable).parent / "icloud"
    if not icloud.exists():
        raise CliError("no-icloud-cli", "pyicloud's `icloud` command is not installed next to this Python")
    cmd = [str(icloud), "auth", "logout" if args.logout else "login", "--session-dir", str(app.paths.session_dir)]
    username = args.username or app.config.values.get("username") or ""
    if username:
        cmd += ["--username", username]
    if args.logout:
        return subprocess.call(cmd)
    if not sys.stdin.isatty():
        raise CliError("needs-terminal", "login needs a terminal for the password and two-factor code; "
                       "run `photos login` yourself", 2)
    rc = subprocess.call(cmd)
    if rc == 0 and args.username:
        app.config.values["username"] = args.username
        app.config.save(app.paths.config_file)
    return rc


def cmd_status(app: App, args: argparse.Namespace) -> int:
    auth: dict[str, Any]
    if args.offline:
        auth = {"checked": False}
    else:
        try:
            auth = app.adapter.auth_status() | {"checked": True}
        except Exception as err:  # noqa: BLE001 - status must never crash
            auth = {"checked": True, "authenticated": False, "reason": f"{type(err).__name__}: {err}"}
    last = app.catalog.get_meta("last_sync")
    progress = app.catalog.get_meta("sync_progress")
    payload = {
        "authenticated": auth.get("authenticated"),
        "auth": auth,
        "catalog": app.catalog.counts(),
        "last_sync": json.loads(last) if last else None,
        "sync_running": _sync_pid(app) is not None,
        "sync_progress": json.loads(progress) if progress else None,
        "cache": app.cache.usage(),
        "paths": {"catalog": str(app.paths.catalog_db), "cache": str(app.paths.cache_dir),
                  "session": str(app.paths.session_dir), "config": str(app.paths.config_file)},
    }

    def text(p: dict[str, Any]) -> str:
        c, u = p["catalog"], p["cache"]
        lines = [
            "auth:     " + ("ok" if p["authenticated"] else "not logged in" if p["auth"].get("checked")
                            else "not checked") + (f" ({p['auth']['reason']})" if p["auth"].get("reason") else ""),
            f"catalog:  {c['assets']} assets ({c['images']} images, {c['movies']} movies, {c['missing']} missing), "
            f"{c['albums']} albums, {c['people']} people, {c['collections']} collections",
            f"cache:    {human_bytes(u['bytes'])} of {human_bytes(u['budget_bytes'])} in {u['files']} files, "
            f"{human_bytes(u['pinned_bytes'])} pinned",
        ]
        if p["last_sync"]:
            ls = p["last_sync"]
            lines.append(f"last sync: {ls.get('finished')} mode={ls.get('mode')} records={ls.get('records')} "
                         f"new={ls.get('new')} changed={ls.get('changed')} missing={ls.get('missing')}")
        else:
            lines.append("last sync: never (run `photos sync`)")
        if p["sync_running"]:
            lines.append(f"sync:     running {p['sync_progress'] or ''}")
        return "\n".join(lines)

    app.emit(payload, text)
    return 0


def _sync_pid(app: App) -> int | None:
    lock = app.paths.sync_lock
    if not lock.exists():
        return None
    try:
        pid = int(lock.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        lock.unlink(missing_ok=True)
        return None


def cmd_sync(app: App, args: argparse.Namespace) -> int:
    if (pid := _sync_pid(app)) is not None:
        raise CliError("sync-running", f"a sync is already running (pid {pid}); see `photos status`")
    if args.background:
        flags = ["--full"] if args.full else []
        log = app.paths.sync_log.open("ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "icloud_photos", "--json", "sync", *flags],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        app.emit({"started": True, "pid": proc.pid, "log": str(app.paths.sync_log)},
                 lambda p: f"sync started in the background (pid {p['pid']}), log at {p['log']}; "
                           f"watch with `photos status`")
        return 0
    app.paths.sync_lock.write_text(str(os.getpid()))
    try:
        def progress(p: dict[str, Any]) -> None:
            if not app.json and p["pages"] % 20 == 0:
                print(f"  records {p['records']} assets new {p['new']} changed {p['changed']} "
                      f"relations {p['relations']} people {p['people']}", file=sys.stderr)
        result = run_sync(app.catalog, app.adapter, full=args.full, progress=progress)
    finally:
        app.paths.sync_lock.unlink(missing_ok=True)
    app.emit(result, lambda r: f"sync {r['mode']}: {r['records']} records; assets new {r['new']}, changed {r['changed']}, "
                               f"missing {r['missing']}; relations {r['relations']}, people {r['people']}, "
                               f"face crops {r['face_crops']}, albums {r['albums']}")
    return 0


def cmd_people(app: App, args: argparse.Namespace) -> int:
    rows = app.catalog.people()
    app.emit(rows, lambda rs: "\n".join(
        f"{r['face_crops']:>4}  {r['name'] or '(unnamed)'}" + (f"  ({r['display_name']})" if r['display_name'] and r['display_name'] != r['name'] else "")
        + ("" if r['verified'] else "  unverified") + f"  {r['id']}" for r in rs)
        or "no people in the catalogue yet; run `photos sync`")
    return 0


def cmd_albums(app: App, args: argparse.Namespace) -> int:
    rows = app.catalog.albums()
    app.emit(rows, lambda rs: "\n".join(f"{r['count']:>6}  {r['fullname']}  ({r['id']})" for r in rs)
             or "no albums in the catalogue; run `photos sync --albums`")
    return 0


def cmd_search(app: App, args: argparse.Namespace) -> int:
    fav = True if args.favorite else (False if args.not_favorite else None)
    rows, next_cursor = app.catalog.search(
        text=args.query, since=parse_date(args.since), until=parse_date(args.until, end=True),
        kind=args.kind, favorite=fav, album=args.album, collection=args.collection,
        located=True if args.located else None, live=True if args.live else None,
        include_missing=args.include_missing, include_hidden=args.include_hidden,
        limit=args.limit, cursor=args.cursor)
    payload = {"results": rows, "count": len(rows), "next_cursor": next_cursor}

    def text(p: dict[str, Any]) -> str:
        lines = [asset_line(a) for a in p["results"]] or ["no matches"]
        if p["next_cursor"]:
            lines.append(f"more: --cursor {p['next_cursor']}")
        return "\n".join(lines)

    app.emit(payload, text)
    return 0


def _assets(app: App, ids: list[str]) -> list[dict[str, Any]]:
    out = []
    for asset_id in ids:
        a = app.catalog.get_asset(asset_id)
        if a is None:
            raise CliError("unknown-asset", f"no asset {asset_id} in the catalogue; run `photos sync` or check the id")
        out.append(a)
    return out


def cmd_info(app: App, args: argparse.Namespace) -> int:
    rows = _assets(app, args.id)
    for a in rows:
        a["cached"] = [{"version": r["version"], "path": r["path"], "bytes": r["bytes"], "pinned": bool(r["pinned"])}
                       for r in app.catalog.cache_rows(asset_id=a["id"])]
        a["albums"] = [r["fullname"] for r in app.catalog.db.execute(
            "SELECT al.fullname FROM album_assets aa JOIN albums al ON al.id = aa.album_id WHERE aa.asset_id=?",
            (a["id"],))]
        a["collections"] = [r["collection"] for r in app.catalog.db.execute(
            "SELECT collection FROM collection_assets WHERE asset_id=?", (a["id"],))]
    app.emit(rows if len(rows) > 1 else rows[0])
    return 0


def _fetch(app: App, asset: dict[str, Any], version: str, pin: bool = False) -> dict[str, Any]:
    if version not in asset["versions"]:
        raise CliError("no-such-version",
                       f"{asset['id']} has no {version} rendition; it has {', '.join(asset['versions']) or 'none'}")
    path = app.cache.get(asset["id"], version)
    cached = path is not None
    if path is None:
        try:
            data = app.adapter.download(asset["id"], version)
        except NotLoggedIn as err:
            raise CliError("not-logged-in", f"{err}; run `photos login`", 3) from err
        if data is None:
            raise CliError("download-failed", f"iCloud returned nothing for {asset['id']} {version}")
        v = asset["versions"][version]
        ext = Path(v.get("filename") or "").suffix.lower() or EXT_BY_TYPE.get(v.get("type") or "", "")
        try:
            path = app.cache.put(asset["id"], version, data, ext, pinned=pin)
        except BudgetExceeded as err:
            raise CliError("cache-full", str(err), 4) from err
    elif pin:
        app.catalog.cache_pin(asset["id"], version, True)
    return {"id": asset["id"], "version": version, "path": str(path), "bytes": path.stat().st_size,
            "cached": cached, "pinned": pin or bool((app.catalog.cache_get(asset["id"], version) or {}).get("pinned"))}


def cmd_show(app: App, args: argparse.Namespace) -> int:
    size = args.size or app.config.values.get("preview_size", "thumb")
    results = []
    for asset in _assets(app, args.id):
        version = size if asset["kind"] == "image" else f"{size}_image"
        results.append(_fetch(app, asset, version))
    app.emit(results, lambda rs: "\n".join(f"{r['id']}\t{r['path']}" for r in rs))
    return 0


def cmd_original(app: App, args: argparse.Namespace) -> int:
    results = []
    for asset in _assets(app, args.id):
        version = args.version or "original"
        r = _fetch(app, asset, version, pin=args.pin)
        if asset["live"] and version == "original":
            r["note"] = "live photo: the video half is version original_video (`photos original --version original_video`)"
        results.append(r)
    app.emit(results, lambda rs: "\n".join(f"{r['id']}\t{r['path']}" for r in rs))
    return 0


def cmd_cache(app: App, args: argparse.Namespace) -> int:
    if args.cache_cmd == "status":
        u = app.cache.usage()
        app.emit(u, lambda u: f"{human_bytes(u['bytes'])} of {human_bytes(u['budget_bytes'])} in {u['files']} files; "
                              f"{human_bytes(u['pinned_bytes'])} pinned; " +
                              ", ".join(f"{k}: {v['files']} files {human_bytes(v['bytes'])}" for k, v in u['by_version'].items()))
    elif args.cache_cmd == "verify":
        app.emit(app.cache.verify(), lambda r: f"dropped {r['dropped']} index entries whose files were gone")
    elif args.cache_cmd in ("pin", "unpin"):
        n = sum(app.catalog.cache_pin(i, args.version, args.cache_cmd == "pin") for i in args.id)
        app.emit({"updated": n}, lambda r: f"{args.cache_cmd}ned {r['updated']} cached files")
    elif args.cache_cmd == "evict":
        if not args.id and not args.all:
            raise CliError("nothing-to-evict", "give asset ids, or --all to evict every unpinned file")
        gone = []
        for asset_id in (args.id or [None]):
            gone += app.cache.evict(asset_id, args.version, include_pinned=args.include_pinned)
        freed = sum(r["bytes"] for r in gone)
        app.emit({"evicted": len(gone), "freed_bytes": freed,
                  "files": [{"id": r["asset_id"], "version": r["version"]} for r in gone]},
                 lambda r: f"evicted {r['evicted']} files, freed {human_bytes(r['freed_bytes'])} (iCloud untouched)")
    return 0


def cmd_collection(app: App, args: argparse.Namespace) -> int:
    c = app.catalog
    cmd = args.coll_cmd
    if cmd == "list":
        rows = c.collections()
        app.emit(rows, lambda rs: "\n".join(f"{r['count']:>5}  {r['name']}" + (f"  {r['note']}" if r['note'] else "")
                                            for r in rs) or "no collections")
        return 0
    if cmd == "create":
        made = c.collection_create(args.name, args.note)
        app.emit({"name": args.name, "created": made}, lambda r: f"{'created' if r['created'] else 'already exists'}: {r['name']}")
        return 0
    if not c.collection_exists(args.name):
        raise CliError("unknown-collection", f"no collection {args.name!r}; `photos collection create {args.name}` first")
    if cmd == "delete":
        c.collection_delete(args.name)
        app.emit({"name": args.name, "deleted": True}, lambda r: f"deleted collection {r['name']} (photos untouched)")
    elif cmd == "add":
        _assets(app, args.id)
        n = c.collection_add(args.name, args.id, args.note)
        app.emit({"name": args.name, "added": n}, lambda r: f"added {r['added']} to {r['name']}")
    elif cmd == "remove":
        n = c.collection_remove(args.name, args.id)
        app.emit({"name": args.name, "removed": n}, lambda r: f"removed {r['removed']} from {r['name']}")
    elif cmd == "show":
        items = c.collection_items(args.name)
        app.emit({"name": args.name, "count": len(items), "items": items},
                 lambda r: "\n".join(f"{i['position']:>3}  " + asset_line(i) for i in r["items"]) or "empty")
    return 0


def cmd_config(app: App, args: argparse.Namespace) -> int:
    if args.config_cmd == "set":
        try:
            app.config.set(args.key, args.value)
        except KeyError:
            raise CliError("unknown-key", f"unknown key {args.key}; keys are {', '.join(DEFAULTS)}") from None
        except ValueError:
            raise CliError("bad-value", f"{args.key} needs a number") from None
        app.config.save(app.paths.config_file)
    app.emit(app.config.values | {"file": str(app.paths.config_file)},
             lambda v: "\n".join(f"{k} = {v[k]}" for k in DEFAULTS) + f"\n# {v['file']}")
    return 0


# --- parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photos",
        description="An iCloud photo library from the command line, built for assistants.\n\n"
                    "The catalogue (metadata for every asset) lives locally and answers every query\n"
                    "without the network. Images are fetched on demand into a bounded cache; `show`\n"
                    "and `original` print the local path of the file they fetched. Asset ids are stable.\n"
                    "Add --json to any command for structured output; errors are one line on stderr\n"
                    "with a non-zero exit (2 needs a terminal, 3 not logged in, 4 cache full).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="typical session:\n"
               "  photos login                      once, interactive (password + two-factor code)\n"
               "  photos sync --background          then `photos status` until it finishes\n"
               "  photos search beach --since 2019-07 --until 2019-08 --json\n"
               "  photos show <id> <id>             fetch small previews, print their paths\n"
               "  photos original <id> --pin        fetch the full file and keep it\n"
               "  photos collection create book && photos collection add book <id>...\n")
    p.add_argument("--json", action="store_true", help="structured output on stdout; errors as JSON on stderr")
    p.add_argument("--version", action="version", version=f"photos {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="command", required=True)

    s = sub.add_parser("login", help="sign in to iCloud (interactive; needs a terminal)",
                       description="Signs in through pyicloud's `icloud auth login` and stores the session under "
                                   "the state directory. Asks for the password and the two-factor code; the "
                                   "password can go in the system keyring so later logins are silent.")
    s.add_argument("--username", help="Apple ID; remembered in the config file")
    s.add_argument("--logout", action="store_true", help="drop the stored session instead")
    s.set_defaults(fn=cmd_login)

    s = sub.add_parser("status", help="auth state, catalogue counts, cache usage, sync progress")
    s.add_argument("--offline", action="store_true", help="do not contact iCloud to check the session")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("sync", help="bring the catalogue up to date with the library",
                       description="Walks iCloud's change feed: the whole photo zone the first time (metadata only, "
                                   "no images; assets, albums, people, face crops), only what changed after that. "
                                   "Safe to interrupt: an interrupted run resumes where it stopped.")
    s.add_argument("--full", action="store_true", help="start from the beginning of the zone again")
    s.add_argument("--background", action="store_true", help="run detached; follow with `photos status`")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("albums", help="list albums known to the catalogue")
    s.set_defaults(fn=cmd_albums)

    s = sub.add_parser("people", help="named people from iCloud's People album, with their face crop counts")
    s.set_defaults(fn=cmd_people)

    s = sub.add_parser("search", help="find assets in the catalogue (no network)",
                       description="Newest capture first. Results are paged: pass the printed cursor back "
                                   "with --cursor for the next page. Dates accept YYYY, YYYY-MM, YYYY-MM-DD "
                                   "or ISO 8601; --until includes the whole of the day, month or year given.")
    s.add_argument("query", nargs="?", help="substring of the filename or caption")
    s.add_argument("--since", metavar="DATE"); s.add_argument("--until", metavar="DATE")
    s.add_argument("--kind", choices=("image", "movie"))
    s.add_argument("--favorite", action="store_true"); s.add_argument("--not-favorite", action="store_true")
    s.add_argument("--album", metavar="NAME_OR_ID"); s.add_argument("--collection", metavar="NAME")
    s.add_argument("--located", action="store_true", help="only assets with GPS coordinates")
    s.add_argument("--live", action="store_true", help="only Live Photos")
    s.add_argument("--include-missing", action="store_true", help="also assets no longer in iCloud")
    s.add_argument("--include-hidden", action="store_true")
    s.add_argument("--limit", type=int, default=50, metavar="N", help="page size (default 50)")
    s.add_argument("--cursor", help="continue from a previous page")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("info", help="everything known about one or more assets")
    s.add_argument("id", nargs="+")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("show", help="fetch previews into the cache and print their paths",
                       description="Prints one `id<TAB>path` line per asset (or JSON). Look at the file to see "
                                   "the picture. Previews are small JPEGs; use `original` for the real file.")
    s.add_argument("id", nargs="+")
    s.add_argument("--size", choices=PREVIEW_SIZES, help="thumb (default, from config) or medium")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("original", help="fetch full-resolution originals into the cache",
                       description="Counts against the cache budget; unpinned originals are the first to be "
                                   "evicted when room is needed. --pin keeps them.")
    s.add_argument("id", nargs="+")
    s.add_argument("--pin", action="store_true", help="never evict automatically")
    s.add_argument("--version", help="a specific rendition (see `info`): original, alternative, original_video, ...")
    s.set_defaults(fn=cmd_original)

    s = sub.add_parser("cache", help="inspect, pin, or evict cached files (never touches iCloud)")
    cs = s.add_subparsers(dest="cache_cmd", metavar="action", required=True)
    cs.add_parser("status", help="usage against the budget")
    cs.add_parser("verify", help="drop index entries whose files are gone")
    for name, hlp in (("pin", "keep these assets' cached files"), ("unpin", "allow eviction again")):
        x = cs.add_parser(name, help=hlp); x.add_argument("id", nargs="+"); x.add_argument("--version")
    x = cs.add_parser("evict", help="delete local copies (iCloud keeps everything)")
    x.add_argument("id", nargs="*"); x.add_argument("--version"); x.add_argument("--all", action="store_true")
    x.add_argument("--include-pinned", action="store_true")
    s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("collection", help="named sets of assets for a project (a photobook, a slideshow)")
    cc = s.add_subparsers(dest="coll_cmd", metavar="action", required=True)
    cc.add_parser("list", help="all collections")
    x = cc.add_parser("create"); x.add_argument("name"); x.add_argument("--note")
    x = cc.add_parser("delete", help="remove the collection only; photos and cache stay"); x.add_argument("name")
    x = cc.add_parser("add"); x.add_argument("name"); x.add_argument("id", nargs="+"); x.add_argument("--note")
    x = cc.add_parser("remove"); x.add_argument("name"); x.add_argument("id", nargs="+")
    x = cc.add_parser("show", help="the collection's assets in order"); x.add_argument("name")
    s.set_defaults(fn=cmd_collection)

    s = sub.add_parser("config", help="show or set settings (" + ", ".join(DEFAULTS) + ")")
    cf = s.add_subparsers(dest="config_cmd", metavar="action", required=True)
    cf.add_parser("show")
    x = cf.add_parser("set"); x.add_argument("key"); x.add_argument("value")
    s.set_defaults(fn=cmd_config)
    return p


def main(argv: list[str] | None = None, adapter_factory: Callable[[App], Adapter] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = App(args.json, adapter_factory)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return args.fn(app, args)
    except CliError as err:
        _fail(app, err.code, str(err))
        return err.exit_code
    except NotLoggedIn as err:
        _fail(app, "not-logged-in", f"{err}; run `photos login`")
        return 3
    except CloudError as err:
        _fail(app, "cloud-error", str(err))
        return 1
    except KeyboardInterrupt:
        _fail(app, "interrupted", "interrupted")
        return 130
    finally:
        app.close()


def _fail(app: App, code: str, message: str) -> None:
    if app.json:
        print(json.dumps({"error": code, "message": message}), file=sys.stderr)
    else:
        print(f"photos: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
