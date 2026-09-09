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
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .adapter import Adapter, CloudError, ICloudAdapter, NotLoggedIn
from .cache import BudgetExceeded, Cache
from .catalog import Catalog, now as catalog_now
from .paths import DEFAULTS, Config, Paths
from .sync import sync as run_sync
from . import index as indexing
from . import lap as lap_export
from . import edit as editing
from .ml import COMPUTE_MODES, ComputeUnavailable, ModelsMissing, OnnxModels, as_blob, fetch_clip_models, from_blob

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

    def __init__(self, json_mode: bool, adapter_factory: Callable[["App"], Adapter] | None = None,
                 models_factory: Callable[["App"], Any] | None = None) -> None:
        self.json = json_mode
        self._models: Any = None
        # Only the index worker takes the GPU: a query's text embedding is tiny, and a
        # second process mapping the models on the GPU while the worker holds it has
        # failed on Vulkan under memory pressure (2026-09-08).
        self.gpu_allowed = False
        self._models_factory = models_factory or (lambda app: OnnxModels(
            app.paths.models_dir, compute=str(app.config.values.get("compute", "auto")) if app.gpu_allowed else "cpu",
            record=app.paths.index_lock.with_name("compute.json") if app.gpu_allowed else None))
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
    def models(self) -> Any:
        if self._models is None:
            self._models = self._models_factory(self)
        return self._models

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
        "index": app.catalog.index_counts() | {"running": _pid(app.paths.index_lock) is not None,
                                                "progress": json.loads(ip) if (ip := app.catalog.get_meta("index_progress")) else None},
        "paths": {"catalog": str(app.paths.catalog_db), "cache": str(app.paths.cache_dir),
                  "session": str(app.paths.session_dir), "config": str(app.paths.config_file)},
        "compute": _compute_status(app),
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
        ix = p["index"]
        lines.append(f"index:    {ix['indexed']} of {ix['images']} images; {ix['faces']} faces, {ix['faces_named']} named; "
                     f"{ix['seeds']} seeds for {ix['seeded_people']} people" + (f"; running {ix['progress']}" if ix["running"] else ""))
        cp = p["compute"]
        lines.append(f"compute:  {cp['resolved'] or 'unknown'} ({cp['detail']}; configured {cp['mode']})")
        return "\n".join(lines)

    app.emit(payload, text)
    return 0


def _compute_status(app: App) -> dict[str, Any]:
    """The configured mode and what the last index run resolved it to. Status
    does not probe the GPU itself: a second process on it while the worker
    runs has pushed this 8 GB machine into the OOM killer."""
    mode = str(app.config.values.get("compute", "auto"))
    rec = app.paths.index_lock.with_name("compute.json")
    if rec.exists():
        try:
            last = json.loads(rec.read_text())
            return {"mode": mode, "resolved": last.get("resolved"), "detail": last.get("detail"), "when": last.get("when")}
        except (OSError, ValueError):
            pass
    return {"mode": mode, "resolved": None, "detail": "not resolved yet; the next `index` run records it", "when": None}


def _sync_pid(app: App) -> int | None:
    return _pid(app.paths.sync_lock)


def _pid(lock: Path) -> int | None:
    if not lock.exists():
        return None
    try:
        pid = int(lock.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        lock.unlink(missing_ok=True)
        return None


def _spawn_worker(name: str, argv: list[str], log_path: Path, lock: Path) -> dict[str, Any]:
    """Start a detached worker for a long job.

    Under systemd the worker gets its own transient user unit: it outlives the
    terminal it was started from, and when memory runs out the kernel kills one
    unit, not the terminal session and every job in it (which is how a 2026-09-08
    index run, a compiler and the assistant's session all died together).
    ICLOUD_PHOTOS_WORKER=plain, or no usable systemd, gives a plain detached process.
    """
    cmd = [sys.executable, "-m", "icloud_photos", "--json", *argv]
    unit = None
    if os.environ.get("ICLOUD_PHOTOS_WORKER", "systemd") == "systemd" and shutil.which("systemd-run"):
        unit = f"icloud-photos-{name}-{int(time.time())}"
        # The unit does not inherit this shell's environment; pass through our own settings only.
        env = [f"--setenv={k}={v}" for k, v in os.environ.items() if k.startswith("ICLOUD_PHOTOS_")]
        run = subprocess.run(["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}",
                              f"--property=StandardOutput=append:{log_path}",
                              f"--property=StandardError=append:{log_path}", *env, "--", *cmd],
                             capture_output=True, text=True)
        if run.returncode != 0:
            unit = None
    if unit is None:
        log = log_path.open("ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        return {"started": True, "pid": proc.pid, "unit": None, "log": str(log_path)}
    pid = None
    for _ in range(40):                       # the worker writes its own pid into the lock as it starts
        if (pid := _pid(lock)) is not None:
            break
        time.sleep(0.05)
    return {"started": True, "pid": pid, "unit": unit, "log": str(log_path)}


def _started_text(name: str) -> Callable[[dict[str, Any]], str]:
    return lambda p: (f"{name} started in the background (pid {p['pid']}"
                      + (f", unit {p['unit']}" if p.get("unit") else "")
                      + f"), log at {p['log']}; watch with `photos status`")


def cmd_sync(app: App, args: argparse.Namespace) -> int:
    if (pid := _sync_pid(app)) is not None:
        raise CliError("sync-running", f"a sync is already running (pid {pid}); see `photos status`")
    if args.background:
        flags = (["--full"] if args.full else []) + (["--no-shared"] if args.no_shared else [])
        app.emit(_spawn_worker("sync", ["sync", *flags], app.paths.sync_log, app.paths.sync_lock),
                 _started_text("sync"))
        return 0
    app.paths.sync_lock.write_text(str(os.getpid()))
    try:
        def progress(p: dict[str, Any]) -> None:
            if not app.json and p["pages"] % 20 == 0:
                print(f"  records {p['records']} assets new {p['new']} changed {p['changed']} "
                      f"relations {p['relations']} people {p['people']}", file=sys.stderr)
        result = run_sync(app.catalog, app.adapter, full=args.full,
                          shared=not args.no_shared, progress=progress)
    finally:
        app.paths.sync_lock.unlink(missing_ok=True)
    def text(r: dict[str, Any]) -> str:
        lines = [f"sync {r['mode']}: {r['records']} records; assets new {r['new']}, changed {r['changed']}, "
                 f"missing {r['missing']}; relations {r['relations']}, people {r['people']}, "
                 f"face crops {r['face_crops']}, albums {r['albums']}"]
        for name, z in (r.get("zones") or {}).items():
            if z.get("error"):
                lines.append(f"  {name}: could not be read ({z['error']})")
            elif z.get("shared"):
                lines.append(f"  {name}: shared library, {z['records']} records, {z['new']} new photos")
        return "\n".join(lines)

    app.emit(result, text)
    return 0


def cmd_lap_export(app: App, args: argparse.Namespace) -> int:
    library = Path(args.library) if args.library else lap_export.default_library()
    if library is None or not library.exists():
        raise CliError("lap-missing", "no lap library found; run lap once (https://github.com/julyx10/lap) or pass --library PATH")
    root = Path(args.root) if args.root else app.paths.cache_dir / "lap"
    try:
        lib = lap_export.LapLibrary(library, root, thumb_size=args.thumb_size)
    except ValueError as err:
        raise CliError("lap-invalid", str(err)) from err
    if lib.app_is_running() and not app.json:
        print("note: the photo app is running; it will show the changes after a restart", file=sys.stderr)

    def progress(p: dict[str, Any]) -> None:
        if not app.json:
            print(f"  {p['done']} of {p['total']}: {p['files']} files, {p['thumbs']} thumbs ({p.get('fetched', 0)} fetched), {p['faces']} faces", file=sys.stderr)
    try:
        fetch = _lap_fetch_command(app)
        result = lap_export.export(app.catalog, app.cache, lib, limit=args.limit, fetch_command=fetch,
                                   cli=_photos_binary(),
                                   fetch_thumbs=args.fetch_thumbs, adapter=app.adapter if args.fetch_thumbs else None,
                                   open_version=args.open_version or str(app.config.values.get("lap_open_version", "original")),
                                   progress=progress)
    finally:
        lib.close()
    if not result.get("managed"):
        print("warning: this build of the app does not support managed albums (its migration 18); it will "
              "delete rows whose file is not cached. Update it before browsing.", file=sys.stderr)
    app.emit(result, lambda r: f"lap library {r['library']}: {r['files']} files ({r['linked']} with a cached file), "
                               f"{r['thumbs']} thumbnails, {r['embeddings']} embeddings, {r['faces']} faces of {r['people']} people, "
                               f"{r['collections']} collections; restart lap to see them")
    return 0


def _asset_for(app: App, ident: str) -> dict[str, Any]:
    """An asset from its id, or from the path of an entry in the GUI album."""
    target = Path(ident)
    if target.exists() or "@" in target.name:
        asset_id = lap_export.asset_id_of(target.name)
        row = app.catalog.db.execute("SELECT id FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None:
            row = next((r for r in app.catalog.db.execute("SELECT id FROM assets")
                        if lap_export.safe_name(r["id"]) == asset_id), None)
        if row is None:
            raise CliError("no-such-asset", f"{target.name} names no asset in the catalogue")
        return _assets(app, [row["id"]])[0]
    return _assets(app, [ident])[0]


def _run_edit(app: App, asset: dict[str, Any], prompt: str, parent: int | None, out: str | None,
              timeout: int | None, quiet: bool) -> dict[str, Any]:
    """Do one edit and record it as a version of the photo."""
    if parent is not None:
        source_row = app.catalog.variant(parent)
        if source_row is None:
            raise CliError("no-such-variant", f"there is no version {parent}")
        if source_row["asset_id"] != asset["id"]:
            raise CliError("no-such-variant", f"version {parent} belongs to another photo")
        source, version = Path(source_row["path"]), f"variant:{parent}"
        if not source.exists():
            raise CliError("missing-variant", f"version {parent} is no longer on disk ({source})")
    else:
        version = "original" if "original" in asset["versions"] else next(iter(asset["versions"]), "")
        if not version:
            raise CliError("no-such-version", f"{asset['id']} has no rendition to edit")
        source = Path(_fetch(app, asset, version)["path"])
    try:
        result = editing.edit(source, prompt,
                              name=asset.get("filename") or asset["id"],
                              root=Path(out) if out else editing.edits_dir(app.config.values.get("edits_dir")),
                              agent=str(app.config.values.get("edit_agent", editing.DEFAULT_AGENT)),
                              timeout=timeout or int(app.config.values.get("edit_timeout_s", editing.DEFAULT_TIMEOUT)),
                              progress=lambda m: None if quiet else print(f"  {m}", file=sys.stderr))
    except editing.EditFailed as err:
        raise CliError("edit-failed", str(err)) from err
    w, h = (result["size"].split("x") + ["", ""])[:2] if result.get("size") else ("", "")
    vid = app.catalog.add_variant(asset["id"], parent, result["path"], prompt, result.get("method"),
                                  version, int(w) if w.isdigit() else None, int(h) if h.isdigit() else None,
                                  result["bytes"])
    return {"id": asset["id"], "variant": vid, "parent": parent} | result


def cmd_edit(app: App, args: argparse.Namespace) -> int:
    """Ask the agent to change one photo; the result is a new version on disk."""
    asset = _asset_for(app, args.id)
    if args.background:
        job = app.catalog.add_job(asset["id"], args.frm, args.prompt)
        info = _spawn_worker("edit", ["edit", "--job", str(job)], app.paths.edit_log, app.paths.edit_lock)
        app.catalog.set_job(job, state="queued", pid=info.get("pid"))
        app.emit({"job": job} | info, lambda p: f"edit queued as job {p['job']}; watch with `photos jobs`")
        return 0
    result = _run_edit(app, asset, args.prompt, args.frm, args.out, args.timeout, app.json)
    app.emit(result, lambda r: f"{r['id']}\tversion {r['variant']}\t{r['path']}\n  {r['method']}, {r['seconds']}s, "
                               f"{human_bytes(r['bytes'])}"
                               + (f", {r['source_size']} -> {r['size']}" if r["size"] and r["size"] != r["source_size"] else ""))
    return 0


def cmd_edit_job(app: App, job_id: int) -> int:
    """Run one queued edit. This is what the background worker does."""
    job = app.catalog.job(job_id)
    if job is None:
        raise CliError("no-such-job", f"there is no job {job_id}")
    app.catalog.set_job(job_id, state="running", started=catalog_now(), pid=os.getpid())
    try:
        result = _run_edit(app, _assets(app, [job["asset_id"]])[0], job["prompt"], job["parent"], None, None, True)
    except CliError as err:
        app.catalog.set_job(job_id, state="failed", error=str(err), finished=catalog_now())
        raise
    app.catalog.set_job(job_id, state="done", variant=result["variant"], finished=catalog_now())
    app.emit({"job": job_id} | result, lambda r: f"job {r['job']} done: version {r['variant']} at {r['path']}")
    return 0


def cmd_variants(app: App, args: argparse.Namespace) -> int:
    """Every version of a photo, and what was asked for each."""
    asset = _asset_for(app, args.id)
    rows = app.catalog.variants_of(asset["id"])
    for r in rows:
        r["exists"] = Path(r["path"]).exists()
    app.emit({"id": asset["id"], "versions": rows, "count": len(rows)},
             lambda p: "\n".join(f"{r['id']:>4}  {'from ' + str(r['parent']) if r['parent'] else 'from original':14s}  "
                                 f"{r['method'] or '?':9s}  {r['prompt'][:48]:48s}  "
                                 f"{'' if r['exists'] else '(file gone) '}{r['path']}" for r in p["versions"])
                       or "no versions yet")
    return 0


def cmd_jobs(app: App, args: argparse.Namespace) -> int:
    """Queued, running and finished edits."""
    states = tuple(args.state) if args.state else ()
    rows = app.catalog.jobs(states, limit=args.limit)
    app.emit({"jobs": rows, "count": len(rows)},
             lambda p: "\n".join(f"{r['id']:>4}  {r['state']:8s}  {r['asset_id'][:8]}  {r['prompt'][:44]:44s}  "
                                 f"{('version ' + str(r['variant'])) if r['variant'] else (r['error'] or '')[:40]}"
                                 for r in p["jobs"]) or "no edits yet")
    return 0


def cmd_refresh(app: App, args: argparse.Namespace) -> int:
    """sync, index what is new, and update the GUI library: what the timer runs."""
    steps: dict[str, Any] = {}
    if (pid := _sync_pid(app)) is not None:
        raise CliError("sync-running", f"a sync is already running (pid {pid}); see `photos status`")
    app.paths.sync_lock.write_text(str(os.getpid()))
    try:
        result = run_sync(app.catalog, app.adapter, full=False)
    finally:
        app.paths.sync_lock.unlink(missing_ok=True)
    steps["sync"] = {k: result[k] for k in ("mode", "records", "new", "changed", "missing") if k in result}
    # Only load the models when there is something to index: they cost ~2 GB, and this
    # runs hourly on a machine that may be doing something else.
    todo = app.catalog.db.execute(
        "SELECT COUNT(*) FROM assets a WHERE a.kind='image' AND a.hidden=0 AND a.missing_since IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM index_state s WHERE s.asset_id=a.id)").fetchone()[0]
    if (pid := _pid(app.paths.index_lock)) is not None:
        steps["index"] = {"todo": todo, "skipped": f"an index run is already going (pid {pid})"}
    elif args.no_index or not todo:
        steps["index"] = {"todo": todo, "skipped": "nothing new" if not todo else "--no-index"}
    else:
        app.paths.index_lock.write_text(str(os.getpid()))
        try:
            steps["index"] = indexing.index(app.catalog, app.adapter, app.cache, app.models, limit=args.index_limit)
        except ModelsMissing as err:
            steps["index"] = {"skipped": str(err)}
        finally:
            app.paths.index_lock.unlink(missing_ok=True)
    library = lap_export.default_library()
    if library is None:
        steps["gui"] = {"skipped": "no GUI library on this machine"}
    else:
        lib = lap_export.LapLibrary(library, app.paths.cache_dir / "lap")
        try:
            steps["gui"] = lap_export.export(app.catalog, app.cache, lib, fetch_command=_lap_fetch_command(app),
                                         cli=_photos_binary(),
                                             open_version=str(app.config.values.get("lap_open_version", "original")))
        finally:
            lib.close()
    app.emit(steps, lambda s: "\n".join(f"{k}: {v}" for k, v in s.items()))
    return 0


def _photos_binary() -> str:
    """This program, as an absolute path when it has one: the app runs it directly."""
    photos_bin = Path(sys.executable).with_name("photos")
    return str(photos_bin) if photos_bin.exists() else "photos"


def _lap_fetch_command(app: App) -> str:
    return f"{_photos_binary()} lap-fetch {{path}}"


def cmd_lap_fetch(app: App, args: argparse.Namespace) -> int:
    """lap's fetch-on-open command for our album: PATH is `<root>/YYYY/MM/<id>__<stem>.<ext>`."""
    target = Path(args.path)
    asset_id = lap_export.asset_id_of(target.name)
    row = app.catalog.db.execute("SELECT id FROM assets WHERE id=?", (asset_id,)).fetchone()
    if row is None:   # the id in the name is filename-safe; real ones are UUIDs, but be complete
        row = next((r for r in app.catalog.db.execute("SELECT id FROM assets") if lap_export.safe_name(r["id"]) == asset_id), None)
    if row is None:
        raise CliError("no-such-asset", f"{target.name} names no asset in the catalogue")
    asset = _assets(app, [row["id"]])[0]
    version = lap_export.version_for_entry(target.name, asset, str(app.config.values.get("lap_open_version", "original")))
    r = _fetch(app, asset, version)
    lap_export.link(target, Path(r["path"]))
    app.emit({"id": asset["id"], "version": version, "path": r["path"], "link": str(target)},
             lambda p: f"{p['id']}\t{p['link']} -> {p['path']}")
    return 0


def cmd_people(app: App, args: argparse.Namespace) -> int:
    rows = app.catalog.people()
    photos = app.catalog.person_photo_counts()
    for r in rows:
        r["photos"] = photos.get(r["id"], 0)
    if not args.all:
        rows = [r for r in rows if r["name"] or r["photos"]]
    app.emit(rows, lambda rs: "\n".join(
        f"{r['photos']:>6} photos {r['face_crops']:>3} crops  {r['name'] or '(unnamed)'}"
        + (f"  ({r['display_name']})" if r['display_name'] and r['display_name'] != r['name'] else "")
        + ("" if r['verified'] else "  unverified") + f"  {r['id']}" for r in rs)
        or "no people in the catalogue yet; run `photos sync`")
    return 0


def cmd_index(app: App, args: argparse.Namespace) -> int:
    if args.fetch_models:
        fetch_clip_models(app.paths.models_dir, report=lambda m: print(m, file=sys.stderr))
        app.emit({"models": str(app.paths.models_dir)}, lambda p: f"models under {p['models']}")
        return 0
    if (pid := _pid(app.paths.index_lock)) is not None:
        raise CliError("index-running", f"an index run is already going (pid {pid}); see `photos status`")
    if args.background:
        flags = [f for f, on in (("--seed", args.seed), ("--rematch", args.rematch)) if on]
        if args.limit:
            flags += ["--limit", str(args.limit)]
        if args.threshold is not None:
            flags += ["--threshold", str(args.threshold)]
        app.emit(_spawn_worker("index", ["index", *flags], app.paths.index_log, app.paths.index_lock),
                 _started_text("index"))
        return 0
    threshold = args.threshold if args.threshold is not None else float(app.config.values.get("face_threshold", indexing.DEFAULT_THRESHOLD))
    app.paths.index_lock.write_text(str(os.getpid()))
    out: dict[str, Any] = {}
    try:
        def progress(p: dict[str, Any]) -> None:
            if not app.json:
                print("  " + " ".join(f"{k} {v}" for k, v in p.items() if isinstance(v, int)), file=sys.stderr)
        try:
            if args.seed or not app.catalog.seeds():
                out["seed"] = indexing.seed_people(app.catalog, app.adapter, app.models, progress=progress)
            if args.rematch:
                out["rematch"] = indexing.rematch(app.catalog, threshold)
            else:
                out["index"] = indexing.index(app.catalog, app.adapter, app.cache, app.models, limit=args.limit,
                                              threshold=threshold, progress=progress)
        except ModelsMissing as err:
            raise CliError("models-missing", str(err), 5) from err
        except ComputeUnavailable as err:
            raise CliError("compute-unavailable", str(err), 6) from err
    finally:
        app.paths.index_lock.unlink(missing_ok=True)

    def text(o: dict[str, Any]) -> str:
        lines = []
        if "seed" in o:
            sd = o["seed"]; lines.append(f"seeds: {sd['seeded']} embedded from {sd['crops']} face crops ({sd['no_face']} without a face, {sd['failed']} failed)")
        if "rematch" in o:
            lines.append(f"rematch: {o['rematch']['changed']} of {o['rematch']['faces']} faces changed")
        if "index" in o:
            ix = o["index"]; lines.append(f"index: {ix['done']} of {ix['todo']} images ({ix['source']}); {ix['faces']} faces, {ix['named']} matched to a person; {ix['failed']} failed")
        return "\n".join(lines)
    app.emit(out, text)
    return 0


def cmd_faces(app: App, args: argparse.Namespace) -> int:
    c = app.catalog
    if args.faces_cmd == "show":
        rows = []
        for asset in _assets(app, args.id):
            rows += c.faces_of(asset["id"])
        app.emit(rows, lambda rs: "\n".join(
            f"face {f['id']:>7}  {f['asset_id']}  box {f['box']}  det {f['det']:.2f}  "
            + (f"{f['person_name'] or f['person_id']} ({f['similarity']:.2f}, {f['assigned']})" if f['person_id'] else
               (f"unassigned (best {f['similarity']:.2f})" if f['similarity'] is not None else "unassigned"))
            for f in rs) or "no faces recorded; is it indexed?")
    elif args.faces_cmd == "assign":
        face = c.get_face(args.face_id)
        if face is None:
            raise CliError("unknown-face", f"no face {args.face_id}")
        person = _person_id(app, args.person)
        c.assign_face(face["id"], person, None, "user")
        # a confirmed face is the best seed there is
        c.put_seed(f"face:{face['id']}", person, face["embedding"], face["det"], "user")
        app.emit({"face": face["id"], "person": person}, lambda r: f"face {r['face']} -> {args.person}; it now seeds that person")
    elif args.faces_cmd == "unassign":
        face = c.get_face(args.face_id)
        if face is None:
            raise CliError("unknown-face", f"no face {args.face_id}")
        c.assign_face(face["id"], None, None, "user-cleared")
        c.db.execute("DELETE FROM person_seeds WHERE id=?", (f"face:{face['id']}",))
        app.emit({"face": face["id"], "person": None}, lambda r: f"face {r['face']} unassigned")
    elif args.faces_cmd == "unassigned":
        rows = [dict(r) for r in c.db.execute(
            "SELECT f.id, f.asset_id, f.det, f.similarity, a.taken, a.filename FROM faces f JOIN assets a ON a.id=f.asset_id "
            "WHERE f.person_id IS NULL AND f.det >= ? ORDER BY f.similarity DESC LIMIT ?", (args.min_det, args.limit))]
        app.emit(rows, lambda rs: "\n".join(
            f"face {r['id']:>7}  {r['asset_id']}  {(r['taken'] or '')[:10]}  det {r['det']:.2f}  best {r['similarity'] if r['similarity'] is None else round(r['similarity'], 2)}  {r['filename']}"
            for r in rs) or "none")
    return 0


def cmd_albums(app: App, args: argparse.Namespace) -> int:
    rows = app.catalog.albums()
    app.emit(rows, lambda rs: "\n".join(f"{r['count']:>6}  {r['fullname']}  ({r['id']})" for r in rs)
             or "no albums in the catalogue; run `photos sync --albums`")
    return 0


def _person_id(app: App, key: str | None) -> str | None:
    if not key:
        return None
    person = app.catalog.find_person(key)
    if person is None:
        raise CliError("unknown-person", f"no person {key!r}; see `photos people`")
    return person["id"]


def cmd_search(app: App, args: argparse.Namespace) -> int:
    fav = True if args.favorite else (False if args.not_favorite else None)
    filters = dict(
        since=parse_date(args.since), until=parse_date(args.until, end=True),
        kind=args.kind, favorite=fav, album=args.album, collection=args.collection,
        person=_person_id(app, args.person),
        located=True if args.located else None, live=True if args.live else None,
        include_missing=args.include_missing, include_hidden=args.include_hidden)
    if args.semantic or args.similar:
        if not app.catalog.vec:
            raise CliError("no-vector-search", "sqlite-vec is not available, so there is no semantic search")
        if args.cursor:
            raise CliError("no-cursor", "semantic results are ranked by score, not paged; raise --limit instead")
        if args.similar:
            q = app.catalog.clip_of(args.similar)
            if q is None:
                raise CliError("not-indexed", f"{args.similar} has no embedding yet; run `photos index`")
        else:
            if not args.query:
                raise CliError("no-query", "--semantic needs a query, e.g. `photos search --semantic \"a beach at sunset\"`")
            try:
                q = as_blob(app.models.embed_text(args.query))
            except ModelsMissing as err:
                raise CliError("models-missing", str(err), 5) from err
        # over-fetch, then apply the metadata filters and keep the order by score
        nearest = app.catalog.nearest_clip(q, max(args.limit * 8, 200))
        score = {aid: 1 - d * d / 2 for aid, d in nearest}       # L2 on unit vectors -> cosine
        rows, _ = app.catalog.search(ids=list(score), limit=len(score) or 1, **filters)
        rows.sort(key=lambda a: -score[a["id"]])
        rows = rows[:args.limit]
        for a in rows:
            a["score"] = round(score[a["id"]], 4)
        payload = {"results": rows, "count": len(rows), "next_cursor": None,
                   "query": args.query if args.semantic else f"similar to {args.similar}"}
    else:
        rows, next_cursor = app.catalog.search(text=args.query, limit=args.limit, cursor=args.cursor, **filters)
        payload = {"results": rows, "count": len(rows), "next_cursor": next_cursor}

    def text(p: dict[str, Any]) -> str:
        lines = [(f"{a['score']:.3f}  " if "score" in a else "") + asset_line(a) for a in p["results"]] or ["no matches"]
        if p["next_cursor"]:
            lines.append(f"more: --cursor {p['next_cursor']}")
        return "\n".join(lines)

    app.emit(payload, text)
    return 0


def cmd_select(app: App, args: argparse.Namespace) -> int:
    """Choose a handful of good photos: the funnel in `select.py`, wired to the catalogue."""
    from . import select as selector

    people = [_person_id(app, k) for k in (args.person or [])]
    fav = True if args.favorite else None
    filters = dict(
        since=parse_date(args.since), until=parse_date(args.until, end=True),
        kind=args.kind or "image", favorite=fav, album=args.album, collection=args.collection,
        people=people or None, located=True if args.located else None)
    controls = selector.Controls(
        variety=args.variety, spread=args.spread, everyone=args.everyone,
        sharp_only=args.sharp_only, duplicates=args.keep_duplicates, floor=args.floor,
        screenshots=args.include_screenshots)

    anchor = None
    if args.similar:
        anchor = app.catalog.clip_of(_asset_for(app, args.similar)["id"])
        if anchor is None:
            raise CliError("not-indexed",
                           f"{args.similar} has no embedding yet; run `photos index`")

    embed = None
    if args.query:
        if not app.catalog.vec:
            raise CliError("no-vector-search", "sqlite-vec is not available, so there is no ranking by meaning")

        def embed(text: str) -> bytes:                                  # noqa: F811
            try:
                return as_blob(app.models.embed_text(text))
            except ModelsMissing as err:
                raise CliError("models-missing", str(err), 5) from err

    result = selector.run(
        app.catalog, query=args.query, query_vector=anchor, count=args.count,
        filters=filters, controls=controls,
        embed=embed, thumb_for=lambda aid: app.cache.peek(aid, "thumb"),
        people_required=people if args.everyone else ())

    payload = result.as_dict()
    if args.into and result.picked:
        app.catalog.collection_create(args.into)
        if args.replace:
            app.catalog.collection_remove(
                args.into, [r["id"] for r in app.catalog.collection_items(args.into)])
        added = app.catalog.collection_add(args.into, [a["id"] for a in result.picked])
        payload["collection"] = {"name": args.into, "added": added}

    def text(pl: dict[str, Any]) -> str:
        lines = [f"{s['kept']:>7,}  {s['stage']:<20} {s['note']}" for s in pl["stages"]]
        lines.append("")
        lines += [f"{a['score']:.3f}  " + asset_line(a) for a in pl["selected"]] or ["nothing selected"]
        if pl.get("collection"):
            lines.append(f"added {pl['collection']['added']} to collection {pl['collection']['name']!r}")
        if pl.get("reason"):
            lines.append(f"note: {pl['reason']}")
        return "\n".join(lines)

    app.emit(payload, text)
    return 0


def _compose_photo(app: App, asset: dict[str, Any], want_original: bool,
                   fetched: list[str]) -> Any:
    """Resolve one asset to a file on disk plus its face boxes, as the layout needs it."""
    from . import compose as composer
    from .lap import rendition_dims

    order = ("original", "medium", "thumb") if want_original else ("original", "medium", "thumb")
    path = None
    if want_original:
        path = app.cache.peek(asset["id"], "original")
        if path is None:
            path = Path(_fetch(app, asset, "original")["path"])
            fetched.append(asset["id"])
    if path is None:
        for version in order:
            path = app.cache.peek(asset["id"], version)
            if path is not None:
                break
    if path is None:
        # nothing cached and no original asked for: a medium is the cheap way in
        path = Path(_fetch(app, asset, "medium")["path"])
        fetched.append(asset["id"])

    width, height = composer.image_size(path)
    faces = app.catalog.faces_of(asset["id"])
    boxes: list[Any] = []
    if faces:
        source = faces[0].get("source") or "thumb"
        sw, sh = rendition_dims(asset, source)
        if not (sw and sh):
            cached = app.cache.peek(asset["id"], source)
            if cached is not None:
                sw, sh = composer.image_size(cached)
        boxes = composer.normalise_faces([f["box"] for f in faces], sw, sh)
    return composer.Photo(asset["id"], path, width, height, boxes)


def cmd_compose(app: App, args: argparse.Namespace) -> int:
    """Turn a set of photos into one picture."""
    from . import compose as composer

    if args.id:
        # ids or the paths of entries in the GUI album, so the app can pass what it has
        assets = [_asset_for(app, i) for i in args.id]
        name = args.name or "selection"
    else:
        if not args.name:
            raise CliError("nothing-to-compose", "name a collection, or give --id ID...")
        if not app.catalog.collection_exists(args.name):
            raise CliError("unknown-collection",
                           f"no collection {args.name!r}; see `photos collection list`")
        assets = app.catalog.collection_items(args.name)
        name = args.name
    assets = [a for a in assets if a["kind"] == "image"]
    if args.count:
        assets = assets[:args.count]
    if not assets:
        raise CliError("nothing-to-compose", "no photos in that set (movies are not composed)")

    if args.originals:
        # Say what it costs before reaching for the network, and never quietly
        # overrun the cache budget for a collage.
        missing = [a for a in assets if app.cache.peek(a["id"], "original") is None]
        need = sum(int(a.get("bytes") or 0) for a in missing)
        usage = app.cache.usage()
        if need and usage["bytes"] + need > usage["budget_bytes"]:
            raise CliError("cache-full",
                           f"{len(missing)} originals would need {human_bytes(need)} and the cache "
                           f"budget has {human_bytes(max(0, usage['budget_bytes'] - usage['bytes']))} "
                           "left; raise cache_budget_mb or evict first", 4)

    fetched: list[str] = []
    photos = [_compose_photo(app, a, args.originals, fetched) for a in assets]

    if args.plan:
        payload = composer.plan(photos, template=args.template, shape=args.shape,
                                gap=args.gap, long_edge=args.long_edge, face_safe=not args.no_face_safe)
        payload["fetched"] = len(fetched)
        app.emit(payload, lambda pl: "\n".join(
            [f"{pl['template']} on {pl['width']}x{pl['height']} ({pl['note']})"] +
            [f"  {s['id'][:8]}  {s['w']:>5}x{s['h']:<5} at +{s['x']}+{s['y']}"
             f"  crop {s['crop']['w']}x{s['crop']['h']}+{s['crop']['x']}+{s['crop']['y']}"
             for s in pl["slots"]]))
        return 0

    root = Path(str(app.config.values.get("collages_dir") or composer.DEFAULT_DIR))
    out = Path(args.out) if args.out else composer.output_path(root, name, args.template, args.shape)
    long_edge = args.long_edge or (None if args.originals else composer.DRAFT_LONG_EDGE)
    try:
        result = composer.render(
            photos, out, template=args.template, shape=args.shape, gap=args.gap,
            long_edge=long_edge, face_safe=not args.no_face_safe, background=args.background,
            progress=(lambda m: None) if app.json else (lambda m: print(m, file=sys.stderr)))
    except composer.ComposeFailed as err:
        raise CliError("compose-failed", str(err)) from err

    result["collection"] = name
    result["fetched"] = len(fetched)
    result["faces_used"] = sum(1 for p in photos if p.faces)
    app.emit(result, lambda r: (
        f"{r['path']}\n{r['template']} - {r['photos']} photos on {r['width']}x{r['height']} "
        f"({r['note']}), {human_bytes(r['bytes'])} in {r['seconds']}s"
        + (f"; fetched {r['fetched']} originals" if r["fetched"] else "")
        + ("; draft resolution, use --originals to print it" if r["draft"] else "")))
    return 0


def _book_or_fail(app: App, name: str) -> dict[str, Any]:
    book = app.catalog.book(name)
    if book is None:
        raise CliError("unknown-book", f"no book {name!r}; see `photos book list`")
    return book


def cmd_book(app: App, args: argparse.Namespace) -> int:
    """An ordered list of pages, exported as one PDF."""
    from . import compose as composer

    cmd = args.book_cmd
    if cmd == "list":
        rows = app.catalog.books()
        app.emit(rows, lambda rs: "\n".join(
            f"{r['pages']:>4} pages  {r['shape']:<14} {r['name']}" + (f"  {r['note']}" if r["note"] else "")
            for r in rs) or "no books yet")
    elif cmd == "create":
        made = app.catalog.book_create(args.name, args.shape, args.note)
        app.emit({"name": args.name, "shape": args.shape, "created": made},
                 lambda r: f"{'created' if r['created'] else 'already exists'}: {r['name']} ({r['shape']})")
    elif cmd == "delete":
        _book_or_fail(app, args.name)
        app.catalog.book_delete(args.name)
        app.emit({"name": args.name, "deleted": True},
                 lambda r: f"deleted book {r['name']} (photos and collages untouched)")
    elif cmd == "add":
        _book_or_fail(app, args.name)
        if not args.collection and not args.id:
            raise CliError("nothing-on-the-page", "a page needs --collection NAME or --id ID...")
        if args.collection and not app.catalog.collection_exists(args.collection):
            raise CliError("unknown-collection",
                           f"no collection {args.collection!r}; see `photos collection list`")
        ids = [_asset_for(app, i)["id"] for i in (args.id or [])] or None
        page = app.catalog.book_add_page(args.name, collection=args.collection, ids=ids,
                                         template=args.template, caption=args.caption)
        pages = app.catalog.book_pages(args.name)
        app.emit({"book": args.name, "page": len(pages), "id": page},
                 lambda r: f"page {r['page']} added to {r['book']}")
    elif cmd == "remove":
        _book_or_fail(app, args.name)
        if not app.catalog.book_remove_page(args.name, args.page):
            raise CliError("no-such-page", f"{args.name} has no page {args.page}")
        app.emit({"book": args.name, "removed": args.page},
                 lambda r: f"removed page {r['removed']} from {r['book']}")
    elif cmd == "move":
        _book_or_fail(app, args.name)
        if not app.catalog.book_move_page(args.name, args.page, args.to):
            raise CliError("no-such-page", f"{args.name} has no page {args.page} or {args.to}")
        app.emit({"book": args.name, "moved": args.page, "to": args.to},
                 lambda r: f"page {r['moved']} is now page {r['to']}")
    elif cmd == "show":
        book = _book_or_fail(app, args.name)
        pages = app.catalog.book_pages(args.name)
        app.emit({"name": args.name, "shape": book["shape"], "pages": pages},
                 lambda r: "\n".join(
                     [f"{r['name']} ({r['shape']})"] +
                     [f"  {p['position']:>3}. {p['template']:<10} "
                      f"{p['collection'] or str(len(p['ids'] or [])) + ' photos'}"
                      + (f"  \u2014 {p['caption']}" if p["caption"] else "")
                      for p in r["pages"]] or ["  no pages yet"]))
    elif cmd == "export":
        book = _book_or_fail(app, args.name)
        pages = app.catalog.book_pages(args.name)
        if not pages:
            raise CliError("empty-book", f"{args.name} has no pages; add one with `photos book add`")
        root = Path(str(app.config.values.get("collages_dir") or composer.DEFAULT_DIR))
        out = Path(args.out).expanduser() if args.out else composer.output_path(
            root, args.name, "book", book["shape"], ".pdf")
        work = out.parent / f".{out.stem}-pages"
        work.mkdir(parents=True, exist_ok=True)
        long_edge = None if args.originals else composer.DRAFT_LONG_EDGE
        drawn: list[Path] = []
        fetched: list[str] = []
        try:
            for page in pages:
                if page["collection"]:
                    assets = app.catalog.collection_items(page["collection"])
                else:
                    assets = [_asset_for(app, i) for i in (page["ids"] or [])]
                assets = [a for a in assets if a["kind"] == "image"]
                if len(assets) < 1:
                    raise CliError("empty-page",
                                   f"page {page['position']} of {args.name} has no photos left")
                photos = [_compose_photo(app, a, args.originals, fetched) for a in assets]
                target = work / f"page-{page['position']:03d}.jpg"
                if not app.json:
                    print(f"page {page['position']} of {len(pages)}", file=sys.stderr)
                composer.render(photos, target, template=page["template"], shape=book["shape"],
                                gap=args.gap, long_edge=long_edge, background=args.background,
                                caption=page["caption"])
                drawn.append(target)
            result = composer.export_pdf(drawn, out, dpi=300 if args.originals else 150)
        except composer.ComposeFailed as err:
            raise CliError("compose-failed", str(err)) from err
        finally:
            if not args.keep_pages:
                shutil.rmtree(work, ignore_errors=True)
        result["book"] = args.name
        result["shape"] = book["shape"]
        result["draft"] = not args.originals
        result["fetched"] = len(fetched)
        app.emit(result, lambda r: (
            f"{r['path']}\n{r['pages']} pages, {human_bytes(r['bytes'])} in {r['seconds']}s"
            + (f"; fetched {r['fetched']} originals" if r["fetched"] else "")
            + ("; draft resolution, use --originals to print it" if r["draft"] else "")))
    return 0


def cmd_faces_cluster(app: App, args: argparse.Namespace) -> int:
    """Group faces that belong to the same person, so naming is done a group at a time."""
    from . import cluster as clustering

    result = clustering.build(app.catalog, threshold=args.threshold, min_size=args.min_size,
                              progress=(lambda m: None) if app.json else
                                       (lambda m: print(m, file=sys.stderr)))
    app.emit(result, lambda r: (
        f"{r['clusters']:,} groups over {r['grouped']:,} of {r['faces']:,} faces "
        f"at {r['threshold']:.2f}; {r['loose']:,} stayed on their own"))
    return 0


def cmd_faces_groups(app: App, args: argparse.Namespace) -> int:
    from . import cluster as clustering

    named = True if args.named else (False if args.unnamed else None)
    rows = [g.as_dict() for g in clustering.groups(
        app.catalog, limit=args.limit, named=named, min_size=args.min_size)]
    app.emit({"clusters": rows, "count": len(rows)}, lambda p: "\n".join(
        f"{g['cluster']:>5}  {g['faces']:>5} faces  "
        f"{(g['first_seen'] or '?')[:7]}..{(g['last_seen'] or '?')[:7]}  "
        f"{g['person'] or '(no name yet)'}" for g in p["clusters"]) or "no groups; run `photos faces cluster`")
    return 0


def cmd_faces_name(app: App, args: argparse.Namespace) -> int:
    from . import cluster as clustering

    person = app.catalog.find_person(args.person)
    if person is None:
        raise CliError("unknown-person", f"no person {args.person!r}; see `photos people`")
    try:
        result = clustering.name(app.catalog, args.cluster, person["id"], seeds=args.seeds,
                                 force=args.force)
    except ValueError as err:
        raise CliError("no-such-cluster", str(err)) from err
    result["person"] = person["name"]
    app.emit(result, lambda r: (
        f"named {r['faces']} faces {r['person']!r} and seeded {r['seeds']} of them "
        f"across {(r['first'] or '?')[:4]}..{(r['last'] or '?')[:4]}"
        + (f"; {r['left_alone']} kept the name they had" if r["left_alone"] else "")
        + "; run `photos index --rematch` to spread it"))
    return 0


def cmd_faces_unname(app: App, args: argparse.Namespace) -> int:
    from . import cluster as clustering

    n = clustering.unname(app.catalog, args.cluster)
    app.emit({"cluster": args.cluster, "cleared": n},
             lambda r: f"cleared {r['cleared']} faces in group {r['cluster']} (hand-made names kept)")
    return 0


def cmd_faces_crop(app: App, args: argparse.Namespace) -> int:
    """Write a JPEG of each face, so an interface can show who it is asking about."""
    import subprocess
    from . import compose as composer
    from .lap import rendition_dims

    magick = shutil.which("magick")
    if magick is None:
        raise CliError("no-imagemagick", "ImageMagick 7 (`magick`) is not installed")
    out_dir = Path(args.out).expanduser() if args.out else (app.paths.cache_dir / "faces")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for face_id in args.face_id:
        face = app.catalog.get_face(face_id)
        if face is None:
            raise CliError("no-such-face", f"no face {face_id}")
        asset = app.catalog.get_asset(face["asset_id"])
        source = app.cache.peek(face["asset_id"], "medium") or app.cache.peek(face["asset_id"], "thumb")
        if source is None:
            results.append({"face_id": face_id, "asset_id": face["asset_id"], "path": None,
                            "reason": "nothing cached for that photo yet"})
            continue
        # the box is in the pixels of whichever rendition the face pass analysed
        sw, sh = rendition_dims(asset, face.get("source") or "thumb")
        if not (sw and sh):
            cached = app.cache.peek(face["asset_id"], face.get("source") or "thumb")
            sw, sh = composer.image_size(cached) if cached else (None, None)
        boxes = composer.normalise_faces([face["box"]], sw, sh)
        fw, fh = composer.image_size(source)
        if not boxes:
            x, y, w, h = fw // 4, fh // 4, fw // 2, fh // 2      # no usable box: the middle
        else:
            x1, y1, x2, y2 = boxes[0]
            pad = 0.35 * max(x2 - x1, y2 - y1)                   # room for hair and chin
            x1, y1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
            x2, y2 = min(1.0, x2 + pad), min(1.0, y2 + pad)
            x, y = int(x1 * fw), int(y1 * fh)
            w, h = max(8, int((x2 - x1) * fw)), max(8, int((y2 - y1) * fh))
        target = out_dir / f"face-{face_id}.jpg"
        run = subprocess.run([magick, str(source), "-auto-orient",
                              "-crop", f"{w}x{h}+{x}+{y}", "+repage",
                              "-resize", f"{args.size}x{args.size}^",
                              "-gravity", "center", "-extent", f"{args.size}x{args.size}",
                              "-quality", "88", str(target)],
                             capture_output=True, text=True, timeout=120)
        if run.returncode != 0 or not target.exists():
            results.append({"face_id": face_id, "asset_id": face["asset_id"], "path": None,
                            "reason": (run.stderr or "").strip().splitlines()[-1:][0][:120] if run.stderr else "crop failed"})
            continue
        results.append({"face_id": face_id, "asset_id": face["asset_id"],
                        "person": face.get("person_name"), "path": str(target)})
    app.emit(results, lambda rs: "\n".join(
        f"{r['face_id']}\t{r['path'] or '- ' + str(r.get('reason'))}" for r in rs))
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
        a["faces"] = [{k: v for k, v in f.items() if k != "asset_id"} for f in app.catalog.faces_of(a["id"])]
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
            data = app.adapter.download(asset["id"], version, asset.get("master_id"),
                                        zone=asset.get("zone"))
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
        if version not in asset["versions"]:
            # PNGs and some imports have no medium: the original if it is small, else the thumb
            original = asset["versions"].get("original", {})
            if asset["kind"] == "image" and (original.get("bytes") or 0) <= 4 * 1024 * 1024:
                version = "original"
            else:
                version = "thumb" if asset["kind"] == "image" else "thumb_image"
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
        if args.key == "compute" and args.value not in COMPUTE_MODES:
            raise CliError("bad-value", f"compute must be one of {', '.join(COMPUTE_MODES)}")
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
                    "with a non-zero exit (2 needs a terminal, 3 not logged in, 4 cache full, 5 models missing,\n"
                    "6 compute=gpu without a usable GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="typical session:\n"
               "  photos login                      once, interactive (password + two-factor code)\n"
               "  photos sync --background          then `photos status` until it finishes\n"
               "  photos search beach --since 2019-07 --until 2019-08 --json\n"
               "  photos show <id> <id>             fetch small previews, print their paths\n"
               "  photos original <id> --pin        fetch the full file and keep it\n"
               "  photos index --background        CLIP + faces for every image; then\n"
               "  photos search --semantic \"kids on a beach\" --person Julie --since 2019\n"
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
    s.add_argument("--no-shared", action="store_true",
                   help="skip shared libraries; by default every zone the account can read is walked")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("albums", help="list albums known to the catalogue")
    s.set_defaults(fn=cmd_albums)

    s = sub.add_parser("people", help="people from iCloud's People album, with photos where their face was found")
    s.add_argument("--all", action="store_true", help="include unnamed people with no matched photos")
    s.set_defaults(fn=cmd_people)

    s = sub.add_parser("index", help="embed images (CLIP) and find faces, matched to named people",
                       description="Pass 1 runs on thumbnails: one CLIP embedding per image for `search --semantic` "
                                   "and `--similar`, and every face with its match to a named person. The first run "
                                   "seeds people from the face crops iCloud keeps for them. Resumable; run it detached "
                                   "with --background and follow `photos status`.")
    s.add_argument("--limit", type=int, metavar="N", help="index at most N images (newest first)")
    s.add_argument("--seed", action="store_true", help="(re)build person seeds from iCloud face crops first")
    s.add_argument("--rematch", action="store_true", help="only re-run person matching over known faces")
    s.add_argument("--threshold", type=float, help="cosine similarity needed to name a face (config face_threshold, default 0.5)")
    s.add_argument("--fetch-models", action="store_true", help="download the CLIP model files (~600 MB) and exit")
    s.add_argument("--background", action="store_true", help="run detached; follow with `photos status`")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("lap-export", help="write the catalogue into a lap library so lap can browse it",
                       description="Writes files, thumbnails, CLIP embeddings, people, faces and collections into a lap "
                                   "library database (https://github.com/julyx10/lap), as one album rooted at a tree of "
                                   "symlinks under the cache. Idempotent; lap needs a restart to show the changes.")
    s.add_argument("--library", help="lap library database (default: lap's default library)")
    s.add_argument("--root", help="album root for the symlink tree (default: <cache>/lap)")
    s.add_argument("--thumb-size", type=int, default=512, help="lap's gallery thumbnail setting: 256, 512 (default) or 1024; a mismatch makes lap regenerate every thumbnail")
    s.add_argument("--limit", type=int, help="export only the newest N assets")
    s.add_argument("--fetch-thumbs", action="store_true", help="download thumbnails that are not cached (movies included); needs iCloud")
    s.add_argument("--open-version", choices=("original", "medium"), help="what opening a photo fetches (config lap_open_version, default original)")
    s.set_defaults(fn=cmd_lap_export)

    s = sub.add_parser("lap-fetch", help="fetch the file behind a lap album entry (lap's fetch-on-open command)",
                       description="Given the path of an entry in the exported lap album, fetches the best rendition "
                                   "into the cache and points the entry at it. lap runs this when a preview is opened "
                                   "and the file is not on disk.")
    s.add_argument("path", help="the entry's path under the album root")
    s.set_defaults(fn=cmd_lap_fetch)

    s = sub.add_parser("edit", help="ask an agent to change a photo; the result is a new file on disk",
                       description="Fetches the original and hands it to the Codex CLI, which signs in with a "
                                   "ChatGPT subscription and edits the photo with the tools on this machine "
                                   "(ImageMagick, ffmpeg). The result goes to <edits_dir>/<date>/, an ordinary "
                                   "folder; nothing is written to iCloud and the original is untouched. Ordinary "
                                   "photo work only: exposure, colour, crop, rotate, resize, borders, text. "
                                   "Inventing content needs an image model, which this does not use.")
    s.add_argument("id", nargs="?", help="asset id, or the path of an entry in the GUI album")
    s.add_argument("prompt", nargs="?", help="what to change, in plain words")
    s.add_argument("--from", dest="frm", type=int, metavar="VERSION",
                   help="build on an earlier version of the photo instead of the original")
    s.add_argument("--background", action="store_true", help="queue it and return at once; watch with `photos jobs`")
    s.add_argument("--out", help="where to put the result (config edits_dir, default ~/Pictures/Photos Edits)")
    s.add_argument("--timeout", type=int, help="seconds to allow the agent (config edit_timeout_s, default 900)")
    s.add_argument("--job", type=int, help=argparse.SUPPRESS)   # the background worker runs one queued job
    s.set_defaults(fn=lambda app, a: cmd_edit_job(app, a.job) if a.job else cmd_edit(app, a))

    s = sub.add_parser("versions", help="the edited versions of a photo",
                       description="Every version made from a photo, oldest first, with the request that produced "
                                   "each one. The original is untouched and stays in iCloud.")
    s.add_argument("id", help="asset id, or the path of an entry in the GUI album")
    s.set_defaults(fn=cmd_variants)

    s = sub.add_parser("jobs", help="queued, running and finished edits",
                       description="Edits queued with `edit --background`, newest first.")
    s.add_argument("--state", action="append", choices=("queued", "running", "done", "failed"),
                   help="only these states (repeatable)")
    s.add_argument("--limit", type=int, default=20, help="how many to show (default 20)")
    s.set_defaults(fn=cmd_jobs)

    s = sub.add_parser("refresh", help="sync, index what is new, and update the GUI library",
                       description="One pass of everything that keeps the library current: an incremental sync, an "
                                   "index run over whatever it added, and an export to the GUI library when one "
                                   "exists. This is what the systemd timer runs.")
    s.add_argument("--no-index", action="store_true", help="skip the index run")
    s.add_argument("--index-limit", type=int, help="index at most N new images this pass")
    s.set_defaults(fn=cmd_refresh)

    s = sub.add_parser("faces", help="faces found in photos; assign or clear a person")
    fs = s.add_subparsers(dest="faces_cmd", metavar="action", required=True)
    x = fs.add_parser("show", help="faces in the given assets"); x.add_argument("id", nargs="+")
    x = fs.add_parser("assign", help="name a face; it becomes a seed for that person")
    x.add_argument("face_id", type=int); x.add_argument("person", help="name, display name or id from `photos people`")
    x = fs.add_parser("unassign"); x.add_argument("face_id", type=int)
    x = fs.add_parser("cluster", help="group faces that look like the same person")
    x.add_argument("--threshold", type=float, default=0.62, metavar="COS",
                   help="how alike two faces must be to be joined (default 0.62)")
    x.add_argument("--min-size", type=int, default=2, metavar="N",
                   help="ignore groups smaller than this (default 2)")
    x.set_defaults(fn=cmd_faces_cluster)
    x = fs.add_parser("groups", help="the groups, largest first")
    x.add_argument("--limit", type=int, default=40, metavar="N")
    x.add_argument("--min-size", type=int, default=2, metavar="N")
    x.add_argument("--named", action="store_true", help="only groups that already have a name")
    x.add_argument("--unnamed", action="store_true", help="only groups still without one")
    x.set_defaults(fn=cmd_faces_groups)
    x = fs.add_parser("name", help="name a whole group at once; its faces become seeds")
    x.add_argument("cluster", type=int); x.add_argument("person")
    x.add_argument("--seeds", type=int, default=12, metavar="N",
                   help="how many faces to seed from, spread over the group's whole range")
    x.add_argument("--force", action="store_true",
                   help="also rename faces that already carry a different person")
    x.set_defaults(fn=cmd_faces_name)
    x = fs.add_parser("unname", help="undo naming a group; hand-made names are kept")
    x.add_argument("cluster", type=int)
    x.set_defaults(fn=cmd_faces_unname)
    x = fs.add_parser("crop", help="write a JPEG of a face, for showing who is being asked about")
    x.add_argument("face_id", nargs="+", type=int)
    x.add_argument("--out", metavar="DIR", help="where to write them (default the cache)")
    x.add_argument("--size", type=int, default=256, metavar="PX")
    x.set_defaults(fn=cmd_faces_crop)
    x = fs.add_parser("unassigned", help="faces no person matched, most nearly matched first")
    x.add_argument("--limit", type=int, default=30); x.add_argument("--min-det", type=float, default=0.7)
    s.set_defaults(fn=cmd_faces)

    s = sub.add_parser("search", help="find assets in the catalogue (no network)",
                       description="Newest capture first. Results are paged: pass the printed cursor back "
                                   "with --cursor for the next page. Dates accept YYYY, YYYY-MM, YYYY-MM-DD "
                                   "or ISO 8601; --until includes the whole of the day, month or year given.")
    s.add_argument("query", nargs="?", help="substring of the filename or caption")
    s.add_argument("--since", metavar="DATE"); s.add_argument("--until", metavar="DATE")
    s.add_argument("--kind", choices=("image", "movie"))
    s.add_argument("--favorite", action="store_true"); s.add_argument("--not-favorite", action="store_true")
    s.add_argument("--album", metavar="NAME_OR_ID"); s.add_argument("--collection", metavar="NAME")
    s.add_argument("--person", metavar="NAME_OR_ID", help="only photos where this person's face was found")
    s.add_argument("--semantic", action="store_true", help="rank by meaning: the query describes the picture (CLIP)")
    s.add_argument("--similar", metavar="ID", help="rank by visual similarity to this asset")
    s.add_argument("--located", action="store_true", help="only assets with GPS coordinates")
    s.add_argument("--live", action="store_true", help="only Live Photos")
    s.add_argument("--include-missing", action="store_true", help="also assets no longer in iCloud")
    s.add_argument("--include-hidden", action="store_true")
    s.add_argument("--limit", type=int, default=50, metavar="N", help="page size (default 50)")
    s.add_argument("--cursor", help="continue from a previous page")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("select", help="choose a handful of good photos out of thousands",
                       description="A funnel, not a ranking: filters, then meaning, then "
                                   "near-duplicates collapsed, then quality, then variety and "
                                   "quotas. Every stage reports what it removed, so an empty "
                                   "result can be explained. Images only unless --kind says otherwise.")
    s.add_argument("query", nargs="?", help="describe the picture you want (ranked by CLIP)")
    s.add_argument("--similar", metavar="ID",
                   help="grow a set around this photo instead of a description; "
                        "an id or a GUI album entry path")
    s.add_argument("--count", type=int, default=9, metavar="N", help="how many to choose (default 9)")
    s.add_argument("--since", metavar="DATE"); s.add_argument("--until", metavar="DATE")
    s.add_argument("--kind", choices=("image", "movie"))
    s.add_argument("--album", metavar="NAME_OR_ID"); s.add_argument("--collection", metavar="NAME")
    s.add_argument("--person", metavar="NAME_OR_ID", action="append",
                   help="only photos with this person; repeat for several (any of them)")
    s.add_argument("--favorite", action="store_true"); s.add_argument("--located", action="store_true")
    s.add_argument("--variety", type=float, default=0.45, metavar="0..1",
                   help="0 is the closest match to the query, 1 is the widest spread (default 0.45)")
    s.add_argument("--spread", choices=("none", "day", "month", "year"), default="none",
                   help="don't take them all from one afternoon, month or year")
    s.add_argument("--everyone", action="store_true",
                   help="every person named with --person appears at least once")
    s.add_argument("--sharp-only", action="store_true", help="drop soft frames")
    s.add_argument("--keep-duplicates", action="store_true",
                   help="keep near-duplicates instead of collapsing each burst to its best frame")
    s.add_argument("--include-screenshots", action="store_true",
                   help="keep screen captures; they are set aside by default, since a PNG in a "
                        "photo library is a screenshot rather than a photograph")
    s.add_argument("--floor", type=float, default=0.0, metavar="SCORE",
                   help="minimum meaning score to survive (default 0)")
    s.add_argument("--into", metavar="COLLECTION", help="add the chosen photos to this collection, in order")
    s.add_argument("--replace", action="store_true", help="with --into, empty the collection first")
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("compose", help="turn a set of photos into one picture",
                       description="Templates: justified fills the page with rows that each span "
                                   "the full width; grid uses uniform cells; hero makes the first "
                                   "photo large; filmstrip runs equal frames across the page; "
                                   "spread lays out two facing pages; scatter drops mounted prints "
                                   "on a table. No crop cuts a face when the catalogue knows where "
                                   "they are. Draft resolution unless --originals.")
    s.add_argument("name", nargs="?", help="a collection (see `photos collection list`)")
    s.add_argument("--id", nargs="+", metavar="ID",
                   help="compose these assets instead of a collection; ids or GUI album entry paths")
    s.add_argument("--template", default="justified",
                   choices=("justified", "grid", "hero", "filmstrip", "spread", "scatter"))
    s.add_argument("--shape", default="3:2",
                   choices=("3:2", "square", "a4-landscape", "a4-portrait", "spread", "16:9"))
    s.add_argument("--gap", type=int, default=8, metavar="N", help="space between photos (default 8)")
    s.add_argument("--count", type=int, metavar="N", help="use only the first N of the set")
    s.add_argument("--background", default="white", metavar="COLOUR")
    s.add_argument("--long-edge", type=int, metavar="PX", help="override the page's pixel size")
    s.add_argument("--originals", action="store_true",
                   help="fetch full-resolution originals and compose at print size")
    s.add_argument("--no-face-safe", action="store_true",
                   help="crop from the centre even when it cuts a face")
    s.add_argument("--plan", action="store_true", help="print the geometry; write no file")
    s.add_argument("--out", metavar="FILE", help="write here instead of the dated folder")
    s.set_defaults(fn=cmd_compose)

    s = sub.add_parser("book", help="ordered pages of collages, exported as one PDF",
                       description="A page is a recipe, not a picture: the photos it uses and how "
                                   "to lay them out. Nothing is drawn until the book is exported, "
                                   "so pages can be reordered at any time. Every page has the "
                                   "book's shape, because a PDF gives its pages one size.")
    bk = s.add_subparsers(dest="book_cmd", required=True)
    bk.add_parser("list", help="all books")
    x = bk.add_parser("create"); x.add_argument("name")
    x.add_argument("--shape", default="a4-landscape",
                   choices=("3:2", "square", "a4-landscape", "a4-portrait", "spread", "16:9"))
    x.add_argument("--note")
    x = bk.add_parser("delete", help="remove the book only; photos and collages stay")
    x.add_argument("name")
    x = bk.add_parser("add", help="append a page")
    x.add_argument("name")
    x.add_argument("--collection", metavar="NAME", help="the page holds this collection")
    x.add_argument("--id", nargs="+", metavar="ID", help="or these photos, by id or GUI entry path")
    x.add_argument("--template", default="justified",
                   choices=("justified", "grid", "hero", "filmstrip", "spread", "scatter"))
    x.add_argument("--caption", help="a line of text under the page")
    x = bk.add_parser("remove", help="drop a page and renumber the rest")
    x.add_argument("name"); x.add_argument("--page", type=int, required=True)
    x = bk.add_parser("move", help="reorder a page")
    x.add_argument("name"); x.add_argument("--page", type=int, required=True)
    x.add_argument("--to", type=int, required=True)
    x = bk.add_parser("show", help="the pages in order"); x.add_argument("name")
    x = bk.add_parser("export", help="draw every page and bind them into a PDF")
    x.add_argument("name")
    x.add_argument("--out", metavar="FILE", help="write here instead of the dated folder")
    x.add_argument("--originals", action="store_true",
                   help="fetch full-resolution originals and export at 300 dpi")
    x.add_argument("--gap", type=int, default=8, metavar="N")
    x.add_argument("--background", default="white", metavar="COLOUR")
    x.add_argument("--keep-pages", action="store_true", help="leave the rendered pages on disk")
    s.set_defaults(fn=cmd_book)

    s = sub.add_parser("info", help="everything known about one or more assets")
    s.add_argument("id", nargs="+")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("show", help="fetch previews into the cache and print their paths",
                       description="Prints one `id<TAB>path` line per asset (or JSON). Look at the file to see "
                                   "the picture. Previews are small JPEGs; use `original` for the real file.")
    s.add_argument("id", nargs="+")
    s.add_argument("--size", choices=PREVIEW_SIZES, help="thumb (default, from config) or medium; an asset without "
                                                          "that rendition gets its original if under 4 MB, else the thumb "
                                                          "(the output says which)")
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


def main(argv: list[str] | None = None, adapter_factory: Callable[[App], Adapter] | None = None,
         models_factory: Callable[[App], Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = App(args.json, adapter_factory, models_factory)
    app.gpu_allowed = args.fn is cmd_index
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
    except Exception as err:  # noqa: BLE001 - the contract is one line, never a stack trace
        if os.environ.get("ICLOUD_PHOTOS_DEBUG"):
            raise
        _fail(app, "internal-error", f"{type(err).__name__}: {err} (set ICLOUD_PHOTOS_DEBUG=1 for the traceback; "
                                     "please report it at https://github.com/Sonstebo/icloud-photos-cli/issues)")
        return 1
    finally:
        app.close()


def _fail(app: App, code: str, message: str) -> None:
    if app.json:
        print(json.dumps({"error": code, "message": message}), file=sys.stderr)
    else:
        print(f"photos: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
