"""Bring the catalogue up to date with the library.

First run: list everything (newest first) and record it. Later
runs: ask iCloud's change feed what happened since the stored cursor and
refetch only those records. `--full` forces a full listing, which is also
the recovery path if the cursor ever stops working. Progress is written to
the catalogue as it goes, so `photos status` can show it and a killed run
loses nothing that was already committed.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .adapter import Adapter, AssetInfo, Change
from .catalog import Catalog, now

Progress = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:
    pass


def sync(catalog: Catalog, adapter: Adapter, *, full: bool = False, limit: int | None = None,
         albums: bool = False, progress: Progress = _noop) -> dict[str, Any]:
    started = now()
    stored_cursor = catalog.get_meta("cursor")
    result: dict[str, Any] = {
        "started": started, "mode": None, "listed": 0, "new": 0, "changed": 0,
        "same": 0, "missing": 0, "albums": 0, "cursor_before": stored_cursor,
    }
    cursor_now = adapter.sync_cursor()
    if not full and stored_cursor and stored_cursor == cursor_now:
        result["mode"] = "unchanged"
    elif not full and stored_cursor:
        result["mode"] = "changes"
        _apply_changes(catalog, adapter, stored_cursor, result, progress)
    else:
        result["mode"] = "full"
        _full_listing(catalog, adapter, cursor_now, result, limit, progress)
    if albums:
        result["albums"] = _sync_albums(catalog, adapter, progress)
    result["finished"] = now()
    result["cursor_after"] = catalog.get_meta("cursor")
    catalog.set_meta("last_sync", json.dumps(result))
    catalog.set_meta("sync_progress", None)
    return result


def _record(catalog: Catalog, info: AssetInfo, result: dict[str, Any]) -> None:
    status = catalog.upsert_asset(info)
    result[status] += 1
    result["listed"] += 1


def _full_listing(catalog: Catalog, adapter: Adapter, cursor_now: str | None,
                  result: dict[str, Any], limit: int | None, progress: Progress) -> None:
    seen: set[str] = set()
    for info in adapter.iter_assets():
        _record(catalog, info, result)
        seen.add(info.id)
        if result["listed"] % 100 == 0:
            catalog.set_meta("sync_progress", json.dumps(result | {"phase": "listing"}))
            progress(result)
        if limit and result["listed"] >= limit:
            break
    if not limit:
        # a complete listing means anything not seen this run has left the library
        gone = catalog.asset_ids() - seen
        stamp = now()
        result["missing"] = sum(catalog.mark_missing(asset_id, stamp) for asset_id in gone)
        # the cursor is only trustworthy after a complete listing
        catalog.set_meta("cursor", cursor_now)


def _apply_changes(catalog: Catalog, adapter: Adapter, since: str, result: dict[str, Any],
                   progress: Progress) -> None:
    changes, new_cursor = adapter.changes_since(since)
    result["events"] = len(changes)
    touched: dict[str, bool] = {}   # asset id -> deleted?
    for ch in changes:
        asset_id = _asset_id_for(catalog, ch)
        if asset_id is None:
            continue
        touched[asset_id] = touched.get(asset_id, False) or ch.deleted
    for n, (asset_id, deleted) in enumerate(touched.items(), 1):
        info = None if deleted else adapter.get_asset(asset_id)
        if info is None:
            if catalog.mark_missing(asset_id):
                result["missing"] += 1
        else:
            _record(catalog, info, result)
        if n % 50 == 0:
            catalog.set_meta("sync_progress", json.dumps(result | {"phase": "changes", "of": len(touched)}))
            progress(result)
    catalog.set_meta("cursor", new_cursor or since)


def _asset_id_for(catalog: Catalog, ch: Change) -> str | None:
    if ch.record_type == "CPLMaster":
        return catalog.asset_id_for_master(ch.record_name)
    if ch.record_type in (None, "CPLAsset"):
        # tombstones carry no type: it is an asset if we know it, a master if
        # we know it that way, otherwise something we never tracked (an album)
        if ch.record_type is None and catalog.get_asset(ch.record_name) is None:
            return catalog.asset_id_for_master(ch.record_name)
        return ch.record_name
    return None


def _sync_albums(catalog: Catalog, adapter: Adapter, progress: Progress) -> int:
    albums = adapter.albums()
    catalog.replace_albums((a.id, a.name, a.fullname) for a in albums)
    for n, album in enumerate(albums, 1):
        catalog.replace_album_members(album.id, adapter.iter_album_asset_ids(album.id))
        catalog.set_meta("sync_progress", json.dumps({"phase": "albums", "done": n, "of": len(albums)}))
        progress({"phase": "albums", "done": n, "of": len(albums)})
    return len(albums)
