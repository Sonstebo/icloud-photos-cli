"""Bring the catalogue up to date with the library, from iCloud's change feed.

The photo zone in CloudKit hands out every record it holds, in modification
order, in pages that each come with a sync token. The first sync walks the
whole zone from the beginning; later syncs ask for what changed after the
stored token; an interrupted sync resumes from the token of the last page
it committed. One walk gives everything: assets and their masters, album
membership (CPLContainerRelation), people (CPLPerson), face crops
(CPLFaceCrop), and tombstones for whatever was deleted.
"""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .adapter import Adapter, RawRecord
from .catalog import Catalog, now

Progress = Callable[[dict[str, Any]], None]
KEPT_TYPES = {"CPLAsset", "CPLMaster", "CPLContainerRelation", "CPLPerson", "CPLFaceCrop"}


# A page's worth of records is one write transaction, and pages come back to
# back. Without a pause between them an interactive fetch waits for the whole
# sync: opening a photo has to be able to get in edgeways.
YIELD_BETWEEN_PAGES_S = 0.01


def _noop(_: dict[str, Any]) -> None:
    pass


def _text(rec: RawRecord, key: str) -> str | None:
    """Apple's *Enc fields are base64 text, not encryption."""
    v = rec.value(key)
    if not isinstance(v, str):
        return None
    try:
        return base64.b64decode(v).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _tokens(catalog: Catalog) -> dict[str, str | None]:
    """Where each zone was left off. The old single `cursor` is the primary zone's."""
    raw = catalog.get_meta("cursors")
    if raw:
        try:
            return dict(json.loads(raw))
        except ValueError:
            pass
    old = catalog.get_meta("cursor")
    return {"": old} if old else {}


def sync(catalog: Catalog, adapter: Adapter, *, full: bool = False, shared: bool = True,
         progress: Progress = _noop) -> dict[str, Any]:
    tokens = {} if full else _tokens(catalog)
    try:
        zones = adapter.zones()
    except Exception:  # noqa: BLE001
        zones = []
    if not zones:                                    # an adapter that predates zones
        zones = [None]
    if not shared:
        zones = [z for z in zones if z is None or not getattr(z, "shared", False)]

    primary_token = tokens.get(zones[0].name if zones[0] is not None else "") or tokens.get("")
    result: dict[str, Any] = {
        "started": now(), "mode": "full" if primary_token is None else "changes",
        "pages": 0, "records": 0,
        "new": 0, "changed": 0, "same": 0, "missing": 0, "relations": 0, "people": 0, "face_crops": 0,
        "tombstones": 0, "resumed": bool(primary_token and catalog.get_meta("sync_progress")),
        "zones": {},
    }
    for i, zone in enumerate(zones):
        name = zone.name if zone is not None else ""
        token = tokens.get(name) if not full else None
        if token is None and i == 0 and not full:
            # Catalogues written before zones kept one unnamed cursor, and it is
            # the user's own library's. Without this the first sync after the
            # upgrade walks all of it again.
            token = tokens.get("")
        before = dict(new=result["new"], records=result["records"])
        try:
            _walk(catalog, adapter, zone, token, tokens, result, progress, primary=(i == 0))
        except Exception as err:  # noqa: BLE001
            # The page loop opens a transaction per page; one that died mid-page
            # has to be closed or nothing after it can start its own.
            try:
                catalog.db.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            if i == 0:
                raise                                # the user's own library is not optional
            # A shared library that will not answer must not cost the user their own.
            result["zones"][name] = {"error": str(err)[:200]}
            continue
        result["zones"][name or "PrimarySync"] = {
            "records": result["records"] - before["records"], "new": result["new"] - before["new"],
            "shared": bool(zone is not None and getattr(zone, "shared", False)),
        }
    albums = adapter.albums()
    catalog.replace_albums((a.id, a.name, a.fullname) for a in albums)
    result["albums"] = len(albums)
    result["finished"] = now()
    catalog.set_meta("last_sync", json.dumps(result))
    catalog.set_meta("sync_progress", None)
    return result


def _walk(catalog: Catalog, adapter: Adapter, zone: Any, token: str | None,
          tokens: dict[str, str | None], result: dict[str, Any], progress: Progress,
          primary: bool = True) -> None:
    name = zone.name if zone is not None else ""
    for page, next_token in adapter.iter_zone(token, zone) if zone is not None else adapter.iter_zone(token):
        catalog.db.execute("BEGIN")
        for rec in page:
            _apply(catalog, adapter, rec, result, name or None)
        result["pages"] += 1
        result["records"] += len(page)
        tokens[name] = next_token
        catalog.set_meta("cursors", json.dumps(tokens))
        if primary:
            catalog.set_meta("cursor", next_token)   # the old key, for anything still reading it
        catalog.set_meta("sync_progress", json.dumps(result))
        catalog.db.execute("COMMIT")
        if YIELD_BETWEEN_PAGES_S:
            time.sleep(YIELD_BETWEEN_PAGES_S)
        progress(result)


def _apply(catalog: Catalog, adapter: Adapter, rec: RawRecord, result: dict[str, Any],
           zone: str | None = None) -> None:
    if rec.deleted:
        result["tombstones"] += 1
        known = catalog.get_record(rec.name)
        rtype = rec.type or (known["type"] if known else None)
        if rtype is None:
            # never seen this record; only an asset id could matter, and it costs nothing to try
            if catalog.mark_missing(rec.name):
                result["missing"] += 1
            return
        catalog.put_record(rec.name, rtype, True, _iso(rec.modified), None, None)
        if rtype == "CPLAsset" and catalog.mark_missing(rec.name):
            result["missing"] += 1
        elif rtype == "CPLContainerRelation" and known and known["json"]:
            r = RawRecord(rec.name, rtype, True, None, known["json"])
            catalog.set_relation(r.value("containerId"), r.value("itemId"), True)
        elif rtype == "CPLPerson":
            catalog.mark_deleted("people", rec.name)
        elif rtype == "CPLFaceCrop":
            catalog.mark_deleted("face_crops", rec.name)
        return
    if rec.type not in KEPT_TYPES:
        return
    catalog.put_record(rec.name, rec.type, False, _iso(rec.modified), rec.master_ref, rec.fields)
    if rec.type == "CPLAsset":
        master = catalog.get_record(rec.master_ref) if rec.master_ref else None
        if master and master["json"] and not master["deleted"]:
            _asset(catalog, adapter, rec, RawRecord(master["name"], "CPLMaster", False, None, master["json"]), result, zone)
    elif rec.type == "CPLMaster":
        for asset in catalog.assets_of_master(rec.name):
            _asset(catalog, adapter, RawRecord(asset["name"], "CPLAsset", False, None, asset["json"]), rec, result, zone)
    elif rec.type == "CPLContainerRelation":
        catalog.set_relation(rec.value("containerId"), rec.value("itemId"), False)
        result["relations"] += 1
    elif rec.type == "CPLPerson":
        catalog.put_person(rec.name, _text(rec, "personFullNameEnc"), _text(rec, "displayName"),
                           bool(rec.value("verifiedType")), rec.value("personType"), _iso(rec.modified))
        result["people"] += 1
    elif rec.type == "CPLFaceCrop":
        ref = rec.value("personRef") or {}
        catalog.put_face_crop(rec.name, ref.get("recordName") if isinstance(ref, dict) else None,
                              rec.value("type"), rec.value("resFaceCropFileSize"), _iso(rec.modified))
        result["face_crops"] += 1


def _asset(catalog: Catalog, adapter: Adapter, asset: RawRecord, master: RawRecord,
           result: dict[str, Any], zone: str | None = None) -> None:
    info = adapter.asset_from_records(asset, master)
    result[catalog.upsert_asset(info, zone=zone)] += 1
