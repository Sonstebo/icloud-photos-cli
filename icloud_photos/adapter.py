"""The cloud side, behind one small interface.

`Adapter` is what the rest of the package talks to. `ICloudAdapter` is the
real one, a thin layer over pyicloud; tests use a fake with the same shape.
Nothing here writes to iCloud: listing, the change feed and downloads only.
Take care with pyicloud's PhotoAsset: favorite(), unfavorite(), set_favorite()
and delete() are writes, not getters. Read fields with record_field_value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol


class NotLoggedIn(Exception):
    """No usable iCloud session; `photos login` is the fix."""


class CloudError(Exception):
    """iCloud answered with something we cannot use."""


@dataclass
class AssetInfo:
    """One photo or video as the catalogue records it.

    `versions` maps a size name (thumb, medium, original, ...) to what is
    known about that rendition: bytes, width, height, type, filename. URLs
    are deliberately not kept: they are signed and short-lived.
    """

    id: str
    master_id: str
    filename: str
    kind: str                      # image | movie
    live: bool
    taken: datetime | None         # capture time, UTC
    added: datetime | None         # when it entered the library, UTC
    width: int | None
    height: int | None
    bytes: int | None              # original size
    favorite: bool
    caption: str | None
    latitude: float | None
    longitude: float | None
    hidden: bool
    versions: dict[str, dict[str, Any]] = field(default_factory=dict)
    deleted: bool = False           # in Recently Deleted

    def fingerprint(self) -> str:
        """A cheap 'has anything I record changed' value."""
        d = asdict(self)
        d["taken"] = self.taken.isoformat() if self.taken else None
        d["added"] = self.added.isoformat() if self.added else None
        return repr(sorted(d.items()))


@dataclass
class AlbumInfo:
    id: str
    name: str
    fullname: str


@dataclass
class RawRecord:
    """One CloudKit record as the zone hands it over: type, fields and all.

    `fields` is the record's JSON dump (pyicloud's CKRecord.model_dump) so the
    catalogue can keep it and the adapter can rebuild a PhotoAsset from it.
    A tombstone has deleted=True and no fields.
    """

    name: str
    type: str | None
    deleted: bool
    modified: datetime | None
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def master_ref(self) -> str | None:
        ref = (self.fields.get("fields") or {}).get("masterRef", {}).get("value") or {}
        return ref.get("recordName") if isinstance(ref, dict) else None

    def value(self, key: str) -> Any:
        f = (self.fields.get("fields") or {}).get(key)
        return f.get("value") if isinstance(f, dict) else None


class Adapter(Protocol):
    def auth_status(self) -> dict[str, Any]: ...
    def iter_zone(self, since: str | None) -> Iterator[tuple[list[RawRecord], str | None]]:
        """Pages of zone changes after `since` (None = from the beginning), each with the token after it."""
        ...
    def asset_from_records(self, asset: RawRecord, master: RawRecord) -> AssetInfo: ...
    def download(self, asset_id: str, version: str, master_id: str | None = None) -> bytes | None: ...
    def download_many(self, items: list[tuple[str, str]], version: str, threads: int = 4) -> Iterator[tuple[str, bytes | None]]:
        """(asset_id, master_id) pairs -> (asset_id, bytes); one lookup per batch, downloads in parallel."""
        ...
    def download_face_crops(self, crop_ids: list[str], threads: int = 4) -> Iterator[tuple[str, bytes | None]]: ...
    def albums(self) -> list[AlbumInfo]: ...


# --- the real one -----------------------------------------------------------

def _utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


class ICloudAdapter:
    """pyicloud, with the session persisted under our state directory.

    Login itself is delegated to pyicloud's own CLI (`icloud auth login`),
    which handles the keyring, two-factor prompts and session trust. This
    class only ever reopens the session that login left behind.
    """

    def __init__(self, session_dir: Path, username: str = "") -> None:
        self.session_dir = Path(session_dir)
        self.username = username
        self._api: Any = None
        self._photos: Any = None

    # pyicloud is imported lazily so `photos --help` and offline commands
    # start fast and tests never touch it
    def _state(self) -> Any:
        from pyicloud.cli.context import CLIState, LogLevel, OutputFormat

        return CLIState(
            username=self.username or None,
            password=None,
            china_mainland=None,
            interactive=False,
            accept_terms=False,
            with_family=False,
            session_dir=str(self.session_dir),
            http_proxy=None,
            https_proxy=None,
            no_verify_ssl=False,
            log_level=LogLevel.ERROR,
            output_format=OutputFormat.JSON,
        )

    def api(self) -> Any:
        if self._api is None:
            from pyicloud.cli.context import CLIAbort

            try:
                self._api = self._state().get_api()
            except CLIAbort as err:
                # pyicloud's message ends with its own login instructions; ours differ
                raise NotLoggedIn(str(err).split(" To log in")[0].rstrip(".")) from err
        return self._api

    def photos(self) -> Any:
        if self._photos is None:
            self._photos = self.api().photos
        return self._photos

    def auth_status(self) -> dict[str, Any]:
        try:
            api = self.api()
        except NotLoggedIn as err:
            return {"authenticated": False, "reason": str(err)}
        status = dict(api.get_auth_status())
        status["username"] = getattr(api, "apple_id", None) or getattr(api, "_apple_id", None)
        return status

    def _library(self) -> Any:
        return self.photos().libraries["root"]

    def iter_zone(self, since: str | None) -> Iterator[tuple[list[RawRecord], str | None]]:
        from pyicloud.common.cloudkit import CKRecord, CKZoneChangesZoneReq, CKZoneID

        library = self._library()
        zone_req = CKZoneChangesZoneReq(zoneID=CKZoneID(**library.zone_id), syncToken=since, reverse=False)
        for zone in library._client.iter_changes(zone_req=zone_req):
            page = []
            for rec in zone.records:
                if isinstance(rec, CKRecord):
                    page.append(RawRecord(rec.recordName, rec.recordType, bool(rec.deleted),
                                          rec.modified.timestamp if rec.modified else None,
                                          rec.model_dump(mode="json")))
                else:
                    page.append(RawRecord(rec.recordName, None, True, None))
            yield page, zone.syncToken

    def asset_from_records(self, asset: RawRecord, master: RawRecord) -> AssetInfo:
        from pyicloud.common.cloudkit import CKRecord

        library = self._library()
        photo = library.asset_type(self.photos(), CKRecord.model_validate(master.fields),
                                   CKRecord.model_validate(asset.fields), library=library)
        return self._info(photo)

    def download(self, asset_id: str, version: str, master_id: str | None = None) -> bytes | None:
        """Fetch fresh records by name (download URLs in stored records expire) and download.

        Not PhotoAlbum.get(): when its index lookup misses, pyicloud walks the
        whole library looking for the id, which is minutes per photo here.
        """
        from pyicloud.common.cloudkit import CKRecord, CKZoneIDReq

        library = self._library()
        names = [asset_id] + ([master_id] if master_id else [])
        found = {r.recordName: r for r in library._client.lookup(record_names=names, zone_id=CKZoneIDReq(**library.zone_id)).records
                 if isinstance(r, CKRecord)}
        asset = found.get(asset_id)
        if asset is None:
            return None
        if master_id is None:
            ref = RawRecord(asset_id, "CPLAsset", False, None, asset.model_dump(mode="json")).master_ref
            if not ref:
                return None
            more = library._client.lookup(record_names=[ref], zone_id=CKZoneIDReq(**library.zone_id)).records
            found.update({r.recordName: r for r in more if isinstance(r, CKRecord)})
            master_id = ref
        master = found.get(master_id)
        if master is None:
            return None
        photo = library.asset_type(self.photos(), master, asset, library=library)
        return photo.download(version)

    def download_many(self, items: list[tuple[str, str]], version: str, threads: int = 4) -> Iterator[tuple[str, bytes | None]]:
        from concurrent.futures import ThreadPoolExecutor

        from pyicloud.common.cloudkit import CKRecord, CKZoneIDReq

        library = self._library()
        zone = CKZoneIDReq(**library.zone_id)
        client = library._client

        def fetch(photo: Any) -> bytes | None:
            url = photo.download_url(version)
            return client.download_asset_bytes(url) if url else None

        for i in range(0, len(items), 50):
            batch = items[i:i + 50]
            names = [n for pair in batch for n in pair]
            found = {r.recordName: r for r in client.lookup(record_names=names, zone_id=zone).records
                     if isinstance(r, CKRecord)}
            photos = []
            for asset_id, master_id in batch:
                asset, master = found.get(asset_id), found.get(master_id)
                photos.append((asset_id, library.asset_type(self.photos(), master, asset, library=library)
                               if asset is not None and master is not None else None))
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = [(asset_id, pool.submit(fetch, p) if p is not None else None) for asset_id, p in photos]
                for asset_id, fut in futures:
                    try:
                        yield asset_id, (fut.result() if fut is not None else None)
                    except Exception:  # noqa: BLE001 - one bad download must not end the run
                        yield asset_id, None

    def download_face_crops(self, crop_ids: list[str], threads: int = 4) -> Iterator[tuple[str, bytes | None]]:
        from concurrent.futures import ThreadPoolExecutor

        from pyicloud.common.cloudkit import CKRecord, CKZoneIDReq

        library = self._library()
        zone = CKZoneIDReq(**library.zone_id)
        client = library._client

        def url_of(rec: Any) -> str | None:
            res = (rec.model_dump(mode="json").get("fields") or {}).get("resFaceCropRes", {}).get("value") or {}
            return res.get("downloadURL")

        for i in range(0, len(crop_ids), 50):
            batch = crop_ids[i:i + 50]
            found = {r.recordName: r for r in client.lookup(record_names=batch, zone_id=zone).records if isinstance(r, CKRecord)}
            urls = {cid: url_of(found[cid]) for cid in batch if cid in found}
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = {cid: pool.submit(client.download_asset_bytes, url) for cid, url in urls.items() if url}
                for cid in batch:
                    fut = futures.get(cid)
                    try:
                        yield cid, (fut.result() if fut is not None else None)
                    except Exception:  # noqa: BLE001
                        yield cid, None

    def albums(self) -> list[AlbumInfo]:
        return [AlbumInfo(a.id, a.name, a.fullname) for a in self.photos().albums]

    @staticmethod
    def _info(photo: Any) -> AssetInfo:
        from pyicloud.services.photos_cloudkit.mappers import decode_encrypted_text, record_field_value
        from pyicloud.services.photos_cloudkit.materialize import _extract_location

        asset_record = photo.asset_record
        width, height = photo.dimensions
        location = _extract_location(asset_record)
        versions = {}
        for key, res in photo.versions.items():
            versions[key] = {
                "bytes": res.get("size"),
                "width": res.get("width"),
                "height": res.get("height"),
                "type": res.get("type"),
                "filename": res.get("filename"),
            }
        return AssetInfo(
            id=photo.id,
            master_id=photo.master_id,
            filename=photo.filename,
            kind=photo.item_type,
            live=bool(photo.is_live_photo),
            taken=_utc(photo.asset_date),
            added=_utc(photo.added_date),
            width=width,
            height=height,
            bytes=photo.size,
            # read the field: PhotoAsset.favorite() is a *setter* in pyicloud 2.7
            # (it marks the asset as a favourite in iCloud), as are unfavorite(),
            # set_favorite(), delete() and PhotoAlbum.add_photo(). Never call them.
            favorite=bool(record_field_value(asset_record, "isFavorite") or 0),
            caption=decode_encrypted_text(asset_record, "captionEnc") or None,
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
            hidden=bool(record_field_value(asset_record, "isHidden") or 0),
            deleted=bool(record_field_value(asset_record, "isDeleted") or 0),
            versions=versions,
        )
