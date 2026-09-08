"""The cloud side, behind one small interface.

`Adapter` is what the rest of the package talks to. `ICloudAdapter` is the
real one, a thin layer over pyicloud; tests use a fake with the same shape.
Nothing here writes to iCloud: listing, the change feed and downloads only.
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
class Change:
    record_name: str
    record_type: str | None      # CPLAsset, CPLMaster, ... or None for tombstones
    deleted: bool


class Adapter(Protocol):
    def auth_status(self) -> dict[str, Any]: ...
    def sync_cursor(self) -> str | None: ...
    def iter_assets(self) -> Iterator[AssetInfo]:
        """Every asset in the library, newest capture date first."""
        ...
    def get_asset(self, asset_id: str) -> AssetInfo | None: ...
    def changes_since(self, cursor: str) -> tuple[list[Change], str | None]: ...
    def download(self, asset_id: str, version: str) -> bytes | None: ...
    def albums(self) -> list[AlbumInfo]: ...
    def iter_album_asset_ids(self, album_id: str) -> Iterator[str]: ...


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

    def sync_cursor(self) -> str | None:
        return self.photos().sync_cursor()

    def iter_assets(self) -> Iterator[AssetInfo]:
        for photo in self.photos().all:
            yield self._info(photo)

    def get_asset(self, asset_id: str) -> AssetInfo | None:
        photo = self.photos().all.get(asset_id)
        return self._info(photo) if photo is not None else None

    def changes_since(self, cursor: str) -> tuple[list[Change], str | None]:
        photos = self.photos()
        changes = [
            Change(ev.record_name, ev.record_type, bool(ev.deleted))
            for ev in photos.iter_changes(since=cursor)
        ]
        return changes, photos.sync_cursor()

    def download(self, asset_id: str, version: str) -> bytes | None:
        photo = self.photos().all.get(asset_id)
        if photo is None:
            return None
        return photo.download(version)

    def albums(self) -> list[AlbumInfo]:
        return [AlbumInfo(a.id, a.name, a.fullname) for a in self.photos().albums]

    def iter_album_asset_ids(self, album_id: str) -> Iterator[str]:
        album = self.photos().albums.get(album_id)
        if album is None:
            return
        for photo in album.photos:
            yield photo.id

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
            favorite=bool(photo.favorite),
            caption=decode_encrypted_text(asset_record, "captionEnc") or None,
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
            hidden=bool(record_field_value(asset_record, "isHidden") or 0),
            versions=versions,
        )
