"""The bounded on-disk cache of previews and originals.

Files live under the cache directory by version (thumb/, medium/,
original/). The catalogue's `cache` table is the index: bytes, last use,
pinned. A write that would exceed the budget first evicts the least
recently used unpinned files; if that is not enough, it refuses. Evicting
removes only the local copy. Nothing in iCloud is ever touched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .catalog import Catalog


class BudgetExceeded(Exception):
    """Storing this file would exceed the cache budget even after eviction."""


def safe_name(asset_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", asset_id)


class Cache:
    def __init__(self, root: Path, catalog: Catalog, budget_bytes: int) -> None:
        self.root = Path(root)
        self.catalog = catalog
        self.budget = budget_bytes

    def usage(self) -> dict[str, Any]:
        u = self.catalog.cache_usage()
        u["budget_bytes"] = self.budget
        u["free_bytes"] = max(self.budget - u["bytes"], 0)
        return u

    def get(self, asset_id: str, version: str) -> Path | None:
        """The cached file if it is on disk; repairs the index if it is gone."""
        row = self.catalog.cache_get(asset_id, version)
        if row is None:
            return None
        path = Path(row["path"])
        if not path.exists():
            self.catalog.cache_delete(asset_id, version)
            return None
        self.catalog.cache_touch(asset_id, version)
        return path

    def peek(self, asset_id: str, version: str) -> Path | None:
        """The cached file if it is on disk, without touching its use time or the
        catalogue: safe while an index worker holds the write lock."""
        row = self.catalog.cache_get(asset_id, version)
        if row is None:
            return None
        path = Path(row["path"])
        return path if path.exists() else None

    def put(self, asset_id: str, version: str, data: bytes, ext: str, pinned: bool = False) -> Path:
        size = len(data)
        self.make_room(size, keep=(asset_id, version))
        directory = self.root / version
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe_name(asset_id)}{ext}"
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        self.catalog.cache_put(asset_id, version, str(path), size, pinned)
        return path

    def make_room(self, needed: int, keep: tuple[str, str] | None = None) -> list[dict[str, Any]]:
        """Evict least-recently-used unpinned entries until `needed` bytes fit."""
        evicted: list[dict[str, Any]] = []
        used = self.catalog.cache_usage()["bytes"]
        if used + needed <= self.budget:
            return evicted
        for row in self.catalog.cache_rows(unpinned_only=True, oldest_first=True):
            if used + needed <= self.budget:
                break
            if keep and (row["asset_id"], row["version"]) == keep:
                continue
            self._remove(row)
            used -= row["bytes"]
            evicted.append(row)
        if used + needed > self.budget:
            raise BudgetExceeded(
                f"need {needed} bytes but only {max(self.budget - used, 0)} free within the "
                f"{self.budget} byte budget after evicting unpinned files; unpin something, "
                f"or raise cache_budget_mb with `photos config set cache_budget_mb N`")
        return evicted

    def evict(self, asset_id: str | None = None, version: str | None = None,
              include_pinned: bool = False) -> list[dict[str, Any]]:
        gone = []
        for row in self.catalog.cache_rows(asset_id=asset_id, unpinned_only=not include_pinned):
            if version and row["version"] != version:
                continue
            self._remove(row)
            gone.append(row)
        return gone

    def _remove(self, row: dict[str, Any]) -> None:
        path = Path(row["path"])
        # only ever delete inside our own cache directory
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            pass
        else:
            path.unlink(missing_ok=True)
        self.catalog.cache_delete(row["asset_id"], row["version"])

    def verify(self) -> dict[str, int]:
        """Drop index rows whose files are gone; report what was found."""
        dropped = 0
        for row in self.catalog.cache_rows():
            if not Path(row["path"]).exists():
                self.catalog.cache_delete(row["asset_id"], row["version"])
                dropped += 1
        return {"dropped": dropped}
