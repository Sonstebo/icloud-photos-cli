"""Where things live on disk, and the small config file.

Everything is under the XDG directories unless ICLOUD_PHOTOS_HOME is set, in
which case all four (config, data, state, cache) sit under that one
directory. Tests use the override; so can a second library on one machine.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

APP = "icloud-photos"

DEFAULTS = {
    "cache_budget_mb": 2048,   # previews + originals together
    "username": "",            # Apple ID; empty means "the one session in session_dir"
    "preview_size": "thumb",   # what `show` fetches by default: thumb or medium
    "face_threshold": 0.5,     # cosine similarity for naming a face from a person's seeds
    "compute": "auto",         # where CLIP and the face models run: auto (GPU when a working one is found), cpu, gpu
    "lap_open_version": "original",   # what opening a photo in the GUI fetches: original or medium
}


def _xdg(var: str, fallback: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / fallback)


@dataclass
class Paths:
    config_dir: Path
    data_dir: Path
    state_dir: Path
    cache_dir: Path

    @classmethod
    def discover(cls) -> "Paths":
        home = os.environ.get("ICLOUD_PHOTOS_HOME")
        if home:
            root = Path(home)
            return cls(root / "config", root / "data", root / "state", root / "cache")
        return cls(
            _xdg("XDG_CONFIG_HOME", ".config") / APP,
            _xdg("XDG_DATA_HOME", ".local/share") / APP,
            _xdg("XDG_STATE_HOME", ".local/state") / APP,
            _xdg("XDG_CACHE_HOME", ".cache") / APP,
        )

    def ensure(self) -> "Paths":
        for d in (self.config_dir, self.data_dir, self.state_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        # the session holds Apple cookies and tokens: keep it to this user
        self.session_dir.mkdir(mode=0o700, exist_ok=True)
        return self

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def catalog_db(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "session"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def edit_log(self) -> Path:
        return self.state_dir / "edit.log"

    @property
    def edit_lock(self) -> Path:
        return self.state_dir / "edit.lock"

    @property
    def index_lock(self) -> Path:
        return self.state_dir / "index.lock"

    @property
    def index_log(self) -> Path:
        return self.state_dir / "index.log"

    @property
    def sync_lock(self) -> Path:
        return self.state_dir / "sync.lock"

    @property
    def sync_log(self) -> Path:
        return self.state_dir / "sync.log"


@dataclass
class Config:
    values: dict = field(default_factory=lambda: dict(DEFAULTS))

    @classmethod
    def load(cls, path: Path) -> "Config":
        cfg = cls()
        if path.exists():
            with path.open("rb") as fh:
                cfg.values.update(tomllib.load(fh))
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for key, value in sorted(self.values.items()):
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f'{key} = "{value}"')
        path.write_text("\n".join(lines) + "\n")

    def set(self, key: str, raw: str) -> None:
        if key not in DEFAULTS:
            raise KeyError(key)
        current = DEFAULTS[key]
        if isinstance(current, bool):
            self.values[key] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            self.values[key] = int(raw)
        elif isinstance(current, float):
            self.values[key] = float(raw)
        else:
            self.values[key] = raw

    @property
    def cache_budget_bytes(self) -> int:
        return int(self.values["cache_budget_mb"]) * 1024 * 1024
