# icloud-photos

An iCloud photo library from the command line, built so an AI assistant can
use it through a shell. Originals stay in iCloud. Metadata for the whole
library lives in a local SQLite catalogue that answers every query without
the network. Images are fetched on demand into a bounded cache, and every
command that fetches one prints the file's path so the assistant can look at
it. Every command has `--json`, bounded output, and one-line errors.

## Built on

[pyicloud](https://github.com/timlaing/pyicloud) 2.7 does the talking to
Apple: login with the two-factor dance and the system keyring, the persisted
session, listing, the CloudKit change feed with its sync cursor, and downloads
by rendition (thumb, medium, original, the video half of a Live Photo). This
project adds only what pyicloud does not have and an assistant needs: the
durable catalogue, the cache with a budget, pinning and eviction, named
collections, and the CLI.

## Install

```sh
cd ~/Work/icloud-photo-workspace
python3 -m venv .venv
.venv/bin/pip install -e .
ln -s "$PWD/.venv/bin/photos" ~/.local/bin/photos   # optional
```

Python 3.11 or newer. Nothing outside the venv.

## First run

```sh
photos login                     # password + two-factor code; offer to store the password in the keyring
photos sync --albums --background
photos status                    # repeat until sync_running is false
```

The first sync lists the whole library, metadata only: no image is
downloaded. Later syncs read the change feed and touch only what changed; a
sync when nothing changed does no listing at all. `photos sync --full`
relists everything and is the recovery path if the cursor ever misbehaves.

To keep the catalogue fresh without thinking about it:

```sh
mkdir -p ~/.config/systemd/user
cp systemd/icloud-photos-sync.* ~/.config/systemd/user/
systemctl --user enable --now icloud-photos-sync.timer
```

## Commands

| Command | What it does | Network |
| --- | --- | --- |
| `login [--username ID] [--logout]` | sign in through pyicloud's CLI; needs a terminal | yes |
| `status [--offline]` | auth state, catalogue counts, cache usage, sync progress | unless `--offline` |
| `sync [--full] [--albums] [--limit N] [--background]` | update the catalogue | yes |
| `albums` | albums in the catalogue with counts | no |
| `search [TEXT] [--since D] [--until D] [--kind image\|movie] [--favorite] [--album A] [--collection C] [--located] [--live] [--limit N] [--cursor C]` | paged search, newest first | no |
| `info ID...` | every field, cached renditions, albums, collections | no |
| `show ID... [--size thumb\|medium]` | fetch previews, print `id<TAB>path` | if not cached |
| `original ID... [--pin] [--version V]` | fetch full files, print paths | if not cached |
| `cache status\|verify\|pin\|unpin\|evict` | manage local copies; iCloud is never touched | no |
| `collection list\|create\|delete\|add\|remove\|show` | ordered sets of ids for a project | no |
| `config show\|set KEY VALUE` | `cache_budget_mb`, `username`, `preview_size` | no |

Dates accept `2019`, `2019-07`, `2019-07-20` or full ISO 8601; `--until`
includes the whole of the period given. Exit codes: 1 error, 2 needs a
terminal, 3 not logged in, 4 cache full. With `--json`, errors are
`{"error": code, "message": text}` on stderr.

## An assistant's session

```sh
photos --json search --since 2019-07 --until 2019-08 --located --limit 20
photos --json show A1B2… C3D4…            # look at the files it names
photos collection create book --note "Norway 2019"
photos collection add book A1B2… C3D4…
photos --json original A1B2… --pin          # the full HEIC for the layout
photos cache evict --all                    # drop unpinned local copies; the catalogue and collections stay
```

## Where things live

| | Path |
| --- | --- |
| catalogue | `~/.local/share/icloud-photos/catalog.db` |
| cache | `~/.cache/icloud-photos/{thumb,medium,original,...}/` |
| session (cookies, tokens) | `~/.local/state/icloud-photos/session/` (mode 700) |
| config | `~/.config/icloud-photos/config.toml` |
| background sync log and lock | `~/.local/state/icloud-photos/` |

Set `ICLOUD_PHOTOS_HOME=/some/dir` to put all four under one directory (the
tests do this). Nothing under any of them belongs in git.

## What eviction can and cannot do

Eviction deletes files under the cache directory and their index rows,
nothing else. It cannot delete anything in iCloud, a collection, or the
catalogue. Pinned files are skipped unless `--include-pinned` is given. A
fetch that would exceed `cache_budget_mb` first evicts the least recently
used unpinned files and, if that is not enough, refuses with exit 4 and says
what to do.

## Not yet

Semantic search, faces, objects, and shared libraries. The catalogue is
designed to hold model output alongside the asset rows (with the model
version and the rendition it was computed from), and previews can be evicted
after analysis without losing what was learned. The candidates to evaluate
first are existing open-source stacks, not new code: see `CLAUDE.md`.

## Tests

```sh
.venv/bin/python -m unittest discover tests
```

The suite runs against a fake iCloud with the adapter's shape: first sync,
change feed with an edit and a deletion, master-record changes, paging,
every search filter, preview and original fetches, the budget with pinning
and eviction, a file removed behind the tool's back, Live Photo halves,
collections, error shapes, and the detached sync.
