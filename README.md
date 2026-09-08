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

## GPU

CLIP and the face models run through onnxruntime. With
[onnxruntime-ggml](https://github.com/thewh1teagle/onnxruntime-ggml) installed
in the same venv (`pip install onnxruntime-ggml`) they run on the GPU instead, through Vulkan or Metal,
at a fraction of the CPU's energy: on an Apple M1 under Asahi Linux the face
pass draws about 12 W instead of 27 W for the same work, and is slightly
faster. `config set compute auto|cpu|gpu` chooses; `auto` (the default) uses
the GPU when the provider imports and a small convolution on it matches the
CPU, and says why in `photos status` when it does not. Only `index` uses the GPU;
queries embed their text on the CPU so they never compete with a running
index worker. Nothing else changes:
embeddings from either path compare at cosine 1.0000.

Until the provider's next release, the published Linux arm64 wheel crashes on
Apple Silicon and the vision models need pull requests
[#4](https://github.com/thewh1teagle/onnxruntime-ggml/pull/4) and
[#6](https://github.com/thewh1teagle/onnxruntime-ggml/pull/6); build from
[this fork](https://github.com/Sonstebo/onnxruntime-ggml) (branch
`aarch64-conv2d`, `docs/BUILDING.md`) and `pip install -e python/`. Without a
working provider everything runs on the CPU.

## Browsing with lap

[lap](https://github.com/julyx10/lap) is an open-source desktop photo manager
(Tauri, local AI, map, faces, editor). `photos lap-export` writes the
catalogue into a lap library: one album of symlinks under the cache, rows with
dates, GPS, favourites and captions, our thumbnails, our CLIP vectors (lap uses
the same ViT-B/32 weights, so its semantic search runs on them), people, faces
and collections. Run lap once so it creates its library, export, restart lap.
Photos whose preview is not cached show their thumbnail but cannot open until
fetched.

## Install

```sh
git clone https://github.com/Sonstebo/icloud-photos-cli
cd icloud-photos-cli
python3 -m venv .venv
.venv/bin/pip install .
ln -s "$PWD/.venv/bin/photos" ~/.local/bin/photos
```

Python 3.11 or newer, Linux or macOS. The symlink puts `photos` on your PATH
and is what the systemd timer below runs. Nothing is installed outside the
venv; the catalogue, cache and state live under the XDG directories
(`photos status` prints the paths). Tests: `python -m unittest discover -s tests`.

## First run

```sh
photos login                     # password + two-factor code; offer to store the password in the keyring
photos sync --background
photos status                    # repeat until sync_running is false
```

`--background` starts the job as a transient systemd user unit
(`icloud-photos-sync-*`, `icloud-photos-index-*`), so it outlives the terminal
and is killed on its own if memory runs out; the log file is where to look, and
`ICLOUD_PHOTOS_WORKER=plain` falls back to a detached process.

The first sync lists the whole library, metadata only: no image is
downloaded. Later syncs read the change feed and touch only what changed; a
sync when nothing changed does no listing at all. `photos sync --full`
relists everything and is the recovery path if the cursor ever misbehaves.

To keep the catalogue fresh without thinking about it:

```sh
mkdir -p ~/.config/systemd/user
cp systemd/icloud-photos-sync.* ~/.config/systemd/user/
systemctl --user enable --now icloud-photos-sync.timer   # hourly, incremental
```

## Commands

| Command | What it does | Network |
| --- | --- | --- |
| `login [--username ID] [--logout]` | sign in through pyicloud's CLI; needs a terminal | yes |
| `status [--offline]` | auth state, catalogue counts, cache usage, sync progress | unless `--offline` |
| `sync [--full] [--background]` | update the catalogue: assets, albums, people, face crops | yes |
| `albums` | albums in the catalogue with counts | no |
| `search [TEXT] [--semantic] [--similar ID] [--person P] [--since D] [--until D] [--kind image\|movie] [--favorite] [--album A] [--collection C] [--located] [--live] [--limit N] [--cursor C]` | paged search, newest first; ranked by score with `--semantic` or `--similar` | no |
| `info ID...` | every field, cached renditions, albums, collections | no |
| `show ID... [--size thumb\|medium]` | fetch previews, print `id<TAB>path` | if not cached |
| `original ID... [--pin] [--version V]` | fetch full files, print paths | if not cached |
| `cache status\|verify\|pin\|unpin\|evict` | manage local copies; iCloud is never touched | no |
| `collection list\|create\|delete\|add\|remove\|show` | ordered sets of ids for a project | no |
| `people [--all]` | people from iCloud's People album, with photos found per person | no |
| `index [--limit N] [--seed] [--rematch] [--threshold T] [--fetch-models] [--background]` | CLIP embeddings and faces | yes |
| `lap-export [--library P] [--root P] [--limit N]` | write the catalogue into a [lap](https://github.com/julyx10/lap) library for browsing | no |
| `faces show\|assign\|unassign\|unassigned` | faces in photos; name or clear one | no |
| `config show\|set KEY VALUE` | `cache_budget_mb`, `username`, `preview_size`, `face_threshold`, `compute` | no |

Dates accept `2019`, `2019-07`, `2019-07-20` or full ISO 8601; `--until`
includes the whole of the period given. Exit codes: 1 error, 2 needs a
terminal, 3 not logged in, 4 cache full, 5 model files missing, 6 `compute=gpu`
without a usable GPU. With `--json`, errors are
`{"error": code, "message": text}` on stderr.

## JSON output

`--json` goes before the command. Every asset, wherever it appears, is the
same object: `id` (stable across runs and reindexing), `master_id`,
`filename`, `kind` (`image` or `movie`), `live`, `taken` and `added` (ISO
8601, UTC), `width`, `height`, `bytes`, `favorite`, `caption`, `latitude`,
`longitude`, `hidden`, and `versions` (the renditions iCloud offers, each with
`bytes`, `width`, `height`, `type`, `filename`). Ranked searches add `score`.

| Command | JSON on stdout |
| --- | --- |
| `search` | `{"results": [asset…], "count", "next_cursor"}` and `"query"` when ranked; pass `next_cursor` back as `--cursor` |
| `info` | the asset plus `cached` (`version`, `path`, `bytes`, `pinned`), `albums`, `collections`, `faces`; an array when several ids are given |
| `show`, `original` | `[{"id", "version", "path", "bytes", "cached", …}]`, one per id |
| `status` | `authenticated`, `auth`, `catalog` counts, `last_sync`, `sync_running`, `sync_progress`, `cache`, `index`, `compute`, `paths` |
| `sync`, `index` | the run's counts, or with `--background` `{"started": true, "pid", "unit", "log"}` |
| `albums`, `people`, `faces …`, `collection …`, `cache …`, `config …` | arrays of rows or the object the text form describes |
| `lap-export`, `lap-fetch` | counts written, or `{"id", "version", "path", "link"}` |

Errors are `{"error": code, "message": text}` on stderr with a non-zero exit;
codes are stable strings (`not-logged-in`, `cache-full`, `models-missing`,
`compute-unavailable`, `internal-error`, …). Fields are only ever added, not
renamed or removed, within a major version.

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
| models | `~/.local/share/icloud-photos/models/` |
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

## Meaning and faces

`photos index` gives every image a CLIP embedding and finds the faces in it.
The models are the ones Immich uses, run in-process: Immich's ONNX export of
CLIP ViT-B-32 (from huggingface.co/immich-app/ViT-B-32__openai, fetched with
`photos index --fetch-models`, about 600 MB) under onnxruntime, and
InsightFace's buffalo_l pack (fetched on first use). CPU only, roughly 60 ms
for the embedding and 300 ms for the faces of one thumbnail on an M1.

```sh
photos index --fetch-models
photos index --background           # pass 1: every image from its thumbnail; resumable
photos search --semantic "kids on a beach at sunset" --since 2019 --limit 10
photos search --similar <id>
photos search --person Julie --since 2024
photos faces show <id>
photos faces unassigned             # nearest misses first
photos faces assign <face-id> Julie # becomes a seed for her
photos people                       # photos found per person
```

Names come from iCloud, not from you: the People album Apple syncs holds a
record per person and a few face crops each. The first index run embeds
those crops as seeds, and a face in a photo is assigned to a person when it
is close enough (cosine 0.5 by default, `config set face_threshold`) to one
of that person's seeds. A face you assign by hand becomes a seed too, and
`photos index --rematch` re-runs matching over every automatically assigned
face after seeds change. What iCloud does not provide is which faces are in
which photo; that is what the index adds.

Pass 1 uses thumbnails (about 480 px wide), which is enough for the
embedding and for faces that fill a fair part of the frame. Small faces in
group shots need the medium rendition; the index records the source of each
result so a later pass can redo those. Movies are not indexed yet.

## Not yet

Pass 2 on medium previews, objects beyond what CLIP understands, shared
libraries, video sampling, GPU or Neural Engine inference. The catalogue
holds model output alongside the asset rows with the model version and the
rendition it came from, so previews can be evicted after analysis and a model
change re-indexes only what it must.

## Tests

```sh
.venv/bin/python -m unittest discover tests
```

The suite runs against a fake iCloud with the adapter's shape: first sync,
change feed with an edit and a deletion, master-record changes, paging,
every search filter, preview and original fetches, the budget with pinning
and eviction, a file removed behind the tool's back, Live Photo halves,
collections, error shapes, and the detached sync.
