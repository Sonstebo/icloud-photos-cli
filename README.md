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

## Browsing with the Photos app

The desktop app is a fork of [lap](https://github.com/julyx10/lap) (Tauri,
Rust and Vue, local AI, map, faces, editor) with two additions offered back
upstream: an album may be **managed**, meaning an external tool owns its rows
and the app never scans or prunes it, and it may name a **fetch-on-open**
command that materialises a file the moment someone opens it.

`photos lap-export` writes the catalogue into the app's library: one album of
symlinks under the cache, rows with dates, GPS, favourites and captions, our
thumbnails, our CLIP vectors (the app uses the same ViT-B/32 weights, so its
semantic search runs on them), people, faces and collections. Each entry is
named after the original file, so the app decodes HEIC and RAW itself, and it
stays a dangling link until you open the photo: then the app runs
`photos lap-fetch`, the full-resolution original is downloaded into the bounded
cache, and eviction removes it again later. `config set lap_open_version medium`
fetches the 2048 px preview instead; `photos original ID --pin` keeps a file
for good.

```sh
Photos                                  # once, so the app creates its library
photos lap-export                       # then restart the app
photos refresh                          # sync + index + export, what the timer runs
```

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
| `select [QUERY] [--similar ID] [--person P] [--include-screenshots] [--count N] [--variety 0..1] [--spread none\|day\|month] [--everyone] [--sharp-only] [--keep-duplicates] [--floor S] [--person P] [--since D] [--until D] [--album A] [--collection C] [--favorite] [--located] [--into COLLECTION] [--replace]` | choose a handful of good photos out of thousands; reports every stage of the funnel | no |
| `compose [COLLECTION] [--id ID...] [--template justified\|grid\|hero\|filmstrip\|spread\|scatter] [--shape 3:2\|square\|a4-landscape\|a4-portrait\|spread\|16:9] [--gap N] [--count N] [--background C] [--long-edge PX] [--originals] [--no-face-safe] [--plan] [--out FILE]` | turn a set of photos into one picture; no crop cuts a face | only with `--originals` |
| `book list\|create\|delete\|add\|remove\|move\|show\|export` | ordered pages of collages, exported as one PDF | only with `--originals` |
| `info ID...` | every field, cached renditions, albums, collections | no |
| `show ID... [--size thumb\|medium]` | fetch previews, print `id<TAB>path` | if not cached |
| `original ID... [--pin] [--version V]` | fetch full files, print paths | if not cached |
| `cache status\|verify\|pin\|unpin\|evict` | manage local copies; iCloud is never touched | no |
| `collection list\|create\|delete\|add\|remove\|show` | ordered sets of ids for a project | no |
| `people [--all]` | people from iCloud's People album, with photos found per person | no |
| `index [--limit N] [--seed] [--rematch] [--threshold T] [--fetch-models] [--background]` | CLIP embeddings and faces | yes |
| `refresh [--no-index] [--index-limit N]` | sync, index what is new, update the GUI library | yes |
| `edit ID PROMPT [--out DIR] [--timeout N]` | ask an agent to change a photo; the result is a new file on disk | yes |
| `lap-export [--library P] [--root P] [--limit N] [--fetch-thumbs] [--open-version V]` | write the catalogue into the GUI's library | only with `--fetch-thumbs` |
| `lap-fetch PATH` | fetch the file behind one GUI entry (the app's fetch-on-open command) | yes |
| `faces show\|assign\|unassign\|unassigned` | faces in photos; name or clear one | no |
| `config show\|set KEY VALUE` | `cache_budget_mb`, `username`, `preview_size`, `face_threshold`, `compute`, `lap_open_version`, `edits_dir`, `edit_agent`, `edit_timeout_s`, `collages_dir` | no |

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

## Choosing a handful out of thousands

Ranking by similarity to a description gives you nine frames of one moment,
which is why automatic selection usually disappoints. `photos select` is a
funnel instead, and every stage says what it removed:

```
$ photos select "children playing outdoors in summer" --count 8 --spread day
 33,161  library              everything in the catalogue
 30,100  filters              dates, people, album, place
    240  meaning              nearest 1,440 to 'children playing outdoors...', then filtered
    153  near-duplicates      87 collapsed into their best frame
      8  variety and quotas   variety 0.45, spread by day
```

The last stage picks greedily: each time it takes the photo that is most
relevant *minus* how much it resembles what has already been chosen. `--variety`
sets that trade-off, from 0 for the tightest match to 1 for the widest spread.

Two photos count as the same picture only when they look alike **and** were
taken within ninety seconds, so a burst collapses to its best frame while the
same wall photographed a year apart does not. The keeper is decided by focus,
except that anything you marked as a favourite outranks any measurement.

`--similar ID` grows a set around one photograph instead of a description,
which is what "more like this" means: the same funnel, anchored on a picture
rather than on words.

**Screen captures are set aside.** A camera does not write PNG, and in a real
33,000-photo library 3,326 of the 3,328 screen captures are PNG while no
photograph is, so the extension is the whole test. Judging by device-screen
dimensions instead was tried and rejected: it would have thrown out a scan
called *Julie og oldefar.jpg*. `--include-screenshots` keeps them, and the
funnel always says how many were set aside.

**The pool spans the whole range.** With nothing to rank by, taking the newest
few hundred cannot answer "from when she was little until now" with anything but
this year. The pool is drawn across the whole filtered set instead, and when a
spread is asked for, round-robin across its buckets, so a year with forty
photographs cannot crowd out one with four.

`--spread day` stops a week's trip coming out as one Tuesday afternoon, and
`--spread year` turns eighteen years into eighteen photographs.
`--person` may be repeated to pool several people, and `--everyone` then
guarantees each of them appears at least once, reporting anyone it could not
place rather than quietly leaving them out. Nothing here touches the network,
and `--into` writes the result to a collection in order.

## Turning a set of photos into one picture

`photos compose` lays a collection out as a collage, a book spread, a contact
strip or a scatter of mounted prints. Geometry is code, not conversation: the
same photos and the same recipe always give the same picture, so a draft on
screen and the file that goes to the printer are the same layout.

```
$ photos select "the winter trip" --count 9 --spread day --into "Winter book"
$ photos compose "Winter book" --template justified          # a draft, seconds, no network
$ photos compose "Winter book" --template justified --originals   # the same page at 300 dpi
```

Composing is fast because it uses whatever is already cached, at draft size.
Only `--originals` reaches for the full files, and it refuses to start if they
would not fit the cache budget rather than overrunning it.

**No crop cuts a face.** Every slot has its own shape and almost no photo
matches it, so something must be cropped away; cropping from the centre
eventually slices somebody's head in half. The catalogue already knows where the
faces are, so the crop window slides just far enough to hold them all. If the
stored boxes cannot be trusted for a photo, its faces are ignored and the crop is
centred, because a wrong crop is worse than a plain one. `--no-face-safe` turns
it off.

The templates: **justified** fills the page with rows that each span the full
width, choosing the row count whose natural height lands closest to the page;
**grid** uses uniform cells and centres a short last row; **hero** makes the
first photo large; **filmstrip** runs equal frames across the whole page;
**spread** lays out two facing pages and lets nothing cross the gutter;
**scatter** drops mounted prints on a table with a fixed, repeatable tilt.
No photograph is ever stretched, only cropped.

`--plan` prints the geometry and writes no file, which is what a preview draws.
Results land in `~/Pictures/Photos Collages/<date>/`, an ordinary folder
(`collages_dir`).

## Books

A book is an ordered list of pages, and a page is a recipe rather than a
picture: the photos it uses and how to lay them out. Nothing is drawn until the
book is exported, so pages can be reordered or relaid at any time and the export
always matches what the pages say.

```
$ photos book create "Summer 2024" --shape a4-landscape
$ photos book add "Summer 2024" --collection "Lofoten" --template justified --caption "The first day"
$ photos book add "Summer 2024" --collection "Lofoten" --template hero
$ photos book move "Summer 2024" --page 2 --to 1
$ photos book export "Summer 2024" --originals
```

Every page has the book's shape, because a PDF gives its pages one size and a
page of a different shape would be squashed into it. A caption is written into a
strip the layout reserves inside the page for exactly that reason: adding one
underneath afterwards would make a captioned page taller than an uncaptioned one.

`export` without `--originals` gives a draft in seconds from cached thumbnails,
at 150 dpi. With it, the originals are fetched and the PDF comes out at 300 dpi,
A4 landscape measuring 842 by 595 points, which is A4.

## Editing a photo by asking

```sh
photos edit <id or album entry> "warmer light and a little more contrast"
```

The original is fetched and handed to [Codex](https://developers.openai.com/codex/cli),
OpenAI's agent CLI, which signs in with a ChatGPT subscription. The result is a
new file in `~/Pictures/Photos Edits/<date>/`, an ordinary folder you can browse,
back up or delete without this tool. Nothing is written to iCloud and the
original is never touched.

The agent picks one of two routes and says which it used:

| Route | For | Cost |
| --- | --- | --- |
| `magick` | exposure, contrast, colour, crop, rotate, resize, sharpen, borders, text, format | seconds, exact, full resolution |
| `generated` | removing or adding objects, restoring an old photo, changing a background | a minute or two, and the image model's own size (about 1024x1536), so a large original comes back smaller |

Both are covered by the subscription. The output line reports the route and,
when the size changed, both sizes. `config set edits_dir`, `edit_agent` and
`edit_timeout_s` change where results go, which agent is asked, and how long it
may take.

## What the GUI does not do

- **Favourites come from iCloud.** Toggling one in the app is overwritten by the
  next refresh; the CLI never writes to iCloud. Ratings, tags and the app's own
  collections are left alone.
- **Collections travel one way**, from here into the app.
- **Keeping a file offline** is `photos original ID --pin`; there is no button
  for it in the app yet.
- **62 RAW imports have no thumbnail** until you open one: iCloud offers only
  their 20 MB original, so nothing small exists to make one from.

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
