# iCloud Photo Workspace — project brief for Claude

## Status and authorization

Implementation started 2026-09-08 on the user's instruction. The user logged in (`photos login`, Apple ID in config) and a full sync of the real library ran the same day. Verified on real data:

- **Library**: 33,159 assets in the catalogue (30,191 images, 2,968 movies; 67 hidden, 26 recently deleted, 222 favourites, 13,875 with GPS, 10,639 Live Photos). The index count Apple reports is 32,963; the ~100 difference is unexplained and low priority.
- **Sync** = one resumable walk of the CloudKit zone change feed (`icloud_photos/sync.py`): 352,694 records in 67 minutes, 141,190 of them tombstones; assets paired with masters, album membership from `CPLContainerRelation`, people from `CPLPerson`, face crops from `CPLFaceCrop`. Later syncs continue from the stored token. Raw records are kept (`records` table), which makes the catalogue 1.2 GB; shrinking that (compress JSON, or drop resource URLs) is an open item.
- **People**: 1,255 person records, 95 named, 535 with face crops; the named ones carry full and display names and a verified flag. Face crops reference a person and hold a ~15 KB crop image, but **no asset reference**: iCloud gives the who and a reference face, not which photo a face is in. Per-asset `people` lists are empty and `facesVersion` is 0 everywhere.
- **Fetching**: `show` two thumbnails 4.5 s, `original` 4.5 s (mostly session setup), evict works. Thumbs ~480x360, medium ~2048x1536, for an ordinary photo.
- **Tests**: `tests/test_cli.py`, 16 tests against a fake zone feed.

pyicloud findings, for an upstream report: (1) `PhotoAlbum.photos` stops when a page pairs fewer than half its assets with masters (the reply's record budget runs out), which ended a listing at July 2024 here; (2) descending album walks can loop at rank 0; (3) `PhotoAlbum.get(id)` falls back to iterating the whole library when the index lookup misses; (4) `PhotoAsset.favorite()` is a setter. The adapter avoids all four by using the change feed and `records/lookup`.

- **Index** (`icloud_photos/index.py`, `ml.py`, chosen 2026-09-08 after benchmarking on this machine): Immich's ONNX export of CLIP ViT-B-32 (huggingface.co/immich-app/ViT-B-32__openai, 600 MB under `~/.local/share/icloud-photos/models/`) under onnxruntime, and InsightFace buffalo_l, both CPU. Measured: 85 ms per CLIP embedding, ~0.4 s per face pass, ~0.58 s per thumbnail all in, 2.1 GB RSS. Docker is unusable from the user's account, which ruled out Immich's container; torch was tried and removed (5 GB, unnecessary). Seeds: 501 embeddings from the 543 iCloud face crops of the 95 named people (42 crops too small). First 40 images: 34 faces, one named at cosine 0.77 (a clear portrait), the rest below 0.5; a 1980s scan scored 0.42 against a person seeded from recent faces, which is the expected age gap and what `faces unassigned` + `faces assign` are for. Pass 1 (thumbnails) over all 30,165 images started 2026-09-08 in the background, roughly 5 hours.

- **GPU inference, tried and parked (2026-09-08).** The only runtime that runs our ONNX files on the M1 GPU under Asahi is onnxruntime-ggml (thewh1teagle, ggml's Vulkan backend behind an onnxruntime execution provider). Its PyPI wheel crashes with SIGILL: it was compiled with SVE, which the M1 lacks. Building it from source fixes that (checkout, venv and models under `~/.cache/ggml-dev/`; `libs/lib` holds ggml built with `-mcpu=apple-m1` and Vulkan, `target/release/libonnxruntime_ggml.so` is the provider, `ONNXRUNTIME_GGML_LIBRARY` points a venv at it). Findings on this machine: the provider's Conv is 1-D only, so the InsightFace detector and recogniser (2-D CNNs, most of the per-image time) cannot run on it at all; CLIP's transformer does run on the GPU once the patch convolution is done in numpy and the final L2 norm is cut off (`clip_gpu.py` there), with identical embeddings, but at 57 ms per image against 53 ms on onnxruntime's CPU provider, so nothing is gained. Partial claims are also rejected by onnxruntime because the provider claims one fused block. Adding 2-D convolution to the provider is feasible (mirror of its 1-D im2col path plus pool_2d, ~300-400 lines of Rust; `~/.cache/ggml-dev/bench/` has the benchmarks), but measured on an idle CPU, ggml's Vulkan conv2d on this GPU is only 1.5-2x faster than onnxruntime's CPU conv at the detector's large layers and equal or slower at the recogniser's deep layers, before per-op dispatch overhead. Expected whole-pipeline gain is well under 2x. Overlapping CLIP on the GPU with faces on the CPU in a second thread was measured too (`bench/pipeline.py`): 605 vs 617 ms per image, a 2% gain, because faces are 531 ms of it. Stay on CPU. The lever is the face pass itself (`bench/facemods.py`, 24 thumbnails): all five InsightFace models 540 ms, detection+recognition only 332 ms with the same 38 faces, and with the detector at 480x352 instead of 640x640 230 ms but 34 faces. The index uses only detection and recognition (age/gender are stored but nothing reads them).
- **Power (2026-09-08, SMC "Total System Power" via hwmon, `bench/power.py`, idle 8.4 W).** Faces det+rec on CPU: 26.7 W, 329 ms, 8.8 J per image. CLIP on CPU: 28.9 W, 47 ms, 1.36 J; the same CLIP on the GPU: 12.8 W, 48 ms, 0.62 J, so the GPU does equal work in equal time at about a quarter of the energy above idle. The conv2d layer set: CPU 23.0 W and 538 ms (12.4 J) against GPU 17.7 W and 262 ms (4.6 J). A full 30k-image face pass on the CPU is roughly 50 Wh of work above idle, most of a 58 Wh battery; on the GPU it would be an estimated 15-25 Wh. This is the case for building 2-D convolution into the provider: not speed, energy. The user requires that everything built is general (auto-detected, falls back to CPU, upstreamed) so other users benefit, not this installation only.
- **Memory (2026-09-08).** 7.5 GB RAM, no swap, and `/tmp` is a 3.7 GB tmpfs. The first pass-1 run died at 3,990 images: the index loop kept every chunk's decoded thumbnails (fixed, 54c3515), and a parallel ggml build plus scratch files in `/tmp` pushed the machine into the OOM killer, which took the terminal's whole cgroup with it. Background workers now run as transient systemd user units (`icloud-photos-index-*`, `icloud-photos-sync-*`); the index was restarted the same day and continues from 3,990.

Not started: pass 2 on medium previews, shared library, video sampling, ANE inference, a bug report to pyicloud.

Publishing to GitHub still requires an instruction. The systemd user timer in `systemd/` is written but not enabled.

## Decisions (2026-09-08)

- **The overriding goal is a perfect fit for AI assistants working through a CLI.** Every other quality (GUI, speed, breadth of features) is secondary. Concretely:
  - Every operation is a subcommand; nothing requires a GUI, an interactive prompt or a browser except the one-time iCloud sign-in.
  - Every subcommand has `--json` output with a documented, stable schema; IDs are stable across runs and reindexing.
  - Output is bounded by default (`--limit`, paging cursors) so an assistant never gets a library dump in its context.
  - Errors are one line on stderr plus a non-zero exit, with a machine-readable code in JSON mode; no stack traces, no silent failures.
  - Every command is idempotent or says what it changed; nothing destructive happens without an explicit flag.
  - `photos --help` and per-command `--help` are complete enough that an assistant can use the tool from them alone; the README is for humans.
  - The tool returns file paths for images it fetched, so an assistant can look at a preview with its own image reading.
  - Long jobs (indexing, downloads) run detached and report progress through `status`; a query never waits on the network.
- **Reuse before building.** Do not implement anything an existing open-source project already provides: iCloud access, catalogue, thumbnailing, face/object models, vector search. Survey candidates first, verify them on this machine, and write only the glue and whatever is genuinely missing. Say explicitly when something has to be built because no candidate fits.
- **CLI, not MCP.** The interface for assistants, scripts and the user is a command-line tool with `--json` output, stable asset IDs, errors on stderr and non-zero exits. No MCP server; a wrapper can be added later if a client without shell access needs it.
- The CLI is the interface, not the whole program. Indexing and preview fetching run in a resumable background worker; the CLI reads and writes the local catalogue and never blocks a query on the network.

## User's goal

Provide a Linux photo experience that keeps full-resolution originals in iCloud, limits local storage, and makes the library useful to AI assistants for photobooks, slideshows, search, and other creative projects.

The user values accurate face and object recognition, local processing where practical, easy access from local applications, and a reproducible setup for a reinstalled or replacement computer. Prefer existing open-source projects and components over building everything from scratch.

## Target environment

- Apple Silicon M1 MacBook Pro running Linux with Asahi support and an Omarchy/Hyprland Wayland desktop.
- Project directory: `/home/sonstebo/Work/icloud-photo-workspace`.
- Existing photo tools observed: imv image viewer and a Google Photos web launcher; neither supplies the desired iCloud catalogue.
- Vulkan acceleration has previously improved Whisper dictation on this machine. This does not prove that a chosen vision model or runtime supports its GPU.
- Recheck actual hardware, packages, kernel and runtime capabilities before implementation. Do not assume macOS frameworks are available on Linux.

## Proposed architecture, not a final technology selection

Separate the cloud source, durable catalogue, disposable image caches, inference workers, and creative projects.

| Component | Responsibility | Persistence |
| --- | --- | --- |
| iCloud adapter | Enumerate assets and available albums/metadata; retrieve available image variants | Authentication state protected separately |
| Local catalogue | Stable cloud asset identifiers, revisions, dates, dimensions and available album membership | Durable |
| AI index | Semantic search vectors, object/scene labels, face groups, confidence and model versions | Durable |
| Thumbnail/preview cache | Fast browsing and images for analysis | Bounded and evictable |
| Original cache | Files needed for editing, export or high-resolution inspection | On demand and evictable unless pinned |
| Project store | Photobook selections, captions, ordering, layout, crop/edit instructions and asset references | Durable |
| AI tool interface | Search, retrieve selected previews, create collections and request originals | Scoped access through the catalogue |

Possible logical flow:

```text
iCloud -> adapter -> catalogue -> search/UI/AI tools -> project selections
             |           ^                               |
             v           |                               v
      bounded previews -> inference workers       selected originals
                                                      -> export
```

Remembering a photo must not require retaining its full image. An analysed asset remains searchable after its preview has been evicted. Offline access to uncached images cannot be promised.

## Storage and synchronization requirements

- Never mirror all originals as the default behavior.
- Define a user-configurable total disk budget covering previews, originals, temporary files and model files; make durable catalogue/index growth visible separately.
- Account for concurrent downloads and exports before starting them. Eviction alone is insufficient to enforce a strict limit when files are open or pinned.
- Evict cached copies without deleting anything in iCloud. Explicit exports and user-created files are not disposable cache entries.
- Allow pinning selected assets/projects for offline work, showing their disk cost. If pinned content exceeds the budget, stop and ask for a storage decision.
- Use stable asset IDs and revisions rather than filenames alone; filenames can collide.
- Make indexing resumable, incremental and restart-safe. Store the source revision and model version used for each analysis result.
- Treat cloud edits and deletions explicitly. Do not remove durable project content silently when the cloud source changes; mark missing or changed sources for review.
- Distinguish unmodified originals, edited renditions, HEIC/RAW assets, and Live Photo image/video pairs. Preserve provenance and choose the appropriate rendition for export.
- Start with read-only cloud access. Two-way album, edit, upload and deletion synchronization is outside the initial scope.
- A complete visual index still requires fetching image data across the library. Bounded local storage does not eliminate initial bandwidth or compute cost.

## Recognition and accuracy

Use progressively better image representations:

1. Small thumbnails for browsing.
2. Medium previews for initial semantic, object and face analysis.
3. Higher-resolution images/crops for small faces, uncertain results and final selection checks.

Do not assume tiny thumbnails are adequate for accurate face recognition or print-quality assessment. Verify what smaller renditions iCloud actually supplies for representative assets before relying on them.

Face detection, face grouping, named identity assignment, object recognition, semantic search and quality ranking are distinct tasks. Preserve user corrections and avoid assigning confident names to ambiguous faces. Keep confidence, model versions and result provenance. Evaluate accuracy on a user-approved representative sample rather than relying only on model marketing or speed.

Initial indexing should be resumable, favour charging/idle periods, and pause or throttle during interactive work. Video analysis requires a separate sampling and storage strategy; do not silently download every video.

## Apple Neural Engine (ANE)

Apple documents using ANE for an on-device people-recognition model in Photos. This is not evidence that every Photos feature always uses ANE, or that Apple's proprietary models and People labels are accessible to Linux.

As checked on 2026-09-07, Asahi's M1 feature table describes ANE support as an out-of-tree kernel module. The `eiln/ane` driver and `eiln/anecc` converter are relevant experimental foundations. Do not treat them as a ready, verified inference backend for this computer or arbitrary models.

Design an interchangeable inference interface:

- CPU as the compatibility baseline.
- GPU acceleration only after verifying the selected runtime, operators, correctness and performance on this Apple GPU.
- ANE as an optional experimental backend after separate driver, model-conversion and numerical-accuracy validation.

Do not make successful ANE deployment a prerequisite for the photo application. Measure accuracy, throughput, memory use, power/thermal behavior and responsiveness for representative workloads. Do not install a custom kernel or driver merely to explore the design.

## AI-native workflow

Expose purposeful tools rather than giving an assistant an unrestricted filesystem scan:

- Search by date, available album/location metadata, confirmed people, objects and natural-language similarity.
- Fetch a bounded batch of previews with stable asset IDs and useful metadata.
- Create and revise a project collection.
- Request a selected original for export, editing or detailed inspection.
- Report cache budget, indexing progress, uncertain matches and missing sources.

A local API or MCP interface is a candidate, not a selected implementation. Default analysis should be local where feasible. Cloud AI should receive only the selected images and metadata needed for an explicitly chosen task; local indexing must not imply permission to upload the library.

Example photobook flow:

1. Search for a family trip using dates, available metadata and visual matches.
2. Inspect candidate previews and group events or near duplicates.
3. Propose diverse selections, captions and page layouts.
4. Let the user correct identities, selections and captions.
5. Fetch the selected high-resolution renditions and check crop/print suitability.
6. Export the book; retain its manifest and explicitly saved output while releasing unpinned cache files.

## Existing open-source candidates and findings

These are research findings, not tested compatibility claims. Recheck versions, maintenance, licensing and source code before choosing dependencies.

### rclone

Documentation describes an iCloud Photos service, read-only album/library hierarchy, listing caches, and FUSE mounting with local caching. This is a promising original-file access component.

Crucial limitation: mounting remote originals does not provide Apple Photos-style thumbnail browsing. A viewer or indexer may read originals to generate thumbnails, triggering substantial downloads. Measure actual network and disk behavior. Verify the installed/released version includes the documented Photos features; do not confuse older iCloud Drive support with Photos support.

The current documentation describes Advanced Data Protection support with trusted-device approval. This differs from icloudpd's documented prerequisites and must be verified for the chosen version/account.

### icloudpy / pyicloud family

These libraries expose iCloud web-service access; icloudpy documents thumbnail, medium and original photo versions. Potential foundation for metadata and preview retrieval. Validate current authentication, pagination, edited renditions, shared libraries, and actual variant availability before committing to one fork.

### icloudpd

A downloader/synchronization tool, not an on-demand photo catalogue. Its repository currently requests a maintainer and documents disabling Advanced Data Protection as a prerequisite. Do not change the user's protection settings to accommodate a dependency. Avoid invoking modes that delete cloud assets.

### iCloud for Linux (`cgillinger/icloud-electron`)

An open-source launcher for Apple's web applications in desktop windows. Despite the repository name, its current description uses a browser launcher. Useful for immediate interactive browsing, but it is not a native catalogue or an AI data-access layer; Apple's web interface is proprietary.

### PhotoHarbor

An existing GUI project describing local iCloud downloads, albums and downloaded-file browsing. Investigate before dismissing it, but its published description does not establish the requested thumbnail-first, bounded-storage behavior.

Do not claim that a complete native open-source replacement has been found. The research so far identifies reusable components and partial applications; it does not prove no such application exists.

## Authentication and private data

- Use the selected application's normal authentication flow; never ask the user to put passwords in chat or source files.
- A prior unrelated Thunderbird task received an approval rejection for extracting/decrypting saved credentials. Do not reuse or bypass that approach for this project.
- Protect session tokens and the sensitive face/index data. Keep credentials, images, caches and account-specific catalogue data out of Git.
- Plan for expired sessions and trusted-device approvals without losing indexing progress.
- Keep source access read-only initially and make any later cloud mutations explicit.

## Suggested next design work

1. Confirm library scale (asset count, photos/videos, approximate bytes), acceptable local storage budget and offline expectations.
2. Establish whether Advanced Data Protection and Shared Photo Library are used, without changing settings.
3. Compare current reusable projects against preview access, stable IDs, incremental listing, authentication and ARM64 compatibility.
4. Propose a concrete component selection and cache policy, clearly separating verified support from assumptions.
5. After implementation is authorized, prove a narrow end-to-end path: enumerate a small sample, fetch previews, index it, select one image, fetch its original and evict only the cached copy.
6. Measure download volume, peak disk/RAM use, restart recovery, cache behavior and recognition quality before scaling.
7. Add reproducible setup and documentation after the chosen path is validated. Publishing to GitHub requires an instruction for this project.

## Initial acceptance criteria

- Browsing/searching does not trigger an original-library download.
- The system clearly distinguishes cached, cloud-only and pinned assets.
- Completed analysis remains searchable after preview eviction.
- Selected originals can be retrieved with correct asset/rendition identity.
- Cache eviction cannot delete cloud originals, project manifests or explicit exports.
- Jobs recover after restart or authentication expiry without restarting the whole library.
- Disk exhaustion is prevented or handled explicitly, including pinned content and temporary exports.
- Local and cloud AI paths have clear, inspectable data boundaries.
- Face accuracy and user corrections survive reindexing; backend changes do not silently change identities.

## Sources checked during discussion

- Apple people recognition and ANE: https://machinelearning.apple.com/research/recognizing-people-photos
- Apple Photos privacy: https://www.apple.com/legal/privacy/data/en/photos/
- Asahi M1 support: https://asahilinux.org/docs/platform/feature-support/m1/#ane-driver
- ANE driver: https://github.com/eiln/ane
- ANE converter: https://github.com/eiln/anecc
- rclone iCloud Drive and Photos: https://rclone.org/iclouddrive/
- icloudpy: https://github.com/mandarons/icloudpy
- pyicloud photo implementation: https://github.com/picklepete/pyicloud/blob/master/pyicloud/services/photos.py
- icloudpd: https://github.com/icloud-photos-downloader/icloud_photos_downloader
- iCloud desktop launcher: https://github.com/cgillinger/icloud-electron
- PhotoHarbor: https://github.com/woutervanwijk/PhotoHarbor
- Apple iCloud Photos web guide: https://support.apple.com/en-gb/guide/icloud/mmbc402b84/icloud
