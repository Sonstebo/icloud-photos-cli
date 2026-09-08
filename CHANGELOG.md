# Changelog

## 0.2.0 (2026-09-08)

First public release, https://github.com/Sonstebo/icloud-photos-cli.

- Catalogue of the whole library from the CloudKit change feed (assets, albums,
  people, face crops), incremental sync, hourly systemd timer.
- Bounded preview and original cache with pinning and eviction.
- `search` by text, date, kind, favourite, album, collection, location, Live
  Photo, `--semantic` (CLIP ViT-B/32) and `--similar`; `--person` from faces
  matched to iCloud's named people; `faces` review commands; collections.
- `index` in a resumable background worker; `compute auto|cpu|gpu`, the GPU
  through onnxruntime-ggml with a self-test and CPU fallback; queries stay on
  the CPU.
- `lap-export` and `lap-fetch` for browsing the library in lap.
- Every command has `--json`; errors are one line with a code and a non-zero
  exit; nothing writes to iCloud.
