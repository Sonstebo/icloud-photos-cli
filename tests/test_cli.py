"""End-to-end through the CLI with a fake iCloud.

The fake has the same shape as the real adapter (see icloud_photos.adapter)
and a small library the tests mutate to simulate edits and deletions in the
cloud. Nothing here needs the network or a session.
"""
from __future__ import annotations

import base64
import io
import json
import numpy as np
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from icloud_photos import cli
from icloud_photos.adapter import AlbumInfo, AssetInfo, CloudError, RawRecord, retry

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2019, 7, 20, 12, 0, tzinfo=timezone.utc)


def make_asset(n: int, **over) -> AssetInfo:
    base = dict(
        id=f"A{n:03d}/x+y==", master_id=f"M{n:03d}", filename=f"IMG_{n:04d}.HEIC", kind="image", live=False,
        taken=T0 + timedelta(days=n), added=T0 + timedelta(days=n, hours=1), width=4032, height=3024,
        bytes=3_000_000, favorite=False, caption=None, latitude=None, longitude=None, hidden=False,
        versions={
            "original": {"bytes": 3_000_000, "width": 4032, "height": 3024, "type": "public.heic", "filename": f"IMG_{n:04d}.HEIC"},
            "medium": {"bytes": 300_000, "width": 1600, "height": 1200, "type": "public.jpeg", "filename": f"IMG_{n:04d}.JPG"},
            "thumb": {"bytes": 30_000 + n, "width": 400, "height": 300, "type": "public.jpeg", "filename": f"IMG_{n:04d}.JPG"},
        })
    base.update(over)
    return AssetInfo(**base)


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def rec(name: str, rtype: str, **fields) -> RawRecord:
    """A raw record in CloudKit's JSON shape: fields are {name: {value: ...}}."""
    return RawRecord(name, rtype, False, T0, {"recordName": name, "recordType": rtype,
                                              "fields": {k: {"value": v} for k, v in fields.items()}})


def tombstone(name: str) -> RawRecord:
    return RawRecord(name, None, True, None)


class FakeCloud:
    """Pretends to be iCloud's photo zones.

    `library` holds AssetInfo objects; the zone feed presents each as a
    CPLMaster and a CPLAsset record (master first or asset first, both
    happen for real). `pending` is what the next incremental sync will see.
    Downloads return `bytes` bytes of filler.
    """

    def __init__(self) -> None:
        self.assets: dict[str, AssetInfo] = {}
        self.token = 0
        self.pending: list[RawRecord] = []
        self.downloads: list[tuple[str, str]] = []
        self.album_list = [AlbumInfo("alb1", "Trip", "Trip"), AlbumInfo("alb2", "Family", "Folder/Family")]
        # a shared library is a second zone; empty unless a test puts photos in one
        self.shared_zones: list[str] = []
        self.shared_pages: dict[str, list[RawRecord]] = {}
        # photos that live only in a shared library, so `whole_zone` never sees them
        self.shared_assets: dict[str, AssetInfo] = {}
        self.relations: list[tuple[str, str]] = []
        self.people: list[tuple[str, str, str]] = [("p1", "Thea-Oline", "Thea-Oline"), ("p2", "Ingjerd Thürmer", "Ingjerd")]
        self.face_crops: list[tuple[str, str]] = [("fc1", "p1"), ("fc2", "p1"), ("fc3", "p2")]
        self.logged_in = True
        self.page_size = 5

    def add(self, *assets: AssetInfo) -> None:
        for a in assets:
            self.assets[a.id] = a

    def records_for(self, a: AssetInfo, master_first: bool = True) -> list[RawRecord]:
        master = rec(a.master_id, "CPLMaster", filenameEnc=b64(a.filename))
        asset = rec(a.id, "CPLAsset", masterRef={"recordName": a.master_id}, isFavorite=int(a.favorite),
                    isHidden=int(a.hidden), isDeleted=int(a.deleted))
        return [master, asset] if master_first else [asset, master]

    def whole_zone(self) -> list[RawRecord]:
        out = []
        for n, a in enumerate(sorted(self.assets.values(), key=lambda a: a.taken)):
            out += self.records_for(a, master_first=(n % 2 == 0))
        out += [rec(f"{i}-IN-{c}", "CPLContainerRelation", itemId=i, containerId=c, position=n * 1024)
                for n, (c, i) in enumerate(self.relations)]
        out += [rec(pid, "CPLPerson", personFullNameEnc=b64(full), displayName=b64(disp), verifiedType=1, personType=0)
                for pid, full, disp in self.people]
        out += [rec(fid, "CPLFaceCrop", personRef={"recordName": pid}, type=5, resFaceCropFileSize=15000)
                for fid, pid in self.face_crops]
        out.append(rec("mem1", "CPLMemory", title=b64("Majorca")))   # a type the sync ignores
        return out

    def push(self, *records: RawRecord) -> None:
        self.pending += records

    # --- Adapter ---
    def auth_status(self):
        return {"authenticated": self.logged_in, "username": "fake@example.com"}

    def _iter_primary(self, since):
        records = self.whole_zone() if since is None else self.pending
        if since is not None:
            self.pending = []
        for i in range(0, len(records), self.page_size):
            self.token += 1
            yield records[i:i + self.page_size], f"t{self.token}"
        if not records:
            return

    def asset_from_records(self, asset, master):
        info = self.assets.get(asset.name) or self.shared_assets[asset.name]
        # what a real adapter would derive from the records rather than remember
        info.favorite = bool(asset.value("isFavorite"))
        return info

    def zones(self):
        from icloud_photos.adapter import ZoneInfo
        return [ZoneInfo("PrimarySync")] + [ZoneInfo(n) for n in self.shared_zones]

    def iter_zone(self, since, zone=None):
        name = getattr(zone, "name", None)
        if name and name != "PrimarySync":
            # like the real feed: what is waiting, once, and then nothing
            page = self.shared_pages.get(name, [])
            self.shared_pages[name] = []
            yield page, f"{name}-token"
            return
        yield from self._iter_primary(since)

    def download(self, asset_id, version, master_id=None, zone=None):
        a = self.assets.get(asset_id) or self.shared_assets.get(asset_id)
        assert master_id == a.master_id, "the CLI passes the catalogue's master id so no index walk is needed"
        if a is None or version not in a.versions:
            return None
        self.downloads.append((asset_id, version))
        return b"x" * a.versions[version]["bytes"]

    def download_many(self, items, version, threads=4, zone=None):
        for asset_id, master_id in items:
            yield asset_id, self.download(asset_id, version, master_id, zone)

    def download_face_crops(self, crop_ids, threads=4):
        for crop_id in crop_ids:
            self.downloads.append((crop_id, "facecrop"))
            yield crop_id, b"CROP:" + crop_id.encode()

    def albums(self):
        return list(self.album_list)


class StubModels:
    """Deterministic stand-ins: an image's embedding is derived from its bytes, faces from a table."""

    clip_name, face_name = "stub-clip", "stub-face"

    def __init__(self):
        self.faces_by_key = {}      # image-bytes-key -> list of face embeddings (np arrays)
        self.text_vectors = {}      # query -> vector

    @staticmethod
    def _unit(seed):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_image(self, bgr):
        return self._unit(int(bgr[0, 0, 0]))          # decode_image is patched to carry a key in pixel 0

    def embed_text(self, text):
        return self.text_vectors.get(text, self._unit(hash(text) % 1000))

    def faces(self, bgr):
        key = int(bgr[0, 0, 0])
        return [{"box": [1, 2, 3, 4], "det": 0.9, "age": 30, "gender": 1, "embedding": e.tobytes()}
                for e in self.faces_by_key.get(key, [])]


def stub_record(**fields):
    """A raw-dict CloudKit record, the legacy shape pyicloud's mappers accept."""
    return {"recordName": "R", "fields": {k: {"value": v} for k, v in fields.items()}}


class StubPhoto:
    """A pyicloud PhotoAsset look-alike whose writes explode."""

    def __init__(self):
        self.asset_record = stub_record(isFavorite=1, isHidden=0, assetDate=T0)
        self.master_record = stub_record()
        self.id, self.master_id, self.filename, self.item_type = "P1", "M1", "IMG_1.JPG", "image"
        self.is_live_photo, self.dimensions, self.size = False, (10, 10), 5
        self.asset_date, self.added_date, self.versions = T0, T0, {}

    def favorite(self):
        raise AssertionError("favorite() writes to iCloud")

    unfavorite = set_favorite = delete = favorite


class AdapterTest(unittest.TestCase):
    def test_reading_an_asset_never_calls_a_writing_method(self):
        try:
            from icloud_photos.adapter import ICloudAdapter
            info = ICloudAdapter._info(StubPhoto())
        except ModuleNotFoundError:
            self.skipTest("pyicloud not installed")
        self.assertTrue(info.favorite)
        self.assertFalse(info.hidden)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ICLOUD_PHOTOS_HOME"] = self.tmp.name
        self.cloud = FakeCloud()
        self.cloud.add(*(make_asset(n) for n in range(1, 8)))
        self.cloud.assets["A003/x+y=="].favorite = True
        self.cloud.assets["A004/x+y=="].caption = "Beach day"
        self.cloud.assets["A005/x+y=="].latitude = 59.9
        self.cloud.assets["A005/x+y=="].longitude = 10.7
        self.cloud.add(make_asset(9, kind="movie", filename="IMG_0009.MOV", versions={
            "original": {"bytes": 50_000_000, "type": "com.apple.quicktime-movie", "filename": "IMG_0009.MOV"},
            "thumb_image": {"bytes": 20_000, "type": "public.jpeg", "filename": "IMG_0009.JPG"},
        }))
        self.cloud.relations = [("alb1", "A001/x+y=="), ("alb1", "A002/x+y==")]

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("ICLOUD_PHOTOS_HOME", None)

    def run_cli(self, *argv, expect=0):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(list(argv), adapter_factory=lambda app: self.cloud)
        self.assertEqual(rc, expect, f"argv={argv}\nstdout={out.getvalue()}\nstderr={err.getvalue()}")
        return out.getvalue(), err.getvalue()

    def j(self, *argv, expect=0):
        out, err = self.run_cli("--json", *argv, expect=expect)
        return json.loads(out) if out.strip() else None, err

    # --- sync and search ---------------------------------------------------
    def test_first_sync_walks_the_zone_then_the_change_feed_takes_over(self):
        r, _ = self.j("sync")
        self.assertEqual((r["mode"], r["new"], r["relations"], r["people"], r["face_crops"], r["albums"]),
                         ("full", 8, 2, 2, 3, 2))
        self.assertEqual(self.j("status", "--offline")[0]["catalog"]["people"], 2)
        # nothing changed: no pages at all
        r, _ = self.j("sync")
        self.assertEqual((r["mode"], r["pages"], r["records"]), ("changes", 0, 0))
        # an edit, a deletion, a relation gone and a new person arrive through the feed
        self.cloud.assets["A002/x+y=="].caption = "renamed"
        edited = self.cloud.records_for(self.cloud.assets["A002/x+y=="])[1]
        self.cloud.push(edited, tombstone("A006/x+y=="), tombstone("A001/x+y==-IN-alb1"),
                        rec("p3", "CPLPerson", personFullNameEnc=b64("Morfar"), displayName=b64("Morfar"), verifiedType=0, personType=0))
        r, _ = self.j("sync")
        self.assertEqual((r["mode"], r["records"], r["changed"], r["missing"], r["tombstones"], r["people"]),
                         ("changes", 4, 1, 1, 2, 1))
        self.assertEqual(self.j("info", "A002/x+y==")[0]["caption"], "renamed")
        info, _ = self.j("info", "A006/x+y==")
        self.assertTrue(info["missing"])
        res, _ = self.j("search")
        self.assertNotIn("A006/x+y==", [a["id"] for a in res["results"]])
        res, _ = self.j("search", "--include-missing")
        self.assertIn("A006/x+y==", [a["id"] for a in res["results"]])
        self.assertEqual([a["id"] for a in self.j("search", "--album", "Trip")[0]["results"]], ["A002/x+y=="])
        people, _ = self.j("people")
        self.assertEqual([(p["name"], p["face_crops"], bool(p["verified"])) for p in people],
                         [("Thea-Oline", 2, True), ("Ingjerd Thürmer", 1, True), ("Morfar", 0, False)])

    def test_an_asset_arriving_before_its_master_is_still_paired(self):
        # whole_zone alternates master-first and asset-first; every asset must land
        self.j("sync")
        self.assertEqual(self.j("status", "--offline")[0]["catalog"]["assets"], 8)
        # a master edited later re-derives its asset
        a = self.cloud.assets["A001/x+y=="]; a.width = 100
        self.cloud.push(self.cloud.records_for(a)[0])
        r, _ = self.j("sync")
        self.assertEqual(r["changed"], 1)
        self.assertEqual(self.j("info", "A001/x+y==")[0]["width"], 100)

    def test_an_interrupted_sync_resumes_from_the_last_committed_page(self):
        class Boom(Exception):
            pass
        real = self.cloud.iter_zone
        def flaky(since, zone=None):
            for n, page in enumerate(real(since, zone)):
                if n == 2:
                    raise Boom()
                yield page
        self.cloud.iter_zone = flaky
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            from unittest import mock
            with mock.patch.dict(os.environ, {"ICLOUD_PHOTOS_DEBUG": "1"}), self.assertRaises(Boom):   # debug mode re-raises
                cli.main(["--json", "sync"], adapter_factory=lambda app: self.cloud)
        s, _ = self.j("status", "--offline")
        self.assertIsNotNone(s["sync_progress"])
        self.assertEqual(s["sync_progress"]["pages"], 2)
        self.cloud.iter_zone = real
        # the fake's incremental feed is `pending`, so hand it the rest of the zone
        self.cloud.pending = self.cloud.whole_zone()[2 * self.cloud.page_size:]
        r, _ = self.j("sync")
        self.assertTrue(r["resumed"])
        self.assertEqual(self.j("status", "--offline")[0]["catalog"]["assets"], 8)

    def test_a_recently_deleted_asset_is_missing_and_hidden_is_hidden(self):
        self.cloud.add(make_asset(30, deleted=True), make_asset(31, hidden=True))
        self.j("sync")
        ids = [a["id"] for a in self.j("search", "--limit", "100")[0]["results"]]
        self.assertNotIn("A030/x+y==", ids); self.assertNotIn("A031/x+y==", ids)
        self.assertIn("A031/x+y==", [a["id"] for a in self.j("search", "--include-hidden", "--limit", "100")[0]["results"]])
        self.assertTrue(self.j("info", "A030/x+y==")[0]["missing"])

    def test_search_filters_and_pages_newest_first(self):
        self.j("sync")
        res, _ = self.j("search", "--limit", "3")
        ids = [a["id"] for a in res["results"]]
        self.assertEqual(ids, ["A009/x+y==", "A007/x+y==", "A006/x+y=="])
        self.assertIsNotNone(res["next_cursor"])
        res2, _ = self.j("search", "--limit", "3", "--cursor", res["next_cursor"])
        self.assertEqual([a["id"] for a in res2["results"]], ["A005/x+y==", "A004/x+y==", "A003/x+y=="])
        page3, _ = self.j("search", "--limit", "3", "--cursor", res2["next_cursor"])
        self.assertEqual(len(page3["results"]), 2)
        self.assertIsNone(page3["next_cursor"])
        self.assertEqual([a["id"] for a in self.j("search", "beach")[0]["results"]], ["A004/x+y=="])
        self.assertEqual([a["id"] for a in self.j("search", "--favorite")[0]["results"]], ["A003/x+y=="])
        self.assertEqual([a["id"] for a in self.j("search", "--located")[0]["results"]], ["A005/x+y=="])
        self.assertEqual([a["id"] for a in self.j("search", "--kind", "movie")[0]["results"]], ["A009/x+y=="])
        self.assertEqual(len(self.j("search", "--album", "Trip")[0]["results"]), 2)
        self.assertEqual(len(self.j("search", "--album", "alb1")[0]["results"]), 2)
        # --since/--until: A003 is taken 2019-07-23, A004 on 07-24; --until includes its whole day
        r, _ = self.j("search", "--since", "2019-07-23", "--until", "2019-07-24")
        self.assertEqual([a["id"] for a in r["results"]], ["A004/x+y==", "A003/x+y=="])
        r, _ = self.j("search", "--since", "2019-07-23", "--until", "2019-07-23")
        self.assertEqual([a["id"] for a in r["results"]], ["A003/x+y=="])
        r, _ = self.j("search", "--since", "2019-07", "--until", "2019-07")
        self.assertEqual(len(r["results"]), 8)
        _, err = self.j("search", "--since", "yesterday", expect=1)
        self.assertEqual(json.loads(err)["error"], "bad-date")

    # --- previews, originals, cache ----------------------------------------
    def test_show_fetches_once_and_returns_paths(self):
        self.j("sync")
        r, _ = self.j("show", "A001/x+y==", "A009/x+y==")
        self.assertEqual([x["version"] for x in r], ["thumb", "thumb_image"])
        for x in r:
            self.assertTrue(Path(x["path"]).exists(), x)
            self.assertFalse(x["cached"])
        self.assertTrue(r[0]["path"].endswith(".jpg"))
        self.assertTrue(r[0]["path"].startswith(self.tmp.name))
        r2, _ = self.j("show", "A001/x+y==")
        self.assertTrue(r2[0]["cached"])
        self.assertEqual(self.cloud.downloads.count(("A001/x+y==", "thumb")), 1)
        out, _ = self.run_cli("show", "A001/x+y==", "--size", "medium")
        self.assertRegex(out, r"^A001/x\+y==\t.*/medium/A001_x_y__\.jpg\n$")

    def test_show_falls_back_when_the_size_is_missing(self):
        small = make_asset(40, versions={"original": {"bytes": 300_000, "type": "public.png", "filename": "a.PNG"},
                                         "thumb": {"bytes": 20_000, "type": "public.jpeg", "filename": "a.JPG"}})
        big = make_asset(41, versions={"original": {"bytes": 30_000_000, "type": "public.heic", "filename": "b.HEIC"},
                                       "thumb": {"bytes": 20_000, "type": "public.jpeg", "filename": "b.JPG"}})
        self.cloud.add(small, big)
        self.j("sync")
        r, _ = self.j("show", "A040/x+y==", "A041/x+y==", "--size", "medium")
        self.assertEqual([x["version"] for x in r], ["original", "thumb"])
        self.assertTrue(r[0]["path"].endswith(".png"))

    def test_original_pin_evict_and_the_budget(self):
        self.j("sync")
        self.j("config", "set", "cache_budget_mb", "7")  # room for two originals, not three
        r, _ = self.j("original", "A001/x+y==", "--pin")
        self.assertTrue(r[0]["pinned"])
        self.j("original", "A002/x+y==")
        # third original: the unpinned A002 goes, the pinned A001 stays
        self.j("original", "A003/x+y==")
        u, _ = self.j("cache", "status")
        self.assertEqual(u["by_version"]["original"]["files"], 2)
        self.assertIsNone(self.j("info", "A002/x+y==")[0]["cached"] or None)
        self.assertEqual([c["version"] for c in self.j("info", "A001/x+y==")[0]["cached"]], ["original"])
        # pin everything: now the fourth cannot fit and says so, exit 4
        self.j("cache", "pin", "A003/x+y==")
        _, err = self.j("original", "A004/x+y==", expect=4)
        self.assertEqual(json.loads(err)["error"], "cache-full")
        self.assertIn("cache_budget_mb", json.loads(err)["message"])
        # evict respects pins unless told otherwise, and never reaches the cloud
        before = len(self.cloud.downloads)
        self.assertEqual(self.j("cache", "evict", "--all")[0]["evicted"], 0)
        self.assertEqual(self.j("cache", "evict", "--all", "--include-pinned")[0]["evicted"], 2)
        self.assertEqual(len(self.cloud.downloads), before)
        self.assertEqual(len(self.cloud.assets), 8)
        self.assertEqual(list((Path(self.tmp.name) / "cache" / "original").iterdir()), [])

    def test_a_file_removed_behind_our_back_is_refetched(self):
        self.j("sync")
        r, _ = self.j("show", "A001/x+y==")
        Path(r[0]["path"]).unlink()
        r, _ = self.j("show", "A001/x+y==")
        self.assertFalse(r[0]["cached"])
        self.assertTrue(Path(r[0]["path"]).exists())

    def test_live_photo_original_points_at_the_video_half(self):
        self.cloud.add(make_asset(20, live=True, versions=make_asset(20).versions | {
            "original_video": {"bytes": 1000, "type": "com.apple.quicktime-movie", "filename": "IMG_0020.MOV"}}))
        self.j("sync")
        r, _ = self.j("original", "A020/x+y==")
        self.assertIn("original_video", r[0]["note"])
        r, _ = self.j("original", "A020/x+y==", "--version", "original_video")
        self.assertTrue(r[0]["path"].endswith(".mov"))
        _, err = self.j("original", "A001/x+y==", "--version", "sidecar", expect=1)
        self.assertEqual(json.loads(err)["error"], "no-such-version")

    # --- collections -------------------------------------------------------
    def test_collections_keep_order_and_survive_eviction(self):
        self.j("sync")
        self.j("collection", "create", "book", "--note", "summer")
        self.j("collection", "add", "book", "A003/x+y==", "A001/x+y==")
        self.j("collection", "add", "book", "A001/x+y==")  # duplicate ignored
        r, _ = self.j("collection", "show", "book")
        self.assertEqual([i["id"] for i in r["items"]], ["A003/x+y==", "A001/x+y=="])
        self.assertEqual(self.j("collection", "list")[0][0]["count"], 2)
        self.j("show", "A003/x+y==")
        self.j("cache", "evict", "--all")
        self.assertEqual(self.j("collection", "show", "book")[0]["count"], 2)
        self.assertEqual(len(self.j("search", "--collection", "book")[0]["results"]), 2)
        _, err = self.j("collection", "add", "nope", "A001/x+y==", expect=1)
        self.assertEqual(json.loads(err)["error"], "unknown-collection")
        _, err = self.j("collection", "add", "book", "A999", expect=1)
        self.assertEqual(json.loads(err)["error"], "unknown-asset")
        self.j("collection", "delete", "book")
        self.assertEqual(self.j("collection", "list")[0], [])
        self.assertEqual(len(self.cloud.assets), 8)

    # --- index, faces, semantic search ---------------------------------------
    def with_models(self):
        from icloud_photos import index as ix
        from icloud_photos import ml
        models = StubModels()
        # decode_image: the stub reads a key from pixel 0; map bytes -> a 2x2 image whose pixel carries the key
        def decode(data):
            key = ((len(data) * 7 + data[-1]) if data else 0) % 250
            return np.full((2, 2, 3), key, dtype=np.uint8)
        self._decode_patch = (ix, ix.decode_image)
        ix.decode_image = decode
        self.addCleanup(lambda: setattr(ix, "decode_image", self._decode_patch[1]))
        return models, decode

    def key_of(self, asset_id, version="thumb"):
        data = self.cloud.download(asset_id, version, self.cloud.assets[asset_id].master_id)
        self.cloud.downloads.pop()
        return (len(data) * 7 + data[-1]) % 250

    def run_ix(self, *argv, models, expect=0):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(["--json", *argv], adapter_factory=lambda app: self.cloud, models_factory=lambda app: models)
        self.assertEqual(rc, expect, f"argv={argv}\nstdout={out.getvalue()}\nstderr={err.getvalue()}")
        return json.loads(out.getvalue()) if out.getvalue().strip() else None, err.getvalue()

    def test_index_seeds_people_embeds_images_and_names_faces(self):
        models, decode = self.with_models()
        thea = StubModels._unit(1); ingjerd = StubModels._unit(2); stranger = StubModels._unit(3)
        # the seed crops: fc1, fc2 -> p1 (Thea), fc3 -> p2 (Ingjerd); a crop's "image" carries the key of its bytes
        for crop, vec in (("fc1", thea), ("fc2", thea), ("fc3", ingjerd)):
            models.faces_by_key[(len(b"CROP:" + crop.encode()) * 7 + ord(crop[-1])) % 250] = [vec]
        self.j("sync")
        # faces in photos: A001 has Thea + a stranger, A002 has Ingjerd (near), A003 none
        near_ingjerd = ingjerd + 0.3 * StubModels._unit(9); near_ingjerd /= np.linalg.norm(near_ingjerd)
        models.faces_by_key[self.key_of("A001/x+y==")] = [thea, stranger]
        models.faces_by_key[self.key_of("A002/x+y==")] = [near_ingjerd]
        r, _ = self.run_ix("index", models=models)
        self.assertEqual((r["seed"]["crops"], r["seed"]["seeded"]), (3, 3))
        # 7 images; the movie is not indexed in pass 1
        self.assertEqual((r["index"]["todo"], r["index"]["done"], r["index"]["faces"], r["index"]["named"]), (7, 7, 3, 2))
        s, _ = self.j("status", "--offline")
        self.assertEqual((s["index"]["indexed"], s["index"]["faces_named"], s["index"]["seeded_people"]), (7, 2, 2))
        # a second run has nothing to do
        r, _ = self.run_ix("index", models=models)
        self.assertEqual(r["index"]["todo"], 0)
        # who is where
        faces, _ = self.j("faces", "show", "A001/x+y==")
        self.assertEqual(sorted(str(f["person_name"]) for f in faces), ["None", "Thea-Oline"])
        self.assertEqual([a["id"] for a in self.j("search", "--person", "Thea-Oline")[0]["results"]], ["A001/x+y=="])
        self.assertEqual([a["id"] for a in self.j("search", "--person", "Ingjerd")[0]["results"]], ["A002/x+y=="])
        people, _ = self.j("people")
        self.assertEqual({p["name"]: p["photos"] for p in people}, {"Thea-Oline": 1, "Ingjerd Thürmer": 1})
        _, err = self.j("search", "--person", "Nobody", expect=1)
        self.assertEqual(json.loads(err)["error"], "unknown-person")
        # the stranger: unassigned, then assigned by hand, which seeds the person and survives re-indexing
        un, _ = self.j("faces", "unassigned")
        self.assertEqual(len(un), 1)
        self.j("faces", "assign", str(un[0]["id"]), "Ingjerd")
        self.assertEqual(len(self.j("search", "--person", "Ingjerd")[0]["results"]), 2)
        self.assertEqual(self.j("status", "--offline")[0]["index"]["seeds"], 4)
        # semantic search: the query vector equals A003's image vector, so A003 ranks first with score ~1
        models.text_vectors["a unicorn"] = StubModels._unit(self.key_of("A003/x+y=="))
        r, _ = self.run_ix("search", "--semantic", "a unicorn", "--limit", "3", models=models)
        self.assertEqual(r["results"][0]["id"], "A003/x+y==")
        self.assertGreater(r["results"][0]["score"], 0.99)
        self.assertEqual(len(r["results"]), 3)
        # ...and filters still apply on top of the ranking
        r, _ = self.run_ix("search", "--semantic", "a unicorn", "--favorite", models=models)
        self.assertEqual([a["id"] for a in r["results"]], ["A003/x+y=="])
        r, _ = self.run_ix("search", "--similar", "A003/x+y==", "--limit", "2", models=models)
        self.assertEqual(r["results"][0]["id"], "A003/x+y==")
        _, err = self.run_ix("search", "--similar", "A999", models=models, expect=1)
        self.assertEqual(json.loads(err)["error"], "not-indexed")

    def test_index_survives_a_failed_download_and_the_cache_budget(self):
        models, _ = self.with_models()
        self.j("sync")
        self.j("config", "set", "cache_budget_mb", "0")
        broken = self.cloud.assets["A005/x+y=="]; broken.versions = {"original": broken.versions["original"]}
        r, _ = self.run_ix("index", models=models)
        self.assertEqual((r["index"]["done"], r["index"]["failed"]), (6, 1))
        self.assertEqual(self.j("cache", "status")[0]["files"], 0)

    # --- status, errors, help ----------------------------------------------
    def test_status_reports_without_a_sync_and_after_one(self):
        s, _ = self.j("status")
        self.assertEqual((s["authenticated"], s["catalog"]["assets"], s["last_sync"]), (True, 0, None))
        self.j("sync")
        s, _ = self.j("status", "--offline")
        self.assertEqual((s["authenticated"], s["catalog"]["assets"], s["last_sync"]["mode"]), (None, 8, "full"))
        out, _ = self.run_cli("status", "--offline")
        self.assertIn("8 assets", out)
        self.cloud.logged_in = False
        s, _ = self.j("status")
        self.assertFalse(s["authenticated"])

    def test_errors_are_one_line_and_json_when_asked(self):
        out, err = self.run_cli("info", "nope", expect=1)
        self.assertEqual(out, "")
        self.assertEqual(err.count("\n"), 1)
        self.assertTrue(err.startswith("photos: no asset nope"))
        _, err = self.j("info", "nope", expect=1)
        self.assertEqual(json.loads(err), {"error": "unknown-asset",
                                           "message": "no asset nope in the catalogue; run `photos sync` or check the id"})

    def test_help_is_complete_from_the_installed_command(self):
        exe = Path(sys.executable).parent / "photos"
        if not exe.exists():
            self.skipTest("package not installed in this interpreter")
        proc = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for word in ("--json", "search", "show", "original", "collection", "login"):
            self.assertIn(word, proc.stdout)
        proc = subprocess.run([str(exe), "search", "--help"], capture_output=True, text=True, timeout=30)
        self.assertIn("--cursor", proc.stdout)

    def test_login_refuses_without_a_terminal(self):
        _, err = self.j("login", expect=2)
        self.assertEqual(json.loads(err)["error"], "needs-terminal")

    def test_background_sync_runs_detached_and_refuses_to_double_up(self):
        import time
        from unittest import mock
        lock = Path(self.tmp.name) / "state" / "sync.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(os.getpid()))
        _, err = self.j("sync", "--background", expect=1)
        self.assertEqual(json.loads(err)["error"], "sync-running")
        lock.unlink()
        # the detached child is a real `photos` with no session: it must end quickly
        # with a not-logged-in error in the log, not hang or prompt
        with mock.patch.dict(os.environ, {"ICLOUD_PHOTOS_WORKER": "plain"}):
            r, _ = self.j("sync", "--background")
        self.assertIsNone(r["unit"])
        for _ in range(300):
            try:
                os.kill(r["pid"], 0)
            except ProcessLookupError:
                break
            try:
                if os.waitpid(r["pid"], os.WNOHANG)[0]:
                    break
            except ChildProcessError:
                break
            time.sleep(0.1)
        else:
            self.fail("background sync did not finish")
        log = Path(r["log"]).read_text()
        self.assertIn('"error": "not-logged-in"', log)
        self.assertFalse(lock.exists())

    def test_background_sync_under_systemd_gets_its_own_unit(self):
        import shutil, time
        if os.environ.get("ICLOUD_PHOTOS_WORKER") == "plain":
            self.skipTest("ICLOUD_PHOTOS_WORKER=plain in this environment")
        if not shutil.which("systemd-run") or subprocess.run(
                ["systemd-run", "--user", "--quiet", "--collect", "--wait", "true"], capture_output=True).returncode:
            self.skipTest("no usable systemd user manager")
        r, _ = self.j("sync", "--background")
        self.assertTrue(r["unit"].startswith("icloud-photos-sync-"))
        for _ in range(300):
            if "not-logged-in" in Path(r["log"]).read_text():
                break
            time.sleep(0.1)
        else:
            self.fail("background sync under systemd did not finish")
        self.assertFalse((Path(self.tmp.name) / "state" / "sync.lock").exists())


if __name__ == "__main__":
    unittest.main()


class RetryTests(unittest.TestCase):
    def test_retry_recovers_from_dropped_connections_then_gives_up(self):
        calls, slept = [], []
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionResetError(104, "Connection reset by peer")
            return "ok"
        self.assertEqual(retry(flaky, "lookup", delays=(1, 2, 3), sleep=slept.append), "ok")
        self.assertEqual(slept, [1, 2])
        def down():
            raise OSError("no route")
        with self.assertRaises(CloudError) as ctx:
            retry(down, "lookup", delays=(1, 2), sleep=slept.append)
        self.assertIn("after 3 attempts", str(ctx.exception))
        self.assertEqual(slept, [1, 2, 1, 2])
        def not_cloud():
            raise CloudError("bad answer")
        with self.assertRaises(CloudError):
            retry(not_cloud, "lookup", delays=(1,), sleep=slept.append)
        self.assertEqual(slept, [1, 2, 1, 2])   # our own errors are not retried


class ComputeTests(unittest.TestCase):
    def test_cpu_mode_and_missing_provider(self):
        import sys
        from unittest import mock

        from icloud_photos.ml import ComputeUnavailable, resolve_compute
        self.assertEqual(resolve_compute("cpu"), ("cpu", "configured"))
        with mock.patch.dict(sys.modules, {"onnxruntime_ggml": None}):   # import fails
            self.assertEqual(resolve_compute("auto")[0], "cpu")
            self.assertIn("not installed", resolve_compute("auto")[1])
            with self.assertRaises(ComputeUnavailable):
                resolve_compute("gpu")
        with self.assertRaises(ValueError):
            resolve_compute("tpu")

    def test_broken_provider_falls_back_with_reason(self):
        import sys, types
        from unittest import mock

        from icloud_photos.ml import ComputeUnavailable, resolve_compute
        fake = types.ModuleType("onnxruntime_ggml")
        fake.__version__ = "0.0"
        def boom(*a, **k):
            raise RuntimeError("device=gpu requested but no gpu backend is available")
        fake.InferenceSession = boom
        with mock.patch.dict(sys.modules, {"onnxruntime_ggml": fake}):
            resolved, detail = resolve_compute("auto")
            self.assertEqual(resolved, "cpu")
            self.assertIn("no gpu backend", detail)
            with self.assertRaises(ComputeUnavailable):
                resolve_compute("gpu")



class LapExportTests(unittest.TestCase):
    """The export is checked against lap's real schema (tests/fixtures/lap_schema.sql)."""

    def setUp(self):
        self.base = IndexTests() if "IndexTests" in globals() else None

    def test_export_writes_files_thumbs_embeddings_faces_and_collections(self):
        import sqlite3
        case = [c for c in globals().values() if isinstance(c, type) and issubclass(c, unittest.TestCase)
                and hasattr(c, "run_ix") and hasattr(c, "with_models")][0]
        t = case("test_index_seeds_people_embeds_images_and_names_faces")
        t.setUp()
        try:
            t.test_index_seeds_people_embeds_images_and_names_faces()
            models, _ = t.with_models()
            t.j("collection", "create", "book")
            first = t.j("search", "--limit", "1")[0]["results"][0]["id"]
            t.j("collection", "add", "book", first)
            lap_db = Path(t.tmp.name) / "lap.db"
            con = sqlite3.connect(lap_db)
            con.executescript(Path(__file__).with_name("fixtures").joinpath("lap_schema.sql").read_text())
            con.close()
            root = Path(t.tmp.name) / "lap-root"
            r, _ = t.run_ix("lap-export", "--library", str(lap_db), "--root", str(root), models=models)
            con = sqlite3.connect(lap_db)
            files = con.execute("SELECT COUNT(*) FROM afiles").fetchone()[0]
            self.assertEqual(files, r["files"])
            self.assertGreater(files, 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM afiles WHERE embeds IS NOT NULL").fetchone()[0], r["embeddings"])
            self.assertEqual(con.execute("SELECT COUNT(*) FROM athumbs WHERE error_code=0").fetchone()[0], r["thumbs"])
            self.assertEqual(con.execute("SELECT COUNT(*) FROM faces").fetchone()[0], r["faces"])
            self.assertGreater(r["faces"], 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM persons").fetchone()[0], r["people"])
            self.assertTrue(con.execute("SELECT COUNT(*) FROM persons WHERE cover_face_id IS NOT NULL").fetchone()[0] >= 1)
            self.assertEqual(con.execute("SELECT name FROM acollections").fetchone()[0], "book")
            self.assertEqual(con.execute("SELECT COUNT(*) FROM acollections_files").fetchone()[0], 1)
            album = con.execute("SELECT path, total FROM albums").fetchone()
            self.assertEqual((album[0], album[1]), (str(root), files))
            links = [p for p in root.rglob("*") if p.is_symlink()]
            self.assertEqual(len(links), r["linked"])
            bbox = json.loads(con.execute("SELECT bbox FROM faces LIMIT 1").fetchone()[0])
            self.assertEqual(set(bbox), {"x", "y", "width", "height", "confidence"})
            # lap-fetch: an entry whose file is missing gets its rendition fetched and linked
            folder, name = con.execute("SELECT b.path, a.name FROM afiles a JOIN afolders b ON a.folder_id=b.id WHERE a.file_type=1 LIMIT 1").fetchone()
            entry = Path(folder) / name
            self.assertFalse(entry.exists())
            f, _ = t.run_ix("lap-fetch", str(entry), models=models)
            self.assertTrue(entry.exists())
            self.assertEqual(os.readlink(entry), f["path"])
            # the album is marked so the app never scans it away, and the entry is
            # named after the rendition it stands for
            self.assertEqual(con.execute("SELECT managed FROM albums").fetchone()[0], 1)
            self.assertTrue(r["managed"])
            names = [n for (n,) in con.execute("SELECT name FROM afiles")]
            self.assertTrue(all("@" in n for n in names), names[:3])
            # idempotent: a second run changes no counts
            r2, _ = t.run_ix("lap-export", "--library", str(lap_db), "--root", str(root), models=models)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM afiles").fetchone()[0], files)
            self.assertEqual(r2["files"], r["files"])
            con.close()
        finally:
            t.tearDown()


class InternalErrorTests(unittest.TestCase):
    def test_unexpected_exception_is_one_line_with_a_code(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"ICLOUD_PHOTOS_HOME": tmp}):
            os.environ.pop("ICLOUD_PHOTOS_DEBUG", None)
            def boom(app):
                raise RuntimeError("models exploded")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = cli.main(["--json", "search", "--semantic", "x"], models_factory=boom)
            self.assertEqual(rc, 1)
            payload = json.loads(err.getvalue().strip().splitlines()[-1])
            self.assertEqual(payload["error"], "internal-error")
            self.assertIn("RuntimeError: models exploded", payload["message"])
            self.assertNotIn("Traceback", err.getvalue())


class EditTests(unittest.TestCase):
    """`edit` hands the photo to an agent; the agent itself is stubbed here."""

    def test_slug_and_naming(self):
        from icloud_photos import edit as editing
        self.assertEqual(editing.slugify("Make it black & white!"), "make-it-black-white")
        self.assertEqual(editing.slugify(""), "edit")
        self.assertEqual(editing.slugify("a" * 80), "a" * 40)

    def test_edit_runs_the_agent_and_files_the_result(self):
        import shutil, tempfile
        from unittest import mock
        from icloud_photos import edit as editing
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "IMG_1.jpg"
            src.write_bytes(b"original")
            fake_agent = tmp / "agent"
            fake_agent.write_text("#!/bin/sh\nd=$(echo \"$@\" | tr ' ' '\\n' | grep -A0 '^/.*input' | head -1)\n"
                                  "dir=$(dirname \"$d\")\nprintf edited > \"$dir/output.jpg\"\necho 'RESULT: output.jpg'\n")
            fake_agent.chmod(0o755)
            with mock.patch.object(editing, "is_image", lambda p: p.exists() and p.stat().st_size > 0):
                r = editing.edit(src, "Make It Grey", root=tmp / "edits", agent=str(fake_agent))
            out = Path(r["path"])
            self.assertTrue(out.exists())
            self.assertEqual(out.read_bytes(), b"edited")
            self.assertEqual(out.parent.parent, tmp / "edits")
            self.assertTrue(out.name.startswith("IMG_1-make-it-grey"))
            self.assertEqual(src.read_bytes(), b"original", "the original must not be touched")
            self.assertFalse((tmp / "edits" / ".work").exists() and any((tmp / "edits" / ".work").iterdir()))

    def test_a_silent_agent_is_an_error_not_a_broken_file(self):
        import tempfile
        from icloud_photos import edit as editing
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "IMG_2.jpg"; src.write_bytes(b"x")
            quiet = tmp / "quiet"; quiet.write_text("#!/bin/sh\necho 'CANNOT: I cannot invent content'\n"); quiet.chmod(0o755)
            with self.assertRaises(editing.EditFailed) as ctx:
                editing.edit(src, "remove the car", root=tmp / "edits", agent=str(quiet))
            self.assertIn("cannot invent", str(ctx.exception).lower())

    def test_a_missing_agent_says_so(self):
        import tempfile
        from icloud_photos import edit as editing
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.jpg"; src.write_bytes(b"x")
            with self.assertRaises(editing.EditFailed) as ctx:
                editing.edit(src, "x", root=Path(tmp), agent="no-such-agent-binary")
            self.assertIn("not installed", str(ctx.exception))


# --- choosing a handful out of thousands ------------------------------------

def svec(*coords: float) -> bytes:
    """A 512-d unit vector whose first coordinates are given, so cosines are exact."""
    v = np.zeros(512, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32).tobytes()


class FakeCatalog:
    """Just the handful of methods the funnel asks for."""

    def __init__(self, rows, vecs=None, people=None):
        self.rows = rows
        self.vecs = vecs or {}
        self.people = people or {}

    def counts(self):
        return {"assets": len(self.rows)}

    def _matching(self, ids=None, favorite=None, screenshots=None, **_ignored):
        out = list(self.rows)
        if ids is not None:
            keep = set(ids)
            out = [a for a in out if a["id"] in keep]
        if favorite is not None:
            out = [a for a in out if bool(a.get("favorite")) == favorite]
        return out

    def count_assets(self, **filters):
        return len(self._matching(**filters))

    def asset_dates(self, **filters):
        rows = self._matching(**filters)
        rows.sort(key=lambda a: a.get("taken") or "")
        return [(a["id"], a.get("taken")) for a in rows]

    def search(self, limit=50, **filters):
        rows = self._matching(**filters)
        rows.sort(key=lambda a: a.get("taken") or "", reverse=True)
        return [dict(a) for a in rows[:limit]], None

    def nearest_clip(self, embedding, limit):
        q = np.frombuffer(embedding, dtype=np.float32)
        out = []
        for aid, blob in self.vecs.items():
            v = np.frombuffer(blob, dtype=np.float32)
            cos = float(np.dot(q, v))
            out.append((aid, float(np.sqrt(max(0.0, 2 - 2 * cos)))))   # cosine -> L2 on unit vectors
        out.sort(key=lambda t: t[1])
        return out[:limit]

    def clip_of(self, asset_id):
        return self.vecs.get(asset_id)

    def faces_of(self, asset_id):
        return [{"person_id": p} for p in self.people.get(asset_id, [])]


def srow(n, taken, **over):
    base = dict(id=f"S{n}", filename=f"IMG_{n}.HEIC", kind="image", taken=taken,
                favorite=False, width=4032, height=3024)
    base.update(over)
    return base


class SelectFunnelTests(unittest.TestCase):
    """The algorithm itself, with no catalogue, no models and no network."""

    def setUp(self):
        from icloud_photos import select
        self.sel = select

    def test_a_burst_collapses_to_one_frame_but_a_different_scene_does_not(self):
        rows = [srow(1, "2024-05-01T10:00:00"), srow(2, "2024-05-01T10:00:30"),
                srow(3, "2024-05-01T10:00:45"), srow(4, "2024-05-01T10:01:00")]
        vecs = {"S1": svec(1, 0), "S2": svec(0.999, 0.045),       # cosine 0.999: the same picture
                "S3": svec(0.99, 0.141),                           # cosine 0.99: still the same
                "S4": svec(0, 1)}                                  # cosine 0: a different scene
        unit = {k: np.frombuffer(v, dtype=np.float32) for k, v in vecs.items()}
        kept, removed = self.sel._collapse_duplicates(rows, unit, lambda a: 0.5)
        self.assertEqual((removed, sorted(a["id"] for a in kept)), (2, ["S1", "S4"]))

    def test_looking_alike_is_not_enough_without_being_close_in_time(self):
        rows = [srow(1, "2024-05-01T10:00:00"), srow(2, "2024-08-14T16:00:00")]
        unit = {"S1": np.frombuffer(svec(1, 0), dtype=np.float32),
                "S2": np.frombuffer(svec(0.999, 0.045), dtype=np.float32)}
        kept, removed = self.sel._collapse_duplicates(rows, unit, lambda a: 0.5)
        self.assertEqual((removed, len(kept)), (0, 2))

    def test_the_best_frame_of_a_burst_is_the_one_that_survives(self):
        rows = [srow(1, "2024-05-01T10:00:00"), srow(2, "2024-05-01T10:00:20", favorite=True)]
        unit = {"S1": np.frombuffer(svec(1, 0), dtype=np.float32),
                "S2": np.frombuffer(svec(0.999, 0.045), dtype=np.float32)}
        quality = {"S1": 0.4, "S2": 0.9}
        kept, removed = self.sel._collapse_duplicates(rows, unit, lambda a: quality[a["id"]])
        self.assertEqual((removed, kept[0]["id"]), (1, "S2"))

    def test_your_own_favourite_outranks_a_crisper_photo_nobody_marked(self):
        favourite_but_soft = self.sel.quality({"favorite": True}, 12.0)
        crisp_but_unmarked = self.sel.quality({"favorite": False}, 4000.0)
        self.assertGreater(favourite_but_soft, crisp_but_unmarked)

    def test_variety_at_zero_takes_the_top_ranked_and_at_one_spreads_out(self):
        rows = [srow(i, f"2024-05-0{i}T10:00:00") for i in range(1, 5)]
        unit = {"S1": np.frombuffer(svec(1, 0), dtype=np.float32),
                "S2": np.frombuffer(svec(0.999, 0.045), dtype=np.float32),   # nearly S1
                "S3": np.frombuffer(svec(0.99, 0.141), dtype=np.float32),    # nearly S1
                "S4": np.frombuffer(svec(0, 1), dtype=np.float32)}           # unlike all of them
        rel = {"S1": 0.99, "S2": 0.98, "S3": 0.97, "S4": 0.10}
        tight = [a["id"] for a in self.sel._mmr(rows, unit, rel, 2, 0.0, "none")]
        wide = [a["id"] for a in self.sel._mmr(rows, unit, rel, 2, 1.0, "none")]
        self.assertEqual(tight, ["S1", "S2"])          # the two best matches, near-identical
        self.assertEqual(wide, ["S1", "S4"])           # the two least alike

    def test_spreading_over_days_avoids_taking_them_all_from_one_afternoon(self):
        rows = ([srow(i, f"2024-05-01T1{i}:00:00") for i in range(1, 5)] +
                [srow(9, "2024-06-20T10:00:00")])
        unit = {a["id"]: np.frombuffer(svec(1, 0.01 * i), dtype=np.float32)
                for i, a in enumerate(rows)}
        rel = {"S1": .99, "S2": .98, "S3": .97, "S4": .96, "S9": .50}
        crowded = [a["id"] for a in self.sel._mmr(rows, unit, rel, 2, 0.0, "none")]
        spread = [a["id"] for a in self.sel._mmr(rows, unit, rel, 2, 0.0, "day")]
        self.assertEqual(crowded, ["S1", "S2"])        # both from the first of May
        self.assertEqual(spread, ["S1", "S9"])         # one from each day

    def test_everyone_asked_for_appears_even_when_the_ranking_left_them_out(self):
        rows = [srow(1, "2024-05-01T10:00:00"), srow(2, "2024-05-02T10:00:00"),
                srow(3, "2024-05-03T10:00:00")]
        who = {"S1": {"thea"}, "S2": {"thea"}, "S3": {"morfar"}}
        picked = [rows[0], rows[1]]
        rel = {"S1": .9, "S2": .8, "S3": .1}
        out, missing = self.sel._ensure_everyone(
            picked, rows, lambda i: who[i], ["thea", "morfar"], rel)
        self.assertEqual(missing, [])
        names = {p for a in out for p in who[a["id"]]}
        self.assertEqual(names, {"thea", "morfar"})

    def test_a_person_with_no_photo_at_all_is_reported_not_invented(self):
        rows = [srow(1, "2024-05-01T10:00:00")]
        out, missing = self.sel._ensure_everyone(
            list(rows), rows, lambda i: {"thea"}, ["thea", "nobody"], {"S1": .9})
        self.assertEqual((missing, len(out)), (["nobody"], 1))

    def test_the_funnel_reports_every_stage_and_narrows_to_what_was_asked(self):
        rows = [srow(i, f"2024-05-{i:02d}T10:00:00") for i in range(1, 13)]
        vecs = {f"S{i}": svec(1, 0.02 * i) for i in range(1, 13)}
        cat = FakeCatalog(rows, vecs)
        got = self.sel.run(cat, count=4, filters={}, thumb_for=None)
        self.assertEqual(len(got.picked), 4)
        self.assertEqual([s.name for s in got.stages],
                         ["library", "filters", "pool", "near-duplicates", "variety and quotas"])
        self.assertEqual(got.stages[0].kept, 12)
        self.assertIsNone(got.reason)

    def test_filters_that_match_nothing_explain_themselves_rather_than_failing(self):
        cat = FakeCatalog([srow(1, "2024-05-01T10:00:00")])
        got = self.sel.run(cat, count=4, filters={"favorite": True})
        self.assertEqual(got.picked, [])
        self.assertIn("no photo matched the filters", got.reason)
        self.assertEqual([s.name for s in got.stages], ["library", "filters"])

    def test_a_description_ranks_the_whole_library_before_the_filters_narrow_it(self):
        # The wanted photo is the oldest, so a pool of "the newest few" would miss it.
        rows = [srow(i, f"2024-05-{i:02d}T10:00:00") for i in range(1, 9)]
        vecs = {f"S{i}": svec(0, 1) for i in range(2, 9)}
        vecs["S1"] = svec(1, 0)
        cat = FakeCatalog(rows, vecs)
        got = self.sel.run(cat, query="the one", count=1, filters={},
                           embed=lambda _t: svec(1, 0))
        self.assertEqual([a["id"] for a in got.picked], ["S1"])
        self.assertIn("meaning", [s.name for s in got.stages])

    def test_a_description_nothing_reaches_is_reported_with_the_floor_that_stopped_it(self):
        rows = [srow(1, "2024-05-01T10:00:00")]
        cat = FakeCatalog(rows, {"S1": svec(0, 1)})
        got = self.sel.run(cat, query="nothing like it", count=3, filters={},
                           controls=self.sel.Controls(floor=0.9), embed=lambda _t: svec(1, 0))
        self.assertEqual(got.picked, [])
        self.assertIn("lower --floor", got.reason)


class SelectCliTests(CliTest):
    def test_select_narrows_the_library_and_can_fill_a_collection_in_order(self):
        self.j("sync")
        r, _ = self.j("select", "--count", "3")
        self.assertEqual(r["count"], 3)
        self.assertEqual([s["stage"] for s in r["stages"]][:2], ["library", "filters"])
        self.assertEqual(r["stages"][0]["kept"], 8)          # 7 images and one movie
        self.assertEqual(r["stages"][1]["kept"], 7)          # --kind defaults to image

        r, _ = self.j("select", "--count", "2", "--into", "Book")
        self.assertEqual(r["collection"], {"name": "Book", "added": 2})
        shown, _ = self.j("collection", "show", "Book")
        self.assertEqual([a["id"] for a in shown["items"]], [a["id"] for a in r["selected"]])

        # --replace empties it first, so the collection is the last answer, not every answer
        r2, _ = self.j("select", "--count", "1", "--into", "Book", "--replace")
        shown2, _ = self.j("collection", "show", "Book")
        self.assertEqual(shown2["count"], 1)
        self.assertEqual([a["id"] for a in shown2["items"]], [a["id"] for a in r2["selected"]])

    def test_select_on_favourites_only_reports_the_true_number_matched(self):
        self.j("sync")
        r, _ = self.j("select", "--favorite", "--count", "5")
        self.assertEqual(r["stages"][1]["kept"], 1)          # A003 is the only favourite
        self.assertEqual(r["count"], 1)
        self.assertIn("only 1 photos survived", r["reason"])


# --- turning a set of photos into one picture --------------------------------

def cphoto(pid, w, h, faces=()):
    from icloud_photos import compose
    return compose.Photo(pid, Path("/nowhere.jpg"), w, h, list(faces))


class ComposeGeometryTests(unittest.TestCase):
    """The layout and the crop, with no files and no ImageMagick."""

    def setUp(self):
        from icloud_photos import compose
        self.c = compose
        # a realistic mix: portrait, square, landscape and a panorama
        self.photos = [cphoto("p1", 2000, 3008), cphoto("p2", 4032, 3024),
                       cphoto("p3", 4096, 4096), cphoto("p4", 3024, 4032),
                       cphoto("p5", 6000, 3376), cphoto("p6", 4032, 3024),
                       cphoto("p7", 2480, 3508), cphoto("p8", 3840, 2160),
                       cphoto("p9", 15336, 3876)]

    def inside(self, slots, W, H):
        return [s for s in slots
                if s.x < -1 or s.y < -1 or s.x + s.w > W + 1 or s.y + s.h > H + 1]

    def test_justified_fills_the_page_and_keeps_every_slot_on_it(self):
        for n in range(1, 10):
            with self.subTest(photos=n):
                slots = self.c.justified(self.photos[:n], 1400.0, 933.0, 8.0)
                self.assertEqual(len(slots), n)
                self.assertEqual(self.inside(slots, 1400.0, 933.0), [])
                covered = sum(s.w * s.h for s in slots) / (1400.0 * 933.0)
                self.assertGreater(covered, 0.90, "a justified page should not be mostly empty")

    def test_justified_never_stretches_a_photograph(self):
        slots = self.c.justified(self.photos, 1400.0, 933.0, 0.0)
        for s in slots:
            # each row scales heights, so a slot is a crop of the photo, never a stretch
            self.assertGreater(s.w, 0)
            self.assertGreater(s.h, 0)
        rows = {}
        for s in slots:
            rows.setdefault(round(s.y), []).append(s)
        for row in rows.values():
            self.assertAlmostEqual(sum(s.w for s in row), 1400.0, delta=1.5)

    def test_a_grid_of_nine_is_three_by_three_not_four_by_three(self):
        slots = self.c.grid(self.photos[:9], 1500.0, 1000.0, 0.0)
        xs = sorted({round(s.x) for s in slots})
        ys = sorted({round(s.y) for s in slots})
        self.assertEqual((len(xs), len(ys)), (3, 3))

    def test_a_short_last_row_is_centred_rather_than_left_hanging(self):
        slots = self.c.grid(self.photos[:8], 1500.0, 1000.0, 0.0)
        last = [s for s in slots if round(s.y) == max(round(x.y) for x in slots)]
        left = min(s.x for s in last)
        right = 1500.0 - max(s.x + s.w for s in last)
        self.assertAlmostEqual(left, right, delta=1.0)

    def test_a_filmstrip_fills_the_page_instead_of_floating_in_it(self):
        slots = self.c.filmstrip(self.photos[:6], 1400.0, 933.0, 6.0)
        self.assertEqual(len({round(s.w) for s in slots}), 1)      # equal frames
        self.assertTrue(all(abs(s.h - 933.0) < 1 for s in slots))
        self.assertEqual(self.inside(slots, 1400.0, 933.0), [])

    def test_a_spread_leaves_a_gutter_that_nothing_crosses(self):
        W, H = 1600.0, 1100.0
        slots = self.c.spread(self.photos[:8], W, H, 6.0)
        gutter = max(6.0 * 2.6, W * 0.035)
        page = (W - gutter) / 2.0
        for s in slots:
            crosses = s.x < page and s.x + s.w > page + gutter
            self.assertFalse(crosses, "no photo may sit in the gutter")

    def test_scatter_keeps_every_print_on_the_page_and_repeats_exactly(self):
        a = self.c.scatter(self.photos, 1400.0, 933.0, 8.0)
        b = self.c.scatter(self.photos, 1400.0, 933.0, 8.0)
        self.assertEqual([s.as_dict() for s in a], [s.as_dict() for s in b])
        self.assertEqual(self.inside(a, 1400.0, 933.0), [])
        self.assertTrue(all(s.mount > 0 for s in a))

    def test_every_template_survives_a_single_photograph(self):
        for name, fn in self.c.TEMPLATES.items():
            with self.subTest(template=name):
                slots = fn(self.photos[:1], 1000.0, 1000.0, 8.0)
                self.assertEqual(len(slots), 1)
                self.assertEqual(self.inside(slots, 1000.0, 1000.0), [])


class CropTests(unittest.TestCase):
    def setUp(self):
        from icloud_photos import compose
        self.c = compose

    def test_with_no_faces_the_crop_is_centred(self):
        p = cphoto("p", 4000, 2000)
        w, h, x, y = self.c.crop_box(p, 1.0)
        self.assertEqual((w, h), (2000, 2000))
        self.assertEqual((x, y), (1000, 0))                 # centred on a 4000 wide frame

    def test_a_square_slot_slides_sideways_to_keep_a_face_whole(self):
        # the face sits far to the right; a centred crop would cut it
        p = cphoto("p", 4000, 2000, [(0.80, 0.30, 0.95, 0.70)])
        w, h, x, y = self.c.crop_box(p, 1.0, face_safe=True)
        self.assertEqual((w, h), (2000, 2000))
        self.assertGreaterEqual(x + w, 0.95 * 4000)         # the whole face is inside
        self.assertLessEqual(x, 0.80 * 4000)
        centred = self.c.crop_box(p, 1.0, face_safe=False)[2]
        self.assertGreater(x, centred, "face-safe should have moved the window right")

    def test_a_tall_photo_slides_up_or_down_to_keep_a_face_whole(self):
        p = cphoto("p", 2000, 4000, [(0.3, 0.05, 0.7, 0.20)])
        w, h, x, y = self.c.crop_box(p, 1.0, face_safe=True)
        self.assertEqual((w, h), (2000, 2000))
        self.assertLessEqual(y, 0.05 * 4000)
        self.assertGreaterEqual(y + h, 0.20 * 4000)

    def test_turning_face_safety_off_always_centres(self):
        p = cphoto("p", 4000, 2000, [(0.80, 0.30, 0.95, 0.70)])
        self.assertEqual(self.c.crop_box(p, 1.0, face_safe=False)[2], 1000)

    def test_a_matching_shape_is_not_cropped_at_all(self):
        p = cphoto("p", 3000, 2000)
        self.assertEqual(self.c.crop_box(p, 1.5), (3000, 2000, 0, 0))

    def test_faces_that_do_not_fit_the_rendition_are_refused_not_guessed(self):
        good = self.c.normalise_faces([[100, 50, 200, 180]], 400, 300)
        self.assertEqual(len(good), 1)
        self.assertAlmostEqual(good[0][0], 0.25)
        # a box beyond the frame means the rendition size is wrong; a bad crop is
        # worse than a centred one, so all of them are dropped
        self.assertEqual(self.c.normalise_faces([[100, 50, 900, 180]], 400, 300), [])
        self.assertEqual(self.c.normalise_faces([[1, 2, 3, 4]], None, None), [])


class ComposePlanTests(unittest.TestCase):
    def setUp(self):
        from icloud_photos import compose
        self.c = compose

    def test_the_plan_reports_the_page_and_a_crop_for_every_photo(self):
        photos = [cphoto(f"p{i}", 4032, 3024) for i in range(6)]
        got = self.c.plan(photos, template="justified", shape="a4-landscape")
        self.assertEqual((got["width"], got["height"]), (3508, 2480))
        self.assertFalse(got["draft"])
        self.assertEqual(len(got["slots"]), 6)
        self.assertTrue(all("crop" in s for s in got["slots"]))

    def test_a_smaller_page_is_marked_as_a_draft_and_says_the_print_size(self):
        got = self.c.plan([cphoto("p", 4032, 3024)], shape="3:2", long_edge=1800)
        self.assertTrue(got["draft"])
        self.assertEqual(got["print_size"], [5400, 3600])
        self.assertIn("5400x3600", got["note"])

    def test_an_unknown_template_or_shape_says_what_there_is(self):
        with self.assertRaises(self.c.ComposeFailed) as e:
            self.c.plan([cphoto("p", 10, 10)], template="mosaic")
        self.assertIn("justified", str(e.exception))
        with self.assertRaises(self.c.ComposeFailed):
            self.c.plan([cphoto("p", 10, 10)], shape="A3")
        with self.assertRaises(self.c.ComposeFailed):
            self.c.plan([], template="grid")

    def test_the_output_name_never_overwrites_and_carries_no_colon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.c.output_path(root, "Summer 2024!", "justified", "3:2")
            self.assertNotIn(":", first.name)
            self.assertEqual(first.name, "summer-2024-justified-3-2.jpg")
            first.parent.mkdir(parents=True, exist_ok=True)
            first.write_bytes(b"x")
            second = self.c.output_path(root, "Summer 2024!", "justified", "3:2")
            self.assertNotEqual(first, second)


@unittest.skipUnless(__import__("shutil").which("magick"), "ImageMagick is not installed")
class ComposeRenderTests(unittest.TestCase):
    """The one path that actually draws pixels."""

    def setUp(self):
        from icloud_photos import compose
        self.c = compose
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.photos = []
        for i, (w, h) in enumerate([(400, 300), (300, 400), (400, 400), (600, 200), (500, 375)]):
            f = self.dir / f"src{i}.jpg"
            subprocess.run(["magick", "-size", f"{w}x{h}",
                            f"gradient:#{i}{i}0000-#00{i}{i}00", str(f)], check=True)
            self.photos.append(self.c.Photo(f"p{i}", f, w, h, []))
        self.addCleanup(self.tmp.cleanup)

    def test_a_page_is_drawn_at_the_size_asked_for(self):
        out = self.dir / "page.jpg"
        got = self.c.render(self.photos, out, template="justified", shape="3:2", long_edge=600)
        self.assertTrue(out.exists())
        self.assertGreater(got["bytes"], 0)
        size = subprocess.run(["magick", "identify", "-format", "%wx%h", str(out)],
                              capture_output=True, text=True).stdout
        self.assertEqual(size, f"{got['width']}x{got['height']}")

    def test_every_template_draws_something(self):
        for name in self.c.TEMPLATES:
            with self.subTest(template=name):
                out = self.dir / f"{name}.jpg"
                self.c.render(self.photos, out, template=name, shape="square", long_edge=400)
                self.assertGreater(out.stat().st_size, 0)

    def test_the_same_recipe_twice_gives_the_same_picture(self):
        a, b = self.dir / "a.png", self.dir / "b.png"
        self.c.render(self.photos, a, template="scatter", shape="3:2", long_edge=500)
        self.c.render(self.photos, b, template="scatter", shape="3:2", long_edge=500)
        # compare the picture, which is what the promise is about
        out = subprocess.run(["magick", "compare", "-metric", "AE", str(a), str(b), "null:"],
                             capture_output=True, text=True)
        self.assertEqual((out.stderr.strip().split()[0], a.read_bytes()), ("0", b.read_bytes()))

    def test_pages_bind_into_one_pdf_and_a_missing_page_is_refused(self):
        pages = []
        for i in range(3):
            out = self.dir / f"p{i}.jpg"
            self.c.render(self.photos, out, template="grid", shape="a4-landscape", long_edge=500)
            pages.append(out)
        book = self.dir / "book.pdf"
        got = self.c.export_pdf(pages, book, dpi=150)
        self.assertEqual(got["pages"], 3)
        self.assertGreater(book.stat().st_size, 0)
        sizes = subprocess.run(["magick", "identify", "-format", "%wx%h\n", str(book)],
                               capture_output=True, text=True).stdout.split()
        self.assertEqual(len(set(sizes)), 1, "every page of a PDF has one size")
        with self.assertRaises(self.c.ComposeFailed):
            self.c.export_pdf([], book)
        with self.assertRaises(self.c.ComposeFailed):
            self.c.export_pdf([self.dir / "never-drawn.jpg"], book)

    def test_a_caption_is_written_inside_the_page(self):
        out = self.dir / "captioned.jpg"
        plain = self.dir / "plain.jpg"
        self.c.render(self.photos, plain, shape="a4-landscape", long_edge=600)
        got = self.c.render(self.photos, out, shape="a4-landscape", long_edge=600,
                            caption="A day at the beach")
        self.assertGreater(got["caption_band"], 0)
        self.assertEqual(self.c.image_size(out), self.c.image_size(plain))

    def test_a_missing_source_is_reported_not_a_blank_page(self):
        gone = [self.c.Photo("x", self.dir / "not-here.jpg", 100, 100, [])]
        with self.assertRaises(self.c.ComposeFailed):
            self.c.render(gone, self.dir / "bad.jpg", long_edge=300)


class ComposeCliTests(CliTest):
    def test_compose_refuses_a_set_it_cannot_use_and_says_why(self):
        self.j("sync")
        _, err = self.j("compose", "Nope", expect=1)
        self.assertIn("unknown-collection", err)
        _, err = self.j("compose", expect=1)
        self.assertIn("nothing-to-compose", err)
        # a movie is not composed, and saying so beats drawing an empty page
        _, err = self.j("compose", "--id", "A009/x+y==", expect=1)
        self.assertIn("nothing-to-compose", err)


# --- books -------------------------------------------------------------------

class CaptionTests(unittest.TestCase):
    def setUp(self):
        from icloud_photos import compose
        self.c = compose

    def test_a_caption_takes_room_inside_the_page_not_below_it(self):
        photos = [cphoto(f"p{i}", 4032, 3024) for i in range(4)]
        plain = self.c.plan(photos, shape="a4-landscape")
        titled = self.c.plan(photos, shape="a4-landscape", caption="A day at the beach")
        # the page is the same size either way, or a PDF could not bind the two
        self.assertEqual((plain["width"], plain["height"]), (titled["width"], titled["height"]))
        self.assertEqual(plain["caption_band"], 0)
        self.assertGreater(titled["caption_band"], 0)
        # and the photographs have moved up to make room
        lowest = max(s["y"] + s["h"] for s in titled["slots"])
        self.assertLessEqual(lowest, titled["height"] - titled["caption_band"] + 1)

    def test_an_empty_caption_reserves_nothing(self):
        got = self.c.plan([cphoto("p", 100, 100)], caption="   ")
        self.assertEqual(got["caption_band"], 0)


class BookTests(CliTest):
    def test_a_book_is_pages_in_order_that_can_be_moved_and_dropped(self):
        self.j("sync")
        self.j("collection", "create", "Trip")
        self.j("collection", "add", "Trip", "A001/x+y==", "A002/x+y==")

        r, _ = self.j("book", "create", "Summer", "--shape", "a4-landscape")
        self.assertTrue(r["created"])
        r, _ = self.j("book", "create", "Summer")
        self.assertFalse(r["created"], "creating twice must not add a second book")

        for template, caption in (("justified", "Day one"), ("grid", None), ("hero", "The last one")):
            args = ["book", "add", "Summer", "--collection", "Trip", "--template", template]
            if caption:
                args += ["--caption", caption]
            self.j(*args)
        shown, _ = self.j("book", "show", "Summer")
        self.assertEqual([p["template"] for p in shown["pages"]], ["justified", "grid", "hero"])
        self.assertEqual([p["position"] for p in shown["pages"]], [1, 2, 3])
        self.assertEqual(shown["shape"], "a4-landscape")

        self.j("book", "move", "Summer", "--page", "3", "--to", "1")
        shown, _ = self.j("book", "show", "Summer")
        self.assertEqual([p["template"] for p in shown["pages"]], ["hero", "justified", "grid"])
        self.assertEqual([p["position"] for p in shown["pages"]], [1, 2, 3])

        self.j("book", "remove", "Summer", "--page", "2")
        shown, _ = self.j("book", "show", "Summer")
        self.assertEqual([p["template"] for p in shown["pages"]], ["hero", "grid"])
        self.assertEqual([p["position"] for p in shown["pages"]], [1, 2], "pages must renumber")

        listed, _ = self.j("book", "list")
        self.assertEqual((listed[0]["name"], listed[0]["pages"]), ("Summer", 2))

    def test_a_page_can_name_its_photos_instead_of_a_collection(self):
        self.j("sync")
        self.j("book", "create", "Direct")
        self.j("book", "add", "Direct", "--id", "A001/x+y==", "A002/x+y==")
        shown, _ = self.j("book", "show", "Direct")
        self.assertEqual(shown["pages"][0]["ids"], ["A001/x+y==", "A002/x+y=="])
        self.assertIsNone(shown["pages"][0]["collection"])

    def test_the_things_that_cannot_work_say_so(self):
        self.j("sync")
        _, err = self.j("book", "show", "Nope", expect=1)
        self.assertIn("unknown-book", err)
        self.j("book", "create", "Empty")
        _, err = self.j("book", "add", "Empty", expect=1)
        self.assertIn("nothing-on-the-page", err)
        _, err = self.j("book", "add", "Empty", "--collection", "Missing", expect=1)
        self.assertIn("unknown-collection", err)
        _, err = self.j("book", "export", "Empty", expect=1)
        self.assertIn("empty-book", err)
        _, err = self.j("book", "remove", "Empty", "--page", "7", expect=1)
        self.assertIn("no-such-page", err)

    def test_deleting_a_book_leaves_the_photos_and_the_collection_alone(self):
        self.j("sync")
        self.j("collection", "create", "Trip")
        self.j("collection", "add", "Trip", "A001/x+y==")
        self.j("book", "create", "Gone")
        self.j("book", "add", "Gone", "--collection", "Trip")
        self.j("book", "delete", "Gone")
        self.assertEqual(self.j("book", "list")[0], [])
        still, _ = self.j("collection", "show", "Trip")
        self.assertEqual(still["count"], 1)


class SelectPoolAndCapturesTests(unittest.TestCase):
    """The two things that made a life story come out as this year's screenshots."""

    def setUp(self):
        from icloud_photos import select
        self.sel = select

    def test_the_pool_spans_the_whole_range_not_the_newest_slice(self):
        # eighteen years, and far more photos than the pool can hold
        rows, vecs = [], {}
        n = 0
        for year in range(2008, 2026):
            for i in range(60):
                n += 1
                rows.append(srow(n, f"{year}-06-{(i % 28) + 1:02d}T10:00:00"))
                vecs[f"S{n}"] = svec(1, 0.001 * n)
        cat = FakeCatalog(rows, vecs)
        got = self.sel.run(cat, count=18, filters={},
                           controls=self.sel.Controls(spread="year"))
        years = {(a["taken"] or "")[:4] for a in got.picked}
        self.assertEqual(len(got.picked), 18)
        self.assertGreaterEqual(len(years), 15, f"a life story cannot be one year: {sorted(years)}")
        self.assertIn("2008", years)
        self.assertIn("2025", years)

    def test_without_a_spread_the_pool_still_reaches_the_oldest_photos(self):
        rows = [srow(i, f"{2008 + i // 40}-06-01T10:00:00") for i in range(1, 400)]
        cat = FakeCatalog(rows, {f"S{i}": svec(1, 0.001 * i) for i in range(1, 400)})
        got = self.sel.run(cat, count=4, filters={})
        pool = next(s for s in got.stages if s.name == "pool")
        self.assertIn("spread over the whole range", pool.note)


class ScreenCaptureTests(CliTest):
    def test_a_png_is_a_screenshot_and_is_set_aside_unless_asked_for(self):
        self.cloud.assets["A002/x+y=="].filename = "IMG_0002.PNG"
        self.cloud.assets["A004/x+y=="].filename = "Screenshot.png"
        self.j("sync")

        r, _ = self.j("select", "--count", "9")
        names = [a["filename"] for a in r["selected"]]
        self.assertNotIn("IMG_0002.PNG", names)
        self.assertNotIn("Screenshot.png", names)
        self.assertIn("2 screen captures set aside", r["stages"][1]["note"])

        r, _ = self.j("select", "--count", "9", "--include-screenshots")
        names = [a["filename"] for a in r["selected"]]
        self.assertIn("IMG_0002.PNG", names)
        self.assertNotIn("screen captures set aside", r["stages"][1]["note"])

    def test_search_is_left_alone_because_it_is_not_choosing_for_a_book(self):
        self.cloud.assets["A002/x+y=="].filename = "IMG_0002.PNG"
        self.j("sync")
        r, _ = self.j("search")
        self.assertIn("IMG_0002.PNG", [a["filename"] for a in r["results"]])


# --- grouping faces so naming is done a group at a time -----------------------

class FakeFaceCatalog:
    """A catalogue with just the tables the clustering touches."""

    def __init__(self, faces):
        import sqlite3
        # autocommit, the way the real catalogue opens it, so BEGIN is ours to give
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE faces (id INTEGER PRIMARY KEY, asset_id TEXT, embedding BLOB,
                                det REAL, person_id TEXT, similarity REAL, assigned TEXT,
                                cluster INTEGER);
            CREATE TABLE assets (id TEXT PRIMARY KEY, taken TEXT);
            CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT);
            CREATE TABLE person_seeds (id TEXT PRIMARY KEY, person_id TEXT, embedding BLOB,
                                       det REAL, origin TEXT);
        """)
        for fid, asset, taken, vec, person in faces:
            self.db.execute("INSERT OR IGNORE INTO assets (id, taken) VALUES (?,?)", (asset, taken))
            self.db.execute(
                "INSERT INTO faces (id, asset_id, embedding, det, person_id, assigned) VALUES (?,?,?,?,?,?)",
                (fid, asset, vec, 0.9, person, "auto" if person else None))
            if person:
                self.db.execute("INSERT OR IGNORE INTO people (id, name) VALUES (?,?)", (person, person))

    def put_seed(self, seed_id, person_id, embedding, det, origin):
        self.db.execute(
            "INSERT OR REPLACE INTO person_seeds (id, person_id, embedding, det, origin) VALUES (?,?,?,?,?)",
            (seed_id, person_id, embedding, det, origin))


def fvec(*coords):
    v = np.zeros(512, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    return (v / float(np.linalg.norm(v))).astype(np.float32).tobytes()


def chain_vec(step, total=1.0):
    """A vector rotated `step` notches; neighbours are close, distant ones are not."""
    import math
    a = step * total
    v = np.zeros(512, dtype=np.float32)
    v[0], v[1] = math.cos(a), math.sin(a)
    return v.tobytes()


class ClusterTests(unittest.TestCase):
    def setUp(self):
        from icloud_photos import cluster
        self.c = cluster

    def test_a_chain_of_ages_becomes_one_group_though_its_ends_do_not_match(self):
        # eight faces, each close to the next, the first and last far apart
        faces = [(i + 1, f"A{i}", f"{2008 + i}-06-01T10:00:00", chain_vec(i, 0.28), None)
                 for i in range(8)]
        cat = FakeFaceCatalog(faces)
        import numpy as _np
        first = _np.frombuffer(faces[0][3], dtype=_np.float32)
        last = _np.frombuffer(faces[-1][3], dtype=_np.float32)
        self.assertLess(float(first @ last), 0.62, "the ends must not match directly")
        got = self.c.build(cat, threshold=0.62)
        self.assertEqual(got["clusters"], 1)
        self.assertEqual(got["grouped"], 8)

    def test_two_people_stay_apart(self):
        faces = ([(i + 1, f"A{i}", "2020-01-01T10:00:00", fvec(1, 0.02 * i), None) for i in range(5)] +
                 [(i + 10, f"B{i}", "2020-01-01T10:00:00", fvec(0, 1, 0.02 * i), None) for i in range(5)])
        got = self.c.build(FakeFaceCatalog(faces), threshold=0.62)
        self.assertEqual(got["clusters"], 2)

    def test_a_lone_face_is_left_out_of_every_group(self):
        faces = [(1, "A", "2020-01-01T10:00:00", fvec(1, 0), None),
                 (2, "B", "2020-01-01T10:00:00", fvec(1, 0.01), None),
                 (3, "C", "2020-01-01T10:00:00", fvec(0, 0, 1), None)]
        got = self.c.build(FakeFaceCatalog(faces), threshold=0.62)
        self.assertEqual((got["clusters"], got["grouped"], got["loose"]), (1, 2, 1))

    def test_a_group_reports_the_person_most_of_its_named_faces_carry(self):
        faces = [(i + 1, f"A{i}", f"{2010 + i}-05-01T10:00:00", fvec(1, 0.01 * i),
                  "julie" if i < 4 else (None if i < 6 else "oline")) for i in range(7)]
        cat = FakeFaceCatalog(faces)
        self.c.build(cat, threshold=0.62)
        g = self.c.groups(cat)[0]
        self.assertEqual(g.person_id, "julie")
        self.assertEqual(g.size, 7)
        self.assertEqual((g.first[:4], g.last[:4]), ("2010", "2016"))

    def test_naming_a_group_leaves_a_face_that_already_names_someone_else(self):
        faces = [(1, "A", "2010-01-01T10:00:00", fvec(1, 0), None),
                 (2, "B", "2014-01-01T10:00:00", fvec(1, 0.01), None),
                 (3, "C", "2018-01-01T10:00:00", fvec(1, 0.02), "oline")]
        cat = FakeFaceCatalog(faces)
        self.c.build(cat, threshold=0.62)
        got = self.c.name(cat, 1, "julie", seeds=4)
        self.assertEqual((got["faces"], got["left_alone"]), (2, 1))
        who = dict(cat.db.execute("SELECT id, person_id FROM faces").fetchall())
        self.assertEqual(who[3], "oline", "a sister must not be renamed by resemblance")
        self.assertEqual((who[1], who[2]), ("julie", "julie"))

    def test_forcing_it_does_rename_them(self):
        faces = [(1, "A", "2010-01-01T10:00:00", fvec(1, 0), None),
                 (2, "B", "2018-01-01T10:00:00", fvec(1, 0.01), "oline")]
        cat = FakeFaceCatalog(faces)
        self.c.build(cat, threshold=0.62)
        got = self.c.name(cat, 1, "julie", force=True)
        self.assertEqual((got["faces"], got["left_alone"]), (2, 0))

    def test_seeds_are_taken_across_the_whole_range_not_from_one_end(self):
        faces = [(i + 1, f"A{i}", f"{2008 + i}-06-01T10:00:00", fvec(1, 0.005 * i), None)
                 for i in range(16)]
        cat = FakeFaceCatalog(faces)
        self.c.build(cat, threshold=0.62)
        self.c.name(cat, 1, "julie", seeds=4)
        taken = [r[0] for r in cat.db.execute(
            "SELECT a.taken FROM person_seeds s JOIN faces f ON ('cluster-1-' || f.id) = s.id "
            "JOIN assets a ON a.id = f.asset_id ORDER BY a.taken")]
        self.assertEqual(len(taken), 4)
        self.assertLess(taken[0][:4], "2011")
        self.assertGreater(taken[-1][:4], "2018")

    def test_unnaming_undoes_the_group_but_not_a_hand_made_name(self):
        faces = [(1, "A", "2010-01-01T10:00:00", fvec(1, 0), None),
                 (2, "B", "2014-01-01T10:00:00", fvec(1, 0.01), None)]
        cat = FakeFaceCatalog(faces)
        self.c.build(cat, threshold=0.62)
        self.c.name(cat, 1, "julie")
        cat.db.execute("UPDATE faces SET assigned='manual' WHERE id=1")
        cleared = self.c.unname(cat, 1)
        who = dict(cat.db.execute("SELECT id, person_id FROM faces").fetchall())
        self.assertEqual((cleared, who[1], who[2]), (1, "julie", None))
        self.assertEqual(cat.db.execute("SELECT COUNT(*) FROM person_seeds").fetchone()[0], 0)


class SharedLibraryTests(CliTest):
    """A shared library is a second zone; a library that ignores it misses its photos."""

    def add_shared(self, *numbers):
        name = "SharedSync-TEST"
        self.cloud.shared_zones = [name]
        pages = []
        for n in numbers:
            asset = make_asset(n, filename=f"SHARED_{n:04d}.HEIC")
            self.cloud.shared_assets[asset.id] = asset
            pages += self.cloud.records_for(asset)
        self.cloud.shared_pages[name] = pages
        return name

    def test_photos_in_a_shared_library_are_synced_and_remember_where_they_live(self):
        name = self.add_shared(21, 22)
        r, _ = self.j("sync")
        self.assertIn(name, r["zones"])
        self.assertTrue(r["zones"][name]["shared"])
        self.assertEqual(r["zones"][name]["new"], 2)

        found, _ = self.j("search", "SHARED_")
        self.assertEqual(found["count"], 2)
        zones = dict(self.j("search", "SHARED_")[0]["results"][0].items()).get("zone")
        self.assertEqual(zones, name, "a photo must remember the zone that holds it")
        own = self.j("search", "IMG_0001")[0]["results"][0]
        self.assertEqual(own["zone"], "PrimarySync")

    def test_each_zone_keeps_its_own_place_so_one_does_not_rewalk_the_other(self):
        name = self.add_shared(21)
        self.j("sync")
        r, _ = self.j("sync")
        self.assertEqual(r["records"], 0, "nothing changed anywhere")
        # a new photo in the shared library alone
        extra = make_asset(31, filename="SHARED_0031.HEIC")
        self.cloud.shared_assets[extra.id] = extra
        self.cloud.shared_pages[name] = self.cloud.records_for(extra)
        r, _ = self.j("sync")
        self.assertEqual(r["zones"][name]["new"], 1)

    def test_no_shared_skips_them(self):
        name = self.add_shared(21, 22)
        r, _ = self.j("sync", "--no-shared")
        self.assertNotIn(name, r["zones"])
        self.assertEqual(self.j("search", "SHARED_")[0]["count"], 0)

    def test_a_shared_library_that_will_not_answer_does_not_cost_the_user_their_own(self):
        name = self.add_shared(21)

        real = self.cloud.iter_zone

        def refuse(since, zone=None):
            if getattr(zone, "name", None) == name:
                raise RuntimeError("shared zone is unavailable")
            yield from real(since, zone)

        self.cloud.iter_zone = refuse
        r, _ = self.j("sync")
        self.assertIn("unavailable", r["zones"][name]["error"])
        self.assertGreater(r["new"], 0, "the user's own library still synced")
        self.assertEqual(self.j("status", "--offline")[0]["catalog"]["assets"], 8)

    def test_a_download_is_asked_for_in_the_zone_that_holds_the_photo(self):
        name = self.add_shared(21)
        self.j("sync")
        asked = []
        real = self.cloud.download

        def note(asset_id, version, master_id=None, zone=None):
            asked.append((asset_id, zone))
            return real(asset_id, version, master_id, zone)

        self.cloud.download = note
        shared_id = next(a["id"] for a in self.j("search", "SHARED_")[0]["results"])
        self.j("show", shared_id)
        self.j("show", "A001/x+y==")
        self.assertEqual(dict(asked)[shared_id], name)
        self.assertEqual(dict(asked)["A001/x+y=="], "PrimarySync")

    def test_a_catalogue_from_before_zones_does_not_rewalk_its_own_library(self):
        self.j("sync")
        # what an older version left behind: one unnamed cursor, no per-zone map
        import sqlite3
        db = next(Path(self.tmp.name).rglob("catalog.db"))
        con = sqlite3.connect(db)
        self.assertIsNotNone(con.execute("SELECT value FROM meta WHERE key='cursor'").fetchone(),
                             "the old key must still be kept for catalogues that predate zones")
        con.execute("DELETE FROM meta WHERE key='cursors'")
        con.commit()
        con.close()
        r, _ = self.j("sync")
        self.assertEqual(r["mode"], "changes", "an old cursor still means an incremental sync")
        self.assertEqual(r["records"], 0, "and nothing is walked again")

    def test_the_catalogue_waits_for_a_lock_instead_of_failing_at_it(self):
        # Opening a photo while a sync runs used to fail with "database is locked",
        # which the app could only report as a missing or corrupt file.
        from icloud_photos.catalog import BUSY_TIMEOUT_S
        self.assertGreaterEqual(BUSY_TIMEOUT_S, 30,
                                "five seconds is not long enough to outlast a sync's page")
        self.j("sync")
        from icloud_photos.cli import App
        app = App(json_mode=True)
        try:
            self.assertEqual(app.catalog.db.execute("PRAGMA busy_timeout").fetchone()[0],
                             int(BUSY_TIMEOUT_S * 1000))
        finally:
            app.close()


# --- getting a book ready for a print shop -----------------------------------

class PreflightTests(unittest.TestCase):
    """The check that comes before the money."""

    def setUp(self):
        from icloud_photos import printing
        self.pr = printing
        self.a4 = printing.PROFILES["a4-landscape"]

    def full_page(self, pid="p"):
        w, h = self.a4.page_px
        return {"width": w, "height": h,
                "slots": [{"id": pid, "x": 0, "y": 0, "w": w, "h": h,
                           "crop": {"w": w, "h": h, "x": 0, "y": 0}}]}

    def test_a_page_size_in_millimetres_becomes_the_right_pixels(self):
        self.assertEqual(self.a4.page_px, (3508, 2480))       # A4 at 300 dpi
        self.assertEqual(self.pr.PROFILES["a4-portrait"].page_px, (2480, 3508))
        self.assertEqual(self.a4.bleed_px, 35)                # 3 mm

    def test_a_photograph_with_enough_pixels_passes(self):
        found = self.pr.check_page(1, self.full_page(), {"p": 4032}, {}, self.a4)
        self.assertEqual(found, [])

    def test_one_that_will_print_soft_is_flagged_and_a_smudge_stops_the_book(self):
        # 2200 px across 297 mm is 188 dpi: past soft, short of hopeless
        soft = self.pr.check_page(1, self.full_page(), {"p": 2200}, {}, self.a4)
        self.assertEqual([f.severity for f in soft], ["look"])
        self.assertIn("soft", soft[0].message)

        bad = self.pr.check_page(1, self.full_page(), {"p": 1200}, {}, self.a4)
        self.assertEqual([f.severity for f in bad], ["stop"])
        self.assertIn("smudge", bad[0].message)

    def test_the_measurement_is_pixels_per_inch_of_the_space_it_fills(self):
        # 3508 px across 297 mm is 300 dpi; half the pixels is half the dpi
        page = self.full_page()
        self.assertEqual(self.pr.check_page(1, page, {"p": 3508}, {}, self.a4), [])
        half = self.pr.check_page(1, page, {"p": 1754}, {}, self.a4)
        self.assertIn("150 dpi", half[0].message)

    def test_a_small_photograph_in_a_small_slot_is_fine(self):
        # the same photograph that fails full-page passes at a quarter of the width
        w, h = self.a4.page_px
        page = {"width": w, "height": h,
                "slots": [{"id": "p", "x": 0, "y": 0, "w": w // 4, "h": h // 4,
                           "crop": {"w": 800, "h": 600, "x": 0, "y": 0}}]}
        self.assertEqual(self.pr.check_page(1, page, {"p": 1200}, {}, self.a4), [])

    def test_a_face_reaching_the_trim_stops_the_book(self):
        safe = self.pr.check_page(1, self.full_page(), {"p": 4032},
                                  {"p": [[0.4, 0.4, 0.6, 0.6]]}, self.a4)
        self.assertEqual(safe, [])
        edge = self.pr.check_page(1, self.full_page(), {"p": 4032},
                                  {"p": [[0.001, 0.4, 0.06, 0.6]]}, self.a4)
        self.assertEqual([f.kind for f in edge], ["trim"])
        self.assertEqual(edge[0].severity, "stop")

    def test_a_face_in_the_spine_of_a_spread_is_worth_a_look(self):
        spread = self.pr.PROFILES["a4-spread"]
        w, h = spread.page_px
        page = {"width": w, "height": h,
                "slots": [{"id": "p", "x": 0, "y": 0, "w": w, "h": h,
                           "crop": {"w": w, "h": h, "x": 0, "y": 0}}]}
        found = self.pr.check_page(1, page, {"p": 6000}, {"p": [[0.47, 0.3, 0.53, 0.5]]}, spread)
        self.assertEqual([f.kind for f in found], ["gutter"])
        self.assertEqual(found[0].severity, "look")

    def test_nothing_cached_to_measure_is_said_rather_than_assumed_fine(self):
        found = self.pr.check_page(1, self.full_page(), {}, {}, self.a4)
        self.assertEqual([f.severity for f in found], ["look"])
        self.assertIn("nothing cached", found[0].message)

    def test_a_book_is_ready_only_when_nothing_stops_it(self):
        stop = self.pr.Finding(1, "resolution", "stop", "too few pixels")
        look = self.pr.Finding(2, "gutter", "look", "in the spine")
        clean = self.pr.check_book([{"findings": [look]}], self.a4)
        self.assertTrue(clean["ready"])
        self.assertEqual((clean["stop"], clean["look"]), (0, 1))
        blocked = self.pr.check_book([{"findings": [stop, look]}], self.a4)
        self.assertFalse(blocked["ready"])

    def test_a_shop_that_counts_pages_is_obeyed(self):
        r = self.pr.check_book([{"findings": []}] * 6, self.a4, multiple_of=4, minimum=8)
        self.assertFalse(r["ready"])
        msgs = " ".join(f["message"] for f in r["findings"])
        self.assertIn("at least 8", msgs)
        self.assertIn("multiple of 4", msgs)


class PreflightCliTests(CliTest):
    def test_preflight_reports_the_page_it_would_print_on(self):
        self.j("sync")
        self.j("collection", "create", "Trip")
        self.j("collection", "add", "Trip", "A001/x+y==", "A002/x+y==")
        self.j("book", "create", "Summer")
        self.j("book", "add", "Summer", "--collection", "Trip")
        _, err = self.j("book", "preflight", "Summer", "--profile", "nonesuch", expect=1)
        self.assertIn("unknown-profile", err)

    def test_preflight_refuses_a_book_with_no_pages(self):
        self.j("sync")
        self.j("book", "create", "Empty")
        _, err = self.j("book", "preflight", "Empty", expect=1)
        self.assertIn("empty-book", err)

    def test_refresh_skips_a_sync_that_is_already_running_instead_of_failing(self):
        self.j("sync")
        app = cli.App(json_mode=True)
        lock = app.paths.sync_lock
        app.close()
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(os.getpid()))          # a sync "in progress"
        try:
            r, _ = self.j("refresh", "--no-index")
            self.assertIn("already running", r["sync"]["skipped"])
        finally:
            lock.unlink(missing_ok=True)
