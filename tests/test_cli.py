"""End-to-end through the CLI with a fake iCloud.

The fake has the same shape as the real adapter (see icloud_photos.adapter)
and a small library the tests mutate to simulate edits and deletions in the
cloud. Nothing here needs the network or a session.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from icloud_photos import cli
from icloud_photos.adapter import AlbumInfo, AssetInfo, RawRecord

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
            "thumb": {"bytes": 30_000, "width": 400, "height": 300, "type": "public.jpeg", "filename": f"IMG_{n:04d}.JPG"},
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
    """Pretends to be iCloud's photo zone.

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

    def iter_zone(self, since):
        records = self.whole_zone() if since is None else self.pending
        if since is not None:
            self.pending = []
        for i in range(0, len(records), self.page_size):
            self.token += 1
            yield records[i:i + self.page_size], f"t{self.token}"
        if not records:
            return

    def asset_from_records(self, asset, master):
        info = self.assets[asset.name]
        # what a real adapter would derive from the records rather than remember
        info.favorite = bool(asset.value("isFavorite"))
        return info

    def download(self, asset_id, version, master_id=None):
        a = self.assets.get(asset_id)
        assert master_id == a.master_id, "the CLI passes the catalogue's master id so no index walk is needed"
        if a is None or version not in a.versions:
            return None
        self.downloads.append((asset_id, version))
        return b"x" * a.versions[version]["bytes"]

    def albums(self):
        return list(self.album_list)


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
        def flaky(since):
            for n, page in enumerate(real(since)):
                if n == 2:
                    raise Boom()
                yield page
        self.cloud.iter_zone = flaky
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(Boom):
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
        lock = Path(self.tmp.name) / "state" / "sync.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(os.getpid()))
        _, err = self.j("sync", "--background", expect=1)
        self.assertEqual(json.loads(err)["error"], "sync-running")
        lock.unlink()
        # the detached child is a real `photos` with no session: it must end quickly
        # with a not-logged-in error in the log, not hang or prompt
        r, _ = self.j("sync", "--background")
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


if __name__ == "__main__":
    unittest.main()
