"""Choose a handful of good photos out of thousands.

Ranking by similarity to a description gives you nine frames of one moment. That
is the single reason automatic selection usually disappoints, and it is what this
module exists to fix. Selection is a funnel, not a ranking: each stage throws
away a different kind of redundancy, and every stage reports what it removed, so
an empty result can always be explained.

    filters -> meaning -> near-duplicates -> quality -> variety and quotas

The last stage is the interesting one. Instead of taking the top N it picks
greedily, each time taking the photo that is most relevant *minus* how much it
resembles what has already been chosen (maximal marginal relevance). One control
sets that trade-off: at 0 you get the tightest possible match to the query, at 1
you get the widest spread.

Nothing here touches the network, and the models are optional: without a text
embedder there is no meaning stage, and the rest of the funnel still works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# How wide to cast before narrowing. Enough that dedupe and quotas have room to
# throw things away, bounded so a query never drags the whole library into memory.
POOL_PER_WANTED = 30
POOL_MIN = 200
POOL_MAX = 2000
# How deep to look for nearest neighbours before the filters narrow them.
NEAREST_SCAN_MAX = 6000

# Two photos are the same picture when they look alike *and* were taken moments
# apart. Either test alone is wrong: a burst is not a duplicate of the same wall
# photographed a year later, and a lens cap is not a duplicate of a sunset.
DUPLICATE_COSINE = 0.93
DUPLICATE_SECONDS = 90.0

# Deliberately large enough to beat a typical gap in relevance: a photo from an
# unused day wins unless the crowded day's photo is more than this much better.
SPREAD_PENALTY = 0.5

Embedder = Callable[[str], bytes]
ThumbFor = Callable[[str], Any]        # asset id -> a path to read, or None


@dataclass
class Controls:
    """The five switches, in the words the interface uses."""

    variety: float = 0.45            # 0 = closest match, 1 = widest spread
    spread: str = "none"             # none | day | month | year: don't take them all from one afternoon
    everyone: bool = False           # every person asked for appears at least once
    sharp_only: bool = False         # drop soft frames
    duplicates: bool = False         # keep near-duplicates instead of collapsing them
    screenshots: bool = False        # keep screen captures, which a photo book does not want
    floor: float = 0.0               # minimum meaning score, when there is a query


@dataclass
class Stage:
    """One step of the funnel, for the report."""

    name: str
    kept: int
    note: str


@dataclass
class Selection:
    picked: list[dict[str, Any]] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    reason: str | None = None        # set when the funnel came out empty or short

    def as_dict(self) -> dict[str, Any]:
        return {"selected": self.picked, "count": len(self.picked),
                "stages": [{"stage": s.name, "kept": s.kept, "note": s.note} for s in self.stages],
                "reason": self.reason}


def _unit(b: bytes | None) -> np.ndarray | None:
    if not b:
        return None
    v = np.frombuffer(b, dtype=np.float32).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def _taken(a: dict[str, Any]) -> datetime | None:
    t = a.get("taken")
    if not t:
        return None
    try:
        return datetime.fromisoformat(str(t)[:19])
    except ValueError:
        return None


def _bucket(a: dict[str, Any], how: str) -> str:
    t = a.get("taken")
    if not t:
        return "undated"
    s = str(t)
    return s[:4] if how == "year" else s[:7] if how == "month" else s[:10] if how == "day" else "all"


def sharpness(path: Any) -> float | None:
    """Variance of the Laplacian, the usual cheap focus measure. None if unreadable."""
    try:
        import cv2
    except Exception:
        return None
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if max(img.shape) > 512:                       # same scale for every photo
            k = 512.0 / max(img.shape)
            img = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))))
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return None


def quality(a: dict[str, Any], sharp: float | None) -> float:
    """0..1, and the user's own mark outranks anything measured.

    Weighted so a favourite always beats a photo nobody marked, however crisp:
    focus can only move the score 0.45, and marking it yourself is worth 0.55, so
    the two bands never overlap.
    """
    focus = 0.5
    if sharp is not None:
        # 0 is a blank wall, ~1000 is crisp detail; the knee sits around 120.
        focus = 1.0 / (1.0 + math.exp(-(math.log10(max(sharp, 1.0)) - 2.08) * 3.2))
    return 0.45 * focus + 0.55 * (1.0 if a.get("favorite") else 0.0)


def _collapse_duplicates(pool: Sequence[dict[str, Any]], vecs: dict[str, np.ndarray],
                         best: Callable[[dict[str, Any]], float]) -> tuple[list[dict[str, Any]], int]:
    """One keeper per burst: alike, and taken within a minute and a half."""
    kept: list[dict[str, Any]] = []
    groups: list[tuple[dict[str, Any], np.ndarray | None, datetime | None]] = []
    removed = 0
    for a in pool:
        v, t = vecs.get(a["id"]), _taken(a)
        hit = None
        if v is not None and t is not None:
            for i, (rep, rv, rt) in enumerate(groups):
                if rv is None or rt is None:
                    continue
                if abs((t - rt).total_seconds()) > DUPLICATE_SECONDS:
                    continue
                if float(np.dot(v, rv)) >= DUPLICATE_COSINE:
                    hit = i
                    break
        if hit is None:
            groups.append((a, v, t))
            continue
        removed += 1
        rep, rv, rt = groups[hit]
        if best(a) > best(rep):                        # the sharper, or the favourite, wins
            groups[hit] = (a, v, t)
    kept = [g[0] for g in groups]
    return kept, removed


def _mmr(pool: list[dict[str, Any]], vecs: dict[str, np.ndarray], rel: dict[str, float],
         want: int, variety: float, spread: str, spread_weight: float = SPREAD_PENALTY) -> list[dict[str, Any]]:
    """Greedy pick: most relevant, least like what is already chosen."""
    lam = 1.0 - max(0.0, min(1.0, variety))
    picked: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    remaining = list(pool)
    while remaining and len(picked) < want:
        best_a, best_score = None, -1e9
        for a in remaining:
            v = vecs.get(a["id"])
            sim = 0.0
            if v is not None and picked:
                sim = max((float(np.dot(v, vecs[p["id"]])) for p in picked if p["id"] in vecs), default=0.0)
            score = lam * rel.get(a["id"], 0.5) - (1.0 - lam) * sim
            if spread != "none":
                score -= spread_weight * used.get(_bucket(a, spread), 0)
            if score > best_score:
                best_a, best_score = a, score
        picked.append(best_a)
        remaining.remove(best_a)
        used[_bucket(best_a, spread)] = used.get(_bucket(best_a, spread), 0) + 1
    return picked


def _ensure_everyone(picked: list[dict[str, Any]], pool: list[dict[str, Any]],
                     people_of: Callable[[str], set[str]], required: Iterable[str],
                     rel: dict[str, float]) -> tuple[list[dict[str, Any]], list[str]]:
    """Swap in a photo for anyone asked for who did not make the cut."""
    missing = []
    for person in required:
        if any(person in people_of(a["id"]) for a in picked):
            continue
        chosen = [a for a in pool if a not in picked and person in people_of(a["id"])]
        if not chosen:
            missing.append(person)
            continue
        best = max(chosen, key=lambda a: rel.get(a["id"], 0.0))
        # give up the weakest photo that is not the only one holding someone else
        droppable = sorted(picked, key=lambda a: rel.get(a["id"], 0.0))
        for cand in droppable:
            others = people_of(cand["id"])
            if all(any(p in people_of(o["id"]) for o in picked if o is not cand) for p in others):
                picked[picked.index(cand)] = best
                break
        else:
            picked.append(best)
    return picked, missing


def _pool(catalog: Any, filters: dict[str, Any], want: int, matched: int,
          spread: str) -> tuple[list[dict[str, Any]], str]:
    """The photographs to choose among when nothing has ranked them.

    Taking the newest few hundred is wrong for the request this exists to serve:
    asked for a life from a first birthday to an eighteenth, a pool of the newest
    photographs can only answer with this year. So the pool is drawn across the
    whole range instead, and when a spread is asked for, across its buckets.
    """
    if matched <= want:
        rows, _ = catalog.search(limit=max(matched, 1), **filters)
        return rows, f"all {matched:,}"

    dated = catalog.asset_dates(**filters)
    if spread != "none":
        buckets: dict[str, list[str]] = {}
        for asset_id, taken in dated:
            buckets.setdefault(_bucket({"taken": taken}, spread), []).append(asset_id)
        # round-robin, so a year with forty photographs cannot crowd out one with four
        picked: list[str] = []
        rounds = 0
        while len(picked) < want and rounds < max((len(v) for v in buckets.values()), default=0):
            for ids in buckets.values():
                if rounds < len(ids) and len(picked) < want:
                    picked.append(ids[rounds])
            rounds += 1
        note = f"{len(picked)} of {matched:,}, across {len(buckets)} {spread}s"
    else:
        step = len(dated) / float(want)
        picked = [dated[min(int(i * step), len(dated) - 1)][0] for i in range(want)]
        note = f"{len(picked)} of {matched:,}, spread over the whole range"

    rows, _ = catalog.search(ids=picked, limit=max(len(picked), 1), **filters)
    return rows, note


def run(catalog: Any, *, query: str | None = None, query_vector: bytes | None = None, count: int = 9,
        filters: dict[str, Any] | None = None, controls: Controls | None = None,
        embed: Embedder | None = None, thumb_for: ThumbFor | None = None,
        people_required: Sequence[str] = ()) -> Selection:
    """Run the funnel and report every stage."""
    controls = controls or Controls()
    filters = dict(filters or {})
    count = max(1, count)
    out = Selection()

    total = catalog.counts().get("assets", 0)
    out.stages.append(Stage("library", total, "everything in the catalogue"))

    pool_size = min(max(count * POOL_PER_WANTED, POOL_MIN), POOL_MAX)

    # 1. Filters. Plain database columns, so this is exact and cheap. Counted over the
    # whole library, not over the pool, or the funnel would report its own cap.
    # A photo book wants photographs. Screen captures are excluded before anything
    # else looks at them, and the funnel says how many that was.
    if not controls.screenshots:
        filters["screenshots"] = False
    matched = catalog.count_assets(**filters)
    note = "dates, people, album, place"
    if not controls.screenshots:
        with_captures = catalog.count_assets(**{**filters, "screenshots": None})
        if with_captures > matched:
            note += f"; {with_captures - matched:,} screen captures set aside"
    out.stages.append(Stage("filters", matched, note))
    if not matched:
        out.reason = "no photo matched the filters; widen the dates or drop a filter"
        return out

    # 2. Meaning. The nearest neighbours are found across the whole library first and
    # the filters applied to those: ranking a pool of the newest few hundred would
    # answer a different question than the one asked.
    rel: dict[str, float] = {}
    # Either a description or a photograph can be the thing to be near; the rest
    # of the funnel does not care which.
    anchor = query_vector if query_vector is not None else (embed(query) if (query and embed) else None)
    if anchor is not None:
        scan = min(max(pool_size * 6, 1200), NEAREST_SCAN_MAX)
        nearest = catalog.nearest_clip(anchor, scan)
        score = {aid: 1.0 - d * d / 2.0 for aid, d in nearest}      # L2 on unit vectors -> cosine
        rows, _ = catalog.search(ids=list(score), limit=max(len(score), 1), **filters)
        rows = [a for a in rows if score.get(a["id"], -1.0) >= controls.floor]
        if not rows:
            out.stages.append(Stage("meaning", 0, f"nothing above {controls.floor:.2f} within the filters"))
            out.reason = ("no photo matched the description well enough; lower --floor, "
                          "widen the filters, or describe the picture differently")
            return out
        rows.sort(key=lambda a: -score.get(a["id"], 0.0))
        rows = rows[:pool_size]
        rel = {a["id"]: score.get(a["id"], 0.0) for a in rows}
        near = repr(query) if query else "this photo"
        out.stages.append(Stage("meaning", len(rows), f"nearest {scan:,} to {near}, then filtered"))
    else:
        rows, note = _pool(catalog, filters, pool_size, matched, controls.spread)
        rel = {a["id"]: 0.5 for a in rows}
        out.stages.append(Stage("pool", len(rows), note))
        if query:
            out.stages.append(Stage("meaning", len(rows), "skipped: no text embedder available"))

    vecs: dict[str, np.ndarray] = {}
    for a in rows:
        v = _unit(catalog.clip_of(a["id"]))
        if v is not None:
            vecs[a["id"]] = v

    # 3. Quality, measured once and reused by the next two stages.
    sharp: dict[str, float | None] = {}
    if thumb_for is not None:
        for a in rows:
            p = thumb_for(a["id"])
            sharp[a["id"]] = sharpness(p) if p is not None else None
    qual = {a["id"]: quality(a, sharp.get(a["id"])) for a in rows}

    # 4. Near-duplicates. Before the quality floor, so a burst is judged as a burst.
    if not controls.duplicates:
        rows, removed = _collapse_duplicates(rows, vecs, lambda a: qual[a["id"]])
        out.stages.append(Stage("near-duplicates", len(rows),
                                f"{removed} collapsed into their best frame"))

    if controls.sharp_only:
        before = len(rows)
        keep = [a for a in rows if sharp.get(a["id"]) is None or qual[a["id"]] >= 0.45]
        if keep:
            rows = keep
        out.stages.append(Stage("sharp only", len(rows), f"{before - len(rows)} soft frames dropped"))

    # Meaning and quality together decide relevance; your own favourites carry weight.
    use_meaning = anchor is not None
    blended = {a["id"]: (0.75 * rel[a["id"]] + 0.25 * qual[a["id"]]) if use_meaning
               else qual[a["id"]] for a in rows}

    # 5. Variety and quotas.
    picked = _mmr(rows, vecs, blended, count, controls.variety, controls.spread)
    note = f"variety {controls.variety:.2f}"
    if controls.spread != "none":
        note += f", spread by {controls.spread}"
    missing: list[str] = []
    if controls.everyone and people_required:
        people_cache: dict[str, set[str]] = {}

        def people_of(asset_id: str) -> set[str]:
            if asset_id not in people_cache:
                people_cache[asset_id] = {f["person_id"] for f in catalog.faces_of(asset_id)
                                          if f.get("person_id")}
            return people_cache[asset_id]

        picked, missing = _ensure_everyone(picked, rows, people_of, people_required, blended)
        note += ", everyone appears"
    out.stages.append(Stage("variety and quotas", len(picked), note))

    for a in picked:
        a["score"] = round(blended.get(a["id"], 0.0), 4)
        if sharp.get(a["id"]) is not None:
            a["sharpness"] = round(float(sharp[a["id"]]), 1)
    out.picked = picked
    if len(picked) < count:
        out.reason = (f"only {len(picked)} photos survived the funnel; "
                      "raise --limit on the filters or turn off --sharp-only")
    if missing:
        out.reason = ((out.reason + "; ") if out.reason else "") + \
            "no photo found for: " + ", ".join(missing)
    return out
