"""Group the faces that belong to the same person, so naming is a hundred
decisions instead of thirty thousand.

This is the piece that makes a person findable across a childhood. Comparing
every face against a handful of reference faces cannot work over eighteen years:
a five-year-old does not resemble an eighteen-year-old closely enough to pass any
threshold that is not also wrong about everyone else.

Clustering sidesteps that. Faces are joined to their nearest neighbours, and the
joins are transitive: five resembles six, six resembles eight, eight resembles
eleven, and so the whole chain becomes one group even though its ends do not
resemble each other at all. A group that contains one already-named face names
the rest of the chain for free.

Nothing here needs a model. The embeddings were computed during indexing; this is
arithmetic over vectors that already exist, and it runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

Progress = Callable[[str], None]

# Cosine similarity at which two faces are taken to be the same person. Higher
# than the naming threshold on purpose: a chain multiplies mistakes, so one bad
# link can weld two people together.
DEFAULT_THRESHOLD = 0.62
# How many neighbours one face may be joined to. Bounds the work and stops a
# single crowded frame from pulling in everything it half-resembles.
MAX_LINKS = 24
# Rows compared at a time. 1024 x 37k floats is about 150 MB, which fits a
# machine with no swap.
BLOCK = 1024


@dataclass
class Group:
    id: int
    size: int
    person_id: str | None
    person_name: str | None
    named_faces: int
    first: str | None
    last: str | None
    faces: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {"cluster": self.id, "faces": self.size, "person_id": self.person_id,
                "person": self.person_name, "named_faces": self.named_faces,
                "first_seen": self.first, "last_seen": self.last,
                "sample_faces": self.faces[:8]}


class _Union:
    """Union-find with path halving; plain Python is fast enough at this size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _load(catalog: Any) -> tuple[list[int], np.ndarray, list[str | None]]:
    ids: list[int] = []
    people: list[str | None] = []
    rows = catalog.db.execute(
        "SELECT id, person_id, embedding FROM faces WHERE embedding IS NOT NULL ORDER BY id")
    vecs = []
    for r in rows:
        v = np.frombuffer(r["embedding"], dtype=np.float32)
        if v.size != 512:
            continue
        n = float(np.linalg.norm(v))
        if n == 0.0:
            continue
        ids.append(int(r["id"]))
        people.append(r["person_id"])
        vecs.append(v / n)
    return ids, (np.stack(vecs) if vecs else np.zeros((0, 512), np.float32)), people


def build(catalog: Any, *, threshold: float = DEFAULT_THRESHOLD, min_size: int = 2,
          progress: Progress = lambda _s: None) -> dict[str, Any]:
    """Join every face to its nearest neighbours and write the groups back."""
    ids, vecs, people = _load(catalog)
    n = len(ids)
    if n == 0:
        return {"faces": 0, "clusters": 0, "grouped": 0}
    progress(f"comparing {n:,} faces")
    uf = _Union(n)
    links = 0
    for start in range(0, n, BLOCK):
        block = vecs[start:start + BLOCK]
        sims = block @ vecs.T
        for i in range(block.shape[0]):
            row = sims[i]
            here = start + i
            row[here] = -1.0                       # never link a face to itself
            near = np.flatnonzero(row >= threshold)
            if near.size > MAX_LINKS:
                near = near[np.argsort(row[near])[-MAX_LINKS:]]
            for j in near:
                uf.union(here, int(j))
                links += 1
        progress(f"  {min(start + BLOCK, n):,} of {n:,}")
    del vecs

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    keep = {root: members for root, members in groups.items() if len(members) >= min_size}

    catalog.db.execute("BEGIN")
    catalog.db.execute("UPDATE faces SET cluster = NULL")
    grouped = 0
    for cluster_id, (root, members) in enumerate(
            sorted(keep.items(), key=lambda kv: -len(kv[1])), start=1):
        catalog.db.executemany("UPDATE faces SET cluster=? WHERE id=?",
                               [(cluster_id, ids[m]) for m in members])
        grouped += len(members)
    catalog.db.execute("COMMIT")
    return {"faces": n, "links": links, "clusters": len(keep), "grouped": grouped,
            "loose": n - grouped, "threshold": threshold}


def groups(catalog: Any, *, limit: int = 50, named: bool | None = None,
           min_size: int = 2) -> list[Group]:
    """The groups, largest first. `named` filters to those that already have a name."""
    rows = catalog.db.execute(
        "SELECT f.cluster, COUNT(*) AS size, "
        "       SUM(CASE WHEN f.person_id IS NOT NULL THEN 1 ELSE 0 END) AS named_faces, "
        "       MIN(a.taken) AS first, MAX(a.taken) AS last "
        "FROM faces f JOIN assets a ON a.id = f.asset_id "
        "WHERE f.cluster IS NOT NULL GROUP BY f.cluster HAVING size >= ? "
        "ORDER BY size DESC", (min_size,))
    out: list[Group] = []
    for r in rows:
        person = catalog.db.execute(
            "SELECT f.person_id, p.name, COUNT(*) AS n FROM faces f "
            "LEFT JOIN people p ON p.id = f.person_id "
            "WHERE f.cluster=? AND f.person_id IS NOT NULL "
            "GROUP BY f.person_id ORDER BY n DESC LIMIT 1", (r["cluster"],)).fetchone()
        if named is True and person is None:
            continue
        if named is False and person is not None:
            continue
        faces = [int(x["id"]) for x in catalog.db.execute(
            "SELECT id FROM faces WHERE cluster=? ORDER BY det DESC LIMIT 8", (r["cluster"],))]
        out.append(Group(int(r["cluster"]), int(r["size"]),
                         person["person_id"] if person else None,
                         person["name"] if person else None,
                         int(r["named_faces"] or 0), r["first"], r["last"], faces))
        if len(out) >= limit:
            break
    return out


def name(catalog: Any, cluster_id: int, person_id: str, *, seeds: int = 12,
         force: bool = False) -> dict[str, Any]:
    """Give the group's faces a person, and seed from a spread of them.

    Faces that already carry a different name are left alone unless `force`: a
    group is a strong hint, not a verdict, and quietly renaming a sister because
    she resembles her sister is worse than leaving one face unnamed.

    The seeds are taken across the group's whole date range rather than from its
    largest faces. The point of naming a group is that it spans ages, and seeds
    drawn from one end would leave the other end unrecognised all over again.
    """
    rows = [dict(r) for r in catalog.db.execute(
        "SELECT f.id, f.person_id, f.embedding, f.det, a.taken FROM faces f "
        "JOIN assets a ON a.id = f.asset_id "
        "WHERE f.cluster=? ORDER BY a.taken IS NULL, a.taken", (cluster_id,))]
    if not rows:
        raise ValueError(f"no cluster {cluster_id}")
    claim = [r for r in rows if force or r["person_id"] is None or r["person_id"] == person_id]
    left = len(rows) - len(claim)
    catalog.db.execute("BEGIN")
    catalog.db.executemany(
        "UPDATE faces SET person_id=?, similarity=NULL, assigned='cluster' WHERE id=?",
        [(person_id, r["id"]) for r in claim])
    catalog.db.execute("COMMIT")

    # seed across the whole range, which is the entire point
    mine = [r for r in rows if force or r["person_id"] in (None, person_id)]
    step = max(1, len(mine) // max(1, seeds))
    added = 0
    for r in mine[::step][:seeds]:
        if r["embedding"]:
            catalog.put_seed(f"cluster-{cluster_id}-{r['id']}", person_id, r["embedding"],
                             r["det"], "cluster")
            added += 1
    return {"cluster": cluster_id, "person_id": person_id, "faces": len(claim),
            "left_alone": left, "seeds": added,
            "first": rows[0]["taken"], "last": rows[-1]["taken"]}


def unname(catalog: Any, cluster_id: int) -> int:
    """Undo a naming: only what the cluster set, never a hand-made assignment."""
    cur = catalog.db.execute(
        "UPDATE faces SET person_id=NULL, assigned=NULL WHERE cluster=? AND assigned='cluster'",
        (cluster_id,))
    catalog.db.execute("DELETE FROM person_seeds WHERE id LIKE ?", (f"cluster-{cluster_id}-%",))
    return cur.rowcount
