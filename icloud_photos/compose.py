"""Turn a set of photos into one picture: a collage, a book spread, a contact sheet.

Geometry here is code, not conversation. The same photos and the same recipe
always produce the same picture, so what was approved on screen at preview size
is what comes out at print resolution. An agent may choose the template, the
order and which photo is the hero; it never draws the collage, and generated
pixels never stand in for a photograph.

Two ideas do most of the work.

**Every row spans the full width.** The justified template picks the number of
rows whose natural height lands closest to the page, then scales the row heights
to land on it exactly. Widths stay in proportion to each photo's aspect ratio, so
no photograph is ever stretched; the slot simply shows a little more or less of
it.

**No crop may cut a face.** Every slot has its own shape and almost no photo
matches it, so something must be cropped away. Cropping from the centre, as
montage tools do, eventually slices somebody's head in half. The catalogue
already knows where the faces are, so a slot can refuse any crop that cuts one.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_DIR = "~/Pictures/Photos Collages"

# Page sizes in pixels. The print ones are at 300 dpi, which is what a photo book
# asks for; the screen ones are what a slideshow or a post wants.
SHAPES: dict[str, tuple[int, int, str]] = {
    "3:2": (5400, 3600, "18 x 12 in at 300 dpi"),
    "square": (4000, 4000, "13.3 in square at 300 dpi"),
    "a4-landscape": (3508, 2480, "A4 wide at 300 dpi"),
    "a4-portrait": (2480, 3508, "A4 tall at 300 dpi"),
    "spread": (4960, 3508, "two A4 pages side by side at 300 dpi"),
    "16:9": (3840, 2160, "a screen, not a print"),
}

# Below this the render is a draft from whatever is cached; above it the sources
# have to be worth the size, which is what --originals is for.
DRAFT_LONG_EDGE = 1800


class ComposeFailed(Exception):
    """The picture could not be made."""


@dataclass
class Photo:
    """One photograph, as the layout needs to see it."""

    id: str
    path: Path
    width: int
    height: int
    faces: list[tuple[float, float, float, float]] = field(default_factory=list)  # x1,y1,x2,y2 as fractions

    @property
    def ar(self) -> float:
        return (self.width / self.height) if self.width and self.height else 1.0


@dataclass
class Slot:
    """Where one photograph sits on the page, in page pixels."""

    photo: Photo
    x: float
    y: float
    w: float
    h: float
    rotate: float = 0.0
    mount: float = 0.0          # white border, for the scatter template

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.photo.id, "x": round(self.x), "y": round(self.y),
                "w": round(self.w), "h": round(self.h),
                "rotate": round(self.rotate, 2), "mount": round(self.mount)}


# --- cropping ---------------------------------------------------------------

def crop_box(photo: Photo, slot_ar: float, face_safe: bool = True) -> tuple[int, int, int, int]:
    """The part of the source to show in a slot of this shape: (w, h, x, y) in source pixels.

    A cover crop: one axis is kept whole and the other is trimmed. The offset is
    centred unless there are faces, in which case it slides just far enough to
    hold all of them.
    """
    W, H = max(1, photo.width), max(1, photo.height)
    pa = W / H
    if abs(pa - slot_ar) < 1e-6:
        return W, H, 0, 0
    if pa > slot_ar:                                  # wider than the slot: trim the sides
        cw, ch = H * slot_ar, float(H)
        span = (cw / W)                               # fraction of the width that survives
        off = _offset(span, [(f[0], f[2]) for f in photo.faces] if face_safe else [])
        return int(round(cw)), int(ch), int(round(off * W)), 0
    cw, ch = float(W), W / slot_ar                    # taller than the slot: trim top and bottom
    span = (ch / H)
    off = _offset(span, [(f[1], f[3]) for f in photo.faces] if face_safe else [])
    return int(cw), int(round(ch)), 0, int(round(off * H))


def _offset(span: float, intervals: Sequence[tuple[float, float]]) -> float:
    """Where to start a window of width `span` in 0..1 so it holds the intervals."""
    room = max(0.0, 1.0 - span)
    if room <= 0.0:
        return 0.0
    if not intervals:
        return room / 2.0
    lo = min(a for a, _ in intervals)
    hi = max(b for _, b in intervals)
    off = _clamp((lo + hi) / 2.0 - span / 2.0, 0.0, room)
    # A face group wider than the window cannot fit; centre on it rather than
    # cutting one edge off arbitrarily.
    if hi - lo <= span:
        if lo < off:
            off = _clamp(lo - 0.01, 0.0, room)
        if hi > off + span:
            off = _clamp(hi - span + 0.01, 0.0, room)
    return off


def _clamp(v: float, a: float, b: float) -> float:
    return a if v < a else b if v > b else v


# --- templates --------------------------------------------------------------

def _partition(items: Sequence[Photo], rows: int) -> list[list[Photo]]:
    """Contiguous groups balanced by aspect weight, so row heights come out near equal."""
    n = len(items)
    if rows >= n:
        return [[p] for p in items]
    total = sum(p.ar for p in items) or 1.0
    out: list[list[Photo]] = []
    cur: list[Photo] = []
    acc = 0.0
    for i, p in enumerate(items):
        cur.append(p)
        acc += p.ar
        rows_left = rows - len(out) - 1
        items_left = n - i - 1
        boundary = (len(out) + 1) * total / rows
        if len(out) < rows - 1 and items_left >= rows_left and (acc >= boundary or items_left == rows_left):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def justified(items: Sequence[Photo], W: float, H: float, gap: float) -> list[Slot]:
    """Rows that each span the full width, filling the page exactly."""
    items = list(items)
    if not items:
        return []
    best = None
    best_err = float("inf")
    for rows in range(1, len(items) + 1):
        groups = _partition(items, rows)
        if len(groups) != rows:
            continue
        heights = [(W - gap * (len(g) - 1)) / (sum(p.ar for p in g) or 1.0) for g in groups]
        err = abs(sum(heights) + gap * (rows - 1) - H)
        if err < best_err:
            best_err, best = err, (groups, heights)
    if best is None:
        return grid(items, W, H, gap)
    groups, heights = best
    avail = H - gap * (len(groups) - 1)
    scale = avail / sum(heights) if sum(heights) else 1.0
    out: list[Slot] = []
    y = 0.0
    for group, rh in zip(groups, heights):
        row_h = rh * scale
        span = sum(p.ar for p in group) or 1.0
        usable = W - gap * (len(group) - 1)
        x = 0.0
        for p in group:
            w = usable * (p.ar / span)
            out.append(Slot(p, x, y, w, row_h))
            x += w + gap
        y += row_h + gap
    return out


def grid(items: Sequence[Photo], W: float, H: float, gap: float) -> list[Slot]:
    """Uniform cells, in the column count that best matches the page."""
    items = list(items)
    n = len(items)
    if not n:
        return []
    want = (W / H) if H else 1.0
    cols, err = 1, float("inf")
    for c in range(1, n + 1):
        r = -(-n // c)
        # A column count that leaves empty cells wastes the page, so it has to be
        # clearly better shaped to win: nine photos belong in three rows of three.
        e = abs((c / r) - want * 0.82) + 0.9 * ((c * r - n) / n)
        if e < err:
            err, cols = e, c
    rows = -(-n // cols)
    cw = (W - gap * (cols - 1)) / cols
    ch = (H - gap * (rows - 1)) / rows
    out = []
    for i, photo in enumerate(items):
        r = i // cols
        in_row = min(cols, n - r * cols)
        # centre a short last row rather than leaving a gap on one side
        indent = (W - (in_row * cw + gap * (in_row - 1))) / 2.0
        out.append(Slot(photo, indent + (i % cols) * (cw + gap), r * (ch + gap), cw, ch))
    return out


def hero(items: Sequence[Photo], W: float, H: float, gap: float) -> list[Slot]:
    """One photograph large, the rest supporting it."""
    items = list(items)
    if len(items) < 2:
        return grid(items, W, H, gap)
    first, rest = items[0], items[1:]
    if W >= H:
        hw = W * 0.615
        out = [Slot(first, 0, 0, hw, H)]
        for s in grid(rest, W - hw - gap, H, gap):
            s.x += hw + gap
            out.append(s)
    else:
        hh = H * 0.6
        out = [Slot(first, 0, 0, W, hh)]
        for s in grid(rest, W, H - hh - gap, gap):
            s.y += hh + gap
            out.append(s)
    return out


def filmstrip(items: Sequence[Photo], W: float, H: float, gap: float) -> list[Slot]:
    """One row of equal frames across the whole page, the way a contact strip runs.

    Frames are the same size rather than proportional to each photo, which is what
    separates this from a one-row grid: the strip fills the page instead of
    floating in the middle of it, and the crop is what varies.
    """
    items = list(items)
    n = len(items)
    if not n:
        return []
    w = (W - gap * (n - 1)) / n
    return [Slot(p, i * (w + gap), 0.0, w, H) for i, p in enumerate(items)]


def spread(items: Sequence[Photo], W: float, H: float, gap: float) -> list[Slot]:
    """Two facing pages with a gutter down the middle; nothing crosses it."""
    items = list(items)
    if len(items) < 2:
        return justified(items, W, H, gap)
    gutter = max(gap * 2.6, W * 0.035)
    page = (W - gutter) / 2.0
    half = -(-len(items) // 2)
    out = justified(items[:half], page, H, gap)
    for s in justified(items[half:], page, H, gap):
        s.x += page + gutter
        out.append(s)
    return out


def scatter(items: Sequence[Photo], W: float, H: float, gap: float) -> list[Slot]:
    """Prints dropped on a table, each with a white mount and a little tilt."""
    items = list(items)
    n = len(items)
    if not n:
        return []
    cols = max(1, int(round((n * W / H) ** 0.5)))
    rows = -(-n // cols)
    cw, ch = W / cols, H / rows
    size = min(cw, ch) * 0.98
    out = []
    for i, p in enumerate(items):
        c, r = i % cols, i // cols
        w = size if p.ar >= 1 else size * p.ar
        h = size / p.ar if p.ar >= 1 else size
        jx = (_noise(i * 7 + 1) - .5) * cw * .26
        jy = (_noise(i * 13 + 5) - .5) * ch * .26
        # a 10-degree tilt grows the bounding box by roughly a sixth of the short side
        room = size * 0.17
        out.append(Slot(
            p,
            _fit(c * cw + (cw - w) / 2 + jx, w, W, room),
            _fit(r * ch + (ch - h) / 2 + jy, h, H, room),
            w, h,
            rotate=(_noise(i * 31 + 3) - .5) * 10.0,
            mount=max(4.0, size * 0.035)))
    return out


def _fit(pos: float, span: float, page: float, room: float) -> float:
    """Keep a tilted print on the page, giving up the margin before the page edge."""
    lo, hi = room, page - span - room
    if hi < lo:                                  # no room to tilt into: centre it instead
        return _clamp((page - span) / 2.0, 0.0, max(0.0, page - span))
    return _clamp(pos, lo, hi)


def _noise(seed: int) -> float:
    """A fixed pseudo-random value: the same recipe must give the same picture.

    A sine-based hash was tried first and correlated badly at these small seeds,
    which piled the prints into one corner.
    """
    import random
    return random.Random(seed).random()


TEMPLATES: dict[str, Callable[..., list[Slot]]] = {
    "justified": justified, "grid": grid, "hero": hero,
    "filmstrip": filmstrip, "spread": spread, "scatter": scatter,
}


# --- the picture ------------------------------------------------------------

def caption_band(page_width: int) -> int:
    """How much of the page a line of text under the photographs takes."""
    return max(28, round(page_width * 0.045))


def plan(photos: Sequence[Photo], *, template: str = "justified", shape: str = "3:2",
         gap: int = 8, long_edge: int | None = None, face_safe: bool = True,
         caption: str | None = None) -> dict[str, Any]:
    """The geometry, without making a file. This is what a preview draws."""
    if template not in TEMPLATES:
        raise ComposeFailed(f"no template {template!r}; try " + ", ".join(sorted(TEMPLATES)))
    if shape not in SHAPES:
        raise ComposeFailed(f"no shape {shape!r}; try " + ", ".join(SHAPES))
    if not photos:
        raise ComposeFailed("nothing to compose")
    pw, ph, note = SHAPES[shape]
    if long_edge:
        k = long_edge / max(pw, ph)
        pw, ph = max(1, int(round(pw * k))), max(1, int(round(ph * k)))
    page_gap = gap * max(pw, ph) / 4000.0                 # the gap is a fraction of the page, not pixels
    # A caption takes room *inside* the page. Adding it afterwards would make a
    # captioned page taller than an uncaptioned one, and a PDF gives its pages one
    # size, so the two would not survive being bound together.
    band = caption_band(pw) if (caption and caption.strip()) else 0
    slots = TEMPLATES[template](photos, float(pw), float(ph - band), page_gap)
    out = []
    for s in slots:
        inner_w = max(1.0, s.w - 2 * s.mount)
        inner_h = max(1.0, s.h - 2 * s.mount)
        cw, ch, cx, cy = crop_box(s.photo, inner_w / inner_h, face_safe)
        d = s.as_dict()
        d["crop"] = {"w": cw, "h": ch, "x": cx, "y": cy}
        out.append(d)
    full_w, full_h, _ = SHAPES[shape]
    drafted = (pw, ph) != (full_w, full_h)
    return {"template": template, "shape": shape, "width": pw, "height": ph,
            "gap": round(page_gap), "face_safe": face_safe,
            "note": f"a draft of {shape}; {full_w}x{full_h} is {note}" if drafted else note,
            "print_size": [full_w, full_h], "draft": drafted, "caption_band": band,
            "photos": len(photos), "slots": out}


def render(photos: Sequence[Photo], out_path: Path, *, template: str = "justified",
           shape: str = "3:2", gap: int = 8, long_edge: int | None = None,
           face_safe: bool = True, background: str = "white", caption: str | None = None,
           memory_limit: str = "512MiB", timeout: int = 900,
           progress: Callable[[str], None] = lambda _s: None) -> dict[str, Any]:
    """Compose the page with ImageMagick and write it to `out_path`."""
    magick = shutil.which("magick")
    if magick is None:
        raise ComposeFailed("ImageMagick 7 (`magick`) is not installed; it is what draws the page")
    laid = plan(photos, template=template, shape=shape, gap=gap,
                long_edge=long_edge, face_safe=face_safe, caption=caption)
    by_id = {p.id: p for p in photos}
    started = time.time()

    args = [magick,
            # A 7.5 GB machine with no swap: bound the pixel cache and let it spill to disk
            # rather than let a nine-photo page at 300 dpi reach the OOM killer.
            "-limit", "memory", memory_limit, "-limit", "map", "1GiB",
            # -geometry on a composite is read against the current gravity, so it is
            # pinned here: a -gravity inside a tile once leaked out and shifted the page.
            "-gravity", "NorthWest",
            "-size", f"{laid['width']}x{laid['height']}", f"xc:{background}"]
    for slot in laid["slots"]:
        p = by_id[slot["id"]]
        c = slot["crop"]
        mount = slot["mount"]
        inner_w = max(1, slot["w"] - 2 * mount)
        inner_h = max(1, slot["h"] - 2 * mount)
        piece = ["(", str(p.path), "-auto-orient",
                 "-crop", f"{c['w']}x{c['h']}+{c['x']}+{c['y']}", "+repage",
                 "-resize", f"{inner_w}x{inner_h}!"]
        if mount:
            piece += ["-bordercolor", "white", "-border", f"{mount}x{mount}",
                      "-background", "white", "-gravity", "South", "-splice", f"0x{int(mount)}",
                      "-gravity", "NorthWest"]
        if slot["rotate"]:
            piece += ["-background", "none", "-rotate", str(slot["rotate"])]
        piece += [")", "-geometry", f"+{slot['x']}+{slot['y']}", "-composite"]
        args += piece
    # ImageMagick stamps the creation time into PNG text chunks, which would make
    # two identical pages differ as files. The picture is the output, not the clock.
    args += ["+set", "date:create", "+set", "date:modify",
             "-define", "png:exclude-chunks=date,time", "-quality", "92", str(out_path)]

    progress(f"composing {len(laid['slots'])} photos at {laid['width']}x{laid['height']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as err:
        raise ComposeFailed(f"the page did not finish within {timeout}s") from err
    if run.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        detail = (run.stderr or "").strip().splitlines()[-1:] or [""]
        raise ComposeFailed(f"ImageMagick could not draw the page: {detail[0][:300]}")

    if laid["caption_band"]:
        add_caption(magick, out_path, caption.strip(), laid["width"],
                    laid["caption_band"], background, timeout)
        laid["caption"] = caption.strip()

    laid["path"] = str(out_path)
    laid["bytes"] = out_path.stat().st_size
    laid["seconds"] = round(time.time() - started, 1)
    return laid


def export_pdf(pages: Sequence[Path], out_path: Path, *, dpi: int = 300,
               memory_limit: str = "512MiB", timeout: int = 1800) -> dict[str, Any]:
    """Bind rendered pages into one PDF.

    Every page of a book is the same shape, which is why the book carries the
    shape and the pages do not: a PDF gives its pages one size, and a page of a
    different shape would be squashed into it.
    """
    magick = shutil.which("magick")
    if magick is None:
        raise ComposeFailed("ImageMagick 7 (`magick`) is not installed; it is what writes the PDF")
    if not pages:
        raise ComposeFailed("the book has no pages")
    missing = [p for p in pages if not p.exists()]
    if missing:
        raise ComposeFailed(f"page {missing[0].name} was not drawn")
    started = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [magick, "-limit", "memory", memory_limit, "-limit", "map", "1GiB",
            "-units", "PixelsPerInch", "-density", str(dpi)]
    args += [str(p) for p in pages]
    args += ["-quality", "92", str(out_path)]
    run = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        detail = ((run.stderr or "").strip().splitlines()[-1:] or [""])[0]
        raise ComposeFailed(f"the PDF could not be written: {detail[:300]}")
    return {"path": str(out_path), "pages": len(pages), "dpi": dpi,
            "bytes": out_path.stat().st_size, "seconds": round(time.time() - started, 1)}


def add_caption(magick: str, path: Path, text: str, page_width: int, band: int,
                background: str, timeout: int = 300) -> None:
    """Write the caption into the strip the layout already left for it.

    Its own run of ImageMagick on purpose: `-gravity` is a setting, and one left
    in force during the compositing loop moves every photograph on the page.
    """
    point = max(10, round(band * 0.42))
    dark = background.lower() in ("black", "#000", "#000000", "none")
    run = subprocess.run(
        [magick, str(path), "-gravity", "South",
         "-fill", "white" if dark else "#333333",
         "-pointsize", str(point), "-annotate", f"+0+{round(band * 0.28)}", text,
         str(path)],
        capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0:
        raise ComposeFailed("the caption could not be drawn: "
                            + ((run.stderr or "").strip().splitlines()[-1:] or [""])[0][:200])


def image_size(path: Path) -> tuple[int, int]:
    """The file's dimensions as they will be seen after `-auto-orient`."""
    out = subprocess.run(["magick", "identify", "-format", "%w %h %[orientation]", str(path)],
                         capture_output=True, text=True, timeout=120)
    parts = out.stdout.strip().split()
    if out.returncode != 0 or len(parts) < 2:
        raise ComposeFailed(f"cannot read {path.name}")
    w, h = int(parts[0]), int(parts[1])
    # 90-degree EXIF orientations mean the stored width and height are swapped
    if len(parts) > 2 and parts[2] in ("RightTop", "LeftBottom", "LeftTop", "RightBottom"):
        w, h = h, w
    return w, h


def normalise_faces(boxes: Sequence[Sequence[float]], src_w: int | None, src_h: int | None
                    ) -> list[tuple[float, float, float, float]]:
    """Face boxes as fractions of the frame, or nothing if they clearly do not fit.

    The catalogue stores boxes in the pixels of whichever rendition the face pass
    analysed. If that rendition's size is unknown or disagrees with the box, a
    wrong crop is worse than a centred one, so the faces are dropped instead.
    """
    if not (src_w and src_h):
        return []
    out = []
    for b in boxes:
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        f = (x1 / src_w, y1 / src_h, x2 / src_w, y2 / src_h)
        if any(v < -0.05 or v > 1.05 for v in f) or f[2] <= f[0] or f[3] <= f[1]:
            return []
        out.append((_clamp(f[0], 0, 1), _clamp(f[1], 0, 1), _clamp(f[2], 0, 1), _clamp(f[3], 0, 1)))
    return out


def output_path(root: Path, name: str, template: str, shape: str, ext: str = ".jpg") -> Path:
    """A dated folder and a name that says what it is, never overwriting."""
    from datetime import date
    import re
    def slugify(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()

    stem = "-".join(x for x in (slugify(name) or "collage", slugify(template), slugify(shape)) if x)
    day = root.expanduser() / date.today().isoformat()
    target = day / f"{stem}{ext}"
    n = 2
    while target.exists():
        target = day / f"{stem}-{n}{ext}"
        n += 1
    return target
