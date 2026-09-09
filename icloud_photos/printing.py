"""Getting a book ready for a print shop, and saying what will go wrong first.

A photo book is bought from somebody else's web site, and every shop wants the
pages in its own shape: its page size, its bleed, its resolution, its file
names. That is data, not code, so each one is a profile here and the export
reads it.

The part worth having is the check that comes before the money. A shop's
uploader will take a page happily and print it soft, and the first anyone knows
is when the book arrives. This module says so first, and it can, because it
knows what went into every page: which photograph, at what size, cropped how,
and where the faces are.

    photos book preflight "Summer 2024" --profile a4-landscape

Three things are worth catching, in the order they ruin a book: a photograph
with too few pixels for the space it was given, a face about to be cut off by
the trim, and a face swallowed by the gutter between two pages. Everything else
is taste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Profile:
    """What one print shop expects. Millimetres, because that is how paper is sold."""

    name: str
    label: str
    page_w_mm: float
    page_h_mm: float
    dpi: int = 300
    bleed_mm: float = 3.0          # printed beyond the trim, then cut off
    safe_mm: float = 5.0           # nothing important closer to the trim than this
    gutter_mm: float = 0.0         # lost into the spine of a spread
    max_file_mb: float = 50.0
    formats: tuple[str, ...] = ("jpg", "pdf")

    @property
    def page_px(self) -> tuple[int, int]:
        return (round(self.page_w_mm / MM_PER_INCH * self.dpi),
                round(self.page_h_mm / MM_PER_INCH * self.dpi))

    @property
    def bleed_px(self) -> int:
        return round(self.bleed_mm / MM_PER_INCH * self.dpi)

    def as_dict(self) -> dict[str, Any]:
        w, h = self.page_px
        return {"profile": self.name, "label": self.label,
                "page_mm": [self.page_w_mm, self.page_h_mm], "page_px": [w, h],
                "dpi": self.dpi, "bleed_mm": self.bleed_mm, "safe_mm": self.safe_mm,
                "gutter_mm": self.gutter_mm, "formats": list(self.formats)}


# Ordinary sizes, named for what they are. A shop that wants something else is a
# new entry here and nothing else changes.
PROFILES: dict[str, Profile] = {
    "a4-landscape": Profile("a4-landscape", "A4 landscape", 297, 210),
    "a4-portrait": Profile("a4-portrait", "A4 portrait", 210, 297),
    "square-210": Profile("square-210", "210 mm square", 210, 210),
    "square-300": Profile("square-300", "300 mm square", 300, 300),
    "a4-spread": Profile("a4-spread", "two A4 pages across the spine", 420, 297,
                         gutter_mm=12.0, safe_mm=8.0),
    "screen": Profile("screen", "for a screen, not a press", 340, 191, dpi=96,
                      bleed_mm=0.0, safe_mm=0.0),
}

# Below this a photograph is soft enough to see; below the second it is a smudge.
SOFT_DPI = 220
BAD_DPI = 150


@dataclass
class Finding:
    page: int
    kind: str
    severity: str            # "stop" or "look"
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"page": self.page, "kind": self.kind, "severity": self.severity,
                "message": self.message}


def _effective_dpi(source_px: int, slot_px: int, page_px: int, page_mm: float, dpi: int) -> float:
    """How many real pixels per inch the photograph brings to the space it fills."""
    if slot_px <= 0 or page_px <= 0:
        return float(dpi)
    slot_mm = slot_px / page_px * page_mm
    if slot_mm <= 0:
        return float(dpi)
    return source_px / (slot_mm / MM_PER_INCH)


def check_page(page: int, plan: dict[str, Any], source_px: dict[str, int],
               faces: dict[str, Sequence[Sequence[float]]], profile: Profile) -> list[Finding]:
    """Everything wrong with one laid-out page, worst first.

    `source_px` is how many pixels of the *original* survive the crop across the
    slot's width. It has to be the original: the page may have been laid out from
    a cached preview, and judging the print by the preview's size would call every
    page fine and every book soft.
    """
    out: list[Finding] = []
    pw, ph = profile.page_px
    laid_w = max(1, int(plan.get("width") or 1))
    laid_h = max(1, int(plan.get("height") or 1))
    for slot in plan.get("slots", []):
        pid = str(slot.get("id"))
        src_w = int(source_px.get(pid, 0))
        slot_w = max(1, int(slot.get("w") or 1))
        if src_w <= 0:
            out.append(Finding(page, "resolution", "look",
                               f"{pid[:8]}: nothing cached to measure; run it again once it is fetched"))
            continue
        # the layout may have been drawn smaller than the page; scale to the page
        slot_on_page = slot_w / laid_w * pw
        eff = _effective_dpi(src_w, round(slot_on_page), pw, profile.page_w_mm, profile.dpi)
        if eff < BAD_DPI:
            out.append(Finding(page, "resolution", "stop",
                               f"{pid[:8]} brings {eff:.0f} dpi to its space; it will print as a smudge"))
        elif eff < SOFT_DPI:
            out.append(Finding(page, "resolution", "look",
                               f"{pid[:8]} brings {eff:.0f} dpi; visibly soft on paper"))

        # a face about to be cut off by the trim, or lost into the spine
        boxes = faces.get(pid) or []
        if not boxes:
            continue
        sx, sy = slot.get("x", 0) / laid_w, slot.get("y", 0) / laid_h
        sw, sh = slot_w / laid_w, max(1, int(slot.get("h") or 1)) / laid_h
        safe_x = profile.safe_mm / profile.page_w_mm
        safe_y = profile.safe_mm / profile.page_h_mm
        for b in boxes:
            # the face's place on the page, given where its slot sits
            fx0, fy0 = sx + b[0] * sw, sy + b[1] * sh
            fx1, fy1 = sx + b[2] * sw, sy + b[3] * sh
            if fx0 < safe_x or fy0 < safe_y or fx1 > 1 - safe_x or fy1 > 1 - safe_y:
                out.append(Finding(page, "trim", "stop",
                                   f"a face in {pid[:8]} reaches the trim; the cut may take part of it"))
                break
            if profile.gutter_mm:
                g = profile.gutter_mm / profile.page_w_mm / 2
                if fx0 < 0.5 + g and fx1 > 0.5 - g:
                    out.append(Finding(page, "gutter", "look",
                                       f"a face in {pid[:8]} sits in the spine and will bend into it"))
                    break
    return out


def check_book(pages: Sequence[dict[str, Any]], profile: Profile,
               *, multiple_of: int = 0, minimum: int = 0) -> dict[str, Any]:
    """The whole book: every page, plus the things a shop counts."""
    findings: list[Finding] = []
    for i, page in enumerate(pages, 1):
        findings += page.get("findings", [])
    n = len(pages)
    if minimum and n < minimum:
        findings.append(Finding(0, "pages", "stop",
                                f"{n} pages; this profile wants at least {minimum}"))
    if multiple_of and n % multiple_of:
        findings.append(Finding(0, "pages", "stop",
                                f"{n} pages; this profile wants a multiple of {multiple_of}"))
    stops = [f for f in findings if f.severity == "stop"]
    return {"profile": profile.as_dict(), "pages": n,
            "findings": [f.as_dict() for f in findings],
            "stop": len(stops), "look": len(findings) - len(stops),
            "ready": not stops}
