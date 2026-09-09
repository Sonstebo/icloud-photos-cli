"""Edit a photo by asking an agent, and keep the result on disk.

The agent is OpenAI's Codex CLI, which signs in with a ChatGPT subscription and
runs locally: it is given the photo and one instruction, and edits it with the
tools on this machine (ImageMagick, ffmpeg). Nothing is uploaded to iCloud and
the original is never touched.

    photos edit <asset id or album entry> "make it black and white"

The result lands in `<edits_dir>/<YYYY-MM-DD>/<name>-<slug>.<ext>`, an ordinary
folder (default `~/Pictures/Photos Edits`) that can be browsed, backed up or
deleted without this tool. Each edit runs in a scratch directory under
`<edits_dir>/.work` that is removed afterwards.

Two kinds of change are possible, and the agent picks between them:

- **Adjustments** (exposure, contrast, colour, crop, rotate, resize, sharpen,
  borders, text, format) are done with ImageMagick: exact, fast, and the full
  resolution of the original is kept.
- **Content changes** (removing an object, restoring and sharpening an old
  photo, adding something that is not there) use the image tool the agent has
  through the ChatGPT subscription. These are regenerated pixels, so the result
  comes back at the image model's own size, usually smaller than the original,
  and the detail is the model's interpretation rather than the camera's.

The result records which route was taken and both sizes, so a drop in
resolution is never a surprise.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Callable

Progress = Callable[[str], None]

DEFAULT_EDITS_DIR = "~/Pictures/Photos Edits"
DEFAULT_AGENT = "codex"
DEFAULT_TIMEOUT = 900

INSTRUCTION = """You are editing one photograph, nothing else.

The photograph is ./{input_name} in this directory, at its full resolution.
./preview.jpg is the same picture as a JPEG, at most 2048 px: it is attached so you
can see it, and it is the file to give an image tool, which cannot read HEIC or RAW.
Apply exactly this change, and only this change:

{prompt}

How to do it:
- If the change is an adjustment of the existing pixels (exposure, contrast, colour,
  white balance, crop, rotate, straighten, resize, sharpen, blur, borders, text,
  format conversion), use `magick` (ImageMagick 7) or `ffmpeg`. This keeps the
  original resolution, so prefer it whenever it can do the job.
- If the change needs pixels that are not in the photo (removing or adding an
  object, restoring or repairing an old or damaged photo, changing a background),
  use your image generation tool with ./preview.jpg as the input image. Keep the
  same people, framing and proportions unless the request says otherwise.

Rules:
- Write the result as ./{output_name} in this directory (any common image format
  is fine; name it output.png if the tool returns PNG).
- Never modify or delete ./{input_name} or ./preview.jpg, and never touch anything
  outside this directory.
- If you genuinely cannot do it, print `CANNOT: <one line why>` and write no file.
- When the file is written, print two lines and nothing after them:
  `METHOD: magick` or `METHOD: generated`
  `RESULT: <the file you wrote>`
"""


class EditFailed(Exception):
    """The agent produced no usable image."""


def slugify(text: str, limit: int = 40) -> str:
    """A short, filename-safe form of the instruction, for the output's name."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:limit].rstrip("-") or "edit")


def agent_available(agent: str = DEFAULT_AGENT) -> str | None:
    """The agent's path, if it is installed."""
    return shutil.which(agent)


def edits_dir(configured: str | None = None) -> Path:
    return Path(configured or DEFAULT_EDITS_DIR).expanduser()


def is_image(path: Path) -> bool:
    """Whether the file is an image this machine can read (ImageMagick decides)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    magick = shutil.which("magick") or shutil.which("identify")
    if magick is None:
        return True
    args = [magick, "identify", str(path)] if magick.endswith("magick") else [magick, str(path)]
    return subprocess.run(args, capture_output=True, timeout=60).returncode == 0


def dimensions(path: Path) -> str:
    """`WIDTHxHEIGHT`, or an empty string if ImageMagick cannot say."""
    magick = shutil.which("magick")
    if magick is None:
        return ""
    out = subprocess.run([magick, "identify", "-format", "%wx%h", str(path)], capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""


def edit(source: Path, prompt: str, *, name: str | None = None, root: Path | None = None,
         agent: str = DEFAULT_AGENT, timeout: int = DEFAULT_TIMEOUT,
         progress: Progress = lambda _: None) -> dict[str, Any]:
    """Ask the agent to apply `prompt` to `source`; return where the result landed."""
    binary = agent_available(agent)
    if binary is None:
        raise EditFailed(f"{agent} is not installed; it is what applies the edit "
                         "(https://developers.openai.com/codex/cli)")
    if not source.exists():
        raise EditFailed(f"{source} is not on disk")
    root = edits_dir(str(root) if root else None)
    day = root / date.today().isoformat()
    work = root / ".work" / f"{int(time.time())}-{slugify(prompt, 16)}"
    work.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".jpg"
    input_name, output_name = f"input{suffix}", f"output{suffix}"
    shutil.copy2(source, work / input_name)
    # A JPEG copy for the image tool and for the agent to look at: it cannot read
    # HEIC or RAW, and a 24-megapixel upload would be slow for no gain.
    preview = work / "preview.jpg"
    magick = shutil.which("magick")
    if magick is not None:
        subprocess.run([magick, str(work / input_name), "-auto-orient", "-resize", "2048x2048>",
                        "-quality", "92", str(preview)], capture_output=True, timeout=300)
    if not preview.exists() and suffix in (".jpg", ".jpeg"):
        shutil.copy2(source, preview)
    attach = preview if preview.exists() else work / input_name

    started = time.time()
    progress(f"asking {agent} to edit {source.name}")
    try:
        run = subprocess.run(
            [binary, "exec", "-C", str(work), "-i", str(attach),
             "-s", "workspace-write", "--skip-git-repo-check",
             INSTRUCTION.format(input_name=input_name, output_name=output_name, prompt=prompt.strip())],
            capture_output=True, text=True, timeout=timeout, cwd=str(work))
    except subprocess.TimeoutExpired as err:
        shutil.rmtree(work, ignore_errors=True)
        raise EditFailed(f"{agent} did not finish within {timeout}s") from err

    tail = (run.stdout or "").strip().splitlines()
    said = next((line for line in reversed(tail) if line.startswith(("RESULT:", "CANNOT:"))), "")
    method = next((line.split(":", 1)[1].strip() for line in reversed(tail) if line.startswith("METHOD:")), "")
    named = said.split(":", 1)[1].strip() if said.startswith("RESULT:") else ""
    produced = work / named if named and (work / named).exists() else work / output_name
    if not produced.exists():
        produced = next((p for p in sorted(work.iterdir())
                         if p.name not in (input_name, preview.name) and p.is_file() and is_image(p)),
                        None) or produced
    if said.startswith("CANNOT:") or not is_image(produced):
        detail = said or (tail[-1] if tail else (run.stderr or "").strip().splitlines()[-1:] or [""])[0]
        shutil.rmtree(work, ignore_errors=True)
        raise EditFailed(f"no edited image was produced: {detail[:300]}" if detail
                         else "no edited image was produced")

    day.mkdir(parents=True, exist_ok=True)
    stem = Path(name or source.name).stem
    target = day / f"{stem}-{slugify(prompt)}{produced.suffix}"
    n = 2
    while target.exists():
        target = day / f"{stem}-{slugify(prompt)}-{n}{produced.suffix}"
        n += 1
    shutil.move(str(produced), target)
    shutil.rmtree(work, ignore_errors=True)
    return {"path": str(target), "bytes": target.stat().st_size, "prompt": prompt,
            "source": str(source), "agent": agent, "method": method or "unknown",
            "size": dimensions(target), "source_size": dimensions(source),
            "seconds": round(time.time() - started, 1)}
