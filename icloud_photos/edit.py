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

What an agent with ImageMagick can do is ordinary photo work: exposure,
contrast, colour, crop, rotate, resize, sharpen, blur, borders, text, format
conversion. It cannot invent content, so "remove the car" or "make her smile"
will fail or disappoint; that needs an image model, which is a different
(metered) service.
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

The file is ./{input_name} in this directory, and it is attached so you can see it.
Apply exactly this change, and only this change:

{prompt}

Rules:
- Write the result as ./{output_name} in this directory.
- Never modify or delete ./{input_name}, and never touch anything outside this directory.
- Keep the original resolution and orientation unless the request asks otherwise.
- Use the tools on this machine: `magick` (ImageMagick 7) and `ffmpeg`.
- If the request needs content that is not in the photo (adding or removing objects,
  changing faces), do not fake it: print `CANNOT: <one line why>` and write no file.
- When the file is written, print `RESULT: {output_name}` as the last line.
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

    started = time.time()
    progress(f"asking {agent} to edit {source.name}")
    try:
        run = subprocess.run(
            [binary, "exec", "-C", str(work), "-i", str(work / input_name),
             "-s", "workspace-write", "--skip-git-repo-check",
             INSTRUCTION.format(input_name=input_name, output_name=output_name, prompt=prompt.strip())],
            capture_output=True, text=True, timeout=timeout, cwd=str(work))
    except subprocess.TimeoutExpired as err:
        shutil.rmtree(work, ignore_errors=True)
        raise EditFailed(f"{agent} did not finish within {timeout}s") from err

    tail = (run.stdout or "").strip().splitlines()
    said = next((line for line in reversed(tail) if line.startswith(("RESULT:", "CANNOT:"))), "")
    produced = work / output_name
    if not produced.exists():
        produced = next((p for p in sorted(work.iterdir())
                         if p.name != input_name and p.is_file() and is_image(p)), None) or produced
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
            "source": str(source), "agent": agent, "seconds": round(time.time() - started, 1)}
