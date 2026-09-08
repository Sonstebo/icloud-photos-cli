"""The models: Immich's ONNX export of CLIP ViT-B-32, and InsightFace buffalo_l.

Nothing is trained or converted here. The CLIP files come from
huggingface.co/immich-app/ViT-B-32__openai (the model Immich uses for its
smart search) and run under onnxruntime; the face models are InsightFace's
own buffalo_l pack, loaded through its FaceAnalysis pipeline. Both are CPU
only for now. Everything is imported lazily so the CLI stays fast when no
model is needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

CLIP_MODEL = "ViT-B-32__openai"
FACE_MODEL = "buffalo_l"
CLIP_URL = "https://huggingface.co/immich-app/ViT-B-32__openai/resolve/main/"
CLIP_FILES = ("visual/model.onnx", "visual/preprocess_cfg.json", "textual/model.onnx", "textual/tokenizer.json")


class ModelsMissing(Exception):
    pass


class Models(Protocol):
    clip_name: str
    face_name: str
    def embed_image(self, bgr: np.ndarray) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...
    def faces(self, bgr: np.ndarray) -> list[dict[str, Any]]: ...


def as_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def decode_image(data: bytes) -> np.ndarray | None:
    """Bytes -> BGR array, or None if it is not an image OpenCV can read (HEIC via pillow-heif)."""
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is not None:
        return arr
    try:
        from PIL import Image
        import io
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img)[:, :, ::-1].copy()
    except Exception:  # noqa: BLE001
        return None


class OnnxModels:
    def __init__(self, models_dir: Path, threads: int | None = None) -> None:
        self.dir = Path(models_dir)
        self.clip_name, self.face_name = CLIP_MODEL, FACE_MODEL
        self.threads = threads
        self._visual = self._textual = self._tokenizer = self._faces = None
        self._cfg: dict[str, Any] | None = None

    # --- clip ---------------------------------------------------------------
    def _clip_dir(self) -> Path:
        d = self.dir / CLIP_MODEL
        missing = [f for f in CLIP_FILES if not (d / f).exists()]
        if missing:
            raise ModelsMissing(f"CLIP model files missing under {d}: {', '.join(missing)}; run `photos index --fetch-models`")
        return d

    def _session(self, path: Path) -> Any:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if self.threads:
            opts.intra_op_num_threads = self.threads
        opts.log_severity_level = 3
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def _load_clip(self) -> None:
        if self._visual is not None:
            return
        d = self._clip_dir()
        self._cfg = json.loads((d / "visual/preprocess_cfg.json").read_text())
        self._visual = self._session(d / "visual/model.onnx")
        self._textual = self._session(d / "textual/model.onnx")
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(d / "textual/tokenizer.json"))

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Shortest side to 224 (bicubic), centre crop, RGB, normalise: what preprocess_cfg.json says."""
        import cv2

        cfg = self._cfg or {}
        size = int((cfg.get("size") or [224])[0])
        h, w = bgr.shape[:2]
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
        top, left = (nh - size) // 2, (nw - size) // 2
        img = img[top:top + size, left:left + size, ::-1].astype(np.float32) / 255.0
        mean = np.array(cfg.get("mean", [0.481, 0.458, 0.408]), dtype=np.float32)
        std = np.array(cfg.get("std", [0.269, 0.261, 0.276]), dtype=np.float32)
        img = (img - mean) / std
        return img.transpose(2, 0, 1)[None]

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        self._load_clip()
        out = self._visual.run(None, {"image": self.preprocess(bgr)})[0][0]
        return out / np.linalg.norm(out)

    def embed_text(self, text: str) -> np.ndarray:
        self._load_clip()
        ids = self._tokenizer.encode(text).ids[:77]
        if len(ids) == 77:
            ids[-1] = 49407          # keep the end-of-text token when truncating
        tokens = np.zeros((1, 77), dtype=np.int32)
        tokens[0, :len(ids)] = ids
        out = self._textual.run(None, {"text": tokens})[0][0]
        return out / np.linalg.norm(out)

    # --- faces --------------------------------------------------------------
    def _load_faces(self) -> None:
        if self._faces is not None:
            return
        from insightface.app import FaceAnalysis

        root = self.dir / "insightface"
        # the pack downloads itself on first use (~280 MB from InsightFace's GitHub release)
        app = FaceAnalysis(name=FACE_MODEL, root=str(root), providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        self._faces = app

    def faces(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        self._load_faces()
        out = []
        for f in self._faces.get(bgr):
            emb = f.normed_embedding
            out.append({
                "box": [int(round(x)) for x in f.bbox.tolist()],   # x1, y1, x2, y2 in the analysed image
                "det": float(f.det_score),
                "age": int(f.age) if getattr(f, "age", None) is not None else None,
                "gender": int(f.gender) if getattr(f, "gender", None) is not None else None,
                "embedding": as_blob(emb),
            })
        return out


def fetch_clip_models(models_dir: Path, report: Any = print) -> None:
    """Download Immich's ONNX CLIP export (about 600 MB) into the models directory."""
    import urllib.request

    d = Path(models_dir) / CLIP_MODEL
    for rel in CLIP_FILES:
        target = d / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        report(f"fetching {rel}")
        urllib.request.urlretrieve(CLIP_URL + rel, target.with_suffix(target.suffix + ".part"))
        target.with_suffix(target.suffix + ".part").replace(target)
