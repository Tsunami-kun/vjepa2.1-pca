from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_DEFAULT_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGENET_DEFAULT_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_ARCHES = {
    "vit_base": "vit_base",
    "vit_large": "vit_large",
    "vit_giant": "vit_giant_xformers",
    "vit_gigantic": "vit_gigantic_xformers",
}
DEFAULT_CKPT_KEYS = ("ema_encoder", "target_encoder", "encoder")


@dataclass(frozen=True)
class UpstreamModules:
    """Handles imported from the upstream vjepa2 checkout."""

    video_transforms: Any
    volume_transforms: Any
    vit: Any
    robust_checkpoint_loader: Any
    video_reader_cls: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render V-JEPA 2.1 dense PCA visualizations from a local checkpoint."
    )
    parser.add_argument(
        "--vjepa-root",
        type=Path,
        default=None,
        help="Path to a local clone of facebookresearch/vjepa2.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to an input video or image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a local V-JEPA 2.1 checkpoint.")
    parser.add_argument("--model", choices=sorted(MODEL_ARCHES.keys()), default="vit_large")
    parser.add_argument("--checkpoint-key", default="auto")
    parser.add_argument("--feature-mode", choices=("last", "hierarchical"), default="last")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vjepa2_1_pca_vis"),
    )
    return parser.parse_args()


def resolve_vjepa_root(vjepa_root: Path | None) -> Path:
    """Find a usable upstream vjepa2 checkout."""

    candidates: list[Path] = []
    if vjepa_root is not None:
        candidates.append(vjepa_root)
    env_root = os.environ.get("VJEPA2_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd() / "vjepa2")
    candidates.append(Path(__file__).resolve().parents[4] / "vjepa2")

    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.expanduser().resolve()
        if (candidate / "app" / "vjepa_2_1").exists() and (candidate / "src").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the upstream vjepa2 repository. Pass --vjepa-root or set VJEPA2_ROOT."
    )


def bootstrap_vjepa_imports(vjepa_root: Path) -> UpstreamModules:
    """Import the modules that stay owned by the upstream repository."""

    if str(vjepa_root) not in sys.path:
        sys.path.insert(0, str(vjepa_root))

    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms
    from app.vjepa_2_1.models import vision_transformer as vit
    from src.utils.checkpoint_loader import robust_checkpoint_loader

    try:
        from decord import VideoReader
    except ImportError as exc:
        raise ImportError("decord is required for video visualization.") from exc

    return UpstreamModules(
        video_transforms=video_transforms,
        volume_transforms=volume_transforms,
        vit=vit,
        robust_checkpoint_loader=robust_checkpoint_loader,
        video_reader_cls=VideoReader,
    )


def build_eval_transform(video_transforms: Any, volume_transforms: Any, img_size: int):
    """Match the upstream evaluation crop and normalization path."""

    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=tuple(IMAGENET_DEFAULT_MEAN.tolist()),
                std=tuple(IMAGENET_DEFAULT_STD.tolist()),
            ),
        ]
    )


def sample_video(
    video_reader_cls: Any,
    path: Path,
    frames: int,
    frame_stride: int,
    fps: float | None,
):
    vr = video_reader_cls(str(path))
    if fps is not None:
        native_fps = vr.get_avg_fps()
        step = max(int(round(native_fps / fps)), 1)
    else:
        step = max(frame_stride, 1)
    frame_idx = np.arange(0, len(vr), step, dtype=np.int64)[:frames]
    if len(frame_idx) == 0:
        raise ValueError(f"No frames could be sampled from {path}")
    return vr.get_batch(frame_idx).asnumpy(), frame_idx


def load_input(
    video_reader_cls: Any,
    path: Path,
    frames: int,
    frame_stride: int,
    fps: float | None,
):
    if path.suffix.lower() in IMAGE_EXTS:
        image = np.array(Image.open(path).convert("RGB"))
        return np.expand_dims(image, axis=0), np.array([0], dtype=np.int64)
    return sample_video(video_reader_cls, path, frames=frames, frame_stride=frame_stride, fps=fps)


def clean_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip wrapper prefixes commonly present in released checkpoints."""

    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def resolve_checkpoint_key(checkpoint: dict[str, Any], checkpoint_key: str) -> str:
    if checkpoint_key != "auto":
        if checkpoint_key not in checkpoint:
            raise KeyError(
                f"Checkpoint key '{checkpoint_key}' not found. Available keys: {sorted(checkpoint.keys())}"
            )
        return checkpoint_key
    for key in DEFAULT_CKPT_KEYS:
        if key in checkpoint:
            return key
    raise KeyError(f"Could not infer encoder key. Available keys: {sorted(checkpoint.keys())}")


def build_model(
    vit: Any,
    model_name: str,
    img_size: int,
    patch_size: int,
    num_frames: int,
    tubelet_size: int,
):
    arch = MODEL_ARCHES[model_name]
    return vit.__dict__[arch](
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=max(num_frames, tubelet_size),
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
    )


def load_model_weights(
    robust_checkpoint_loader: Any,
    model: torch.nn.Module,
    checkpoint_path: Path,
    checkpoint_key: str,
):
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location="cpu")
    resolved_key = resolve_checkpoint_key(checkpoint, checkpoint_key)
    state_dict = clean_encoder_state_dict(checkpoint[resolved_key])
    msg = model.load_state_dict(state_dict, strict=False)
    return resolved_key, msg


def compute_temporal_tokens(num_input_frames: int, tubelet_size: int):
    if num_input_frames == 1:
        return 1
    return num_input_frames // tubelet_size


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_pca_component_signs(components: torch.Tensor) -> torch.Tensor:
    """Canonicalize PCA signs so repeated runs keep the same channel orientation."""

    anchor_idx = components.abs().argmax(dim=0)
    anchor_vals = components.gather(0, anchor_idx.unsqueeze(0)).squeeze(0)
    signs = torch.where(anchor_vals < 0, -1.0, 1.0).to(components.dtype)
    return components * signs.unsqueeze(0)


def run_pca(features: torch.Tensor) -> torch.Tensor:
    """Project dense token features to a stable 3-channel PCA view."""

    centered = (features - features.mean(dim=0, keepdim=True)).float().cpu()
    if min(centered.shape[0], centered.shape[1]) < 3:
        raise ValueError(f"Need at least 3 samples/features for RGB PCA, got shape {tuple(features.shape)}")

    # Full SVD keeps the PCA colors stable across runs, which is important for qualitative inspection.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = fix_pca_component_signs(vh[:3].T)
    proj = centered @ components
    mins = proj.amin(dim=0, keepdim=True)
    maxs = proj.amax(dim=0, keepdim=True)
    return (proj - mins) / (maxs - mins).clamp_min(1e-6)


def denormalize_clip(clip: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_DEFAULT_MEAN[:, None, None, None].to(clip.device)
    std = IMAGENET_DEFAULT_STD[:, None, None, None].to(clip.device)
    return (clip * std + mean).clamp(0.0, 1.0)


def representative_frame_index(vis_idx: int, clip_length: int, tubelet_size: int) -> int:
    """Choose the source frame that best represents a tubelet output."""

    return min(vis_idx * max(tubelet_size, 1) + max(tubelet_size // 2, 0), clip_length - 1)


def save_visualizations(
    clip: torch.Tensor,
    pca_rgb: torch.Tensor,
    out_dir: Path,
    frame_indices: np.ndarray,
    tubelet_size: int,
):
    """Write side-by-side original and PCA panels to disk."""

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_rgb = denormalize_clip(clip).permute(1, 2, 3, 0).cpu()
    pca_rgb = pca_rgb.permute(0, 3, 1, 2)
    pca_rgb = F.interpolate(pca_rgb, size=clip_rgb.shape[1:3], mode="nearest")
    pca_rgb = pca_rgb.permute(0, 2, 3, 1).cpu().clamp(0.0, 1.0)

    for t in range(pca_rgb.shape[0]):
        clip_idx = representative_frame_index(t, clip_rgb.shape[0], tubelet_size)
        original = (clip_rgb[clip_idx].numpy() * 255.0).astype(np.uint8)
        pca = (pca_rgb[t].numpy() * 255.0).astype(np.uint8)
        panel = np.concatenate([original, pca], axis=1)
        source_frame = int(frame_indices[min(clip_idx, len(frame_indices) - 1)])
        Image.fromarray(panel).save(out_dir / f"frame_{t:03d}_src_{source_frame:04d}.png")


def extract_tokens(
    model: torch.nn.Module,
    clip: torch.Tensor,
    device: torch.device,
    feature_mode: str,
):
    """Run the upstream encoder and optionally enable hierarchical outputs."""

    model = model.to(device).eval()
    clip = clip.to(device)
    if feature_mode == "hierarchical":
        model.return_hierarchical = True
    with torch.inference_mode():
        return model(clip), device


def main():
    args = parse_args()
    vjepa_root = resolve_vjepa_root(args.vjepa_root)
    upstream = bootstrap_vjepa_imports(vjepa_root)

    raw_frames, frame_indices = load_input(
        upstream.video_reader_cls,
        args.input,
        frames=args.frames,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    transform = build_eval_transform(
        upstream.video_transforms,
        upstream.volume_transforms,
        args.img_size,
    )
    clip = transform(list(raw_frames))
    clip = clip[:, : args.frames].unsqueeze(0)

    num_frames = int(clip.shape[2])
    model = build_model(
        upstream.vit,
        args.model,
        args.img_size,
        args.patch_size,
        num_frames,
        args.tubelet_size,
    )
    checkpoint_key, load_msg = load_model_weights(
        upstream.robust_checkpoint_loader,
        model,
        args.checkpoint,
        args.checkpoint_key,
    )

    requested_device = resolve_device(args.device)
    try:
        tokens, device = extract_tokens(model, clip, requested_device, args.feature_mode)
    except torch.OutOfMemoryError:
        if requested_device.type != "cuda" or args.device == "cuda":
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tokens, device = extract_tokens(model.cpu(), clip.cpu(), torch.device("cpu"), args.feature_mode)
        print("CUDA OOM during PCA visualization; retried on CPU.")

    temporal_tokens = compute_temporal_tokens(num_frames, args.tubelet_size)
    spatial_tokens = args.img_size // args.patch_size
    expected_tokens = temporal_tokens * spatial_tokens * spatial_tokens
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"Unexpected token count {tokens.shape[1]}; expected {expected_tokens} from "
            f"T={temporal_tokens}, H=W={spatial_tokens}."
        )

    token_grid = tokens[0].reshape(
        temporal_tokens,
        spatial_tokens,
        spatial_tokens,
        tokens.shape[-1],
    )
    pca_rgb = run_pca(token_grid.reshape(-1, tokens.shape[-1])).reshape(
        temporal_tokens,
        spatial_tokens,
        spatial_tokens,
        3,
    )
    save_visualizations(clip[0], pca_rgb, args.output_dir, frame_indices, args.tubelet_size)

    print(f"Upstream V-JEPA root: {vjepa_root}")
    print(f"Loaded encoder weights from key '{checkpoint_key}' with msg: {load_msg}")
    print(f"Device used: {device}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Token grid: T={temporal_tokens}, H={spatial_tokens}, W={spatial_tokens}, C={tokens.shape[-1]}")
    print(f"Saved {temporal_tokens} visualization frames to {args.output_dir}")


if __name__ == "__main__":
    main()
