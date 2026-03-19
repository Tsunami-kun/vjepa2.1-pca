from pathlib import Path

import torch

from vjepa2_1_pca.cli import (
    DEFAULT_CKPT_KEYS,
    clean_encoder_state_dict,
    compute_temporal_tokens,
    fix_pca_component_signs,
    representative_frame_index,
    resolve_checkpoint_key,
    run_pca,
)


def test_clean_encoder_state_dict_strips_expected_prefixes():
    state_dict = {
        "module.backbone.layer.weight": torch.tensor([1.0]),
        "module.bias": torch.tensor([2.0]),
    }

    cleaned = clean_encoder_state_dict(state_dict)

    assert set(cleaned) == {"layer.weight", "bias"}


def test_resolve_checkpoint_key_prefers_standard_order():
    checkpoint = {
        "encoder": {},
        "ema_encoder": {},
    }

    assert resolve_checkpoint_key(checkpoint, "auto") == DEFAULT_CKPT_KEYS[0]


def test_compute_temporal_tokens_handles_image_and_video():
    assert compute_temporal_tokens(1, 2) == 1
    assert compute_temporal_tokens(16, 2) == 8


def test_fix_pca_component_signs_makes_anchor_positive():
    components = torch.tensor(
        [
            [-3.0, 1.0, -2.0],
            [2.0, -4.0, 1.0],
            [1.0, 2.0, 5.0],
        ]
    )

    fixed = fix_pca_component_signs(components)
    anchor_idx = fixed.abs().argmax(dim=0)
    anchor_vals = fixed.gather(0, anchor_idx.unsqueeze(0)).squeeze(0)

    assert torch.all(anchor_vals >= 0)


def test_run_pca_is_deterministic_and_normalized():
    features = torch.tensor(
        [
            [3.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
        ]
    )

    proj_a = run_pca(features)
    proj_b = run_pca(features)

    assert proj_a.shape == (4, 3)
    assert torch.allclose(proj_a, proj_b)
    assert torch.all(proj_a >= 0.0)
    assert torch.all(proj_a <= 1.0)


def test_representative_frame_index_matches_tubelet_centering():
    assert representative_frame_index(vis_idx=0, clip_length=16, tubelet_size=2) == 1
    assert representative_frame_index(vis_idx=7, clip_length=16, tubelet_size=2) == 15
    assert representative_frame_index(vis_idx=3, clip_length=1, tubelet_size=2) == 0
