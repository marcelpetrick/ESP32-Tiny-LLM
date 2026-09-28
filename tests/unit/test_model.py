# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""PyTorch reference model: shapes, causality, parameter accounting, variants."""

import pytest
import torch

from training.model import ModelConfig, TinyLM, apply_rope, rope_tables

VARIANTS = [
    ModelConfig(
        vocab_size=50, ctx_len=16, n_layers=2, d_model=16, n_heads=4, n_kv_heads=4, d_ff=32
    ),
    ModelConfig(
        vocab_size=50,
        ctx_len=16,
        n_layers=1,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=24,
        mlp_type="swiglu",
        pos_type="rope",
    ),
    ModelConfig(
        vocab_size=50, ctx_len=16, n_layers=1, d_model=16, n_heads=4, n_kv_heads=1, d_ff=24
    ),
]


@pytest.mark.parametrize("cfg", VARIANTS)
def test_param_count_matches_module(cfg: ModelConfig) -> None:
    model = TinyLM(cfg)
    assert cfg.param_count() == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("cfg", VARIANTS)
def test_forward_shape_and_causality(cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    model = TinyLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, cfg.vocab_size)
    changed = idx.clone()
    changed[:, 7] = (changed[:, 7] + 1) % cfg.vocab_size
    other = model(changed)
    assert torch.allclose(logits[:, :7], other[:, :7], atol=1e-6)
    assert not torch.allclose(logits[:, 7:], other[:, 7:])


def test_generate_is_greedy_and_stops() -> None:
    torch.manual_seed(1)
    cfg = VARIANTS[0]
    model = TinyLM(cfg).eval()
    out = model.generate([1, 2, 3], max_new=5)
    assert len(out) == 5
    logits = model(torch.tensor([[1, 2, 3]]))[0, -1]
    assert out[0] == int(torch.argmax(logits))
    assert model.generate([1, 2, 3], max_new=5, stop_ids=(out[0],)) == [out[0]]
    assert len(model.generate(list(range(15)), max_new=5)) == 1


def test_too_long_sequence_raises() -> None:
    model = TinyLM(VARIANTS[0])
    with pytest.raises(ValueError, match="ctx_len"):
        model(torch.zeros((1, 17), dtype=torch.long))


def test_rope_rotation_preserves_norm_and_position_zero() -> None:
    cfg = VARIANTS[1]
    cos, sin = rope_tables(cfg)
    x = torch.randn(1, 1, 5, cfg.head_dim)
    y = apply_rope(x, cos[:5], sin[:5])
    assert torch.allclose(y[..., 0, :], x[..., 0, :])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_config_validation_and_json() -> None:
    cfg = VARIANTS[1]
    assert ModelConfig.from_json(cfg.to_json()) == cfg
    bad_configs: list[tuple[dict[str, object], str]] = [
        ({"d_model": 15}, "divisible by n_heads"),
        ({"n_kv_heads": 3}, "divisible by n_kv_heads"),
        ({"mlp_type": "relu"}, "mlp_type"),
        ({"pos_type": "alibi"}, "pos_type"),
        ({"d_model": 12, "n_heads": 4, "pos_type": "rope"}, "even head_dim"),
    ]
    for bad, message in bad_configs:
        with pytest.raises(ValueError, match=message):
            ModelConfig(**bad)  # type: ignore[arg-type]
