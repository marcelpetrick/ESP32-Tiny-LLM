# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Python <-> C parity: tokenizer, logits (all architecture variants), device rules.

This is the "Python, desktop C ... agree numerically within defined tolerance" exit
criterion of vision M2.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tools import runtime
from training import export
from training.data.dialogue import DialogueGenerator
from training.data.noise import add_typos
from training.data.textformat import SPECIAL_TOKENS, prompt_text, target_text
from training.model import ModelConfig, TinyLM
from training.quantize import fake_quantize
from training.tokenizer import Tokenizer
from training.world import DeviceState, apply_action, parse_action, render_state, validate_action
from training.world.actions import ActionParseError


@pytest.fixture(scope="module")
def tokenizer(tiny_data: Path) -> Tokenizer:
    return Tokenizer.load(tiny_data / "tokenizer.json")


def _export(
    tmp: Path, tok: Tokenizer, cfg: ModelConfig, dtype: str = "f32", seed: int = 0
) -> tuple[Path, TinyLM]:
    torch.manual_seed(seed)
    model = TinyLM(cfg).eval()
    path = tmp / f"m_{cfg.mlp_type}_{cfg.pos_type}_{cfg.n_kv_heads}_{dtype}.tllm"
    export.write_tllm(path, cfg, export.state_arrays(model), tok, dtype)
    return path, model


def _texts() -> list[str]:
    gen = DialogueGenerator(77, noise=0.5)
    rng = random.Random(5)
    texts = []
    for _ in range(150):
        s = gen.sample()
        texts.append(s.turn.user)
        texts.append(add_typos(s.turn.reply, rng, 2))
    texts += [
        "  spaced   out  ",
        "UPPER Case",
        "unicode é☃ \U0001f600",
        "t=31.2 h=na",
        "a\tb\nc",
        "",
    ]
    return texts


def test_tokenizer_parity(tmp_path: Path, tokenizer: Tokenizer) -> None:
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=32,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=2,
        d_ff=16,
    )
    path, _ = _export(tmp_path, tokenizer, cfg)
    with runtime.Runtime(path) as rt:
        for text in _texts():
            assert rt.tokenize(text) == tokenizer.encode_text(text), text
        for token in range(tokenizer.vocab_size):
            assert rt.token_bytes(token) == tokenizer.pieces[token]
        with pytest.raises(runtime.TinyLLMError):
            rt.token_bytes(tokenizer.vocab_size)


VARIANTS: list[dict[str, Any]] = [
    {"mlp_type": "gelu", "pos_type": "learned", "n_kv_heads": 4},
    {"mlp_type": "swiglu", "pos_type": "rope", "n_kv_heads": 2},
    {"mlp_type": "gelu", "pos_type": "rope", "n_kv_heads": 1},
]


@pytest.mark.parametrize("variant", VARIANTS)
def test_logits_parity_f32(tmp_path: Path, tokenizer: Tokenizer, variant: dict[str, Any]) -> None:
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=96,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=48,
        **variant,
    )
    path, model = _export(tmp_path, tokenizer, cfg)
    sample = DialogueGenerator(3).sample()
    ids = tokenizer.encode(prompt_text(sample) + target_text(sample.turn))[:96]
    with runtime.Runtime(path) as rt:
        for n in (1, 7, len(ids)):
            c_logits = np.array(rt.logits(ids[:n]))
            with torch.no_grad():
                ref = model(torch.tensor([ids[:n]]))[0, -1].numpy()
            np.testing.assert_allclose(c_logits, ref, rtol=1e-4, atol=1e-4)


def test_int8_matches_fake_quant_and_a8_agrees_on_argmax(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        ctx_len=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=48,
    )
    path, model = _export(tmp_path, tokenizer, cfg, dtype="i8", seed=4)
    reference = fake_quantize(model)
    sample = DialogueGenerator(8).sample()
    ids = tokenizer.encode(prompt_text(sample, history=0))[:60]
    with torch.no_grad():
        ref = reference(torch.tensor([ids]))[0, -1].numpy()
    with runtime.Runtime(path) as rt:
        np.testing.assert_allclose(np.array(rt.logits(ids)), ref, rtol=1e-3, atol=1e-3)
    with runtime.Runtime(path, act_int8=True, kv_int8=True) as rt:
        a8 = np.array(rt.logits(ids))
    assert np.corrcoef(a8, ref)[0, 1] > 0.99


def _random_state(rng: random.Random) -> DeviceState:
    return DeviceState(
        t=rng.choice([None, round(rng.uniform(-5, 42), 1)]),
        h=rng.choice([None, rng.randint(0, 100)]),
        soil=rng.randint(0, 100),
        fan=rng.randint(0, 3),
        heat=rng.randint(0, 1),
        pump=rng.randint(0, 1),
        light=rng.randrange(0, 101, 10),
        win=rng.randint(0, 1),
        pa=round(rng.uniform(0, 3.5), 1),
        vib=rng.randint(0, 1),
        err=rng.choice([0, 3, 5, 7]),
    )


def _settings(state: DeviceState) -> str:
    return " ".join(part for part in render_state(state)[4:-4].split())


def _random_action(rng: random.Random) -> str:
    pairs = []
    for key in rng.sample(
        ["fan", "heat", "pump", "light", "win", "ack", "diag", "bogus"], rng.randint(1, 3)
    ):
        if key == "diag":
            pairs.append(f"diag={rng.choice(['too_hot', 'soil_dry', 'nope'])}")
        elif key == "light":
            pairs.append(f"light={rng.choice([0, 30, 100, 35, 120])}")
        else:
            pairs.append(f"{key}={rng.randint(0, 8)}")
    return " ".join(pairs)


def test_device_rules_parity() -> None:
    rng = random.Random(2026)
    for _ in range(3000):
        state = _random_state(rng)
        action = _random_action(rng)
        c_verdict, c_state = runtime.device_eval(_settings(state), action)
        try:
            parsed = parse_action(action)
        except ActionParseError:
            assert c_verdict == "malformed", action
            continue
        verdict = validate_action(state, parsed)
        assert c_verdict == verdict.reason, (state, action)
        expected = apply_action(state, parsed) if verdict.approved else state
        assert c_state == render_state(expected), (state, action)


def test_special_tokens_match_textformat(tmp_path: Path, tokenizer: Tokenizer) -> None:
    assert tokenizer.specials == SPECIAL_TOKENS
    with pytest.raises(runtime.TinyLLMError):
        runtime.Runtime(tmp_path / "missing.tllm")
    with pytest.raises(runtime.TinyLLMError):
        runtime.device_eval("bogus=1", "fan=1")
