# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Train a :class:`TinyLM` on a generated dataset (vision §29 reproducible training).

Usage::

    python -m training.train --data data/generated --out runs/m --preset M --steps 6000

Loss is computed on the assistant target tokens only. AdamW with linear warm-up and
cosine decay; the checkpoint with the best validation loss is kept as ``best.pt`` and
every run writes ``manifest.json`` (git revision, dataset/tokenizer hashes, seed,
architecture, optimiser, schedule, metrics).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias

from training import project_version
from training.model import ModelConfig, TinyLM
from training.tokenizer import Tokenizer

PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"n_layers": 1, "d_model": 32, "n_heads": 2, "n_kv_heads": 2, "d_ff": 64},
    "S": {"n_layers": 2, "d_model": 96, "n_heads": 4, "n_kv_heads": 4, "d_ff": 192},
    "M": {"n_layers": 4, "d_model": 128, "n_heads": 4, "n_kv_heads": 4, "d_ff": 256},
    "L": {"n_layers": 6, "d_model": 192, "n_heads": 6, "n_kv_heads": 6, "d_ff": 512},
    "XL": {"n_layers": 8, "d_model": 256, "n_heads": 8, "n_kv_heads": 8, "d_ff": 768},
}


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and schedule settings."""

    steps: int = 6000
    batch_size: int = 256
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    patience: int = 8
    seed: int = 1337


def load_split(path: Path, ctx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs, next-token targets, and loss mask (1 on assistant targets) for a JSONL split."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    ids = torch.zeros((n, ctx + 1), dtype=torch.long)
    mask = torch.zeros((n, ctx + 1), dtype=torch.float32)
    for i, row in enumerate(rows):
        prompt, target = row["prompt_ids"], row["target_ids"]
        seq = prompt + target
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, len(prompt) : len(seq)] = 1.0
    return ids[:, :-1], ids[:, 1:], mask[:, 1:]


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over masked positions."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    flat = mask.reshape(-1)
    return (loss * flat).sum() / flat.sum().clamp_min(1.0)


def masked_kl(
    student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL(teacher || student) on softened distributions, averaged over masked positions (Hinton et al.)."""
    s = F.log_softmax(student / temperature, dim=-1)
    t = F.log_softmax(teacher / temperature, dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    flat = mask.reshape(kl.shape)
    return (kl * flat).sum() / flat.sum().clamp_min(1.0)


def _load_teacher(path: Path, device: torch.device) -> TinyLM:
    """A frozen teacher (e.g. a bigger model trained on the same tokenizer)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    teacher = TinyLM(ModelConfig.from_json(ckpt["config"])).to(device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up then cosine decay to ``min_lr_ratio * lr``."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.lr * (
        cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    )


@torch.no_grad()
def evaluate_loss(
    model: TinyLM, data: tuple[torch.Tensor, ...], device: torch.device, batch: int
) -> float:
    """Masked validation loss over a whole split."""
    model.eval()
    x, y, m = data
    total, weight = 0.0, 0.0
    for i in range(0, x.size(0), batch):
        xb, yb, mb = (t[i : i + batch].to(device) for t in (x, y, m))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(xb)
        loss = masked_loss(logits.float(), yb, mb)
        total += float(loss) * float(mb.sum())
        weight += float(mb.sum())
    model.train()
    return total / max(weight, 1.0)


def git_revision() -> str:
    """Current git revision, suffixed with ``-dirty`` if the tree has changes."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev + ("-dirty" if dirty.strip() else "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optimizer(model: TinyLM, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def train(
    data_dir: Path,
    out_dir: Path,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    device_name: str | None = None,
    init_from: Path | None = None,
    log: bool = True,
    teacher_ckpt: Path | None = None,
    kd_alpha: float = 0.5,
    kd_temp: float = 2.0,
) -> dict[str, object]:
    """Run training; returns the run manifest (also saved as ``out_dir/manifest.json``)."""
    torch.manual_seed(cfg.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.load(data_dir / "tokenizer.json")
    train_data = load_split(data_dir / "train.jsonl", model_cfg.ctx_len)
    val_data = load_split(data_dir / "val.jsonl", model_cfg.ctx_len)
    model = TinyLM(model_cfg).to(device)
    teacher = _load_teacher(teacher_ckpt, device) if teacher_ckpt is not None else None
    if init_from is not None:
        model.load_state_dict(
            torch.load(init_from, map_location=device, weights_only=False)["model"]
        )
    optimizer = _optimizer(model, cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    history: list[dict[str, float]] = []
    best_val, best_step, bad_evals = float("inf"), -1, 0
    manifest: dict[str, object] = {
        "project_version": project_version(),
        "git_revision": git_revision(),
        "dataset_manifest_sha256": _sha256(data_dir / "manifest.json"),
        "tokenizer_sha256": _sha256(data_dir / "tokenizer.json"),
        "model_config": json.loads(model_cfg.to_json()),
        "params": model_cfg.param_count(),
        "train_config": dataclasses.asdict(cfg),
        "optimizer": "AdamW(betas=(0.9, 0.95)), decay on 2-D weights only",
        "schedule": "linear warm-up + cosine decay",
        "device": str(device),
        "init_from": str(init_from) if init_from else None,
        "distillation": None
        if teacher is None
        else {
            "teacher": str(teacher_ckpt),
            "alpha": kd_alpha,
            "temperature": kd_temp,
            "loss": "CE + a*T^2*KL",
        },
    }
    started = time.time()
    x_all, y_all, m_all = train_data
    model.train()
    for step in range(cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        idx = torch.randint(0, x_all.size(0), (cfg.batch_size,), generator=generator)
        xb, yb, mb = x_all[idx].to(device), y_all[idx].to(device), m_all[idx].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(xb)
        loss = masked_loss(logits.float(), yb, mb)
        if teacher is not None:
            with (
                torch.no_grad(),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                t_logits = teacher(xb)
            kd = masked_kl(logits.float(), t_logits.float(), mb, kd_temp)
            loss = (1.0 - kd_alpha) * loss + kd_alpha * kd_temp**2 * kd
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last = step == cfg.steps - 1
        if (step + 1) % cfg.eval_every == 0 or last:
            val = evaluate_loss(model, val_data, device, 512)
            history.append({"step": step + 1, "train_loss": loss.item(), "val_loss": val})
            if log:
                lr_now = lr_at(step, cfg)
                print(
                    f"step {step + 1:6d}  train {loss.item():.4f}  val {val:.4f}  lr {lr_now:.2e}"
                )
            if val < best_val:
                best_val, best_step, bad_evals = val, step + 1, 0
                _save(out_dir / "best.pt", model, model_cfg, tokenizer, manifest)
            else:
                bad_evals += 1
                if bad_evals >= cfg.patience:
                    break
    manifest.update(
        {
            "best_val_loss": best_val,
            "best_step": best_step,
            "steps_run": history[-1]["step"] if history else 0,
            "history": history,
            "train_seconds": round(time.time() - started, 1),
        }
    )
    best = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    best["manifest"] = manifest
    torch.save(best, out_dir / "best.pt")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _save(
    path: Path, model: TinyLM, cfg: ModelConfig, tok: Tokenizer, manifest: dict[str, object]
) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"config": cfg.to_json(), "model": state, "tokenizer": tok.to_json(), "manifest": manifest},
        path,
    )


def model_config_from_args(args: argparse.Namespace, vocab_size: int) -> ModelConfig:
    """Resolve preset + overrides into a :class:`ModelConfig`."""
    params: dict[str, int] = dict(PRESETS[args.preset])
    for key in ("n_layers", "d_model", "n_heads", "n_kv_heads", "d_ff"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    return ModelConfig(
        vocab_size=vocab_size, ctx_len=args.ctx, mlp_type=args.mlp, pos_type=args.pos, **params
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="M")
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--mlp", choices=("gelu", "swiglu"), default="gelu")
    parser.add_argument("--pos", choices=("learned", "rope"), default="learned")
    for key in ("n_layers", "d_model", "n_heads", "n_kv_heads", "d_ff"):
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=int, default=None)
    defaults = TrainConfig()
    for field in dataclasses.fields(TrainConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            dest=field.name,
            type=type(getattr(defaults, field.name)),
            default=getattr(defaults, field.name),
        )
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument(
        "--teacher", type=Path, default=None, help="checkpoint for logit distillation"
    )
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--kd-temp", type=float, default=2.0)
    args = parser.parse_args(argv)
    tokenizer = Tokenizer.load(args.data / "tokenizer.json")
    model_cfg = model_config_from_args(args, tokenizer.vocab_size)
    train_cfg = TrainConfig(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)}
    )
    manifest = train(
        args.data,
        args.out,
        model_cfg,
        train_cfg,
        args.device,
        args.init_from,
        teacher_ckpt=args.teacher,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("params", "best_val_loss", "best_step", "train_seconds")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
