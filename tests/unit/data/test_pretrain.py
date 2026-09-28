# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""TinyStories pretraining data builder."""

import json
from pathlib import Path

from training.data import pretrain


def test_stories_and_pack() -> None:
    assert pretrain.stories("A dog.<|endoftext|>\n The Cat!<|endoftext|> ") == [
        "a dog.",
        "the cat!",
    ]
    assert pretrain.pack([[1, 2, 3], [4, 5]], 2) == [[1, 2], [3, 4]]


def test_build(tiny_data: Path, tmp_path: Path) -> None:
    text = tmp_path / "stories.txt"
    text.write_text(
        "<|endoftext|>".join(["Once upon a time there was a fan."] * 60), encoding="utf-8"
    )
    out = tmp_path / "pre"
    assert (
        pretrain.main(
            [
                "--text",
                str(text),
                "--tokenizer",
                str(tiny_data / "tokenizer.json"),
                "--out",
                str(out),
                "--ctx",
                "16",
                "--max-tokens",
                "400",
            ]
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["lowercased"] is True
    row = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert row["prompt_ids"] == []
    assert len(row["target_ids"]) == 17
    assert (out / "tokenizer.json").exists()
