# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Sequence-level distillation (D1): paraphrases of user requests from a local LLM teacher.

Usage::

    python -m training.data.teacher --out data/teacher/paraphrases.json \\
        [--model qwen3.5:4b] [--per-key 24] [--host http://localhost:11434]

Design (docs/04-distillation.md): the teacher only supplies *wording*. Every paraphrase is
bound to a semantic key (intent, device, value) whose label the rule-based oracle
computes; a strict filter rejects lines that lose the device/number or flip the meaning,
and anything resembling a held-out frame is dropped so the held-out suite stays clean.
Only teachers whose licence allows training on outputs are used (Qwen3.5-4B: Apache-2.0).
Responses are cached by prompt hash so reruns are reproducible and cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.data.lexicon import DEVICE_WORDS, FRAMES, NUMBER_WORDS

DEFAULT_MODEL = "qwen3.5:4b"
TEACHER_LICENSE = "Apache-2.0 (Qwen/Qwen3.5-4B model card)"
SYSTEM = (
    "You write training data for a tiny voice assistant that controls a greenhouse. "
    "Output only the requested lines, one per line, lowercase, no numbering, no quotes, no explanations."
)
DEVICE_NAMES = {
    "fan": "the fan",
    "heat": "the heater",
    "pump": "the water pump",
    "light": "the grow light",
    "win": "the roof window",
}
_ALL_DEVICE_WORDS = {w for words in DEVICE_WORDS.values() for w in words}
_REFERENCE_WORDS = ("it", "that", "this", "one")

Key = tuple[str, str, str]  # (intent, device or "", value or "")


@dataclass(frozen=True)
class KeySpec:
    """What to ask for and how to validate the answers."""

    key: Key
    task: str
    require_device: str | None = None
    forbid_devices: bool = True
    require_number: int | None = None
    require_reference: bool = False
    require_any: tuple[str, ...] = ()
    forbid_words: tuple[str, ...] = ()


def key_str(key: Key) -> str:
    return "|".join(key)


def key_specs() -> list[KeySpec]:
    """All semantic keys the generator can draw teacher paraphrases for."""
    specs: list[KeySpec] = []
    for dev in ("fan", "heat", "pump", "light"):
        name = DEVICE_NAMES[dev]
        specs.append(
            KeySpec(("on", dev, ""), f"switch {name} on", dev, forbid_words=("off", "stop"))
        )
        specs.append(
            KeySpec(("off", dev, ""), f"switch {name} off", dev, forbid_words=("on", "start"))
        )
    specs.append(
        KeySpec(("open", "win", ""), "open the roof window", "win", forbid_words=("close", "shut"))
    )
    specs.append(
        KeySpec(("close", "win", ""), "close the roof window", "win", forbid_words=("open",))
    )
    up_words = (
        "lower",
        "less",
        "down",
        "weaker",
        "reduce",
        "decrease",
        "dim",
        "off",
        "shut",
        "stop",
    )
    down_words = (
        "more",
        "higher",
        "up",
        "stronger",
        "increase",
        "raise",
        "brighter",
        "off",
        "shut",
        "stop",
    )
    for dev in ("fan", "light"):
        name = DEVICE_NAMES[dev]
        specs.append(
            KeySpec(
                ("up", dev, ""), f"make {name} a bit stronger or higher", dev, forbid_words=up_words
            )
        )
        specs.append(
            KeySpec(
                ("down", dev, ""),
                f"make {name} a bit weaker or lower",
                dev,
                forbid_words=down_words,
            )
        )
        specs.append(
            KeySpec(("max", dev, ""), f"set {name} to its maximum", dev, forbid_words=up_words)
        )
    for level in (0, 1, 2, 3, 4, 5, 7):
        specs.append(
            KeySpec(
                ("fan_set", "fan", str(level)),
                f"set the fan to level {level}",
                "fan",
                require_number=level,
            )
        )
    for pct in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 35):
        specs.append(
            KeySpec(
                ("light_set", "light", str(pct)),
                f"set the grow light to {pct} percent",
                "light",
                require_number=pct,
            )
        )
    simple = {
        "read_t": ("ask for the current temperature", ("temp", "warm", "hot", "cold", "degree")),
        "read_h": ("ask for the current humidity", ("humid", "damp", "moist")),
        "read_soil": (
            "ask how moist or dry the soil is",
            ("soil", "plant", "earth", "ground", "water"),
        ),
        "read_pa": (
            "ask how much electric current the water pump draws",
            ("current", "amp", "power", "draw"),
        ),
        "status": ("ask for an overview of how everything in the greenhouse is", ()),
        "diagnose": ("ask the assistant to check whether anything is wrong or faulty", ()),
        "complain_humid": (
            "complain that the air feels humid, sticky or muggy",
            ("humid", "sticky", "muggy", "damp", "wet", "clammy", "moist"),
        ),
        "complain_hot": (
            "complain that it is too hot in the greenhouse",
            ("hot", "warm", "heat", "sweat", "boil", "cool"),
        ),
        "complain_cold": (
            "complain that it is too cold in the greenhouse",
            ("cold", "chilly", "freez", "warm", "cool"),
        ),
        "complain_dry": (
            "say the plants look dry or thirsty, or ask to water them",
            ("plant", "water", "dry", "thirst", "soil", "wilt"),
        ),
        "ack": (
            "ask to clear, reset or acknowledge the current error or alarm",
            ("error", "alarm", "fault", "warning", "alert"),
        ),
        "greet": ("greet the assistant", ()),
        "help": ("ask what the assistant can do or who it is", ()),
        "thanks": ("thank the assistant", ()),
        "ood": (
            "ask something completely unrelated to the greenhouse (general knowledge, music, food, travel, sports)",
            (),
        ),
        "what_about": ("ask about the humidity as a short follow-up question", ("humid",)),
    }
    for intent, (task, words) in simple.items():
        specs.append(KeySpec((intent, "", ""), task, require_any=words))
    for dev in ("fan", "heat", "pump", "light", "win"):
        name = DEVICE_NAMES[dev]
        specs.append(
            KeySpec(("read_dev", dev, ""), f"ask whether {name} is on or how it is set", dev)
        )
        specs.append(
            KeySpec(
                ("diag_dev", dev, ""),
                f"say {name} seems to behave strangely or sounds odd and ask to check it",
                dev,
            )
        )
    for degrees in (18, 20, 22, 24, 25, 28, 30):
        specs.append(
            KeySpec(
                ("set_temp", "", str(degrees)),
                f"set the greenhouse temperature to {degrees} degrees",
                require_number=degrees,
            )
        )
    ref = "referring to a device mentioned just before, without naming it (use 'it' or 'that')"
    specs.append(
        KeySpec(
            ("it_on", "", ""),
            f"switch it on, {ref}",
            require_reference=True,
            forbid_words=("off", "up", "down"),
        )
    )
    specs.append(
        KeySpec(
            ("it_off", "", ""),
            f"switch it off, {ref}",
            require_reference=True,
            forbid_words=("on", "up", "down"),
        )
    )
    specs.append(KeySpec(("it_up", "", ""), f"turn it up a bit, {ref}", forbid_words=up_words))
    specs.append(
        KeySpec(("it_down", "", ""), f"turn it down a bit, {ref}", forbid_words=down_words)
    )
    for value in (0, 1, 2, 3, 20, 40, 60, 80):
        specs.append(
            KeySpec(("it_set", "", str(value)), f"set it to {value}, {ref}", require_number=value)
        )
    for dev in ("fan", "heat", "pump", "light"):
        name = DEVICE_NAMES[dev]
        specs.append(
            KeySpec(
                ("and_dev", dev, ""),
                f"ask to do the same with {name} too, as a short follow-up",
                dev,
            )
        )
        specs.append(
            KeySpec(("correct", dev, ""), f"correct the assistant: they actually meant {name}", dev)
        )
    return specs


def _heldout_patterns() -> list[re.Pattern[str]]:
    """Regexes for every held-out frame ({dev} -> any device word, {n} -> any number)."""
    devices = "|".join(sorted((re.escape(w) for w in _ALL_DEVICE_WORDS), key=len, reverse=True))
    numbers = r"\d+|" + "|".join(NUMBER_WORDS.values())
    patterns = []
    for frames in FRAMES.values():
        for frame in frames:
            if not frame.startswith("!"):
                continue
            parts = re.split(r"(\{[a-z0-9]+\})", frame[1:])
            regex = "".join(
                f"(?:{numbers})"
                if part == "{n}"
                else f"(?:{devices})"
                if part.startswith("{")
                else re.escape(part)
                for part in parts
            )
            patterns.append(re.compile(rf"(?<![a-z]){regex}(?![a-z])"))
    return patterns


_HELDOUT = _heldout_patterns()


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text) is not None


_UNITS = {
    w: i
    for i, w in enumerate(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    )
}
_TEENS = {
    w: i + 10
    for i, w in enumerate(
        [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    w: (i + 2) * 10
    for i, w in enumerate(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    )
}
_NUMBER_PHRASE = re.compile(
    r"\b(?:(?:a|one) hundred(?: and)?(?: (?:"
    + "|".join([*_TENS, *_TEENS, *_UNITS])
    + r"))?(?:[ -](?:"
    + "|".join(_UNITS)
    + r"))?|(?:"
    + "|".join(_TENS)
    + r")(?:[ -](?:"
    + "|".join(_UNITS)
    + r"))?|"
    + "|".join([*_TEENS, *_UNITS])
    + r")\b"
)


def _phrase_value(phrase: str) -> int:
    total = 0
    for word in re.split(r"[ -]", phrase):
        if word == "hundred":
            total = max(total, 1) * 100
        elif word in _TENS:
            total += _TENS[word]
        elif word in _TEENS:
            total += _TEENS[word]
        elif word in _UNITS:
            total += _UNITS[word]
    return total


def digits_for_number_words(text: str) -> str:
    """Rewrite spelled-out numbers (0..199) as digits: "fifty percent" -> "50 percent"."""
    return _NUMBER_PHRASE.sub(lambda m: str(_phrase_value(m.group(0).replace(" and", ""))), text)


def clean(line: str) -> str:
    """Normalise one teacher line (strip bullets, numbering, quotes, whitespace)."""
    line = line.strip().lower()
    line = re.sub(r"^(\d+[.)]|[-*•])\s*", "", line)
    return line.strip(" \"'`").strip()


def accept(spec: KeySpec, text: str) -> bool:
    """Strict semantic filter; the oracle label for ``spec.key`` must stay valid."""
    words = text.split()
    if not 1 <= len(words) <= 14 or not text.isascii() or "<" in text or ">" in text:
        return False
    if any(pattern.search(text) for pattern in _HELDOUT):
        return False
    if any(_has_word(text, w) for w in spec.forbid_words):
        return False
    mentioned = {w for w in _ALL_DEVICE_WORDS if _has_word(text, w)}
    if spec.require_device is not None:
        own = set(DEVICE_WORDS[spec.require_device])
        if not mentioned & own or (spec.forbid_devices and mentioned - own):
            return False
    elif spec.forbid_devices and mentioned:
        return False
    if spec.require_number is not None:
        n = spec.require_number
        numbers = re.findall(r"\d+", text)
        if numbers != [str(n)] and not (
            not numbers and n in NUMBER_WORDS and _has_word(text, NUMBER_WORDS[n])
        ):
            return False
    elif re.search(r"\d", text):
        return False
    if spec.require_reference and not any(_has_word(text, w) for w in _REFERENCE_WORDS):
        return False
    return not spec.require_any or any(w in text for w in spec.require_any)


def prompt_for(spec: KeySpec, count: int, examples: list[str]) -> str:
    rules = []
    if spec.require_device:
        rules.append(
            "every line must mention "
            + DEVICE_NAMES[spec.require_device]
            + " (you may use: "
            + ", ".join(DEVICE_WORDS[spec.require_device])
            + ")"
        )
    if spec.require_number is not None:
        rules.append(f"every line must contain the number {spec.require_number} written as digits")
    if spec.require_reference:
        rules.append("do not name any device")
    rule_text = "; ".join(rules)
    return (
        f"Write {count} different short ways a person might {spec.task}. Vary wording, politeness and word order; "
        f"casual spoken english; 2 to 12 words each{'; ' + rule_text if rule_text else ''}. "
        f"Examples of the style: {'; '.join(examples)}."
    )


AskFn = Callable[[str, int], str]


def ollama_ask(host: str, model: str, system: str = SYSTEM) -> AskFn:
    """A function sending one prompt (with seed) to a local Ollama server."""

    def ask(prompt: str, seed: int) -> str:
        body = {
            "model": model,
            "think": False,
            "stream": False,
            "options": {"temperature": 0.9, "seed": seed, "num_predict": 700},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload: dict[str, Any] = json.loads(response.read())
        return str(payload["message"]["content"])

    return ask


def _examples(spec: KeySpec, rng: random.Random) -> list[str]:
    frames = [f for f in FRAMES.get(spec.key[0], ()) if not f.startswith("!")]
    dev = spec.key[1] or "fan"
    out = []
    for frame in rng.sample(frames, min(3, len(frames))):
        word = DEVICE_WORDS.get(dev, ("fan",))[0]
        out.append(frame.format(dev=word, dev2="light", n=spec.key[2] or "2"))
    return out


def _ask_cached(
    ask: AskFn, prompt: str, seed: int, cache: Path | None, model: str, system: str = SYSTEM
) -> str:
    """Ask once per (teacher model, system prompt, prompt, seed); answers are cached on disk."""
    digest = hashlib.sha256(f"{model}\n{system}\n{prompt}\n{seed}".encode()).hexdigest()[:16]
    cached = cache / f"{digest}.txt" if cache else None
    if cached is not None and cached.exists():
        return cached.read_text(encoding="utf-8")
    text = ask(prompt, seed)
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
    return text


def generate(
    ask: AskFn,
    per_key: int = 24,
    cache: Path | None = None,
    seeds: tuple[int, ...] = (7, 11),
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Ask the teacher for every key (once per seed), filter, split 80/20 into train/test."""
    result: dict[str, dict[str, list[str]]] = {}
    stats = {"asked": 0, "accepted": 0, "rejected": 0}
    rngs = {seed: random.Random(seed) for seed in seeds}
    for spec in key_specs():
        raw: set[str] = set()
        for seed in seeds:
            prompt = prompt_for(spec, per_key, _examples(spec, rngs[seed]))
            text = _ask_cached(ask, prompt, seed, cache, model)
            cleaned = {clean(line) for line in text.splitlines() if clean(line)}
            if spec.require_number is not None:  # only numeric intents: "the other one" stays text
                cleaned = {digits_for_number_words(line) for line in cleaned}
            raw |= cleaned
        lines = sorted(raw)
        rng = random.Random(key_str(spec.key))
        good = [line for line in lines if accept(spec, line)]
        stats["asked"] += len(lines)
        stats["accepted"] += len(good)
        stats["rejected"] += len(lines) - len(good)
        rng.shuffle(good)
        cut = max(1, len(good) // 5) if len(good) >= 3 else 0
        result[key_str(spec.key)] = {"train": sorted(good[cut:]), "test": sorted(good[:cut])}
    return {"paraphrases": result, "stats": stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--per-key", type=int, default=24)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/teacher"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    args = parser.parse_args(argv)
    started = time.time()
    data = generate(
        ollama_ask(args.host, args.model), args.per_key, args.cache, tuple(args.seeds), args.model
    )
    data["manifest"] = {
        "teacher": args.model,
        "teacher_license": TEACHER_LICENSE,
        "system_prompt": SYSTEM,
        "per_key": args.per_key,
        "seeds": args.seeds,
        "seconds": round(time.time() - started, 1),
        "note": "labels come from the rule-based oracle; the teacher only supplies wording",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(data["stats"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
