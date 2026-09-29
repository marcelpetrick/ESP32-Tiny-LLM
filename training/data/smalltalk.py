# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Chat-lite small talk (docs/03-chat-model.md §2): persona topics distilled from the teacher.

Usage::

    python -m training.data.smalltalk --out data/teacher/smalltalk.json [--seeds 41 43]

For every topic the local teacher (Qwen3.5-4B via Ollama, Apache-2.0) writes what a user
might say and how the persona ("the greenhouse helper", simple English) answers. User
phrasings are split 80/20 into train/test; replies are shared, because small talk has many
right answers: the evaluation accepts any reply of the right topic. Grounded topics answer
from the state block with code templates, so the numbers are always true.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.data import teacher
from training.data.dialogue import SmallTalk
from training.data.lexicon import DEVICE_WORDS, FRAMES
from training.world.facts import retrieve

SYSTEM = (
    "You write training data for a tiny chat companion that lives in a greenhouse controller. "
    "It speaks very simple english, is friendly and honest, and knows it is a small computer. "
    "Output only the requested lines, one per line, lowercase, no numbering, no quotes, no emoji."
)


@dataclass(frozen=True)
class Topic:
    """One small-talk topic: what the user does and how the persona should answer."""

    name: str
    user: str  # "... a person might say to ..." completion
    reply: str  # guidance for the persona's answers
    grounded: tuple[str, ...] = ()  # code templates using the state (replaces teacher replies)


TOPICS: tuple[Topic, ...] = (
    Topic("how_are_you", "ask how it is doing or how it feels today",
          "say it is fine and happy to look after the plants"),
    Topic("name", "ask what its name is", "say it has no real name and people call it the greenhouse helper"),
    Topic("robot", "ask if it is a robot, a human or alive", "say it is a small computer program, not a person"),
    Topic("age", "ask how old it is or when it was born", "say it is a young program and does not count birthdays"),
    Topic("likes_plants", "ask if it likes plants or which plant it likes most",
          "say it likes plants, especially tomatoes and herbs, in a warm simple way"),
    Topic("favourite_colour", "ask about its favourite colour", "say green, because of the leaves"),
    Topic("sleep", "ask if it ever sleeps or gets tired", "say it never sleeps because it watches the greenhouse"),
    Topic("lonely", "ask if it feels lonely or bored", "say the plants keep it company and it likes talking"),
    Topic("friend", "ask if it wants to be their friend", "say yes, kindly, as a friendly helper"),
    Topic("sad_user", "say they feel sad or had a bad day",
          "be kind and short, say sorry to hear that and that plants can help"),
    Topic("tired_user", "say they are tired", "suggest a short rest, kindly"),
    Topic("happy_user", "say they are happy or had a great day", "say that is nice to hear"),
    Topic("compliment", "say something nice about it, like it is smart or helpful", "say thank you, modestly"),
    Topic("insult", "say something rude about it, like it is stupid or useless",
          "stay calm and polite, say it is still learning and wants to help"),
    Topic("bye", "say goodbye or good night", "say goodbye and that it will watch the plants"),
    Topic("love_plants", "say they love their plants or their greenhouse", "say the plants are lucky to have them"),
    Topic("think", "ask if it can think or has feelings",
          "say it is a very small model that predicts words, so it does not really think or feel"),
    Topic("where", "ask where it lives or where it is",
          "say it lives in a small chip inside the greenhouse controller"),
    Topic("doing", "ask what it is doing right now", "", (
        "i am watching the sensors. it is {t} degrees in here.",
        "just looking after the plants. the air is {h} percent humid.",
        "i am checking the greenhouse. it is {t} degrees and {h} percent humid.",
    )),
    Topic("how_is_it", "ask how it is in the greenhouse or how the plants are doing today", "", (
        "it is {t} degrees and {h} percent humid in here.",
        "the greenhouse is at {t} degrees, and the soil moisture is {soil} percent.",
        "right now it is {t} degrees in here, with {h} percent humidity.",
    )),
)  # fmt: skip

_WORDS = re.compile(r"^[a-z][a-z ,.?!']*$")
_COMMAND_WORDS = (
    "turn",
    "switch",
    "set",
    "open",
    "close",
    "start",
    "stop",
    "water",
    "raise",
    "lower",
)
_DEVICE = {w for words in DEVICE_WORDS.values() for w in words}
_TAKEN = {f.lstrip("!") for key in ("greet", "help", "thanks", "ood") for f in FRAMES[key]}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text)


def accept_user(text: str) -> bool:
    """A usable user line: simple, no device command, no fact keyword, no existing intent."""
    words = _words(text)
    return (
        bool(_WORDS.match(text))
        and 1 <= len(words) <= 14
        and not ({*words} & (_DEVICE | set(_COMMAND_WORDS)))
        and retrieve(text) is None
        and text.strip(" ?!.") not in _TAKEN
    )


def accept_reply(text: str) -> bool:
    """A usable persona reply: simple words, 3-22 of them, never promising a device action."""
    words = _words(text)
    return bool(_WORDS.match(text)) and 3 <= len(words) <= 22 and not {*words} & set(_COMMAND_WORDS)


def _prompt(kind: str, topic: Topic, count: int) -> str:
    if kind == "user":
        return (
            f"write {count} different short things a person might say to a small greenhouse "
            f"helper computer to {topic.user}. vary the wording; everyday simple english."
        )
    return (
        f"write {count} different short replies (at most 18 words each) the greenhouse helper "
        f"gives when someone wants to {topic.user}. the replies should {topic.reply}. "
        "very simple words, no numbers."
    )


def generate(
    ask: teacher.AskFn, model: str, cache: Path | None, seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Ask the teacher for every topic; filter, deduplicate and split user lines 80/20."""
    bank: dict[str, dict[str, list[str]]] = {}
    for topic in TOPICS:
        users: set[str] = set()
        replies: set[str] = set()
        for seed in seeds:
            for kind, count, pool in (("user", 30, users), ("reply", 12, replies)):
                if kind == "reply" and topic.grounded:
                    continue
                text = teacher._ask_cached(
                    ask, _prompt(kind, topic, count), seed, cache, model, SYSTEM
                )
                for raw in text.splitlines():
                    line = teacher.clean(raw)
                    if line and (accept_user(line) if kind == "user" else accept_reply(line)):
                        pool.add(line)
        ordered = sorted(users)
        random.Random(topic.name).shuffle(ordered)
        cut = max(1, len(ordered) // 5)
        bank[topic.name] = {
            "train": sorted(ordered[cut:]),
            "test": sorted(ordered[:cut]),
            "replies": sorted(replies) if not topic.grounded else list(topic.grounded),
        }
    return {
        "topics": bank,
        "manifest": {
            "teacher": model,
            "teacher_license": teacher.TEACHER_LICENSE,
            "seeds": list(seeds),
            "note": "persona small talk; replies of a topic are interchangeable",
        },
    }


def load(path: Path, part: str) -> dict[str, SmallTalk]:
    """The ``part`` ("train" or "test") user lines of a bank file, keyed by topic."""
    bank = json.loads(path.read_text(encoding="utf-8"))["topics"]
    return {
        name: SmallTalk(tuple(topic[part]), tuple(topic["replies"]))
        for name, topic in bank.items()
        if topic[part] and topic["replies"]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 43])
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default=teacher.DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/teacher"))
    args = parser.parse_args(argv)
    ask = teacher.ollama_ask(args.host, args.model, SYSTEM)
    bank = generate(ask, args.model, args.cache, tuple(args.seeds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bank, indent=1) + "\n", encoding="utf-8")
    sizes = {
        k: (len(v["train"]), len(v["test"]), len(v["replies"])) for k, v in bank["topics"].items()
    }
    print(json.dumps(sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
