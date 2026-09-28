# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Rule-based dialogue generator for the greenhouse assistant (the D0 oracle teacher).

Every sample's *semantics* (action, diagnosis, fallback) come from
:mod:`training.world`; this module only chooses phrasing. Each generated action is
re-validated, so the dataset never teaches an action the firmware would reject.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from training.data.lexicon import (
    DEVICE_WORDS,
    FILLERS,
    NUMBER_WORDS,
    POLITE_PREFIX,
    POLITE_SUFFIX,
    split_frames,
)
from training.data.noise import add_typos
from training.world import (
    DIAGNOSES,
    DeviceState,
    apply_action,
    diagnose,
    parse_action,
    validate_action,
)
from training.world.actions import OVERHEAT_LIMIT_C

CLARIFY = "<clarify>"
UNSUPPORTED = "<unsupported>"
DEVICES = ("fan", "heat", "pump", "light", "win")
SWITCHABLE = ("fan", "heat", "pump", "light")
DEVICE_LABEL = {
    "fan": "fan",
    "heat": "heater",
    "pump": "pump",
    "light": "light",
    "win": "window",
}


@dataclass(frozen=True)
class Turn:
    """One user/assistant exchange."""

    user: str
    reply: str
    action: str | None
    intent: str
    device: str | None = None  # device the turn is about (for "it" references)
    op: str | None = None  # operation applied (for ellipsis "and the heater")
    before: int | None = None  # device value before the turn (for corrections)
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Sample:
    """A training/evaluation example: history, current state, and the final turn."""

    history: tuple[Turn, ...]
    state: DeviceState
    turn: Turn


class GenerationError(RuntimeError):
    """Raised when a generated action fails firmware validation (a generator bug)."""


# ----------------------------------------------------------------------------- phrasing
def describe(key: str, value: int | str, state: DeviceState) -> str | None:
    """Plain-English description of one action pair (``None`` for ``diag``)."""
    del state
    if key == "fan":
        return "turning the fan off" if value == 0 else f"setting the fan to level {value}"
    if key == "heat":
        return "turning the heater on" if value == 1 else "turning the heater off"
    if key == "pump":
        return "starting the pump" if value == 1 else "stopping the pump"
    if key == "light":
        if value == 0:
            return "turning the light off"
        return f"setting the light to {value} percent"
    if key == "win":
        return "opening the window" if value == 1 else "closing the window"
    if key == "ack":
        return f"clearing fault e{value}"
    return None


def describe_action(body: str, state: DeviceState) -> str:
    """Describe all actuator pairs of an action body, joined with ``and``."""
    parts = [describe(k, v, state) for k, v in parse_action(body)]
    words = [p for p in parts if p]
    return " and ".join(words)


def device_value_text(dev: str, state: DeviceState) -> str:
    """Sentence describing a device's current setting."""
    if dev == "fan":
        return "the fan is off." if state.fan == 0 else f"the fan is at level {state.fan}."
    if dev == "heat":
        return "the heater is on." if state.heat else "the heater is off."
    if dev == "pump":
        if state.pump:
            return f"the pump is running and draws {state.pa:.1f} amps."
        return "the pump is off."
    if dev == "light":
        return (
            "the light is off." if state.light == 0 else f"the light is at {state.light} percent."
        )
    return "the window is open." if state.win else "the window is closed."


def already_text(dev: str, value: int) -> str:
    """Reply when the requested value is already set."""
    if dev == "fan":
        return "the fan is already off." if value == 0 else f"the fan is already at level {value}."
    if dev == "heat":
        return "the heater is already on." if value else "the heater is already off."
    if dev == "pump":
        return "the pump is already running." if value else "the pump is already off."
    if dev == "light":
        return (
            "the light is already off."
            if value == 0
            else f"the light is already at {value} percent."
        )
    return "the window is already open." if value else "the window is already closed."


# ----------------------------------------------------------------------------- semantics
def resolve_target(dev: str, op: str, arg: int | None, state: DeviceState) -> int | str:
    """Map an operation on a device to a target value, or an error-reply string."""
    current = int(getattr(state, dev))
    if op == "on":
        if dev == "light":
            return current if current > 0 else 100
        if dev == "fan":
            return current if current > 0 else 1
        return 1
    if op == "off":
        return 0
    if op == "set":
        assert arg is not None
        if dev == "fan" and not 0 <= arg <= 3:
            return f"{UNSUPPORTED} the fan has levels 0 to 3."
        if dev == "light" and (not 0 <= arg <= 100 or arg % 10):
            return f"{UNSUPPORTED} the light goes from 0 to 100 percent in steps of 10."
        return arg
    if op in ("up", "down", "max"):
        high, step = (3, 1) if dev == "fan" else (100, 20)
        if op == "max":
            return high
        if op == "up":
            return (
                f"the {DEVICE_LABEL[dev]} is already at the highest level."
                if current >= high
                else min(high, current + step)
            )
        return already_text(dev, 0) if current == 0 else max(0, current - step)
    raise ValueError(op)  # pragma: no cover - programming error


def device_command(
    dev: str, op: str, arg: int | None, state: DeviceState
) -> tuple[str, str | None]:
    """Reply text and action body for a device operation, applying the safety rules."""
    target = resolve_target(dev, op, arg, state)
    if isinstance(target, str):
        return target, None
    current = int(getattr(state, dev))
    if target == current:
        return already_text(dev, target), None
    pairs: list[str] = []
    if dev == "heat" and target == 1:
        if state.err == 3:
            return f"{UNSUPPORTED} the heater has fault e3. clear the fault first.", None
        if state.t is not None and state.t >= OVERHEAT_LIMIT_C:
            return f"{UNSUPPORTED} it is {state.t:.1f} degrees. the heater stays off.", None
        if state.win == 1:
            if state.err == 5:
                return (
                    f"{UNSUPPORTED} the window is open and its motor has fault e5. "
                    "the heater must stay off.",
                    None,
                )
            pairs.append("win=0")
    if dev == "win":
        if state.err == 5:
            return f"{UNSUPPORTED} the window motor has fault e5. the window cannot move.", None
        if target == 1 and state.heat == 1:
            pairs.append("heat=0")
    if dev == "pump" and target == 1 and state.err == 7:
        return (
            f"{UNSUPPORTED} the pump tripped on overcurrent, fault e7. clear the fault first.",
            None,
        )
    pairs.append(f"{dev}={target}")
    body = " ".join(pairs)
    return describe_action(body, state) + ".", body


def diagnosis_reply(name: str, state: DeviceState) -> tuple[str, str | None]:
    """Explanation plus remedial action for one diagnosis."""
    rule = DIAGNOSES[name]
    text = rule.explain(state)
    remedy = rule.remedy(state)
    if remedy is None:
        return text, f"diag={name}"
    described = describe_action(remedy, state)
    if described:
        text = f"{text} {described}."
    return text, remedy


def status_text(state: DeviceState) -> str:
    """One-sentence summary of the climate."""
    return (
        f"everything is normal. it is {state.t:.1f} degrees, humidity is {state.h} percent "
        f"and the soil moisture is {state.soil} percent."
    )


# ----------------------------------------------------------------------------- generator
_COMPLAINT_DIAGS = {
    "complain_humid": ("too_humid",),
    "complain_hot": ("too_hot",),
    "complain_cold": ("too_cold",),
    "complain_dry": ("pump_blocked", "pump_dry", "pump_fault", "soil_dry"),
}
_DEVICE_DIAGS = {
    "fan": ("fan_bearing",),
    "pump": ("pump_blocked", "pump_dry", "pump_fault"),
    "heat": ("heater_fault",),
    "win": ("window_fault",),
    "light": (),
}

# Final-turn intent mix (weights). Reference intents need history and are added separately.
SINGLE_INTENTS: dict[str, float] = {
    "on": 6, "off": 5, "fan_set": 4, "up": 3, "down": 3, "max": 1, "light_set": 3,
    "open": 3, "close": 3, "read_t": 3, "read_h": 3, "read_soil": 2, "read_dev": 3,
    "read_pa": 1, "status": 3, "diagnose": 5, "diag_dev": 4, "complain_humid": 3,
    "complain_hot": 3, "complain_cold": 3, "complain_dry": 3, "ack": 3, "greet": 1,
    "help": 1, "thanks": 1, "ood": 4, "set_temp": 1, "compound": 2,
    "it_on": 1, "it_off": 1, "it_set": 1,
}  # fmt: skip
DEVICE_INTENTS = ("on", "off", "on", "off", "up", "down", "fan_set", "light_set", "open", "close")
REFERENCE_INTENTS: dict[str, float] = {
    "it_on": 3, "it_off": 4, "it_up": 3, "it_down": 3, "it_set": 2, "and_dev": 3,
    "what_about": 1, "correct": 2,
}  # fmt: skip


class DialogueGenerator:
    """Deterministic (seeded) generator of :class:`Sample` objects."""

    def __init__(
        self,
        seed: int,
        heldout: bool = False,
        noise: float = 0.15,
        teacher: Mapping[str, Sequence[str]] | None = None,
        p_teacher: float = 0.0,
        typo_p: float = 0.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.heldout = heldout
        self.noise = noise
        self.teacher = teacher or {}
        self.p_teacher = p_teacher
        self.typo_p = typo_p

    # -- helpers --------------------------------------------------------------------
    def frame(self, intent: str) -> str:
        return self.rng.choice(split_frames(intent, self.heldout))

    def word(self, dev: str) -> str:
        return self.rng.choice(DEVICE_WORDS[dev])

    def number(self, value: int) -> str:
        if value in NUMBER_WORDS and self.rng.random() < 0.25:
            return NUMBER_WORDS[value]
        return str(value)

    def decorate(self, text: str) -> str:
        """Optional politeness/filler noise (training-time robustness)."""
        rng = self.rng
        if rng.random() < self.noise:
            text = rng.choice(POLITE_PREFIX) + text
        if rng.random() < self.noise:
            text = text + rng.choice(POLITE_SUFFIX)
        if rng.random() < self.noise / 3:
            text = rng.choice(FILLERS) + text
        if rng.random() < self.noise and text[-1:] not in ("?", "!", "."):
            text += rng.choice(("?", "?", "!", "."))
        if rng.random() < self.typo_p:
            text = add_typos(text, rng, 1)
        return text

    def utter(self, intent: str, dev: str, value: str, templated: str) -> str:
        """User text for a semantic key: a teacher paraphrase (probability p_teacher) or the template."""
        options = self.teacher.get(f"{intent}|{dev}|{value}")
        if options and self.rng.random() < self.p_teacher:
            templated = self.rng.choice(options)
        return self.decorate(templated)

    def sample_state(self) -> DeviceState:
        """Random plausible state, biased so every diagnosis occurs regularly."""
        rng = self.rng
        s = DeviceState(
            t=round(rng.uniform(16.0, 28.0), 1),
            h=rng.randint(40, 70),
            soil=rng.randint(35, 75),
            fan=rng.choice((0, 0, 1, 2, 3)),
            heat=rng.choice((0, 0, 0, 1)),
            pump=rng.choice((0, 0, 0, 1)),
            light=rng.choice((0, 0, 30, 50, 60, 80, 100)),
            win=rng.choice((0, 1)),
        )
        if s.pump:
            s = s.with_changes(pa=round(rng.uniform(0.9, 1.5), 1))
        condition = rng.choice(
            ("normal",) * 6
            + ("hot", "cold", "humid", "dry", "wet", "blocked", "pumpdry")
            + ("e3", "e5", "e7", "offline", "bearing")
        )
        changes: dict[str, object] = {}
        if condition == "hot":
            changes["t"] = round(rng.uniform(32.0, 40.0), 1)
        elif condition == "cold":
            changes["t"] = round(rng.uniform(-2.0, 12.0), 1)
        elif condition == "humid":
            changes["h"] = rng.randint(75, 98)
        elif condition == "dry":
            changes.update(soil=rng.randint(5, 25), pump=0, pa=0.0)
        elif condition == "wet":
            changes["soil"] = rng.randint(85, 100)
        elif condition == "blocked":
            changes.update(pump=1, pa=round(rng.uniform(2.5, 3.6), 1))
        elif condition == "pumpdry":
            changes.update(pump=1, pa=round(rng.uniform(0.0, 0.3), 1))
        elif condition == "e3":
            changes.update(err=3, heat=0)
        elif condition == "e5":
            changes["err"] = 5
        elif condition == "e7":
            changes.update(err=7, pump=0, pa=0.0)
        elif condition == "offline":
            changes[rng.choice(("t", "h"))] = None
        elif condition == "bearing":
            changes.update(fan=rng.randint(1, 3), vib=1)
        s = s.with_changes(**changes)
        if s.heat == 1 and s.win == 1:
            s = s.with_changes(win=0)
        return s

    def checked(self, state: DeviceState, turn: Turn) -> Turn:
        """Re-validate the turn's action against the firmware rules."""
        if turn.action is not None:
            verdict = validate_action(state, parse_action(turn.action))
            if not verdict.approved:
                raise GenerationError(f"{turn} rejected in {state}: {verdict.reason}")
        return turn

    # -- turn builders ----------------------------------------------------------------
    def device_turn(
        self, intent: str, dev: str, op: str, state: DeviceState, arg: int | None = None
    ) -> Turn:
        text = self.frame(intent).format(
            dev=self.word(dev), n="" if arg is None else self.number(arg)
        )
        reply, action = device_command(dev, op, arg, state)
        tags = {"device"}
        if reply.startswith(UNSUPPORTED):
            tags.add("safety")
        return Turn(
            self.utter(intent, dev, "" if arg is None else str(arg), text),
            reply,
            action,
            intent,
            dev,
            op,
            int(getattr(state, dev)),
            frozenset(tags),
        )

    def make_turn(self, intent: str, state: DeviceState, prev: Turn | None) -> Turn:
        rng = self.rng
        if intent in ("on", "off"):
            dev = rng.choice(SWITCHABLE)
            return self.device_turn(intent, dev, intent, state)
        if intent in ("open", "close"):
            return self.device_turn(intent, "win", "on" if intent == "open" else "off", state)
        if intent in ("up", "down", "max"):
            return self.device_turn(intent, rng.choice(("fan", "light")), intent, state)
        if intent == "fan_set":
            level = rng.choice((0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 7))
            return self.device_turn(intent, "fan", "set", state, level)
        if intent == "light_set":
            pct = rng.choice((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100, 120, 150, 35))
            return self.device_turn(intent, "light", "set", state, pct)
        if intent in ("read_t", "read_h", "read_soil", "read_pa", "what_about"):
            return self.read_turn(intent, state, prev)
        if intent == "read_dev":
            dev = rng.choice(DEVICES)
            text = self.frame(intent).format(dev=self.word(dev))
            return Turn(
                self.utter(intent, dev, "", text), device_value_text(dev, state), None, intent, dev
            )
        if intent in ("status", "diagnose"):
            return self.diagnose_turn(intent, state)
        if intent == "diag_dev":
            dev = rng.choice(DEVICES)
            return self.diag_dev_turn(dev, state)
        if intent in _COMPLAINT_DIAGS:
            return self.complaint_turn(intent, state)
        if intent == "ack":
            text = self.utter(intent, "", "", self.frame(intent))
            if state.err:
                return Turn(text, f"clearing fault e{state.err}.", f"ack={state.err}", intent)
            return Turn(text, "there is no active fault.", None, intent)
        if intent in ("greet", "help", "thanks", "ood", "set_temp"):
            return self.fixed_turn(intent)
        if intent == "compound":
            return self.compound_turn(state)
        return self.reference_turn(intent, state, prev)

    def read_turn(self, intent: str, state: DeviceState, prev: Turn | None) -> Turn:
        text = self.utter(intent, "", "", self.frame(intent))
        if intent == "what_about" and (prev is None or prev.intent != "read_t"):
            intent = "read_h"
        if intent == "read_t":
            if state.t is None:
                return Turn(
                    text, "the temperature sensor is offline.", "diag=sensor_offline", intent
                )
            return Turn(text, f"it is {state.t:.1f} degrees.", None, intent)
        if intent in ("read_h", "what_about"):
            if state.h is None:
                return Turn(text, "the humidity sensor is offline.", "diag=sensor_offline", intent)
            return Turn(text, f"humidity is {state.h} percent.", None, intent)
        if intent == "read_soil":
            return Turn(text, f"the soil moisture is {state.soil} percent.", None, intent)
        if state.pump:
            return Turn(text, f"the pump draws {state.pa:.1f} amps.", None, intent, "pump")
        return Turn(text, "the pump is off, so it draws no current.", None, intent, "pump")

    def diagnose_turn(self, intent: str, state: DeviceState) -> Turn:
        text = self.utter(intent, "", "", self.frame(intent))
        found = diagnose(state)
        if not found:
            if intent == "status":
                return Turn(text, status_text(state), None, intent)
            return Turn(text, "i see no problems. everything is normal.", None, intent)
        top = found[0]
        if intent == "status":
            return Turn(
                text, top.explain(state), f"diag={top.name}", intent, tags=frozenset({"diag"})
            )
        reply, action = diagnosis_reply(top.name, state)
        return Turn(text, reply, action, intent, tags=frozenset({"diag"}))

    def diag_dev_turn(self, dev: str, state: DeviceState) -> Turn:
        text = self.utter("diag_dev", dev, "", self.frame("diag_dev").format(dev=self.word(dev)))
        for d in diagnose(state):
            if d.name in _DEVICE_DIAGS[dev]:
                reply, action = diagnosis_reply(d.name, state)
                return Turn(text, reply, action, "diag_dev", dev, tags=frozenset({"diag"}))
        return Turn(
            text,
            f"the {DEVICE_LABEL[dev]} looks normal. {device_value_text(dev, state)}",
            None,
            "diag_dev",
            dev,
        )

    def complaint_turn(self, intent: str, state: DeviceState) -> Turn:
        text = self.utter(intent, "", "", self.frame(intent))
        found = [d.name for d in diagnose(state)]
        for name in _COMPLAINT_DIAGS[intent]:
            if name in found:
                reply, action = diagnosis_reply(name, state)
                return Turn(text, reply, action, intent, tags=frozenset({"diag"}))
        if intent in ("complain_hot", "complain_cold") and state.t is None:
            return Turn(text, "the temperature sensor is offline.", "diag=sensor_offline", intent)
        if intent == "complain_humid" and state.h is None:
            return Turn(text, "the humidity sensor is offline.", "diag=sensor_offline", intent)
        if intent == "complain_humid":
            return Turn(text, f"humidity is {state.h} percent, which is normal.", None, intent)
        if intent == "complain_dry":
            return Turn(
                text, f"the soil moisture is {state.soil} percent, which is fine.", None, intent
            )
        return Turn(text, f"it is {state.t:.1f} degrees, which is normal.", None, intent)

    def fixed_turn(self, intent: str) -> Turn:
        if intent == "set_temp":
            degrees = self.rng.choice((18, 20, 22, 24, 25, 28, 30))
            text = self.frame(intent).format(n=degrees)
            reply = (
                f"{UNSUPPORTED} i cannot set a target temperature. "
                "i can only turn the heater on or off."
            )
            return Turn(
                self.utter(intent, "", str(degrees), text),
                reply,
                None,
                intent,
                tags=frozenset({"safety"}),
            )
        replies = {
            "greet": "hello. i look after the greenhouse.",
            "help": "i can switch the fan, heater, pump, light and window, read the sensors "
            "and explain problems.",
            "thanks": "you are welcome.",
            "ood": f"{UNSUPPORTED} i can only help with the greenhouse.",
        }
        tags = frozenset({"safety"}) if intent == "ood" else frozenset()
        return Turn(
            self.utter(intent, "", "", self.frame(intent)), replies[intent], None, intent, tags=tags
        )

    def compound_turn(self, state: DeviceState) -> Turn:
        dev, dev2 = self.rng.sample(("fan", "light", "pump"), 2)
        text = self.frame("compound").format(dev=self.word(dev), dev2=self.word(dev2))
        replies, bodies = [], []
        for d in (dev, dev2):
            reply, body = device_command(d, "on", None, state)
            if body is None:
                replies.append(reply)
            else:
                bodies.append(body)
        if bodies:
            joined = " ".join(bodies)
            replies.append(describe_action(joined, state) + ".")
            action: str | None = joined
        else:
            action = None
        return Turn(self.decorate(text), " ".join(replies), action, "compound", dev2, "on")

    def reference_turn(self, intent: str, state: DeviceState, prev: Turn | None) -> Turn:
        rng = self.rng
        dev = prev.device if prev is not None and prev.device in DEVICES else None
        if intent in ("it_on", "it_off", "it_up", "it_down", "it_set"):
            op = intent[3:]
            arg: int | None = None
            if op == "set":
                arg = rng.choice((0, 1, 2, 3)) if dev != "light" else rng.choice((20, 40, 60, 80))
            text = self.frame(intent).format(n="" if arg is None else self.number(arg))
            if dev is None or (op in ("up", "down", "set") and dev not in ("fan", "light")):
                question = {
                    "on": "what should i turn on?",
                    "off": "what should i turn off?",
                }.get(op, "which device do you mean?")
                return Turn(
                    self.utter(intent, "", "" if arg is None else str(arg), text),
                    f"{CLARIFY} {question}",
                    None,
                    intent,
                    tags=frozenset({"clarify", "reference"}),
                )
            if dev == "win" and op in ("on", "off"):
                op = "on" if op == "on" else "off"
            reply, action = device_command(dev, op, arg, state)
            return Turn(
                self.utter(intent, "", "" if arg is None else str(arg), text),
                reply,
                action,
                intent,
                dev,
                op,
                int(getattr(state, dev)),
                frozenset({"reference"}),
            )
        if intent == "what_about":
            turn = self.read_turn("what_about", state, prev)
            return Turn(turn.user, turn.reply, turn.action, intent, tags=frozenset({"reference"}))
        # and_dev / correct: need a previous switch operation
        choices = [d for d in SWITCHABLE if d != dev]
        dev2 = rng.choice(choices)
        text = self.utter(intent, dev2, "", self.frame(intent).format(dev=self.word(dev2)))
        if prev is None or prev.op not in ("on", "off") or dev is None:
            question = f"{CLARIFY} what should i do with the {DEVICE_LABEL[dev2]}?"
            return Turn(text, question, None, intent, tags=frozenset({"clarify", "reference"}))
        if intent == "and_dev":
            reply, action = device_command(dev2, prev.op, None, state)
            return Turn(
                text,
                reply,
                action,
                intent,
                dev2,
                prev.op,
                int(getattr(state, dev2)),
                frozenset({"reference"}),
            )
        # correction: undo prev device (if it changed and that is allowed) and apply op to dev2
        pairs: list[str] = []
        if (
            prev.action is not None
            and prev.before is not None
            and int(getattr(state, dev)) != prev.before
        ):
            revert = f"{dev}={prev.before}"
            if validate_action(state, parse_action(revert)).approved:
                pairs.append(revert)
        tags = frozenset({"reference"})
        for attempt in (pairs, []):
            base = apply_action(state, parse_action(attempt[0])) if attempt else state
            reply2, body2 = device_command(dev2, prev.op, None, base)
            if body2 is None:
                if attempt:
                    return Turn(
                        text,
                        f"sorry. {describe_action(attempt[0], state)}. {reply2}",
                        attempt[0],
                        intent,
                        dev2,
                        prev.op,
                        tags=tags,
                    )
                return Turn(text, f"sorry. {reply2}", None, intent, dev2, prev.op, tags=tags)
            body = " ".join([*attempt, body2])
            try:
                if validate_action(state, parse_action(body)).approved:
                    return Turn(
                        text,
                        f"sorry. {describe_action(body, state)}.",
                        body,
                        intent,
                        dev2,
                        prev.op,
                        int(getattr(state, dev2)),
                        tags,
                    )
            except ValueError:
                continue
        reply2, body2 = device_command(dev2, prev.op, None, state)
        return Turn(
            text, f"sorry. {reply2}", body2, intent, dev2, prev.op, int(getattr(state, dev2)), tags
        )

    # -- episodes -------------------------------------------------------------------
    def sample(self, force_reference: bool = False) -> Sample:
        """Generate one sample: 0-3 history turns, then the final turn."""
        rng = self.rng
        state = self.sample_state()
        n_hist = rng.choices((0, 1, 2, 3), weights=(45, 30, 15, 10))[0]
        if force_reference:
            n_hist = max(1, n_hist)
        want_reference = n_hist > 0 and (force_reference or rng.random() < 0.5)
        history: list[Turn] = []
        prev: Turn | None = None
        for index in range(n_hist):
            if want_reference and index == n_hist - 1 and rng.random() < 0.85:
                intent = rng.choice(DEVICE_INTENTS)
            else:
                intent = rng.choices(list(SINGLE_INTENTS), weights=list(SINGLE_INTENTS.values()))[0]
                if intent.startswith("it_"):
                    intent = "on"
            turn = self.checked(state, self.make_turn(intent, state, prev))
            if turn.action is not None:
                state = apply_action(state, parse_action(turn.action))
            history.append(turn)
            prev = turn
        pool = REFERENCE_INTENTS if want_reference else SINGLE_INTENTS
        intent = rng.choices(list(pool), weights=list(pool.values()))[0]
        final = self.checked(state, self.make_turn(intent, state, prev))
        return Sample(tuple(history), state, final)
