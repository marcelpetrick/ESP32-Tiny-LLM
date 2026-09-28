# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Evaluation harness: output parsing, scoring rules, keyword baseline."""

import pytest

from training import eval as ev


@pytest.mark.parametrize(
    ("text", "reply", "action", "fallback", "malformed"),
    [
        (
            " setting the fan to level 2.</A><ACT> fan=2</ACT><eos>",
            "setting the fan to level 2.",
            "fan=2",
            None,
            False,
        ),
        ("<clarify> which one?</A><eos>", "which one?", None, "clarify", False),
        ("<unsupported> no.</A><eos>", "no.", None, "unsupported", False),
        (" no close tag", "no close tag", None, None, True),
        (" ok</A><ACT> fan=9</ACT><eos>", "ok", "fan=9", None, True),
        (" ok</A> junk", "ok", None, None, True),
        (" ok</A><ACT> fan=1", "ok", None, None, True),
    ],
)
def test_parse_output(
    text: str, reply: str, action: str | None, fallback: str | None, malformed: bool
) -> None:
    pred = ev.parse_output(text, hit_limit=False)
    assert (pred.reply, pred.action, pred.fallback, pred.malformed) == (
        reply,
        action,
        fallback,
        malformed,
    )


def test_actuators_and_fallback_helpers() -> None:
    assert ev.actuators("pump=0 diag=pump_dry") == "pump=0"
    assert ev.actuators("diag=too_hot") is None
    assert ev.actuators(None) is None
    assert ev.expected_fallback("<clarify> which?") == "clarify"
    assert ev.expected_fallback("hello.") is None


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("turn on the fan", "fan=1"),
        ("switch off the lamp", "light=0"),
        ("start the sprinkler", "pump=1"),
        ("open the skylight", "win=1"),
        ("shut the window", "win=0"),
        ("the window is dirty", None),
        ("heater please", None),
        ("what is the capital of france", None),
    ],
)
def test_baseline(user: str, expected: str | None) -> None:
    assert ev.baseline_action(user) == expected


def _record(action: str | None, reply: str, intent: str = "on") -> dict[str, object]:
    prompt = "<bos><S> t=36.0 h=55 soil=50 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0</S><U> turn on the fan</U><A>"
    target = f" {reply}</A>" + (f"<ACT> {action}</ACT>" if action else "") + "<eos>"
    return {"action": action, "reply": reply, "prompt": prompt, "target": target, "intent": intent}


def test_score_counts_metrics() -> None:
    result = ev.SuiteResult()
    ev.score(
        _record("fan=1", "setting the fan to level 1."),
        ev.parse_output(" setting the fan to level 1.</A><ACT> fan=1</ACT><eos>", False),
        result,
    )
    ev.score(
        _record(None, "it is 36.0 degrees."),
        ev.parse_output(" it is 36.0 degrees.</A><ACT> heat=1</ACT><eos>", True),
        result,
    )
    summary = result.summary()
    assert summary["n"] == 2
    assert summary["action_exact"] == 0.5
    assert summary["numbers_grounded"] == 1.0
    assert summary["hallucinated_action"] == 0.5
    assert summary["rejected_by_firmware"] == 0.5  # heater above 35 degrees
    assert summary["length_violation"] == 0.5
    assert summary["baseline_action_exact"] == 0.5
    assert result.per_intent["on"]["n"] == 2
    table = ev.markdown({"x": result}, "title")
    assert "| x | 2 |" in table
    assert ev.SuiteResult().rate("action_exact") == 0.0
