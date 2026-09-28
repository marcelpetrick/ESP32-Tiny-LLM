# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Vocabulary and sentence frames for user turns.

Frames prefixed with ``!`` are **held out**: they never appear in training data and
form the held-out-template evaluation suite (vision §19, docs/04 §5).
"""

from __future__ import annotations

# Device nouns used in user text. Every synonym appears in training.
DEVICE_WORDS: dict[str, tuple[str, ...]] = {
    "fan": ("fan", "ventilator", "blower", "air fan"),
    "heat": ("heater", "heating", "radiator", "warmer"),
    "pump": ("pump", "water pump", "irrigation", "sprinkler"),
    "light": ("light", "lights", "grow light", "lamp"),
    "win": ("window", "roof window", "skylight", "vent"),
}

NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

POLITE_PREFIX = ("please ", "could you ", "can you ", "hey, ", "hi, ", "would you ", "kindly ")
POLITE_SUFFIX = (" please", " thanks", " thank you", " now", " for me", " right away")
FILLERS = ("um ", "uh ", "so ", "ok ", "well ")

# fmt: off
FRAMES: dict[str, tuple[str, ...]] = {
    "on": (
        "turn on the {dev}", "switch on the {dev}", "turn the {dev} on", "start the {dev}",
        "{dev} on", "put the {dev} on", "activate the {dev}", "i want the {dev} on",
        "!get the {dev} going", "!power up the {dev}", "!could the {dev} be on",
    ),
    "off": (
        "turn off the {dev}", "switch off the {dev}", "turn the {dev} off", "stop the {dev}",
        "{dev} off", "shut off the {dev}", "deactivate the {dev}", "i want the {dev} off",
        "!kill the {dev}", "!power down the {dev}", "!no more {dev}",
    ),
    "fan_set": (
        "set the {dev} to {n}", "fan level {n}", "put the {dev} on level {n}",
        "{dev} to {n}", "change the {dev} to level {n}", "make the {dev} level {n}",
        "!i want level {n} on the {dev}", "!run the {dev} at {n}",
    ),
    "up": (
        "turn up the {dev}", "{dev} higher", "increase the {dev}", "more {dev}",
        "raise the {dev}", "{dev} up a bit", "make the {dev} stronger",
        "!crank up the {dev}", "!a notch more {dev}",
    ),
    "down": (
        "turn down the {dev}", "{dev} lower", "decrease the {dev}", "less {dev}",
        "reduce the {dev}", "{dev} down a bit", "make the {dev} weaker",
        "!bring the {dev} down", "!a notch less {dev}",
    ),
    "max": ("{dev} to max", "full {dev}", "{dev} at maximum", "!{dev} as high as it goes"),
    "light_set": (
        "set the {dev} to {n} percent", "{dev} to {n} percent", "dim the {dev} to {n} percent",
        "make the {dev} {n} percent", "!i want the {dev} at {n} percent",
    ),
    "open": (
        "open the {dev}", "open up the {dev}", "{dev} open", "let some air in",
        "!crack the {dev} open", "!get the {dev} open",
    ),
    "close": (
        "close the {dev}", "shut the {dev}", "{dev} closed", "close it up",
        "!pull the {dev} shut", "!get the {dev} closed",
    ),
    "read_t": (
        "what is the temperature", "how warm is it", "temperature?", "how hot is it in here",
        "tell me the temperature", "what's the temp", "!how many degrees is it",
        "!is it warm in here",
    ),
    "read_h": (
        "what is the humidity", "how humid is it", "humidity?", "tell me the humidity",
        "how damp is the air", "!what does the humidity sensor say",
    ),
    "read_soil": (
        "how wet is the soil", "soil moisture?", "what is the soil moisture",
        "do the plants have enough water", "is the soil dry", "!how moist is the soil",
    ),
    "read_dev": (
        "is the {dev} on", "what is the {dev} doing", "{dev} status", "how is the {dev} set",
        "!tell me about the {dev}",
    ),
    "read_pa": (
        "how much current does the pump draw", "pump current?", "what is the pump current",
        "!how many amps does the pump use",
    ),
    "status": (
        "how is everything", "status report", "give me a status", "is everything ok",
        "how are things in the greenhouse", "any news", "!what is going on in here",
        "!quick overview please",
    ),
    "diagnose": (
        "is something wrong", "any problems", "do you see a problem", "check for faults",
        "run a diagnosis", "why is something off", "!what is broken", "!find the problem",
    ),
    "diag_dev": (
        "why is the {dev} so loud", "the {dev} sounds strange", "is the {dev} ok",
        "something is wrong with the {dev}", "why does the {dev} act weird",
        "!the {dev} makes a funny noise", "!check the {dev}",
    ),
    "complain_humid": (
        "why is it so sticky in here", "it is so humid", "the air feels damp",
        "why is it still uncomfortable in here", "it feels muggy",
        "!the air is heavy and wet", "!why is it so clammy",
    ),
    "complain_hot": (
        "it is too hot", "why is it so hot in here", "i am sweating", "it is boiling in here",
        "cool it down", "!way too warm in here", "!it feels like an oven",
    ),
    "complain_cold": (
        "it is too cold", "why is it so cold", "i am freezing", "warm it up",
        "it is chilly in here", "!brr it is cold", "!the plants will freeze",
    ),
    "complain_dry": (
        "the plants look thirsty", "the plants are wilting", "water the plants",
        "the soil looks dry", "!the leaves are drooping", "!give the plants a drink",
    ),
    "ack": (
        "clear the error", "reset the alarm", "acknowledge the fault", "clear the fault",
        "reset the error", "!dismiss the warning", "!the fault is fixed, reset it",
    ),
    "greet": ("hello", "hi", "good morning", "hey there", "!greetings", "!hi there"),
    "help": (
        "what can you do", "help", "what are you", "how can you help me",
        "!what are your skills", "!who are you",
    ),
    "thanks": ("thanks", "thank you", "great, thanks", "perfect", "!cheers", "!much appreciated"),
    "ood": (
        "what is the capital of france", "play some music", "order a pizza",
        "tell me a joke", "what time is it", "who won the football game", "write a poem",
        "how do i bake bread", "what is the weather tomorrow", "call my mother",
        "!translate this to spanish", "!what is two plus two", "!book a flight to rome",
    ),
    "set_temp": (
        "set the temperature to {n} degrees", "make it {n} degrees", "heat to {n} degrees",
        "!i want {n} degrees",
    ),
    "it_on": ("turn it on", "switch it on", "start it", "!power it up"),
    "it_off": ("turn it off", "switch it off", "stop it", "!shut it down"),
    "it_up": ("a bit more", "higher", "turn it up", "more", "!a little stronger"),
    "it_down": ("a bit less", "lower", "turn it down", "less", "!a little weaker"),
    "it_set": ("set it to {n}", "make it {n}", "{n} please", "!put it on {n}"),
    "and_dev": ("and the {dev}", "the {dev} too", "same for the {dev}", "!also the {dev}"),
    "what_about": ("what about the humidity", "and the humidity", "!humidity as well"),
    "correct": (
        "no, the {dev}", "no, i meant the {dev}", "not that, the {dev}", "!wrong one, the {dev}",
    ),
    "compound": (
        "turn on the {dev} and the {dev2}", "switch on the {dev} and {dev2}",
        "!i want the {dev} and the {dev2} on",
    ),
}
# fmt: on


def split_frames(intent: str, heldout: bool) -> tuple[str, ...]:
    """Return the training (``heldout=False``) or held-out frames of ``intent``."""
    frames = FRAMES[intent]
    if heldout:
        return tuple(f[1:] for f in frames if f.startswith("!"))
    return tuple(f for f in frames if not f.startswith("!"))
