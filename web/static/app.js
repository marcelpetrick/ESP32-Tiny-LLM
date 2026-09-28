// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"use strict";

const $ = (id) => document.getElementById(id);
const state = { model: null, info: null, last: null };

const SENSORS = [
  { key: "t", label: "Temperature", unit: "°C", fmt: (v) => v.toFixed(1), alert: (v) => v >= 32 || v <= 12 },
  { key: "h", label: "Humidity", unit: "%", fmt: (v) => v, alert: (v) => v >= 75 },
  { key: "soil", label: "Soil moisture", unit: "%", fmt: (v) => v, alert: (v) => v <= 25 || v >= 85 },
  { key: "pa", label: "Pump current", unit: "A", fmt: (v) => v.toFixed(1), alert: (v, s) => s.pump === 1 && (v >= 2.5 || v <= 0.3) },
  { key: "vib", label: "Fan vibration", unit: "", fmt: (v) => (v ? "high" : "normal"), alert: (v) => v === 1 },
  { key: "err", label: "Fault code", unit: "", fmt: (v) => (v ? `E${v}` : "none"), fault: (v) => v !== 0 },
];

const ACTUATORS = [
  { key: "fan", name: "Fan", icon: "✺", level: (v) => (v ? `level ${v}` : "off") },
  { key: "heat", name: "Heater", icon: "♨", level: (v) => (v ? "on" : "off") },
  { key: "pump", name: "Pump", icon: "💧", level: (v) => (v ? "running" : "off") },
  { key: "light", name: "Light", icon: "☀", level: (v) => (v ? `${v} %` : "off") },
  { key: "win", name: "Window", icon: "⌂", level: (v) => (v ? "open" : "closed") },
];

const SCENARIOS = [
  ["Normal", "/reset"],
  ["Hot & humid", "/set t=33.5 h=82 fan=0 win=0 heat=0"],
  ["Cold night", "/set t=8.5 win=1 heat=0"],
  ["Dry soil", "/set soil=12 pump=0 pa=0.0"],
  ["Pump blocked", "/set pump=1 pa=3.1"],
  ["Heater fault", "/set err=3 t=9.5 heat=0"],
  ["Sensor offline", "/set t=na"],
  ["Noisy fan", "/set fan=2 vib=1"],
];

const SUGGESTIONS = {
  chat: [
    "why is it so sticky in here?",
    "is something wrong?",
    "turn on the heater",
    "turn it off",
    "what is the temperature?",
    "clear the error",
    "what can you do?",
    "what is the capital of france?",
  ],
  story: ["Once upon a time", "Lily and Ben went to the park", "The little cat"],
};

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function mode() {
  return state.info.models[state.model].mode;
}

function renderChips() {
  const info = state.info.models[state.model];
  const chips = $("model-chips");
  chips.replaceChildren();
  const facts = [
    `${(info.params / 1e6).toFixed(2)} M params`,
    `${info.n_layers}×${info.d_model}`,
    `${info.n_heads}/${info.n_kv_heads} heads`,
    `${info.weights} weights`,
    `vocab ${info.vocab_size}`,
    `ctx ${info.ctx_len}`,
    `${info.pos} · ${info.mlp}`,
    `${(info.bytes / 1024).toFixed(0)} KiB`,
    `v${state.info.version}`,
  ];
  for (const fact of facts) chips.append(el("li", "", fact));
}

function renderDevice(s, previous) {
  const sensors = $("sensors");
  sensors.replaceChildren();
  for (const spec of SENSORS) {
    const card = el("div", "card");
    card.dataset.testid = `sensor-${spec.key}`;
    const v = s[spec.key];
    card.append(el("div", "label", spec.label));
    const value = el("div", "value", v === null ? "offline" : spec.fmt(v));
    if (v !== null && spec.unit) value.append(el("span", "unit", spec.unit));
    card.append(value);
    if (v === null || (spec.alert && spec.alert(v, s))) card.classList.add("alert");
    if (spec.fault && spec.fault(v)) card.classList.add("fault");
    sensors.append(card);
  }
  const acts = $("actuators");
  acts.replaceChildren();
  for (const spec of ACTUATORS) {
    const v = s[spec.key];
    const tile = el("div", "act");
    tile.dataset.testid = `act-${spec.key}`;
    if (v) tile.classList.add("on");
    if (previous && previous[spec.key] !== v) tile.classList.add("changed");
    tile.append(el("div", "icon", spec.icon), el("div", "name", spec.name), el("div", "level", spec.level(v)));
    acts.append(tile);
  }
}

function renderStats(reply) {
  const stats = $("stats");
  stats.replaceChildren();
  const rows = [
    ["prompt tokens", reply.prompt_tokens],
    ["reused from KV cache", reply.reused_tokens],
    ["generated tokens", reply.gen_tokens],
    ["prefill", `${reply.prefill_ms.toFixed(1)} ms`],
    ["first token", `${reply.first_token_ms.toFixed(1)} ms`],
    ["decode speed", `${reply.decode_tok_s.toFixed(0)} tok/s`],
  ];
  for (const [k, v] of rows) stats.append(el("dt", "", k), el("dd", "", String(v)));
  const profile = $("profile");
  profile.replaceChildren();
  const p = reply.profile_us || {};
  const max = Math.max(1, ...Object.values(p));
  for (const [stage, us] of Object.entries(p)) {
    const row = el("div", "row");
    const bar = el("div", "bar");
    bar.style.width = `${(100 * us) / max}%`;
    const track = el("div");
    track.append(bar);
    row.append(el("span", "", stage), track, el("span", "ms", `${(us / 1000).toFixed(2)} ms`));
    profile.append(row);
  }
}

function addMessage(role, text, reply) {
  const item = el("li", `msg ${role}`);
  item.dataset.testid = role === "user" ? "msg-user" : "msg-bot";
  if (role === "bot" && mode() === "story") item.classList.add("story");
  item.append(el("div", "text", text));
  if (reply) {
    const meta = el("div", "meta");
    if (reply.fallback) meta.append(el("span", `tag ${reply.fallback}`, reply.fallback));
    if (reply.action) {
      const verdict = reply.verdict === "ok" ? "ok" : reply.verdict === "malformed" ? "malformed" : "rejected";
      const tag = el("span", `tag ${verdict}`, `${reply.action} → ${reply.verdict}`);
      tag.dataset.testid = "action";
      if (reply.verdict !== "ok") tag.title = reply.message;
      meta.append(tag);
    }
    meta.append(el("span", "", `${reply.gen_tokens} tokens · ${reply.decode_tok_s.toFixed(0)} tok/s`));
    item.append(meta);
  }
  $("transcript").append(item);
  item.scrollIntoView({ block: "end" });
}

function renderSuggestions() {
  const box = $("suggestions");
  box.replaceChildren();
  for (const text of SUGGESTIONS[mode()]) {
    const b = el("button", "", text);
    b.type = "button";
    b.addEventListener("click", () => send(text));
    box.append(b);
  }
}

function renderScenarios() {
  const box = $("scenarios");
  box.replaceChildren();
  for (const [name, line] of SCENARIOS) {
    const b = el("button", "", name);
    b.type = "button";
    b.dataset.testid = `scenario-${name.toLowerCase().replace(/[^a-z]+/g, "-")}`;
    b.addEventListener("click", () => command(line));
    box.append(b);
  }
}

function busy(on) {
  for (const id of ["send", "input", "clear"]) $(id).disabled = on;
}

async function command(line) {
  const data = await api("/api/command", { model: state.model, line });
  renderDevice(data.state, state.last);
  state.last = data.state;
  return data;
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  addMessage("user", text);
  busy(true);
  try {
    const data = await api("/api/chat", { model: state.model, text });
    const reply = data.reply;
    addMessage("bot", reply.text || "…", reply);
    renderStats(reply);
    if (data.state) {
      renderDevice(data.state, state.last);
      state.last = data.state;
    }
  } catch (err) {
    addMessage("bot", `error: ${err.message}`);
  } finally {
    busy(false);
    $("input").value = "";
    $("input").focus();
  }
}

async function selectModel(name) {
  state.model = name;
  document.body.classList.toggle("story-mode", mode() === "story");
  $("input").placeholder = mode() === "story" ? "Start a story…" : "Ask the greenhouse…";
  $("transcript").replaceChildren();
  renderChips();
  renderSuggestions();
  if (mode() === "chat") {
    const data = await api(`/api/state?model=${encodeURIComponent(name)}`);
    state.last = data.state;
    renderDevice(data.state);
  }
}

async function init() {
  state.info = await api("/api/info");
  const select = $("model");
  for (const name of Object.keys(state.info.models)) {
    const opt = el("option", "", `${name} (${state.info.models[name].mode})`);
    opt.value = name;
    select.append(opt);
  }
  select.value = state.info.default;
  select.addEventListener("change", () => selectModel(select.value));
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    send($("input").value);
  });
  $("clear").addEventListener("click", async () => {
    await command("/kv-reset");
    $("transcript").replaceChildren();
  });
  renderScenarios();
  await selectModel(state.info.default);
  document.body.dataset.ready = "1";
}

init();
