// Zero-dependency Node test for the frame/phase viewer
// (Training/webapp_api/static/app.js), same fake-DOM harness as its two
// companions -- the REAL app.js is loaded and driven, never reimplemented.
//
// Run with:  node Tests/webapp_api/test_viewer_navigation.mjs
//
// What it pins:
//   - phase band selection, and jumping to a band's first / last frame;
//   - manual previous / next stepping;
//   - sequence boundaries (no request past the video's real window bounds);
//   - empty / non-existent phase bands;
//   - the frame NUMBER shown always matches the frame URL actually requested;
//   - the time unit is echoed from the backend, never invented;
//   - return-to-selection, and keyboard control.

import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP_JS_PATH = path.join(HERE, "..", "..", "Training", "webapp_api", "static", "app.js");

class FakeTextNode {
  constructor(text) { this.nodeType = 3; this.textContent = String(text); this.children = []; }
}

class FakeElement {
  constructor(tag) {
    this.nodeType = 1;
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this._text = "";
    this.value = "";
    this.title = "";
    this.src = "";
    this.hidden = false;
    this.disabled = false;
    this.style = {};
    this.classList = { add() {}, remove() {}, contains() { return false; } };
    this.listeners = {};
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...nodes) { this.children = nodes; this._text = ""; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  async click() { for (const fn of this.listeners.click || []) await fn({ target: this }); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

function makeFakeDocument(registry, documentListeners) {
  return {
    getElementById(id) {
      if (!registry.has(id)) registry.set(id, new FakeElement("div"));
      return registry.get(id);
    },
    createElement(tag) { return new FakeElement(tag); },
    createTextNode(text) { return new FakeTextNode(text); },
    querySelectorAll() { return []; },
    addEventListener(type, fn) { (documentListeners[type] ||= []).push(fn); },
  };
}

// --- Fake backend, shaped exactly like the real endpoints --------------
//
// Window bounds 0..20. Two phase bands, deliberately of different sizes and
// with a one-window band to exercise the degenerate case.
const FIRST_WINDOW = 0;
const LAST_WINDOW = 20;
const SEGMENTS = [
  { phase: "t7", start_window: 0, end_window: 9, n_windows: 10, mean_probability: 0.8,
    start_time: 1.0, end_time: 10.0 },
  { phase: "t8", start_window: 10, end_window: 19, n_windows: 10, mean_probability: 0.9,
    start_time: 10.0, end_time: 20.0 },
  { phase: "t9+", start_window: 20, end_window: 20, n_windows: 1, mean_probability: 0.7,
    start_time: 20.0, end_time: 21.0 },
];

// Raw frame numbers are deliberately NON-contiguous across windows -- this
// mirrors the real Val data (Patient_319 window 156 -> frames 167..174 but
// window 157 -> 175..182) and would break any client that computed a
// neighbouring window's frame number instead of reading it.
function framesFor(windowStart) {
  const base = 100 + windowStart * (windowStart >= 10 ? 8 : 1);
  return { first: base, last: base + 7, n: 8 };
}

function windowResponse(windowStart, options = {}) {
  const f = framesFor(windowStart);
  const base = `/videos/Patient_1/window/${windowStart}/frame?split=val`;
  return {
    sample: { video_name: "Patient_1" },
    window: {
      window_start: windowStart,
      window_start_time: options.timeAvailable === false ? null : windowStart * 1.0,
      window_end_time: options.timeAvailable === false ? null : windowStart * 1.0 + 1.0,
      time_available: options.timeAvailable !== false,
      time_unit: "unknown/unverified",
    },
    current_state: { current_phase: windowStart < 10 ? "t7" : "t8", phase_probability: 0.5, entropy: 0.1 },
    frame: options.frameAvailable === false
      ? { available: false, frame_number: null, frame_url: null, first_frame_number: null,
          last_frame_number: null, n_frames: 0, first_frame_url: null, last_frame_url: null }
      : { available: true, frame_number: f.last, frame_url: base,
          first_frame_number: f.first, last_frame_number: f.last, n_frames: f.n,
          first_frame_url: `${base}&which=first`, last_frame_url: `${base}&which=last` },
    warnings: options.warnings || [],
  };
}

function loadAppJs(overrides = {}) {
  const source = fs.readFileSync(APP_JS_PATH, "utf8");
  const registry = new Map();
  const calls = [];
  const documentListeners = {};
  const fetchImpl = async (url) => {
    calls.push(url);
    if (url.startsWith("/videos?")) {
      return { ok: true, json: async () => ({ split: "val", videos: ["Patient_1"] }) };
    }
    if (/^\/videos\/[^/]+\?/.test(url)) {
      return { ok: true, json: async () => ({
        video_name: "Patient_1", split: "val", n_windows: LAST_WINDOW + 1,
        first_window: FIRST_WINDOW, last_window: LAST_WINDOW }) };
    }
    if (url.includes("/timeline")) {
      return { ok: true, json: async () => ({
        video_name: "Patient_1", split: "val", segments: overrides.segments || SEGMENTS }) };
    }
    const m = url.match(/\/window\/(\d+)\?/);
    if (m) {
      const w = Number(m[1]);
      if (w < FIRST_WINDOW || w > LAST_WINDOW) {
        return { ok: false, json: async () => ({ message: `window ${w} does not exist` }) };
      }
      return { ok: true, json: async () => windowResponse(w, overrides) };
    }
    return { ok: false, json: async () => ({ message: `unexpected ${url}` }) };
  };
  const sandbox = {
    document: makeFakeDocument(registry, documentListeners),
    window: {},
    fetch: fetchImpl,
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "app.js" });
  return { sandbox, registry, calls, documentListeners };
}

async function boot(overrides = {}) {
  const ctx = loadAppJs(overrides);
  await ctx.sandbox.loadVideos();
  return ctx;
}

const shownWindow = (r) => r.get("cw-window").textContent;
const shownFrame = (r) => r.get("cw-frame-number").textContent;
const shownSrc = (r) => r.get("cw-frame").src;
const windowRequests = (calls) =>
  calls.filter((u) => /\/window\/\d+\?/.test(u)).map((u) => Number(u.match(/\/window\/(\d+)\?/)[1]));

// The real keydown listener fires the navigation without awaiting it (a DOM
// listener cannot be awaited), so the test flushes the pending promise chain
// before asserting.
async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
}

async function press(documentListeners, key, extra = {}) {
  for (const fn of documentListeners.keydown || []) {
    await fn({ key, target: { tagName: "BODY" }, preventDefault() {}, ...extra });
  }
  await flush();
}

// --- phase selection --------------------------------------------------

test("every timeline band is offered as a selectable phase", async () => {
  const { registry } = await boot();
  const opts = Array.from(registry.get("phase-select").children, (o) => o.textContent);
  assert.equal(opts.length, SEGMENTS.length);
  assert.match(opts[0], /t7 — fenêtres 0–9 \(10 fenêtres\)/);
  assert.match(opts[2], /t9\+ — fenêtres 20–20 \(1 fenêtre\)/);
});

test("bands are addressed by index so a repeated phase name stays distinct", async () => {
  const repeated = [
    { phase: "t7", start_window: 0, end_window: 4, n_windows: 5, mean_probability: 0.8 },
    { phase: "t8", start_window: 5, end_window: 9, n_windows: 5, mean_probability: 0.8 },
    { phase: "t7", start_window: 10, end_window: 20, n_windows: 11, mean_probability: 0.8 },
  ];
  const { sandbox, registry } = await boot({ segments: repeated });
  const values = Array.from(registry.get("phase-select").children, (o) => o.value);
  assert.deepEqual(values, ["0", "1", "2"]);

  registry.get("phase-select").value = "2";
  await sandbox.jumpToPhaseEdge("first");
  assert.equal(shownWindow(registry), "10", "the THIRD band must be used, not the first t7");
});

test("jumping to a phase start loads that band's first window at its FIRST frame", async () => {
  const { sandbox, registry } = await boot();
  registry.get("phase-select").value = "1";
  await sandbox.jumpToPhaseEdge("first");

  assert.equal(shownWindow(registry), "10");
  assert.equal(shownFrame(registry), `${framesFor(10).first} (première de la fenêtre)`);
  assert.ok(shownSrc(registry).endsWith("&which=first"), shownSrc(registry));
});

test("jumping to a phase end loads that band's last window at its LAST frame", async () => {
  const { sandbox, registry } = await boot();
  registry.get("phase-select").value = "0";
  await sandbox.jumpToPhaseEdge("last");

  assert.equal(shownWindow(registry), "9");
  assert.equal(shownFrame(registry), `${framesFor(9).last} (dernière de la fenêtre)`);
  assert.ok(shownSrc(registry).endsWith("&which=last"), shownSrc(registry));
});

test("a one-window band has the same window for its start and its end", async () => {
  const { sandbox, registry } = await boot();
  registry.get("phase-select").value = "2";
  await sandbox.jumpToPhaseEdge("first");
  assert.equal(shownWindow(registry), "20");
  await sandbox.jumpToPhaseEdge("last");
  assert.equal(shownWindow(registry), "20");
});

test("an empty timeline yields no phase options and no viewer request", async () => {
  const { registry, calls } = await boot({ segments: [] });
  assert.deepEqual(Array.from(registry.get("phase-select").children), []);
  assert.deepEqual(windowRequests(calls), [], "no window may be loaded when there is no band");
  assert.equal(registry.get("btn-phase-first").disabled, true);
  assert.equal(registry.get("btn-phase-last").disabled, true);
});

test("a non-existent band index is refused instead of requesting a made-up window", async () => {
  const { sandbox, registry, calls } = await boot();
  const before = windowRequests(calls).length;
  registry.get("phase-select").value = "99";
  await sandbox.jumpToPhaseEdge("first");
  assert.equal(windowRequests(calls).length, before, "no request may be issued for an unknown band");
});

// --- manual stepping --------------------------------------------------

test("next and previous step exactly one window", async () => {
  const { sandbox, registry } = await boot();
  registry.get("phase-select").value = "0";
  await sandbox.jumpToPhaseEdge("first");
  assert.equal(shownWindow(registry), "0");

  await sandbox.stepWindow(1);
  assert.equal(shownWindow(registry), "1");
  await sandbox.stepWindow(1);
  assert.equal(shownWindow(registry), "2");
  await sandbox.stepWindow(-1);
  assert.equal(shownWindow(registry), "1");
});

test("stepping reads each window's own frame numbers, never computes them", async () => {
  // Windows 10+ are deliberately non-contiguous in raw frame numbers.
  const { sandbox, registry } = await boot();
  registry.get("phase-select").value = "1";
  await sandbox.jumpToPhaseEdge("first");   // window 10, shown at its FIRST frame
  const at10 = framesFor(10).first;
  await sandbox.stepWindow(1);              // window 11
  const at11 = framesFor(11).first;
  assert.notEqual(at11, at10 + 1, "the fixture must exercise a non-contiguous jump");
  assert.equal(shownFrame(registry), `${at11} (première de la fenêtre)`);
});

test("the chosen first/last view mode is kept while stepping", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", 5, { which: "first" });
  await sandbox.stepWindow(1);
  assert.equal(shownFrame(registry), `${framesFor(6).first} (première de la fenêtre)`);
  assert.ok(shownSrc(registry).endsWith("&which=first"));

  await sandbox.selectWindow("Patient_1", 6, { which: "last" });
  await sandbox.stepWindow(1);
  assert.equal(shownFrame(registry), `${framesFor(7).last} (dernière de la fenêtre)`);
  assert.ok(shownSrc(registry).endsWith("&which=last"));
});

test("the displayed frame number always matches the image URL requested", async () => {
  const { sandbox, registry } = await boot();
  for (const w of [0, 5, 10, 19, 20]) {
    await sandbox.selectWindow("Patient_1", w, { which: "last" });
    const f = framesFor(w);
    assert.equal(shownFrame(registry), `${f.last} (dernière de la fenêtre)`);
    assert.ok(shownSrc(registry).includes(`/window/${w}/frame`), shownSrc(registry));
    assert.ok(shownSrc(registry).endsWith("&which=last"));

    await sandbox.selectWindow("Patient_1", w, { which: "first" });
    assert.equal(shownFrame(registry), `${f.first} (première de la fenêtre)`);
    assert.ok(shownSrc(registry).endsWith("&which=first"));
  }
});

test("the window's full frame range is displayed", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", 3, {});
  const f = framesFor(3);
  assert.equal(registry.get("cw-frame-range").textContent, `${f.first}–${f.last} (8 images)`);
});

// --- sequence boundaries ----------------------------------------------

test("stepping back from the first window issues no request", async () => {
  const { sandbox, registry, calls } = await boot();
  await sandbox.selectWindow("Patient_1", FIRST_WINDOW, {});
  const before = windowRequests(calls).length;
  await sandbox.stepWindow(-1);
  assert.equal(windowRequests(calls).length, before, "must not request window -1");
  assert.equal(shownWindow(registry), String(FIRST_WINDOW));
});

test("stepping forward from the last window issues no request", async () => {
  const { sandbox, registry, calls } = await boot();
  await sandbox.selectWindow("Patient_1", LAST_WINDOW, {});
  const before = windowRequests(calls).length;
  await sandbox.stepWindow(1);
  assert.equal(windowRequests(calls).length, before, `must not request window ${LAST_WINDOW + 1}`);
  assert.equal(shownWindow(registry), String(LAST_WINDOW));
});

test("the boundary buttons are disabled exactly at the boundaries", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", FIRST_WINDOW, {});
  assert.equal(registry.get("btn-prev-window").disabled, true);
  assert.equal(registry.get("btn-next-window").disabled, false);

  await sandbox.selectWindow("Patient_1", LAST_WINDOW, {});
  assert.equal(registry.get("btn-prev-window").disabled, false);
  assert.equal(registry.get("btn-next-window").disabled, true);

  await sandbox.selectWindow("Patient_1", 5, {});
  assert.equal(registry.get("btn-prev-window").disabled, false);
  assert.equal(registry.get("btn-next-window").disabled, false);
});

// --- return to selection ----------------------------------------------

test("back-to-selection returns to the anchored window after stepping away", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", 12, { anchor: true });
  await sandbox.stepWindow(1);
  await sandbox.stepWindow(1);
  assert.equal(shownWindow(registry), "14");

  await sandbox.backToAnchor();
  assert.equal(shownWindow(registry), "12");
  assert.equal(registry.get("btn-back-anchor").disabled, true, "already at the anchor");
});

test("stepping does not move the anchor", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", 4, { anchor: true });
  await sandbox.stepWindow(1);
  assert.match(registry.get("frame-position").textContent, /\+1 par rapport à la sélection/);
  await sandbox.stepWindow(-2);
  assert.match(registry.get("frame-position").textContent, /-1 par rapport à la sélection/);
  await sandbox.backToAnchor();
  assert.equal(shownWindow(registry), "4");
});

// --- keyboard ---------------------------------------------------------

test("arrow keys step and Home returns to the selection", async () => {
  const { sandbox, registry, documentListeners } = await boot();
  await sandbox.selectWindow("Patient_1", 8, { anchor: true });

  await press(documentListeners, "ArrowRight");
  assert.equal(shownWindow(registry), "9");
  await press(documentListeners, "ArrowLeft");
  await press(documentListeners, "ArrowLeft");
  assert.equal(shownWindow(registry), "7");
  await press(documentListeners, "Home");
  assert.equal(shownWindow(registry), "8");
});

test("keyboard stepping is inert while typing and with a modifier held", async () => {
  const { sandbox, registry, documentListeners } = await boot();
  await sandbox.selectWindow("Patient_1", 8, { anchor: true });

  for (const fn of documentListeners.keydown || []) {
    await fn({ key: "ArrowRight", target: { tagName: "INPUT" }, preventDefault() {} });
  }
  assert.equal(shownWindow(registry), "8", "arrow keys must stay available to a text field");

  await press(documentListeners, "ArrowRight", { ctrlKey: true });
  assert.equal(shownWindow(registry), "8", "a modified key must keep its browser meaning");
});

// --- honesty ----------------------------------------------------------

test("the confirmed unit is shown next to the value, without converting it", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", 3, {});
  // windowResponse(3) -> window_start_time 3.0, window_end_time 4.0
  assert.equal(registry.get("cw-time").textContent, "3.00 → 4.00 h",
    "the unit must sit next to the value and the numbers must be the API's own");
});

test("the tooltip keeps the unit's provenance and the API's own flag", async () => {
  const { sandbox, registry } = await boot();
  await sandbox.selectWindow("Patient_1", 3, {});
  const tip = registry.get("cw-time").title;
  assert.ok(tip.includes("heures"), tip);
  assert.ok(/confirmée par le responsable du projet/i.test(tip),
    `the tooltip must say where the unit comes from, got: ${tip}`);
  assert.ok(tip.includes("unknown/unverified"),
    `the API's own flag must stay visible for traceability, got: ${tip}`);
});

test("the displayed numbers are exactly the API's, never rescaled", async () => {
  const { sandbox, registry } = await boot();
  for (const w of [0, 7, 15]) {
    await sandbox.selectWindow("Patient_1", w, {});
    const expected = `${(w * 1.0).toFixed(2)} → ${(w * 1.0 + 1.0).toFixed(2)} h`;
    assert.equal(registry.get("cw-time").textContent, expected);
  }
});

test("time metadata absent is stated, not faked -- and carries no unit", async () => {
  const { sandbox, registry } = await boot({ timeAvailable: false });
  await sandbox.selectWindow("Patient_1", 3, {});
  assert.equal(registry.get("cw-time").textContent, "indisponible");
  assert.ok(!registry.get("cw-time").textContent.includes("h"),
    "a unit must never be shown next to a value that does not exist");
});

test("a missing frame degrades to the fallback without breaking navigation", async () => {
  const { sandbox, registry } = await boot({ frameAvailable: false });
  await sandbox.selectWindow("Patient_1", 3, {});
  assert.equal(registry.get("cw-frame").hidden, true);
  assert.equal(registry.get("cw-frame-fallback").hidden, false);
  assert.equal(shownFrame(registry), "-");
  assert.equal(registry.get("cw-frame-range").textContent, "-");

  await sandbox.stepWindow(1);
  assert.equal(shownWindow(registry), "4", "navigation must still work without an image");
});

// The fake DOM's `textContent` getter returns only text assigned directly,
// not the text of appended child nodes -- and renderWarningsList() builds its
// output from real text nodes. Flatten the subtree the way the DOM would.
function flattenText(node) {
  if (node.nodeType === 3) return node.textContent;
  const own = node._text || "";
  return own + (node.children || []).map(flattenText).join("");
}

// --- Duration warning: hidden in the viewer, computation untouched -------
//
// `reporting/inference_service.py` still emits it and the API still serves
// it; only this viewer declines to display it. These tests drive the REAL
// app.js rendering path, so they prove the suppression behaviourally rather
// than by reading the source.

test("the duration-censoring warning is not displayed in the viewer", async () => {
  const censored = "duration not estimable for phase 'tPB2' " +
    "(structural censoring -- 0 observed interior segments in Train)";
  const { sandbox, registry } = await boot({ warnings: [censored] });
  await sandbox.selectWindow("Patient_1", 3, {});
  const shown = flattenText(registry.get("cw-warnings"));
  assert.ok(!shown.includes("duration not estimable"),
    `the censoring warning reached the user: ${shown}`);
  assert.equal(shown.trim(), "", "no warning should be rendered for this window");
});

test("other warnings are still displayed in the viewer", async () => {
  const { sandbox, registry } = await boot({
    warnings: ["time metadata incomplete for this window"],
  });
  await sandbox.selectWindow("Patient_1", 3, {});
  const shown = flattenText(registry.get("cw-warnings"));
  assert.ok(shown.includes("time metadata incomplete"),
    `an unrelated warning was wrongly suppressed: ${shown}`);
});

test("only the censoring warning is dropped when several are returned", async () => {
  const censored = "duration not estimable for phase 'tPB2' (structural censoring -- 0 ...)";
  const { sandbox, registry } = await boot({
    warnings: [censored, "repeated timestamps in this window"],
  });
  await sandbox.selectWindow("Patient_1", 3, {});
  const shown = flattenText(registry.get("cw-warnings"));
  assert.ok(!shown.includes("duration not estimable"), shown);
  assert.ok(shown.includes("repeated timestamps"), shown);
});
