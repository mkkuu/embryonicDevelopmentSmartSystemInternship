// Zero-dependency Node test for the patient-selection UX
// (Training/webapp_api/static/app.js), same pattern and same fake-DOM
// approach as test_chat_dom_safety.mjs -- the REAL app.js is loaded into a
// self-written fake DOM, never a reimplementation of its logic.
//
// Run with:  node Tests/webapp_api/test_patient_selection.mjs
// (not collected by pytest -- different ecosystem, like its companion.)
//
// What it pins:
//   - patients are ordered NATURALLY (Patient_2 before Patient_10), stably;
//   - the backend's own response array is never mutated;
//   - every option.value is the EXACT backend identifier;
//   - the filter is a view only -- no refetch, no reorder, selection kept;
//   - the selected patient is what gets sent to the backend;
//   - the patient summary comes from the existing GET /videos/<id>;
//   - a failing summary never blocks the viewer.

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
    this.hidden = false;
    this.style = {};
    this.classList = { add() {}, remove() {}, contains() { return false; } };
    this.listeners = {};
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...nodes) { this.children = nodes; this._text = ""; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
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
    // app.js binds page-level keyboard shortcuts on `document` itself; the
    // fake DOM has to model that API or the module cannot even load.
    addEventListener(type, fn) { ((documentListeners || {})[type] ||= []).push(fn); },
  };
}

function loadAppJs(fetchImpl) {
  const source = fs.readFileSync(APP_JS_PATH, "utf8");
  const registry = new Map();
  const calls = [];
  const documentListeners = {};
  const sandbox = {
    document: makeFakeDocument(registry, documentListeners),
    window: {},
    fetch: async (url, options) => { calls.push({ url, options }); return fetchImpl(url, options); },
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "app.js" });
  return { sandbox, registry, calls, documentListeners };
}

const NOT_FOUND = { ok: false, json: async () => ({ message: "not used in this test" }) };

// A backend response deliberately in NON-alphabetical, NON-numeric order --
// exactly what trajectory_service.list_videos()'s documented "cache order
// (stable, not sorted)" produces.
const BACKEND_ORDER = ["Patient_10", "Patient_2", "Patient_319", "Patient_1", "Patient_100", "Patient_20"];
const NATURAL_ORDER = ["Patient_1", "Patient_2", "Patient_10", "Patient_20", "Patient_100", "Patient_319"];

function routedFetch(overrides = {}) {
  return async (url) => {
    if (url.startsWith("/videos?")) {
      return { ok: true, json: async () => ({ split: "val", videos: [...BACKEND_ORDER] }) };
    }
    if (/^\/videos\/[^/]+\?/.test(url)) {
      if (overrides.detailFails) return { ok: false, json: async () => ({ message: "boom" }) };
      return {
        ok: true,
        json: async () => ({
          video_name: decodeURIComponent(url.slice("/videos/".length).split("?")[0]),
          split: "val", n_windows: 42, first_window: 0, last_window: 41,
        }),
      };
    }
    if (url.includes("/timeline")) {
      return { ok: true, json: async () => ({ segments: [] }) };
    }
    return NOT_FOUND;
  };
}

// Values crossing back from the vm realm are copied into host-realm arrays:
// `node:assert/strict`'s deepEqual is deepStrictEqual, which compares
// prototypes, and a vm-created Array has a different Array.prototype.
function optionValues(registry) {
  return Array.from(registry.get("video-select").children || [], (o) => o.value);
}

function hostArray(value) {
  return Array.from(value || []);
}

// --- ordering ---------------------------------------------------------

test("compareVideoIds sorts numerically, not lexicographically", () => {
  const { sandbox } = loadAppJs(routedFetch());
  const sorted = [...BACKEND_ORDER].sort(sandbox.compareVideoIds);
  assert.deepEqual(sorted, NATURAL_ORDER);
});

test("compareVideoIds is a total, stable, deterministic order", () => {
  const { sandbox } = loadAppJs(routedFetch());
  const cmp = sandbox.compareVideoIds;
  assert.equal(cmp("Patient_7", "Patient_7"), 0, "equal ids must compare equal");
  assert.equal(Math.sign(cmp("Patient_2", "Patient_10")), -1);
  assert.equal(Math.sign(cmp("Patient_10", "Patient_2")), 1, "must be antisymmetric");
  // A second sort of an already-sorted list must not move anything.
  const once = [...BACKEND_ORDER].sort(cmp);
  assert.deepEqual([...once].sort(cmp), once);
});

test("mixed / non-numeric identifiers still order deterministically", () => {
  const { sandbox } = loadAppJs(routedFetch());
  const ids = ["Video_B", "Patient_9", "Video_A", "Patient_10"];
  const sorted = [...ids].sort(sandbox.compareVideoIds);
  assert.deepEqual(sorted, ["Patient_9", "Patient_10", "Video_A", "Video_B"]);
});

test("loadVideos renders the list in natural order", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  assert.deepEqual(optionValues(registry), NATURAL_ORDER);
});

test("loadVideos does not mutate the backend's own response order", async () => {
  const seen = [];
  const fetchImpl = async (url) => {
    if (url.startsWith("/videos?")) {
      const videos = [...BACKEND_ORDER];
      seen.push(videos);
      return { ok: true, json: async () => ({ split: "val", videos }) };
    }
    return routedFetch()(url);
  };
  const { sandbox } = loadAppJs(fetchImpl);
  await sandbox.loadVideos();
  assert.deepEqual(hostArray(seen[0]), BACKEND_ORDER, "the response array must be sorted as a COPY");
});

// --- identifiers are preserved exactly --------------------------------

test("every option value is the exact backend identifier", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  const opts = hostArray(registry.get("video-select").children);
  for (const opt of opts) {
    assert.ok(BACKEND_ORDER.includes(opt.value), `${opt.value} is not a backend id`);
    assert.equal(opt.textContent, opt.value, "label and identifier must be the same string");
  }
  assert.equal(opts.length, BACKEND_ORDER.length, "no patient lost, none invented");
});

// --- filtering is a view only -----------------------------------------

test("filter narrows the list without refetching or reordering", async () => {
  const { sandbox, registry, calls } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  const callsAfterLoad = calls.length;

  registry.get("patient-filter").value = "1";
  const visible = hostArray(sandbox.renderVideoOptions());

  assert.equal(calls.length, callsAfterLoad, "filtering must not issue any request");
  assert.deepEqual(visible, ["Patient_1", "Patient_10", "Patient_100", "Patient_319"]);
  assert.deepEqual(optionValues(registry), visible, "filtered list keeps the natural order");
});

test("filter is case-insensitive and clearing it restores the full list", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadVideos();

  registry.get("patient-filter").value = "PATIENT_31";
  assert.deepEqual(hostArray(sandbox.renderVideoOptions()), ["Patient_319"]);

  registry.get("patient-filter").value = "";
  assert.deepEqual(hostArray(sandbox.renderVideoOptions()), NATURAL_ORDER);
});

test("filtering keeps the current selection when it still matches", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  await sandbox.selectPatient("Patient_319");

  registry.get("patient-filter").value = "3";
  sandbox.renderVideoOptions();

  assert.equal(registry.get("video-select").value, "Patient_319",
    "filtering must not switch patient");
});

test("patient count reflects the filter", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  assert.equal(registry.get("patient-count").textContent, "6 patients");

  registry.get("patient-filter").value = "Patient_319";
  sandbox.renderVideoOptions();
  assert.match(registry.get("patient-count").textContent, /^1 patient sur 6 correspond à/);
});

// --- selection reaches the backend ------------------------------------

test("selecting a patient sends that exact id to the backend endpoints", async () => {
  const { sandbox, calls } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  calls.length = 0;

  await sandbox.selectPatient("Patient_319");

  const urls = calls.map((c) => c.url);
  assert.ok(urls.some((u) => u === "/videos/Patient_319?split=val"), `summary not requested: ${urls}`);
  assert.ok(urls.some((u) => u.startsWith("/videos/Patient_319/timeline")), `timeline not requested: ${urls}`);
});

test("the first patient in natural order is auto-selected on load", async () => {
  const { sandbox, registry, calls } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  assert.equal(registry.get("video-select").value, "Patient_1",
    "the auto-selection must be the first NATURAL id, not the backend's first");
  assert.equal(registry.get("pi-id").textContent, "Patient_1");
  assert.ok(calls.some((c) => c.url.startsWith("/videos/Patient_1/timeline")),
    "the auto-selected patient must be the one loaded from the backend");
});

// --- the patient summary panel ----------------------------------------

test("patient summary is populated from the existing GET /videos/<id>", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadVideos();
  await sandbox.selectPatient("Patient_319");

  assert.equal(registry.get("pi-id").textContent, "Patient_319");
  assert.equal(registry.get("pi-split").textContent, "val");
  assert.equal(registry.get("pi-windows").textContent, "42");
  assert.equal(registry.get("pi-range").textContent, "0 → 41");
});

test("a failing patient summary degrades honestly and never blocks the viewer", async () => {
  const { sandbox, registry, calls } = loadAppJs(routedFetch({ detailFails: true }));
  await sandbox.loadVideos();

  assert.equal(registry.get("pi-windows").textContent, "indisponible");
  assert.equal(registry.get("pi-range").textContent, "indisponible");
  assert.ok(calls.some((c) => c.url.includes("/timeline")), "the timeline must still be requested");
});

test("an empty split renders no options and selects nothing", async () => {
  const fetchImpl = async (url) => {
    if (url.startsWith("/videos?")) return { ok: true, json: async () => ({ split: "val", videos: [] }) };
    return NOT_FOUND;
  };
  const { sandbox, registry } = loadAppJs(fetchImpl);
  await sandbox.loadVideos();
  assert.deepEqual(optionValues(registry), []);
  assert.equal(registry.get("patient-count").textContent, "0 patients");
  assert.equal(registry.has("pi-id"), false,
    "with no patients, the summary panel must be left untouched, not filled with a placeholder");
});

// --- identifiers stay inert text (same guarantee as the chat panel) ----

test("a malicious identifier is rendered as inert text, never markup", async () => {
  const evil = "<img src=x onerror=alert(1)>";
  const fetchImpl = async (url) => {
    if (url.startsWith("/videos?")) return { ok: true, json: async () => ({ split: "val", videos: [evil] }) };
    if (/^\/videos\/[^/]+\?/.test(url)) {
      return { ok: true, json: async () => ({ video_name: evil, split: "val", n_windows: 1, first_window: 0, last_window: 0 }) };
    }
    if (url.includes("/timeline")) return { ok: true, json: async () => ({ segments: [] }) };
    return NOT_FOUND;
  };
  const { sandbox, registry } = loadAppJs(fetchImpl);
  await sandbox.loadVideos();

  const opts = hostArray(registry.get("video-select").children);
  assert.equal(opts.length, 1);
  assert.equal(opts[0].tagName, "OPTION", "no extra element was parsed out of the payload");
  assert.equal(opts[0].textContent, evil, "payload present as literal text");
  assert.equal(registry.get("pi-id").textContent, evil);
});
