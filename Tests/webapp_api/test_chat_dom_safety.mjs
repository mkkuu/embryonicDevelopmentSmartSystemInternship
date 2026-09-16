// Zero-dependency DOM-safety test for Training/webapp_api/static/app.js
// (docs/WEBAPP_SECURITY.md). Uses ONLY Node built-ins (node:test,
// node:assert, node:vm, node:fs) -- no npm package, no jsdom, matching
// this project's "no npm/build tooling anywhere" discipline (CLAUDE.md)
// while still loading and exercising the REAL app.js source verbatim,
// not a reimplementation of its logic.
//
// Run with: node Tests/webapp_api/test_chat_dom_safety.mjs
// (Node's built-in test runner auto-executes test() blocks defined in
// the file and exits non-zero on any failure -- no separate test-runner
// invocation needed. NOT part of `pytest Tests/`, a different ecosystem;
// documented as a required companion check for this fix.)
//
// Strategy: this is a SINK-based test, not an origin-based one. Every
// value that reaches the DOM in the answer panel -- whether it originated
// from the LLM's generated text (via the grounding checker's warnings),
// the Reporting API (phase/video names), or RAG chunk metadata
// (source/section) -- flows through the same small set of safe-
// construction helpers (`textSpan`, `renderWarningsList`,
// `renderSources`, and `renderChatContext`'s own local `strong()`).
// Testing those sinks directly with malicious payloads therefore covers
// every origin at once; a payload landing in the DOM as inert text is
// the same safe outcome regardless of where it came from.

import { test } from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_JS_PATH = path.join(__dirname, "..", "..", "Training", "webapp_api", "static", "app.js");

// --- Minimal, dependency-free fake DOM -------------------------------

class FakeNode {
  constructor(nodeType) {
    this.nodeType = nodeType;
    this.children = [];
  }
}

class FakeTextNode extends FakeNode {
  constructor(text) {
    super(3);
    this.textContent = String(text);
  }
}

class FakeElement extends FakeNode {
  constructor(tag) {
    super(1);
    this.tagName = tag.toUpperCase();
    this.className = "";
    this.style = {};
    this.hidden = false;
    this.title = "";
    this._listeners = {};
  }

  get classList() {
    const self = this;
    return {
      add(c) { if (!self.className.split(/\s+/).includes(c)) self.className = (self.className + " " + c).trim(); },
      remove(c) { self.className = self.className.split(/\s+/).filter((x) => x && x !== c).join(" "); },
    };
  }

  set textContent(v) { this.children = [new FakeTextNode(v)]; }
  get textContent() { return this.children.map(flattenText).join(""); }

  appendChild(node) { this.children.push(node); return node; }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, fn) { this._listeners[type] = fn; }
  querySelectorAll() { return []; }
  querySelector() { return new FakeElement("button"); }
}

function flattenText(node) {
  if (node.nodeType === 3) return node.textContent;
  return node.children.map(flattenText).join("");
}

// Recursively collects every element's tagName in a subtree -- used to
// assert a payload never became a real SCRIPT/IMG element, only inert
// text inside a SPAN/DIV/BR/STRONG.
function collectTagNames(node, out = []) {
  if (node.nodeType === 1) {
    out.push(node.tagName);
    node.children.forEach((c) => collectTagNames(c, out));
  }
  return out;
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
  const documentListeners = {};
  const sandbox = {
    document: makeFakeDocument(registry, documentListeners),
    window: {},
    fetch: fetchImpl || (async () => { throw new Error("fetch disabled in this test"); }),
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "app.js" });
  return { sandbox, registry, documentListeners };
}

// --- Payload categories (task's own required minimum) -----------------

const PAYLOADS = [
  ["script tag", "<script>alert(1)</script>"],
  ["img onerror", '<img src=x onerror="alert(1)">'],
  ["arbitrary HTML", "<b>bold</b><i>italic</i>"],
  ["user-supplied HTML", '<div onclick="steal()">click me</div>'],
  ["special characters", `< > & " '`],
];

// --- 1-3: normal/happy-path content (behavior must be unchanged) ------

test("textSpan: normal text renders as expected, unmodified", () => {
  const { sandbox } = loadAppJs();
  const span = sandbox.textSpan("route: DOCUMENTARY", "grounded-true");
  assert.equal(span.tagName, "SPAN");
  assert.equal(span.className, "grounded-true");
  assert.equal(span.textContent, "route: DOCUMENTARY");
});

test("renderSources: a normal source/section renders the same visual text as before", () => {
  const { sandbox } = loadAppJs();
  const container = new FakeElement("div");
  sandbox.renderSources(container, [{ source: "docs/HANDOFF.md", section: "3", status: "current" }]);
  assert.equal(container.children.length, 1);
  assert.equal(container.children[0].tagName, "SPAN");
  assert.equal(container.children[0].className, "source-chip");
  assert.equal(container.children[0].textContent, "docs/HANDOFF.md §3");
});

test("renderWarningsList: a normal warning renders with the same '⚠ ' prefix as before", () => {
  const { sandbox } = loadAppJs();
  const container = new FakeElement("div");
  sandbox.renderWarningsList(container, ["Duration unavailable for this phase."]);
  assert.equal(flattenText(container), "⚠ Duration unavailable for this phase.");
});

test("renderWarningsList: multiple warnings are <br>-separated, same layout as before", () => {
  const { sandbox } = loadAppJs();
  const container = new FakeElement("div");
  sandbox.renderWarningsList(container, ["first warning", "second warning"]);
  const tags = collectTagNames(container);
  assert.ok(tags.includes("BR"), "a real <br> element separates the two warnings");
  assert.equal(flattenText(container), "⚠ first warning⚠ second warning");
});

// --- 4-8: malicious/adversarial payloads -------------------------------

for (const [label, payload] of PAYLOADS) {
  test(`textSpan: ${label} is rendered as inert text, never a real element`, () => {
    const { sandbox } = loadAppJs();
    const span = sandbox.textSpan(payload);
    assert.equal(span.textContent, payload, "payload preserved verbatim as text");
    const tags = collectTagNames(span);
    assert.ok(!tags.includes("SCRIPT"), `${label}: no SCRIPT element created`);
    assert.ok(!tags.includes("IMG"), `${label}: no IMG element created`);
    assert.deepEqual(tags, ["SPAN"], `${label}: only the wrapper SPAN exists, nothing parsed from the payload`);
  });

  test(`renderSources: ${label} in a source/section field is rendered as inert text`, () => {
    const { sandbox } = loadAppJs();
    const container = new FakeElement("div");
    sandbox.renderSources(container, [{ source: payload, section: null, status: "current" }]);
    assert.equal(flattenText(container), `${payload} `);
    const tags = collectTagNames(container);
    assert.ok(!tags.includes("SCRIPT"), `${label}: no SCRIPT element created`);
    assert.ok(!tags.includes("IMG"), `${label}: no IMG element created`);
  });

  test(`renderWarningsList: ${label} in a warning is rendered as inert text`, () => {
    const { sandbox } = loadAppJs();
    const container = new FakeElement("div");
    sandbox.renderWarningsList(container, [payload]);
    assert.equal(flattenText(container), `⚠ ${payload}`);
    const tags = collectTagNames(container);
    assert.ok(!tags.includes("SCRIPT"), `${label}: no SCRIPT element created`);
    assert.ok(!tags.includes("IMG"), `${label}: no IMG element created`);
  });
}

// --- End-to-end: the exact historically-vulnerable path -----------------
// renderChatContext() (the original CVE-shaped line: `el.innerHTML =
// \`...${currentVideo}...${phase}...\`) is reached only through
// selectWindow()'s closure state -- exercised here via a fake /window/<w>
// response carrying a malicious video_id and phase, driving the REAL
// data flow (fetch -> selectWindow -> renderChatContext), not a
// reimplementation of it. Since the free-question panel was removed, the
// element this writes into is `fq-context` (the fixed-questions context
// line); the sink and the guarantee are unchanged.

function fakeWindowResponse(phase, warnings) {
  return {
    window: { window_start: 0, time_available: false },
    current_state: { current_phase: phase, phase_probability: 0.5, entropy: 0.1 },
    frame: { available: false },
    warnings: warnings || [],
  };
}

test("end-to-end: selectWindow -> renderChatContext renders a malicious phase/video as inert text", async () => {
  const maliciousPhase = "<script>alert('phase')</script>";
  const maliciousVideo = "<img src=x onerror=alert('video')>";
  const fetchImpl = async (url) => ({
    ok: true,
    json: async () => fakeWindowResponse(maliciousPhase, ["<script>alert('warning')</script>"]),
  });
  const { sandbox, registry } = loadAppJs(fetchImpl);

  await sandbox.selectWindow(maliciousVideo, 0);

  const questionContext = registry.get("fq-context");
  const cwWarnings = registry.get("cw-warnings");

  assert.ok(questionContext, "fq-context element was touched");
  assert.ok(flattenText(questionContext).includes(maliciousVideo), "video id present as literal text");
  assert.ok(flattenText(questionContext).includes(maliciousPhase), "phase present as literal text");
  assert.ok(!collectTagNames(questionContext).includes("SCRIPT"), "no SCRIPT element in fq-context");
  assert.ok(!collectTagNames(questionContext).includes("IMG"), "no IMG element in fq-context");
  assert.ok(!collectTagNames(cwWarnings).includes("SCRIPT"), "no SCRIPT element in cw-warnings");
});

test("end-to-end: selectWindow -> renderChatContext with normal data matches the expected visual text", async () => {
  const fetchImpl = async () => ({
    ok: true,
    json: async () => fakeWindowResponse("tPNf", []),
  });
  const { sandbox, registry } = loadAppJs(fetchImpl);

  await sandbox.selectWindow("Patient_319", 0);

  const questionContext = registry.get("fq-context");
  assert.equal(
    flattenText(questionContext),
    "La question sera posée sur : Patient_319 · fenêtre 0 · phase (modèle) tPNf (50.0%)",
    "unchanged visual output vs. the pre-fix innerHTML template literal",
  );
});
