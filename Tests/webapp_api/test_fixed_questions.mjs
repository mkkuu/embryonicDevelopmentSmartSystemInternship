// Zero-dependency Node test for the fixed-questions panel
// (Training/webapp_api/static/app.js) -- same fake-DOM approach as
// test_patient_selection.mjs / test_chat_dom_safety.mjs: the REAL app.js is
// loaded verbatim, never a reimplementation of its logic.
//
// Run with:  node Tests/webapp_api/test_fixed_questions.mjs
// (not collected by pytest -- different ecosystem, like its companions; the
// pytest-side guard is Tests/webapp_api/test_fixed_questions_api.py.)
//
// What it pins:
//   - the list is built from GET /questions, one entry per served question,
//     with the VERBATIM wording as inert text and the id as the radio value;
//   - a selected question is sent to POST /chat by `question_id` only, with
//     the LIVE patient/window/split -- the wording never travels back;
//   - the answer is rendered (text, grounding, provenance, sources);
//   - loading state: the button is disabled and relabelled while pending;
//   - a network / HTTP error is shown as an error, the button recovers;
//   - changing patient (or window) removes the previous answer;
//   - a response for a superseded request is discarded;
//   - the button is disabled without a patient/window or without a question.

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
    this.disabled = false;
    this.checked = false;
    this.className = "";
    this.style = {};
    this.listeners = {};
    const self = this;
    this.classList = {
      add(c) { if (!self.className.split(/\s+/).includes(c)) self.className = (self.className + " " + c).trim(); },
      remove(c) { self.className = self.className.split(/\s+/).filter((x) => x && x !== c).join(" "); },
      contains(c) { return self.className.split(/\s+/).includes(c); },
    };
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text || this.children.map(flatten).join(""); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...nodes) { this.children = nodes; this._text = ""; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

function flatten(node) {
  if (node.nodeType === 3) return node.textContent;
  return node.textContent;
}

function tagsIn(node, out = []) {
  if (node.nodeType === 1) {
    out.push(node.tagName);
    node.children.forEach((c) => tagsIn(c, out));
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

// A served list in the exact shape of GET /questions. The texts are
// deliberately NOT the real benchmark wording: this test proves the
// frontend renders whatever the backend serves verbatim (the real wording
// is pinned server-side, in the pytest companion).
const SERVED = [
  { id: "Q1", number: 1, question: "Première question de test ?", category: "A" },
  { id: "Q2", number: 2, question: "Deuxième question, avec l’apostrophe typographique ?", category: "A" },
  { id: "Q3", number: 3, question: "<b>Troisième</b> question <script>alert(1)</script> ?", category: "B" },
];

function windowResponse(phase) {
  return {
    window: { window_start: 0, time_available: false },
    current_state: { current_phase: phase, phase_probability: 0.5, entropy: 0.1 },
    frame: { available: false },
    warnings: [],
  };
}

const NOT_FOUND = { ok: false, json: async () => ({ message: "not used in this test" }) };

function chatResponse(overrides = {}) {
  return {
    question_id: "Q1", question: SERVED[0].question,
    answer: "La réponse.", grounded: true, confidence: "medium",
    sources: [{ source: "docs/HANDOFF.md", section: "3", status: "current" }],
    warnings: [], tools_called: ["get_current_inference", "retrieve_documents"], route: "HYBRID",
    ...overrides,
  };
}

// `chatImpl(body)` decides what POST /chat returns; the default answers at
// once. Tests needing a slow or failing backend pass their own.
function routedFetch(chatImpl) {
  const chat = chatImpl || (async () => ({ ok: true, json: async () => chatResponse() }));
  return async (url, options) => {
    if (url === "/questions") return { ok: true, json: async () => ({ n_questions: SERVED.length, questions: SERVED }) };
    if (url === "/chat") return chat(JSON.parse(options.body));
    // Empty on purpose: app.js auto-selects the first patient at boot, which
    // would race the explicit selections these tests make. Patients are
    // selected explicitly below instead.
    if (url.startsWith("/videos?")) return { ok: true, json: async () => ({ split: "val", videos: [] }) };
    if (/^\/videos\/[^/]+\?/.test(url)) {
      return { ok: true, json: async () => ({ video_name: "x", split: "val", n_windows: 3, first_window: 0, last_window: 2 }) };
    }
    if (url.includes("/timeline")) return { ok: true, json: async () => ({ segments: [] }) };
    if (/\/window\/\d+\?/.test(url)) return { ok: true, json: async () => windowResponse("t4") };
    return NOT_FOUND;
  };
}

function listItems(registry) {
  return registry.has("fq-list") ? Array.from(registry.get("fq-list").children || []) : [];
}

function submitFixedQuestion(registry) {
  const listeners = registry.get("fq-form").listeners.submit || [];
  assert.equal(listeners.length, 1, "exactly one submit handler is bound to the fixed-question form");
  let prevented = false;
  listeners[0]({ preventDefault() { prevented = true; }, target: registry.get("fq-form") });
  assert.ok(prevented, "the native form submission must be prevented");
}

async function settle() {
  // Let every pending microtask/promise chain in the sandbox run.
  for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));
}

// --- list rendering ----------------------------------------------------

test("the list is built from GET /questions with one entry per served question", async () => {
  const { sandbox, registry, calls } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  assert.ok(calls.some((c) => c.url === "/questions"), "the list must come from GET /questions");
  const items = listItems(registry);
  assert.equal(items.length, SERVED.length, "no question lost, none invented");
  items.forEach((item, i) => {
    assert.equal(item.tagName, "LI");
    assert.equal(item._radio.value, SERVED[i].id, "the radio value is the id, the ONLY thing sent back");
    assert.equal(item._radio.name, "fixed-question", "one shared radio group");
  });
});

test("the wording is rendered VERBATIM as inert text, never markup", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  const items = listItems(registry);
  items.forEach((item, i) => {
    const label = item.children[0];
    const texts = label.children.map((c) => c.textContent);
    assert.ok(texts.includes(SERVED[i].question), `wording of ${SERVED[i].id} rendered verbatim`);
    assert.ok(texts.includes(`${SERVED[i].number}.`), "1-based number shown");
    const tags = tagsIn(item);
    assert.ok(!tags.includes("SCRIPT") && !tags.includes("B"), "payload never parsed as HTML");
  });
});

test("a failed /questions load is reported and leaves the button disabled", async () => {
  const fetchImpl = async (url) => (url === "/questions"
    ? { ok: false, status: 500, json: async () => ({ message: "boom" }) }
    : routedFetch()(url));
  const { sandbox, registry } = loadAppJs(fetchImpl);
  await settle();
  assert.match(registry.get("fq-status").textContent, /Échec du chargement des questions fixes/);
  assert.equal(listItems(registry).length, 0);
  assert.equal(registry.get("fq-submit").disabled, true);
  assert.equal(typeof sandbox.askFixedQuestion, "function");
});

// --- enabling rules ----------------------------------------------------

test("the button stays disabled until a patient, a window AND a question are selected", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  assert.equal(registry.get("fq-submit").disabled, true, "no selection at all");

  sandbox.selectFixedQuestion("Q2");
  assert.equal(registry.get("fq-submit").disabled, true, "a question but no patient/window");

  await sandbox.selectWindow("Patient_1", 0);
  assert.equal(registry.get("fq-submit").disabled, false, "question + patient + window");
  assert.equal(registry.get("fq-submit").textContent, "Poser la question");
});

test("selecting a question marks exactly one entry as selected", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  sandbox.selectFixedQuestion("Q2");
  const selected = listItems(registry).filter((i) => i.classList.contains("selected"));
  assert.equal(selected.length, 1);
  assert.equal(selected[0]._questionId, "Q2");
  assert.equal(selected[0]._radio.checked, true);
  sandbox.selectFixedQuestion("Q3");
  assert.deepEqual(listItems(registry).filter((i) => i.classList.contains("selected")).map((i) => i._questionId), ["Q3"]);
});

test("an unknown question id cannot be selected", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q99");
  assert.equal(registry.get("fq-submit").disabled, true);
});

// --- the request that reaches the backend ------------------------------

test("submitting sends the question by id with the LIVE patient/window/split -- never the wording", async () => {
  const { sandbox, registry, calls } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_2", 7);
  sandbox.selectFixedQuestion("Q2");

  submitFixedQuestion(registry);
  await settle();

  const post = calls.find((c) => c.url === "/chat");
  assert.ok(post, "POST /chat must be issued");
  assert.equal(post.options.method, "POST");
  const body = JSON.parse(post.options.body);
  assert.deepEqual(body, { question_id: "Q2", video_id: "Patient_2", window: 7, split: "val" });
  assert.equal("question" in body, false, "the wording must never travel back to the backend");
});

test("the answer is rendered with its question, context, grounding, provenance and sources", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();

  assert.equal(registry.get("fq-answer-block").hidden, false);
  assert.equal(registry.get("fq-answer").textContent, "La réponse.");
  assert.equal(registry.get("fq-answer-question").textContent, `Question 1 : ${SERVED[0].question}`);
  assert.equal(registry.get("fq-sent-context").textContent, "Posée sur Patient_1 · fenêtre 0");
  const grounding = registry.get("fq-grounding").children.map((c) => c.textContent).join(" | ");
  assert.match(grounding, /Vérification : les valeurs citées proviennent/);
  assert.match(grounding, /Confiance : moyenne/);
  const provenance = registry.get("fq-provenance").textContent;
  assert.match(provenance, /inférence du modèle sur la fenêtre sélectionnée/);
  assert.match(provenance, /documentation du projet/);
  assert.ok(!/get_current_inference/.test(provenance), "raw tool names are not shown in the provenance line");
  const chips = registry.get("fq-sources").children[1].children;
  assert.equal(chips.length, 1);
  assert.equal(chips[0].textContent, "docs/HANDOFF.md §3");
  assert.equal(registry.get("fq-status").textContent, "");
  assert.equal(registry.get("fq-submit").disabled, false, "the button is usable again");
});

test("an ungrounded answer is labelled as such, and a malicious answer stays inert text", async () => {
  const evil = "<img src=x onerror=alert(1)>";
  const { sandbox, registry } = loadAppJs(routedFetch(async () => ({
    ok: true, json: async () => chatResponse({ answer: evil, grounded: false, warnings: [evil], sources: [] }),
  })));
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();

  assert.equal(registry.get("fq-answer").textContent, evil);
  assert.ok(!tagsIn(registry.get("fq-warnings")).includes("IMG"));
  const grounding = registry.get("fq-grounding").children.map((c) => c.textContent).join(" | ");
  assert.match(grounding, /n'ont pas pu être retrouvées/);
  assert.equal(registry.get("fq-sources").children.length, 0, "no source block without sources");
});

// --- loading state -----------------------------------------------------

test("while the request is pending the button is disabled and relabelled, then restored", async () => {
  let resolveChat;
  const pending = new Promise((r) => { resolveChat = r; });
  const { sandbox, registry } = loadAppJs(routedFetch(async () => pending));
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");

  submitFixedQuestion(registry);
  await settle();
  assert.equal(registry.get("fq-submit").disabled, true, "disabled while pending");
  assert.equal(registry.get("fq-submit").textContent, "Interrogation en cours…");
  assert.match(registry.get("fq-status").textContent, /Question 1 en cours de traitement/);
  assert.equal(registry.get("fq-answer-block").hidden, true, "no stale answer shown while pending");

  resolveChat({ ok: true, json: async () => chatResponse() });
  await settle();
  assert.equal(registry.get("fq-submit").disabled, false);
  assert.equal(registry.get("fq-submit").textContent, "Poser la question");
  assert.equal(registry.get("fq-answer-block").hidden, false);
});

// --- errors ------------------------------------------------------------

test("a network failure is shown as an error and the button recovers", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch(async () => { throw new Error("Failed to fetch"); }));
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();

  assert.equal(registry.get("fq-status").textContent, "Erreur : Failed to fetch");
  assert.equal(registry.get("fq-status").className, "fq-status fq-status-error");
  assert.equal(registry.get("fq-answer-block").hidden, true);
  assert.equal(registry.get("fq-submit").disabled, false);
});

test("an HTTP error body's message is surfaced", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch(async () => ({
    ok: false, status: 400, json: async () => ({ error: "bad_request", message: "Unknown question_id 'Q1'" }),
  })));
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();
  assert.equal(registry.get("fq-status").textContent, "Erreur : Unknown question_id 'Q1'");
});

// --- stale answers -----------------------------------------------------

test("changing patient removes the previous answer", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();
  assert.equal(registry.get("fq-answer-block").hidden, false);

  await sandbox.selectPatient("Patient_2");
  assert.equal(registry.get("fq-answer-block").hidden, true, "the answer for Patient_1 must not survive");
  assert.equal(registry.get("fq-answer").textContent, "");
  assert.match(registry.get("fq-status").textContent, /La sélection a changé/);
});

test("changing window removes the previous answer; toggling the frame of the same window keeps it", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();

  await sandbox.selectWindow("Patient_1", 0, { which: "first" });
  assert.equal(registry.get("fq-answer-block").hidden, false, "same (patient, window): the answer stays");

  await sandbox.selectWindow("Patient_1", 1);
  assert.equal(registry.get("fq-answer-block").hidden, true, "another window: the answer goes");
});

test("a response arriving after the selection moved is discarded, never shown", async () => {
  let resolveChat;
  const pending = new Promise((r) => { resolveChat = r; });
  const { sandbox, registry } = loadAppJs(routedFetch(async () => pending));
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();

  await sandbox.selectWindow("Patient_2", 3);      // user moves on while the LLM is still working
  resolveChat({ ok: true, json: async () => chatResponse() });
  await settle();

  assert.equal(registry.get("fq-answer-block").hidden, true, "the late answer belongs to Patient_1/0");
  assert.match(registry.get("fq-status").textContent, /a été ignorée/);
  assert.equal(registry.get("fq-submit").disabled, false);
});

test("after a discarded late response, a fresh question on the new selection is answered normally", async () => {
  let resolveFirst;
  let nth = 0;
  const { sandbox, registry } = loadAppJs(routedFetch(async (body) => {
    nth += 1;
    if (nth === 1) return new Promise((r) => { resolveFirst = r; });
    return { ok: true, json: async () => chatResponse({ question_id: body.question_id, answer: `réponse ${body.question_id}` }) };
  }));
  await sandbox.loadFixedQuestions();
  await sandbox.selectWindow("Patient_1", 0);
  sandbox.selectFixedQuestion("Q1");
  submitFixedQuestion(registry);
  await settle();

  await sandbox.selectWindow("Patient_2", 3);
  assert.equal(registry.get("fq-submit").disabled, true, "still pending: no second ask yet");
  resolveFirst({ ok: true, json: async () => chatResponse({ answer: "PÉRIMÉE" }) });
  await settle();
  assert.equal(registry.get("fq-answer-block").hidden, true);
  assert.equal(registry.get("fq-submit").disabled, false, "settled: the button is usable again");

  sandbox.selectFixedQuestion("Q2");
  submitFixedQuestion(registry);
  await settle();
  assert.equal(registry.get("fq-answer").textContent, "réponse Q2");
  assert.equal(registry.get("fq-answer-question").textContent, `Question 2 : ${SERVED[1].question}`);
  assert.equal(registry.get("fq-sent-context").textContent, "Posée sur Patient_2 · fenêtre 3");
});

// --- context line ------------------------------------------------------

test("the fixed-questions context line follows the live selection", async () => {
  const { sandbox, registry } = loadAppJs(routedFetch());
  await sandbox.loadFixedQuestions();
  sandbox.renderChatContext();
  assert.match(registry.get("fq-context").textContent, /Aucun patient ni fenêtre sélectionné/);
  await sandbox.selectWindow("Patient_1", 5);
  const ctx = registry.get("fq-context").children.map((c) => c.textContent).join("");
  assert.equal(ctx, "La question sera posée sur : Patient_1 · fenêtre 5 · phase (modèle) t4 (50.0%)");
});
