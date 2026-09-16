// Minimal MVP frontend -- vanilla JS, no build step, no framework.
// Talks ONLY to this Flask app's own endpoints (/videos, /videos/<id>/timeline,
// /videos/<id>/window/<w>, /questions, /chat). Never calls Ollama/ChromaDB/the model directly.

const SPLIT = "val";

// --- Time unit ---------------------------------------------------------
//
// CONFIRMED BY THE PROJECT OWNER (2026-09-02): the `time` column of
// `Data/embryo_dataset_time_elapsed/{patient}_timeElapsed.csv` is in HOURS.
//
// This is the ONLY place the unit enters the interface, and it is stated
// here rather than derived, because nothing upstream carries it:
//   - the raw CSV header is `frame_index,time` -- verified across all 704
//     files, no unit anywhere in the source data;
//   - `Embeddings/resnet18/{split}/metadata_with_time.manifest.json` records
//     `time_unit = "unknown/unverified - raw values copied as-is from
//     timeElapsed.csv column time, not converted or assumed to be hours"`;
//   - `Training/reporting/schemas.py` deliberately never converts it.
//
// That backend metadata is NOT edited from here: it is an embeddings-cache
// artifact and the pipeline's own source of truth, out of scope for a UI
// change. So the API keeps reporting `time_unit="unknown/unverified"`, and
// this layer keeps that fact visible in the field's tooltip instead of
// hiding it -- the displayed value stays traceable to both its source and
// its confirmation. NO numeric conversion is applied anywhere: the values
// rendered are exactly the ones the API returns.
//
// Durable follow-up (deliberately not done here, it would be a data-lineage
// change and would contradict the Reporting API's own tests): record the
// confirmation in `metadata_with_time.manifest.json`, after which
// `window.time_unit` itself becomes authoritative and this constant can be
// dropped in favour of it.
const TIME_UNIT = {
  symbol: "h",
  label: "heures",
  provenance: "Unité confirmée par le responsable du projet (2026-09-02). Les valeurs sont affichées " +
    "exactement telles que renvoyées par la Reporting API, sans aucune conversion. Les métadonnées de " +
    "l'API rapportent toujours time_unit=\"unknown/unverified\", car le CSV brut du jeu de données " +
    "n'indique aucune unité ; ce drapeau est conservé ici pour la traçabilité.",
};

// Deterministic phase -> color mapping (a small fixed palette, cycled by
// phase name hash) -- purely cosmetic, never used for any scientific
// interpretation.
const PALETTE = [
  "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
  "#EEBA30", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
  "#5C7A99", "#D67C1C", "#8B5FBF", "#3E8E7E", "#C4553D",
];

function colorForPhase(phase) {
  let hash = 0;
  for (let i = 0; i < phase.length; i++) hash = (hash * 31 + phase.charCodeAt(i)) >>> 0;
  return PALETTE[hash % PALETTE.length];
}

// --- Safe DOM construction helpers ------------------------------------
// Every value that reaches these helpers may originate from the backend
// (Reporting API fields, RAG chunk source/section metadata) or, via the
// grounding checker's warning text, indirectly from the LLM's own
// generated answer -- docs/WEBAPP_SECURITY.md. None of it is ever
// inserted as HTML; only textContent/createElement are used, so no
// string here can ever be parsed as markup, regardless of what
// characters it contains.

function textSpan(text, className) {
  const span = document.createElement("span");
  if (className) span.className = className;
  span.textContent = text;
  return span;
}

// Renders a list of warning strings as "⚠ <text>" lines, one per <br>-
// separated line -- same visual layout as the previous
// `warnings.map(w => "⚠ " + w).join("<br>")`, but each warning's text is
// a genuine DOM text node, never HTML the browser parses.
// Warnings the VIEWER does not surface to the user. Purely a display filter:
// the backend keeps computing and returning them unchanged
// (`reporting/inference_service.py`), the Reporting API keeps serving them,
// and nothing scientific is altered -- this layer only decides what is worth
// putting in front of a clinician. `duration not estimable ...` reports a
// structural property of the training data (a phase with no observed interior
// segment), not a property of the patient being viewed, so it is noise here.
const HIDDEN_VIEWER_WARNING_PREFIXES = [
  "duration not estimable for phase",
];

function isHiddenViewerWarning(text) {
  return HIDDEN_VIEWER_WARNING_PREFIXES.some((prefix) => String(text).startsWith(prefix));
}

function renderWarningsList(container, warnings) {
  container.replaceChildren();
  (warnings || []).forEach((w, i) => {
    if (i > 0) container.appendChild(document.createElement("br"));
    container.appendChild(document.createTextNode(`⚠ ${w}`));
  });
}

// Renders /chat's `sources` array as citation chips -- same visual
// output as the previous `sources.map(s => \`<span class="source-chip">
// ${s.source} ${s.section}</span>\`).join("")`, but each chip's label is
// assigned via textContent, never parsed as HTML.
function renderSources(container, sources) {
  container.replaceChildren(...(sources || []).map((s) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = `${s.source || "?"} ${s.section ? "§" + s.section : ""}`;
    return chip;
  }));
}

// Traduction d'AFFICHAGE des niveaux de confiance renvoyés par le backend.
// La valeur transmise par l'API n'est ni modifiée ni réinterprétée : seule son
// étiquette visible change. Une valeur inconnue est affichée telle quelle.
const CONFIDENCE_LABELS = { high: "élevée", medium: "moyenne", low: "faible" };

function confidenceLabel(value) {
  return CONFIDENCE_LABELS[value] || String(value);
}

let currentVideo = null;
let currentWindow = null;
let currentWindowData = null; // full /window/<w> response for the current selection, used to
// render the fixed-questions "Current context" line without a second fetch.

// --- Viewer state (Phase 7.11) ----------------------------------------
//
// The addressable unit of this API is the WINDOW, not the raw frame:
// /videos/<id>/window/<w> and .../frame?which=first|last are the only frame
// entry points, and a window holds 8 raw frames of which two are served.
// So "step" here means "step one window", and the frame numbers shown are
// always READ from the backend for that window.
//
// This matters: raw frame numbers are NOT contiguous across consecutive
// windows. Verified on real Val data -- Patient_319 window 156 covers frames
// 167..174 but window 157 covers 175..182. Any client-side arithmetic on
// frame numbers would therefore be wrong, and none is done anywhere below.
let videoWindows = { first: null, last: null }; // real bounds from GET /videos/<id>
let timelineSegments = [];   // the /timeline response, reused by the phase picker
let anchorWindow = null;     // the window the user selected (timeline click / phase jump);
// stepping with prev/next moves away from it, "Back to selection" returns to it.
let frameWhich = "last";     // "first" | "last" -- which of the window's frames is displayed.
// "last" matches the causal "current" semantics the Reporting API itself uses.

// Renders the context a fixed question WOULD be sent with right now --
// always derived from the live currentVideo/currentWindow/currentWindowData
// state (Phase 7.9: "the question context must automatically follow the
// viewer's selection, never a hardcoded or stale window_start"). Called
// every time selectWindow() finishes, so switching windows immediately
// updates what's shown here.
function renderChatContext() {
  const fqEl = document.getElementById("fq-context");
  if (!currentVideo || currentWindow === null || !currentWindowData) {
    fqEl.textContent = "Aucun patient ni fenêtre sélectionné — sélectionnez un patient et une fenêtre " +
      "dans le visualiseur avant de poser une question fixe.";
    fqEl.classList.add("chat-context-empty");
    renderFixedQuestionControls();
    return;
  }
  fqEl.classList.remove("chat-context-empty");
  const phase = currentWindowData.current_state.current_phase;
  const probability = currentWindowData.current_state.phase_probability;

  function strong(text) {
    const node = document.createElement("strong");
    node.textContent = text;
    return node;
  }

  // The live state, rendered for the fixed-questions panel: what is shown
  // here is exactly what a fixed question is sent with.
  fqEl.replaceChildren(
    document.createTextNode("La question sera posée sur : "), strong(currentVideo),
    document.createTextNode(" · fenêtre "), strong(String(currentWindow)),
    document.createTextNode(" · phase (modèle) "), strong(phase),
    document.createTextNode(` (${(probability * 100).toFixed(1)}%)`),
  );
  renderFixedQuestionControls();
}

function showAppError(message) {
  const el = document.getElementById("app-error");
  el.textContent = message;
  el.hidden = false;
}

function clearAppError() {
  document.getElementById("app-error").hidden = true;
}

async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  const body = await resp.json();
  if (!resp.ok) throw new Error(body.message || `Échec de la requête ${url} (${resp.status})`);
  return body;
}

// --- Patient selection ------------------------------------------------
//
// `/videos` returns `trajectory_service.list_videos()`, whose own docstring
// states the order is "cache order (stable, not sorted)" and that callers
// cross-referencing indices depend on it. That order is meaningless to a
// human scanning a ~100-entry list, so the ORDERING IS APPLIED HERE, in the
// presentation layer only: the backend response is never reordered in place,
// no endpoint changed, and every `option.value` stays the exact backend
// identifier (`Patient_319`), which is what is sent back to /videos/<id>,
// /timeline, /window/<w> and /chat. Display and identity are the same string;
// only the row order and the surrounding layout change.

let allVideos = []; // sorted copy of /videos' `videos`, cached so filtering
// never needs a second request and the order can never drift between renders.

// Natural (numeric-aware) comparison so Patient_2 sorts before Patient_10
// instead of after it, which a plain lexicographic sort would do. Splits each
// id into digit / non-digit chunks and compares chunk by chunk: digits
// numerically, everything else lexicographically. Deterministic and total --
// equal ids compare equal, so the result is stable across renders.
function compareVideoIds(a, b) {
  const chunks = (s) => s.match(/\d+|\D+/g) || [];
  const ca = chunks(a);
  const cb = chunks(b);
  const n = Math.min(ca.length, cb.length);
  for (let i = 0; i < n; i++) {
    const x = ca[i];
    const y = cb[i];
    const bothNumeric = /^\d/.test(x) && /^\d/.test(y);
    if (bothNumeric) {
      const diff = Number(x) - Number(y);
      if (diff !== 0) return diff;
    } else if (x !== y) {
      return x < y ? -1 : 1;
    }
  }
  return ca.length - cb.length;
}

function matchesFilter(videoId, filterText) {
  if (!filterText) return true;
  return videoId.toLowerCase().includes(filterText.toLowerCase());
}

// Rebuilds the <select> from the sorted cache, honouring the current filter.
// Keeps the current selection whenever it survives the filter, so typing in
// the filter box never silently switches which patient is loaded.
function renderVideoOptions() {
  const select = document.getElementById("video-select");
  const filterText = (document.getElementById("patient-filter").value || "").trim();
  const visible = allVideos.filter((name) => matchesFilter(name, filterText));

  select.replaceChildren(...visible.map((name) => {
    const opt = document.createElement("option");
    opt.value = name;      // the EXACT backend identifier -- never a display-only form
    opt.textContent = name;
    return opt;
  }));

  if (currentVideo !== null && visible.includes(currentVideo)) {
    select.value = currentVideo;
  }

  const countEl = document.getElementById("patient-count");
  countEl.textContent = filterText
    ? `${visible.length} patient${visible.length === 1 ? "" : "s"} sur ${allVideos.length} ` +
      `correspond${visible.length === 1 ? "" : "ent"} à « ${filterText} »`
    : `${allVideos.length} patient${allVideos.length === 1 ? "" : "s"}`;
  return visible;
}

// The patient's own summary, kept in a panel PHYSICALLY SEPARATE from the
// picker: the list answers "which patient", this answers "what is this
// patient". Uses the already-existing GET /videos/<id> endpoint (n_windows /
// first_window / last_window) -- no backend change, no new endpoint, no
// scientific computation.
async function loadPatientInfo(videoId) {
  const idEl = document.getElementById("pi-id");
  const splitEl = document.getElementById("pi-split");
  const windowsEl = document.getElementById("pi-windows");
  const rangeEl = document.getElementById("pi-range");

  idEl.textContent = videoId;
  splitEl.textContent = SPLIT;
  windowsEl.textContent = "…";
  rangeEl.textContent = "…";

  try {
    const info = await fetchJSON(`/videos/${encodeURIComponent(videoId)}?split=${SPLIT}`);
    idEl.textContent = info.video_name;
    splitEl.textContent = info.split;
    windowsEl.textContent = String(info.n_windows);
    rangeEl.textContent = (info.first_window === null || info.last_window === null)
      ? "aucune"
      : `${info.first_window} → ${info.last_window}`;
    // The viewer's step bounds come from the backend's real window list --
    // never guessed from a count or assumed to start at 0.
    videoWindows = { first: info.first_window, last: info.last_window };
  } catch (err) {
    // A failed summary must never block the timeline/viewer below it.
    windowsEl.textContent = "indisponible";
    rangeEl.textContent = "indisponible";
  }
}

async function selectPatient(videoId) {
  currentVideo = videoId;
  document.getElementById("video-select").value = videoId;
  // A previous fixed-question answer was computed for ANOTHER patient: it
  // must never stay on screen next to the new one (see
  // invalidateFixedQuestionAnswerIfStale for the window-level rule).
  invalidateFixedQuestionAnswerIfStale();
  await loadPatientInfo(videoId);
  await loadTimeline(videoId);
}

async function loadVideos() {
  const data = await fetchJSON(`/videos?split=${SPLIT}`);
  // Sort a COPY -- `data.videos` is the backend's own ordered response and is
  // left untouched, so nothing here can be mistaken for a backend reordering.
  allVideos = [...data.videos].sort(compareVideoIds);
  const visible = renderVideoOptions();
  if (visible.length > 0) {
    await selectPatient(visible[0]);
  }
}

// Labels one timeline segment for the phase picker. The segment IS the
// existing /timeline band -- the picker is a second view of the SAME data,
// never a second phase-segmentation logic.
function phaseOptionLabel(seg, index) {
  return `${index + 1}. ${seg.phase} — fenêtres ${seg.start_window}–${seg.end_window} ` +
    `(${seg.n_windows} fenêtre${seg.n_windows === 1 ? "" : "s"})`;
}

function renderPhaseOptions() {
  const select = document.getElementById("phase-select");
  select.replaceChildren(...timelineSegments.map((seg, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);              // index into timelineSegments, not a phase name:
    opt.textContent = phaseOptionLabel(seg, i);  // the same phase can occur in several bands
    return opt;
  }));
}

// Keeps the phase picker in sync with whatever window is being viewed, so the
// timeline, the picker and the frame panel can never disagree about which
// band is on screen. Returns the segment index, or -1 if the window falls in
// no band (possible only if the two sources ever disagreed).
function segmentIndexForWindow(windowStart) {
  return timelineSegments.findIndex(
    (seg) => windowStart >= seg.start_window && windowStart <= seg.end_window);
}

async function loadTimeline(videoId) {
  const data = await fetchJSON(`/videos/${encodeURIComponent(videoId)}/timeline?split=${SPLIT}`);
  timelineSegments = data.segments || [];
  renderPhaseOptions();
  const bar = document.getElementById("timeline-bar");
  const legend = document.getElementById("timeline-legend");
  bar.replaceChildren();
  legend.replaceChildren();
  const seenPhases = new Set();

  for (const seg of data.segments) {
    const div = document.createElement("div");
    div.className = "timeline-segment";
    div.style.flexGrow = seg.n_windows;
    div.style.backgroundColor = colorForPhase(seg.phase);
    div.textContent = seg.phase;
    div.title = `${seg.phase} : fenêtres ${seg.start_window}-${seg.end_window} ` +
      `(${seg.n_windows} fenêtres, probabilité moyenne ${seg.mean_probability.toFixed(3)})`;
    div.addEventListener("click", async () => {
      try {
        // A timeline click is a deliberate user SELECTION -> it becomes the
        // anchor that "Back to selection" returns to.
        await selectWindow(videoId, seg.start_window, { anchor: true, which: "last" });
        clearAppError();
      } catch (err) {
        showAppError(`Échec du chargement de la fenêtre ${seg.start_window} : ${err.message}`);
      }
    });
    bar.appendChild(div);

    if (!seenPhases.has(seg.phase)) {
      seenPhases.add(seg.phase);
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = colorForPhase(seg.phase);
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(seg.phase));
      legend.appendChild(item);
    }
  }

  if (timelineSegments.length > 0) {
    await selectWindow(videoId, timelineSegments[0].start_window, { anchor: true, which: "last" });
  } else {
    // No band at all: selectWindow() -- the usual path into
    // renderViewerControls() -- never runs, so the viewer controls would keep
    // their markup default (enabled) and offer navigation for a video that
    // has nothing to navigate. Refresh them explicitly instead.
    currentWindow = null;
    anchorWindow = null;
    renderViewerControls();
  }
}

async function selectWindow(videoId, windowStart, options) {
  const opts = options || {};
  currentVideo = videoId;
  currentWindow = windowStart;
  if (opts.anchor) anchorWindow = windowStart;
  if (opts.which === "first" || opts.which === "last") frameWhich = opts.which;
  invalidateFixedQuestionAnswerIfStale();

  document.querySelectorAll(".timeline-segment").forEach((el) => el.classList.remove("selected"));
  const bar = document.getElementById("timeline-bar");
  const segments = Array.from(bar.children);
  for (const el of segments) {
    // Highlight the segment whose title range contains this window (cheap
    // re-derivation instead of storing extra DOM state for this MVP).
    if (el.title.includes(`fenêtres ${windowStart}-`) || el.title.match(new RegExp(`-${windowStart}\\D`))) {
      el.classList.add("selected");
    }
  }

  const data = await fetchJSON(`/videos/${encodeURIComponent(videoId)}/window/${windowStart}?split=${SPLIT}`);
  currentWindowData = data;
  document.getElementById("cw-window").textContent = data.window.window_start;
  document.getElementById("cw-phase").textContent = data.current_state.current_phase;
  document.getElementById("cw-probability").textContent = data.current_state.phase_probability.toFixed(4);
  document.getElementById("cw-entropy").textContent = data.current_state.entropy.toFixed(4);

  const frameImg = document.getElementById("cw-frame");
  const frameFallback = document.getElementById("cw-frame-fallback");
  const frameNumberEl = document.getElementById("cw-frame-number");
  const frameRangeEl = document.getElementById("cw-frame-range");
  const frame = data.frame || {};
  if (frame.available) {
    // Both the number shown and the URL loaded come from the SAME backend
    // `frame` block for THIS window, picked by `frameWhich` -- the displayed
    // number can therefore never drift from the image actually fetched, and
    // no frame number is ever computed client-side.
    const showFirst = frameWhich === "first";
    const shownNumber = showFirst
      ? (frame.first_frame_number !== undefined ? frame.first_frame_number : frame.frame_number)
      : (frame.last_frame_number !== undefined ? frame.last_frame_number : frame.frame_number);
    const shownUrl = showFirst
      ? (frame.first_frame_url || frame.frame_url)
      : (frame.last_frame_url || frame.frame_url);

    frameNumberEl.textContent =
      `${shownNumber} (${showFirst ? "première" : "dernière"} de la fenêtre)`;
    if (frame.first_frame_number !== undefined && frame.first_frame_number !== null) {
      frameRangeEl.textContent =
        `${frame.first_frame_number}–${frame.last_frame_number} (${frame.n_frames} images)`;
    } else {
      frameRangeEl.textContent = "-";
    }
    frameImg.hidden = false;
    frameFallback.hidden = true;
    frameImg.onerror = () => { frameImg.hidden = true; frameFallback.hidden = false; };
    frameImg.src = shownUrl;
  } else {
    frameNumberEl.textContent = "-";
    frameRangeEl.textContent = "-";
    frameImg.hidden = true;
    frameFallback.hidden = false;
  }

  const timeEl = document.getElementById("cw-time");
  if (data.window.time_available) {
    // The numbers are the API's own, unrounded values formatted for display
    // only (.toFixed(2), unchanged from before) -- never converted.
    timeEl.textContent =
      `${data.window.window_start_time.toFixed(2)} → ${data.window.window_end_time.toFixed(2)} ` +
      `${TIME_UNIT.symbol}`;
    // The API's own time_unit flag is preserved here rather than dropped, so
    // the displayed unit always stays traceable to what the backend actually
    // reported for THIS record.
    timeEl.title = `${TIME_UNIT.label} — ${TIME_UNIT.provenance} ` +
      `(time_unit renvoyé par l'API pour cet enregistrement : "${data.window.time_unit}")`;
  } else {
    timeEl.textContent = "indisponible";
    timeEl.title = "Aucune métadonnée temporelle n'est disponible pour cette fenêtre.";
  }

  renderWarningsList(document.getElementById("cw-warnings"),
    (data.warnings || []).filter((w) => !isHiddenViewerWarning(w)));

  renderViewerControls();
  renderChatContext();
}

// --- Viewer navigation -------------------------------------------------

function atFirstWindow() {
  return videoWindows.first === null || currentWindow === null || currentWindow <= videoWindows.first;
}

function atLastWindow() {
  return videoWindows.last === null || currentWindow === null || currentWindow >= videoWindows.last;
}

// Enables/disables the controls against the video's REAL window bounds and
// keeps the phase picker pointed at the band being viewed.
function renderViewerControls() {
  document.getElementById("btn-prev-window").disabled = atFirstWindow();
  document.getElementById("btn-next-window").disabled = atLastWindow();
  document.getElementById("btn-which-first").disabled = frameWhich === "first";
  document.getElementById("btn-which-last").disabled = frameWhich === "last";
  document.getElementById("btn-back-anchor").disabled =
    anchorWindow === null || anchorWindow === currentWindow;

  const positionEl = document.getElementById("frame-position");
  if (currentWindow === null || videoWindows.first === null) {
    positionEl.textContent = "-";
  } else {
    const offset = anchorWindow === null ? null : currentWindow - anchorWindow;
    const offsetText = (offset === null || offset === 0) ? "" :
      ` · ${offset > 0 ? "+" : ""}${offset} par rapport à la sélection`;
    positionEl.textContent =
      `fenêtre ${currentWindow} sur ${videoWindows.first}–${videoWindows.last}${offsetText}`;
  }

  const index = segmentIndexForWindow(currentWindow);
  const phaseSelect = document.getElementById("phase-select");
  if (index >= 0) phaseSelect.value = String(index);
  document.getElementById("btn-phase-first").disabled = index < 0;
  document.getElementById("btn-phase-last").disabled = index < 0;
}

// One guarded step. Refuses to step past the video's real bounds rather than
// requesting a window that does not exist.
async function stepWindow(delta) {
  if (currentVideo === null || currentWindow === null) return;
  const target = currentWindow + delta;
  if (videoWindows.first !== null && target < videoWindows.first) return;
  if (videoWindows.last !== null && target > videoWindows.last) return;
  await selectWindow(currentVideo, target, {});  // keeps the anchor AND `frameWhich`
}

// Jumps to a phase band's own boundary window, showing the boundary frame:
// the band's FIRST window shown at its first frame, or its LAST window shown
// at its last frame. Both windows come from the /timeline segment itself.
async function jumpToPhaseEdge(edge) {
  const index = Number(document.getElementById("phase-select").value);
  const seg = timelineSegments[index];
  if (!seg) return;
  const target = edge === "first" ? seg.start_window : seg.end_window;
  await selectWindow(currentVideo, target, { anchor: true, which: edge });
}

async function backToAnchor() {
  if (currentVideo === null || anchorWindow === null) return;
  await selectWindow(currentVideo, anchorWindow, { which: "last" });
}

document.getElementById("video-select").addEventListener("change", async (e) => {
  try {
    await selectPatient(e.target.value);
    clearAppError();
  } catch (err) {
    showAppError(`Échec du chargement de ${e.target.value} : ${err.message}`);
  }
});

// Filtering is purely a view over the cached, sorted list -- it never
// refetches, never reorders, and never changes the selected patient unless
// the current one is filtered out (in which case the selection simply stays
// where it was until the user picks another row).
document.getElementById("patient-filter").addEventListener("input", () => {
  renderVideoOptions();
});

// --- Viewer controls wiring -------------------------------------------

async function guarded(action, describe) {
  try {
    await action();
    clearAppError();
  } catch (err) {
    showAppError(`${describe} : ${err.message}`);
  }
}

document.getElementById("btn-prev-window").addEventListener("click", () => {
  guarded(() => stepWindow(-1), "Échec du chargement de la fenêtre précédente");
});

document.getElementById("btn-next-window").addEventListener("click", () => {
  guarded(() => stepWindow(1), "Échec du chargement de la fenêtre suivante");
});

document.getElementById("btn-which-first").addEventListener("click", () => {
  guarded(() => selectWindow(currentVideo, currentWindow, { which: "first" }),
    "Échec du chargement de la première image de cette fenêtre");
});

document.getElementById("btn-which-last").addEventListener("click", () => {
  guarded(() => selectWindow(currentVideo, currentWindow, { which: "last" }),
    "Échec du chargement de la dernière image de cette fenêtre");
});

document.getElementById("btn-phase-first").addEventListener("click", () => {
  guarded(() => jumpToPhaseEdge("first"), "Échec du chargement de la première image de la phase");
});

document.getElementById("btn-phase-last").addEventListener("click", () => {
  guarded(() => jumpToPhaseEdge("last"), "Échec du chargement de la dernière image de la phase");
});

document.getElementById("btn-back-anchor").addEventListener("click", () => {
  guarded(backToAnchor, "Échec du retour à la fenêtre sélectionnée");
});

// Selecting a band jumps to its start -- the same action a timeline click
// performs, so the two controls can never mean different things.
document.getElementById("phase-select").addEventListener("change", () => {
  guarded(() => jumpToPhaseEdge("first"), "Échec du chargement de la phase sélectionnée");
});

// Keyboard stepping, deliberately inert while the user is typing: the patient
// filter is a text field in the same page, and stealing its arrow keys would
// break editing. Also ignores any modified key so browser/OS shortcuts
// (Ctrl+Home, Alt+Left = Back, ...) keep their normal meaning.
function isTypingTarget(target) {
  if (!target) return false;
  const tag = (target.tagName || "").toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable === true;
}

document.addEventListener("keydown", (e) => {
  if (isTypingTarget(e.target)) return;
  if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return;
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    guarded(() => stepWindow(-1), "Échec du chargement de la fenêtre précédente");
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    guarded(() => stepWindow(1), "Échec du chargement de la fenêtre suivante");
  } else if (e.key === "Home") {
    e.preventDefault();
    guarded(backToAnchor, "Échec du retour à la fenêtre sélectionnée");
  }
});

// --- Fixed questions (the frozen benchmark Q1-Q15) ---------------------
//
// The wording is NEVER written here. The list comes from GET /questions,
// which serves `orchestrator.fixed_question_benchmark.FIXED_QUESTIONS` --
// the exact objects the benchmark runner iterates -- and a question is
// sent back to POST /chat by its `question_id` only, so the backend resolves
// the wording from that same single source. The frontend therefore cannot
// hold, reword or drift a second copy of any question.
//
// State rules:
//   - a question can only be asked with a patient AND a window selected;
//   - an answer is tied to the (patient, window) it was asked on: as soon as
//     either changes, the previous answer is removed, never left next to a
//     different selection;
//   - a response arriving for a request that is no longer the latest one
//     (the user changed selection or asked again meanwhile) is discarded.

let fixedQuestions = [];               // [{id, number, question, category}] as served
let selectedFixedQuestionId = null;
let fixedQuestionRequestSeq = 0;       // increments per request; late responses are dropped
let fixedQuestionPending = false;
let fixedQuestionAnsweredFor = null;   // {video, window} the visible answer belongs to

// Plain-language provenance for the tools the pipeline actually used. The
// raw tool names stay available in the "Détails techniques" block; this map
// only decides how they are worded for a non-developer reader. An unknown
// tool name is shown as-is rather than hidden.
const TOOL_PROVENANCE_LABELS = {
  get_current_inference: "inférence du modèle sur la fenêtre sélectionnée",
  get_inference_history: "inférence du modèle sur les fenêtres voisines",
  get_transition_events: "historique des transitions annotées",
  get_trajectory: "trajectoire complète prédite par le modèle",
  get_model_info: "informations sur le modèle",
  retrieve_documents: "documentation du projet",
};

function toolProvenanceLabel(name) {
  return TOOL_PROVENANCE_LABELS[name] || String(name);
}

function fixedQuestionById(id) {
  return fixedQuestions.find((q) => q.id === id) || null;
}

function canAskFixedQuestion() {
  return currentVideo !== null && currentWindow !== null && selectedFixedQuestionId !== null
    && !fixedQuestionPending;
}

function renderFixedQuestionControls() {
  const submit = document.getElementById("fq-submit");
  submit.disabled = !canAskFixedQuestion();
  submit.textContent = fixedQuestionPending ? "Interrogation en cours…" : "Poser la question";
}

function setFixedQuestionStatus(text, kind) {
  const el = document.getElementById("fq-status");
  el.textContent = text || "";
  el.className = "fq-status" + (kind ? ` fq-status-${kind}` : "");
}

function selectFixedQuestion(id) {
  selectedFixedQuestionId = fixedQuestionById(id) ? id : null;
  const list = document.getElementById("fq-list");
  for (const item of Array.from(list.children || [])) {
    const isSelected = item._questionId === selectedFixedQuestionId;
    if (isSelected) item.classList.add("selected"); else item.classList.remove("selected");
    if (item._radio) item._radio.checked = isSelected;
  }
  renderFixedQuestionControls();
}

// Builds one <li> per served question: a radio input (keyboard/screen-reader
// friendly, one shared `name` so exactly one can be checked) with the 1-based
// number and the VERBATIM wording as inert text nodes.
function renderFixedQuestionList() {
  const list = document.getElementById("fq-list");
  list.replaceChildren(...fixedQuestions.map((q) => {
    const item = document.createElement("li");
    item.className = "fq-item";
    item._questionId = q.id;   // plain property: the id never round-trips through markup

    const label = document.createElement("label");
    label.className = "fq-label";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "fixed-question";
    radio.value = q.id;                      // the id is the ONLY thing sent back
    radio.addEventListener("change", () => selectFixedQuestion(q.id));
    item._radio = radio;

    const number = document.createElement("span");
    number.className = "fq-number";
    number.textContent = `${q.number}.`;

    const text = document.createElement("span");
    text.className = "fq-text";
    text.textContent = q.question;           // verbatim, as text -- never markup

    label.appendChild(radio);
    label.appendChild(number);
    label.appendChild(text);
    item.appendChild(label);
    return item;
  }));
}

async function loadFixedQuestions() {
  const data = await fetchJSON(`/questions`);
  fixedQuestions = Array.isArray(data.questions) ? data.questions.slice() : [];
  renderFixedQuestionList();
  if (selectedFixedQuestionId !== null && !fixedQuestionById(selectedFixedQuestionId)) {
    selectedFixedQuestionId = null;
  }
  renderFixedQuestionControls();
}

function clearFixedQuestionAnswer() {
  document.getElementById("fq-answer-block").hidden = true;
  for (const id of ["fq-answer-question", "fq-sent-context", "fq-answer"]) {
    document.getElementById(id).textContent = "";
  }
  for (const id of ["fq-grounding", "fq-provenance", "fq-sources", "fq-warnings"]) {
    document.getElementById(id).replaceChildren();
  }
  fixedQuestionAnsweredFor = null;
}

// Called on every selection change. The answer on screen belongs to exactly
// one (patient, window); if the live selection no longer matches it, the
// answer is removed. Toggling first/last frame within the same window keeps
// it (the frame shown is not part of the question's context).
function invalidateFixedQuestionAnswerIfStale() {
  if (fixedQuestionAnsweredFor === null) return;
  const stale = fixedQuestionAnsweredFor.video !== currentVideo
    || fixedQuestionAnsweredFor.window !== currentWindow;
  if (!stale) return;
  clearFixedQuestionAnswer();
  setFixedQuestionStatus("La sélection a changé : la réponse précédente a été retirée.", "info");
}

function renderFixedQuestionAnswer(data, asked, sentVideo, sentWindow) {
  const block = document.getElementById("fq-answer-block");
  document.getElementById("fq-answer-question").textContent =
    `Question ${asked.number} : ${asked.question}`;
  document.getElementById("fq-sent-context").textContent =
    `Posée sur ${sentVideo} · fenêtre ${sentWindow}`;
  document.getElementById("fq-answer").textContent = data.answer;

  const groundingEl = document.getElementById("fq-grounding");
  groundingEl.replaceChildren(
    textSpan(data.grounded
      ? "Vérification : les valeurs citées proviennent des données fournies au modèle."
      : "Vérification : certaines valeurs citées n'ont pas pu être retrouvées dans les données fournies.",
      data.grounded ? "grounded-true" : "grounded-false"),
    textSpan(`Confiance : ${confidenceLabel(data.confidence)}`),
  );

  const provenanceEl = document.getElementById("fq-provenance");
  const tools = Array.from(data.tools_called || []);
  provenanceEl.replaceChildren(
    document.createTextNode(tools.length
      ? `Informations utilisées : ${tools.map(toolProvenanceLabel).join(" ; ")}.`
      : "Informations utilisées : aucune donnée n'a été consultée pour cette réponse."),
  );

  const sourcesEl = document.getElementById("fq-sources");
  const sources = Array.from(data.sources || []);
  if (sources.length) {
    const title = document.createElement("span");
    title.className = "answer-sources-title";
    title.textContent = "Sources documentaires : ";
    const chips = document.createElement("span");
    renderSources(chips, sources);
    sourcesEl.replaceChildren(title, chips);
  } else {
    sourcesEl.replaceChildren();
  }

  const warningsEl = document.getElementById("fq-warnings");
  const technical = [`outils : ${tools.join(", ") || "aucun"}`, `route : ${data.route}`]
    .concat(Array.from(data.warnings || []));
  renderWarningsList(warningsEl, technical);

  fixedQuestionAnsweredFor = { video: sentVideo, window: sentWindow };
  block.hidden = false;
}

async function askFixedQuestion() {
  if (!canAskFixedQuestion()) return;
  const asked = fixedQuestionById(selectedFixedQuestionId);
  // Read the LIVE selection at send time -- this is exactly what the request
  // carries, and what the answer will be tied to.
  const sentVideo = currentVideo;
  const sentWindow = currentWindow;
  const seq = ++fixedQuestionRequestSeq;

  fixedQuestionPending = true;
  clearFixedQuestionAnswer();
  setFixedQuestionStatus(`Question ${asked.number} en cours de traitement…`, "pending");
  renderFixedQuestionControls();

  try {
    const data = await fetchJSON("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question_id: asked.id, video_id: sentVideo, window: sentWindow, split: SPLIT,
      }),
    });
    if (seq !== fixedQuestionRequestSeq) return;  // superseded: drop silently
    if (sentVideo !== currentVideo || sentWindow !== currentWindow) {
      // The selection moved while the answer was being produced: it no
      // longer describes what is on screen.
      setFixedQuestionStatus("La sélection a changé pendant le traitement : la réponse a été ignorée.", "info");
      return;
    }
    renderFixedQuestionAnswer(data, asked, sentVideo, sentWindow);
    setFixedQuestionStatus("", null);
  } catch (err) {
    if (seq !== fixedQuestionRequestSeq) return;
    setFixedQuestionStatus(`Erreur : ${err.message}`, "error");
  } finally {
    if (seq === fixedQuestionRequestSeq) {
      fixedQuestionPending = false;
      renderFixedQuestionControls();
    }
  }
}

document.getElementById("fq-form").addEventListener("submit", (e) => {
  e.preventDefault();
  askFixedQuestion();
});

loadFixedQuestions()
  .catch((err) => {
    setFixedQuestionStatus(`Échec du chargement des questions fixes : ${err.message}`, "error");
  });

loadVideos()
  .catch((err) => {
    document.getElementById("timeline-bar").textContent = `Échec du chargement des patients : ${err.message}`;
  })
  .finally(() => renderChatContext()); // covers the zero-videos/zero-segments edge case too,
  // where selectWindow() (the usual trigger) never runs.
