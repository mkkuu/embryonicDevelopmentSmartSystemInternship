"""Static-source guard for the patient-selection panel
(Training/webapp_api/templates/index.html + static/app.js).

Pure source-text checks -- no Flask, no torch, no browser -- so this runs
inside the normal `pytest Tests/` suite. The behavioural proof (natural
ordering, filtering, identifier preservation, the selected patient
actually reaching the backend) lives in the companion, dependency-free
Node test `node Tests/webapp_api/test_patient_selection.mjs`, alongside
`test_chat_dom_safety.mjs` -- a different ecosystem, not collected by
pytest, same split as the existing DOM-safety pair.

What is pinned here is the contract that a future edit could silently
break without any test noticing: the element ids the JS and the backend
round-trip depend on, and the fact that the ordering is applied in the
frontend rather than by changing the backend's documented cache order.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_HTML = ROOT / "Training" / "webapp_api" / "templates" / "index.html"
APP_JS = ROOT / "Training" / "webapp_api" / "static" / "app.js"
STYLE_CSS = ROOT / "Training" / "webapp_api" / "static" / "style.css"
TRAJECTORY_SERVICE = ROOT / "Training" / "reporting" / "trajectory_service.py"


def _index() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


def _app_js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_patient_panel_elements_exist_in_the_template():
    source = _index()
    for element_id in ("patient-section", "patient-panel", "patient-picker",
                       "patient-filter", "patient-count", "patient-info"):
        assert f'id="{element_id}"' in source, f"missing element id={element_id!r}"


def test_patient_summary_fields_exist_in_the_template():
    source = _index()
    for element_id in ("pi-id", "pi-split", "pi-windows", "pi-range"):
        assert f'id="{element_id}"' in source, f"missing summary field id={element_id!r}"


def test_video_select_id_is_preserved():
    """`video-select` is the id app.js binds its change handler to and the
    one the existing tests reference -- renaming it would silently break
    patient selection."""
    assert 'id="video-select"' in _index()


def test_selection_and_information_are_separate_containers():
    """The picker and the patient's own summary must be two distinct
    elements, not one merged blob -- the whole point of the panel."""
    source = _index()
    picker_at = source.index('id="patient-picker"')
    info_at = source.index('id="patient-info"')
    assert picker_at != info_at
    # the summary fields must live inside the info container, after it
    for element_id in ("pi-id", "pi-windows"):
        assert source.index(f'id="{element_id}"') > info_at


def test_frontend_owns_the_ordering_not_the_backend():
    source = _app_js()
    assert "function compareVideoIds(" in source, "the natural-order comparator must exist"
    assert "sort(compareVideoIds)" in source, "the video list must actually be sorted"
    # Sorting a COPY is what keeps the backend's documented order intact.
    assert "[...data.videos].sort(compareVideoIds)" in source, (
        "the backend's own response array must be sorted as a copy, never in place"
    )


def test_backend_list_videos_order_contract_is_untouched():
    """The UX change must not have "fixed" the ordering in the backend --
    list_videos()'s docstring says other callers depend on cache order."""
    source = TRAJECTORY_SERVICE.read_text(encoding="utf-8")
    assert "in cache order (stable," in source
    assert "not sorted" in source


def test_identifiers_are_used_verbatim_as_option_values():
    source = _app_js()
    assert "opt.value = name;" in source, "option value must be the exact backend identifier"
    assert "opt.textContent = name;" in source, "label and identifier must be the same string"


def test_patient_summary_uses_the_existing_videos_endpoint():
    source = _app_js()
    assert "function loadPatientInfo(" in source
    assert "`/videos/${encodeURIComponent(videoId)}?split=${SPLIT}`" in source, (
        "the summary must come from the existing GET /videos/<id>, not a new endpoint"
    )


def test_filter_and_render_helpers_exist():
    source = _app_js()
    for helper in ("function matchesFilter(", "function renderVideoOptions(", "function selectPatient("):
        assert helper in source, f"expected helper {helper!r} in app.js"


def test_patient_panel_has_styles():
    source = STYLE_CSS.read_text(encoding="utf-8")
    for selector in ("#patient-panel", "#patient-picker", "#patient-info", "#video-select"):
        assert selector in source, f"missing style rule for {selector}"


def test_new_markup_introduces_no_unsafe_dom_sink():
    """Same guarantee as test_frontend_dom_safety_audit.py, restated for the
    files this change touched."""
    for path in (INDEX_HTML, APP_JS):
        source = path.read_text(encoding="utf-8")
        for pattern in ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write"):
            assert pattern not in source, f"{path.name} contains unsafe DOM sink {pattern!r}"


# --- ISEN logo (presentation only) -------------------------------------

LOGO_FILENAME = "logo-LabISEN_2024.png"
STATIC_DIR = ROOT / "Training" / "webapp_api" / "static"
SOURCE_ASSET = ROOT / "Ressources" / LOGO_FILENAME


def test_logo_asset_is_present_in_the_served_static_directory():
    """Flask serves this app's own `static/` dir; an asset referenced from
    the template but absent there would render as a broken image."""
    assert (STATIC_DIR / LOGO_FILENAME).is_file(), (
        f"expected {LOGO_FILENAME} in {STATIC_DIR}"
    )


def test_logo_is_a_real_png_not_a_placeholder():
    header = (STATIC_DIR / LOGO_FILENAME).read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n", "static logo is not a PNG"


def test_logo_is_the_project_asset_byte_for_byte():
    """The logo must come from the in-project `Ressources/` asset, copied
    verbatim -- never re-encoded, resized or fetched from anywhere else."""
    assert SOURCE_ASSET.is_file(), f"source asset {SOURCE_ASSET} is missing"
    assert (STATIC_DIR / LOGO_FILENAME).read_bytes() == SOURCE_ASSET.read_bytes(), (
        "the served logo differs from Ressources/ -- it must be a verbatim copy"
    )


def test_template_references_the_logo_through_url_for_static():
    source = _index()
    assert f"filename='{LOGO_FILENAME}'" in source, (
        "the logo must be referenced via url_for('static', ...), not a hardcoded path"
    )
    assert "http://" not in source and "https://" not in source, (
        "the frontend must not load the logo (or anything else) from an external URL"
    )


def test_logo_has_alt_text_and_intrinsic_dimensions():
    source = _index()
    assert 'id="isen-logo"' in source
    assert 'alt="LabISEN"' in source, "the logo needs alt text for accessibility"
    # Intrinsic width/height reserve the box before load -> no layout shift.
    # They must be the asset's real pixel size, or the reserved box is wrong.
    assert 'width="285"' in source and 'height="177"' in source


def test_logo_is_sized_down_by_css_and_keeps_its_ratio():
    source = STYLE_CSS.read_text(encoding="utf-8")
    assert "#isen-logo" in source, "the logo must be constrained by CSS"
    logo_rule = source.split("#isen-logo")[1].split("}")[0]
    assert "height:" in logo_rule, "a header logo must be height-constrained"
    assert "width: auto" in logo_rule, "width must stay auto so the aspect ratio is preserved"


def test_logo_does_not_displace_the_existing_header_controls():
    """The split tag must still be in the header, after the brand block --
    the logo is additive, it must not push existing controls out."""
    source = _index()
    brand_at = source.index('class="brand"')
    controls_at = source.index('class="controls"')
    split_at = source.index("split : val")
    assert brand_at < controls_at < split_at
    assert source.index("<h1>") > brand_at, "the title must sit inside the brand block"


# --- Viewer: phase / frame navigation (presentation only) ---------------

def test_viewer_controls_exist_in_the_template():
    source = _index()
    for element_id in ("viewer-section", "phase-controls", "phase-select", "btn-phase-first",
                       "btn-phase-last", "frame-controls", "btn-prev-window", "btn-next-window",
                       "btn-which-first", "btn-which-last", "btn-back-anchor",
                       "frame-position", "cw-frame-range"):
        assert f'id="{element_id}"' in source, f"missing viewer element id={element_id!r}"


def test_timeline_is_kept_and_lives_inside_the_viewer():
    """Requirement: integrate the existing timeline into the viewer rather
    than building a second, independent navigation logic."""
    source = _index()
    assert 'id="timeline-bar"' in source and 'id="timeline-legend"' in source
    viewer_at = source.index('id="viewer-section"')
    assert source.index('id="timeline-bar"') > viewer_at
    # ...and there is exactly one timeline, not a second copy.
    assert source.count('id="timeline-bar"') == 1


def test_phase_picker_is_built_from_the_existing_timeline_segments():
    source = _app_js()
    assert "timelineSegments = data.segments" in source, (
        "the phase picker must reuse the /timeline response, not re-segment anything"
    )
    assert "function renderPhaseOptions(" in source
    assert "function segmentIndexForWindow(" in source


def test_viewer_navigation_helpers_exist():
    source = _app_js()
    for helper in ("function stepWindow(", "function jumpToPhaseEdge(",
                   "function backToAnchor(", "function renderViewerControls("):
        assert helper in source, f"expected viewer helper {helper!r} in app.js"


def _app_js_code_only() -> str:
    """app.js with whole-line `//` comments stripped, so a rule about what the
    CODE does is never satisfied or broken by prose in a comment."""
    return "\n".join(line for line in _app_js().splitlines()
                      if not line.lstrip().startswith("//"))


def test_viewer_never_builds_a_frame_url_client_side():
    """The viewer navigates with the existing endpoints, and every frame URL
    comes from the backend's own `frame` block (frame_url / first_frame_url /
    last_frame_url) -- the client never assembles one."""
    code = _app_js_code_only()
    assert "/frame" not in code, (
        "a frame URL is being constructed in app.js; it must come from the backend"
    )
    for provided in ("frame.frame_url", "frame.first_frame_url", "frame.last_frame_url"):
        assert provided in code, f"expected the backend-provided {provided}"


def test_viewer_requests_only_the_existing_endpoint_shapes():
    code = _app_js_code_only()
    fetched = sorted(set(re.findall(r"fetchJSON\(`([^`]+)`", code)))
    assert fetched == [
        # V1 fixed questions: the frozen Q1-Q15 list, read-only (see
        # Tests/webapp_api/test_fixed_questions_api.py). The viewer itself
        # still uses nothing but the four original endpoints.
        "/questions",
        "/videos/${encodeURIComponent(videoId)}/timeline?split=${SPLIT}",
        "/videos/${encodeURIComponent(videoId)}/window/${windowStart}?split=${SPLIT}",
        "/videos/${encodeURIComponent(videoId)}?split=${SPLIT}",
        "/videos?split=${SPLIT}",
    ], fetched


def test_viewer_never_computes_a_frame_number():
    """Raw frame numbers are NOT contiguous across consecutive windows
    (verified on real Val data), so any arithmetic on them would be wrong."""
    source = _app_js()
    for forbidden in ("frame_number + 1", "frame_number - 1",
                      "frameNumber + 1", "frameNumber - 1"):
        assert forbidden not in source, f"client-side frame arithmetic found: {forbidden!r}"


def test_stepping_is_bounded_by_the_videos_real_window_list():
    source = _app_js()
    assert "videoWindows.first" in source and "videoWindows.last" in source, (
        "step bounds must come from GET /videos/<id>, not be assumed"
    )
    assert "info.first_window" in source and "info.last_window" in source


def test_keyboard_shortcuts_do_not_steal_keys_from_text_inputs():
    source = _app_js()
    assert "function isTypingTarget(" in source
    assert '"ArrowLeft"' in source and '"ArrowRight"' in source and '"Home"' in source


def test_viewer_explanatory_banner_is_gone():
    """The banner above the viewer was removed on request (UI-only change).
    Neither its text nor its container may come back."""
    source = _index()
    for fragment in ("Browse the recorded frames", "model's own prediction",
                     "not the dataset annotation", "viewer-note"):
        assert fragment not in source, f"the removed viewer banner is back: {fragment!r}"
    assert "viewer-note" not in STYLE_CSS.read_text(encoding="utf-8"), (
        "the banner's stylesheet rule outlived the banner"
    )


def test_time_unit_is_declared_once_with_its_provenance():
    """The unit was CONFIRMED BY THE PROJECT OWNER (2026-09-02) as hours.
    Nothing upstream carries it (the raw CSV header is `frame_index,time`;
    the manifest records "unknown/unverified"), so the frontend states it --
    but exactly once, in a single named constant carrying its provenance,
    never sprinkled as a bare literal through the rendering code."""
    source = _app_js()
    assert "const TIME_UNIT = {" in source, "the unit must live in one named constant"
    assert 'label: "heures"' in source
    assert "confirmée par le responsable du projet" in source.lower(), (
        "the constant must record WHERE the unit comes from"
    )
    # Exactly one declaration, and the renderer uses the constant.
    assert source.count("const TIME_UNIT") == 1
    assert "TIME_UNIT.symbol" in source


def test_the_api_time_unit_flag_is_still_surfaced_for_traceability():
    """Displaying the confirmed unit must not hide the fact that the API
    itself still reports "unknown/unverified" for the record."""
    source = _app_js()
    assert "data.window.time_unit" in source, (
        "the API's own time_unit must still reach the UI (tooltip), not be dropped"
    )


def test_no_time_value_is_converted_in_the_frontend():
    """Only display formatting is allowed -- never arithmetic on a timestamp."""
    code = _app_js_code_only()
    for forbidden in ("window_start_time *", "window_end_time *", "window_start_time /",
                      "window_end_time /", "* 60", "/ 60", "* 3600", "/ 3600"):
        assert forbidden not in code, f"time conversion found in the frontend: {forbidden!r}"


def test_the_backend_time_unit_contract_was_not_rewritten():
    """The UI change must NOT have silently edited the pipeline's own honesty
    flag -- that is a data-lineage decision, not a presentation one."""
    schemas = (ROOT / "Training" / "reporting" / "schemas.py").read_text(encoding="utf-8")
    trajectory = TRAJECTORY_SERVICE.read_text(encoding="utf-8")
    assert '"unknown/unverified"' in trajectory, (
        "trajectory_service must still report the API's own unverified flag"
    )
    assert "never converted to" in schemas


# --- Backend: the frame block addition is additive only -----------------

APP_PY = ROOT / "Training" / "webapp_api" / "app.py"


def test_frame_block_keeps_its_documented_fields():
    """docs/WEBAPP_VIEWER.md documents `available`/`frame_number`/`frame_url`
    -- the viewer additions must not remove or rename them."""
    source = APP_PY.read_text(encoding="utf-8")
    for field in ('"available"', '"frame_number"', '"frame_url"'):
        assert field in source, f"documented frame field {field} disappeared"


def test_frame_block_exposes_the_windows_own_frame_bounds():
    source = APP_PY.read_text(encoding="utf-8")
    for field in ('"first_frame_number"', '"last_frame_number"', '"n_frames"',
                  '"first_frame_url"', '"last_frame_url"'):
        assert field in source, f"missing viewer frame field {field}"


def test_no_new_route_was_added_for_the_viewer():
    """The viewer must reuse the existing endpoints; the route set is frozen.
    The one addition since is `/questions` (V1 fixed questions, a read-only
    feed of the frozen benchmark list -- not a viewer route)."""
    source = APP_PY.read_text(encoding="utf-8")
    routes = sorted(set(re.findall(r'@app\.route\("([^"]+)"', source)))
    assert routes == [
        "/",
        "/chat",
        "/health",
        "/questions",
        "/videos",
        "/videos/<video_id>",
        "/videos/<video_id>/timeline",
        "/videos/<video_id>/window/<int:window_start>",
        "/videos/<video_id>/window/<int:window_start>/frame",
    ], routes


# --- Models badge: REMOVED on request (UI only) -------------------------
#
# The badge and its `_models_in_use()` backend helper were deleted. The
# scientific models themselves are untouched: the Semi-HMM still produces
# every phase/probability/entropy the app displays, and the chat LLM is
# still configured exactly as before. Only the explanatory UI element is
# gone. What is pinned below is that it stays gone, and that removing it
# did not drag any model reference into the templates.

def test_models_badge_is_gone_from_the_template():
    source = _index()
    for fragment in ('id="model-info"', "model-info-body", "mi-science", "mi-llm",
                     "Scientific model", "does <strong>not</strong> produce"):
        assert fragment not in source, f"the removed models badge is back: {fragment!r}"


def test_models_badge_template_values_are_gone():
    source = _index()
    for placeholder in ("{{ llm_model }}", "{{ llm_num_ctx }}",
                        "{{ science_model_name }}", "{{ science_model_version }}"):
        assert placeholder not in source, f"leftover badge template value {placeholder}"


def test_models_badge_backend_helper_is_gone_and_index_renders_plainly():
    source = APP_PY.read_text(encoding="utf-8")
    assert "def _models_in_use(" not in source, "the badge's backend helper survived"
    assert "_models_in_use()" not in source, "the badge helper is still called"
    assert 'return render_template("index.html")' in source, (
        "the index view must render the template with no badge context"
    )


def test_models_badge_styles_are_gone():
    css = STYLE_CSS.read_text(encoding="utf-8")
    for fragment in (".model-info", ".mi-tag", ".mi-science", ".mi-llm", ".mi-warn"):
        assert fragment not in css, f"dead badge style left behind: {fragment}"


def test_removing_the_badge_did_not_break_the_chat_provider():
    """The badge read the live provider; deleting it must not delete the
    provider itself, which actually answers /chat."""
    source = APP_PY.read_text(encoding="utf-8")
    # Built through the shared resolver (orchestrator/llm_config.py, `LLM_MODEL`)
    # rather than a model name written here -- see Tests/orchestrator/test_llm_config.py.
    assert "_CHAT_PROVIDER = llm_config.build_ollama_provider()" in source
    assert "llm=_CHAT_PROVIDER" in source


def test_no_model_identifier_is_hardcoded_in_the_template_or_app_js():
    """Kept from the badge era, and still worth pinning: neither the template
    nor the frontend may name a model."""
    for source in (_index(), _app_js_code_only()):
        for hardcoded in (r"llama\d", r"mistral[-:]", r"nemo", r"gpt-", r":latest",
                          r":\d+b", r"dmax\s*=", r"semi_hmm"):
            assert not re.search(hardcoded, source, re.IGNORECASE), (
                f"a model identifier matching {hardcoded!r} is hardcoded in the UI"
            )


def test_serving_code_still_uses_neither_gru_nor_identity_dynamics():
    """Independent of the badge that used to state it: the app must keep
    serving only the Semi-HMM it loads, never the research ladder's models."""
    for path in (ROOT / "Training" / "webapp_api" / "app.py",
                 ROOT / "Training" / "reporting" / "inference_service.py"):
        code = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                          if not line.lstrip().startswith("#"))
        for forbidden in (r"evaluation\.models\.gru", r"\bGRUDynamicsModel\b",
                          r"evaluation\.models\.identity_dynamics", r"\bIdentityDynamics\w*\b"):
            assert not re.search(forbidden, code), f"{path.name} now uses {forbidden!r}"


# --- Duration warning: hidden in the VIEWER only ------------------------
#
# `reporting/inference_service.py` still computes and returns it, and the
# Reporting API still serves it. Only the viewer's rendering drops it.

def test_viewer_filters_the_duration_warning_out_of_its_display():
    code = _app_js()
    assert "HIDDEN_VIEWER_WARNING_PREFIXES" in code
    assert "duration not estimable for phase" in code, (
        "the filter must name the warning it hides, so the reason stays readable"
    )
    assert "isHiddenViewerWarning" in code
    assert "filter((w) => !isHiddenViewerWarning(w))" in code, (
        "the filter must actually be applied to the viewer's warning list"
    )


def test_the_duration_warning_is_still_computed_by_the_backend():
    """The suppression is presentational. The scientific computation, its
    wording and the API contract must be untouched."""
    service = (ROOT / "Training" / "reporting" / "inference_service.py").read_text(encoding="utf-8")
    assert "duration not estimable for phase" in service
    assert "structural censoring" in service
    schemas = (ROOT / "Training" / "reporting" / "schemas.py").read_text(encoding="utf-8")
    assert "warnings: List[str]" in schemas, "the API's warnings field must still exist"


def test_only_the_viewer_hides_warnings_not_the_answer_panel():
    """The answer panel's own warnings (grounding) must keep reaching the
    user. Since the free-question panel was removed, the fixed-question
    answer block is where they are rendered -- unfiltered, appended to the
    technical lines inside "Détails techniques"."""
    code = _app_js()
    assert "renderWarningsList(warningsEl, technical);" in code, (
        "the answer panel must still render its warnings unfiltered"
    )
    assert ".concat(Array.from(data.warnings || []));" in code, (
        "every backend warning must reach that list, none filtered out"
    )


# --- French UI ----------------------------------------------------------

def test_page_declares_french():
    assert '<html lang="fr">' in _index()


def test_visible_ui_strings_are_in_french():
    source = _index()
    for expected in ("Filtrer", "Identifiant", "Fenêtres", "Plage de fenêtres",
                     "Visualiseur", "Bande de phase", "Première image", "Dernière image",
                     "Retour à la sélection", "Poser la question",
                     "Probabilité", "Entropie", "Temps", "Phase (modèle)"):
        assert expected in source, f"missing French UI string: {expected!r}"


def test_no_leftover_english_control_labels():
    source = _index()
    for english in (">Filter<", ">Identifier<", ">Windows<", ">Window range<",
                    ">Viewer<", ">Phase band<", ">First frame<", ">Last frame<",
                    ">Send<", "Ask a question", "Back to selection",
                    "Loading context", "no frame available"):
        assert english not in source, f"untranslated UI string left: {english!r}"


def test_frontend_user_facing_strings_are_in_french():
    code = _app_js_code_only()
    # "Contexte actuel"/"Envoi…" belonged to the removed free-question panel;
    # their fixed-question counterparts are checked in their place.
    for expected in ("La question sera posée sur", "indisponible", "Échec du chargement",
                     "par rapport à la sélection", "en cours de traitement…", "Erreur",
                     "fenêtre", "images"):
        assert expected in code, f"missing French frontend string: {expected!r}"
    for english in ('"Current context: "', '"unavailable"', '"Sending…"',
                    '"Send"', "from selection", "of window)", "patients match"):
        assert english not in code, f"untranslated frontend string left: {english!r}"


def test_phase_labels_and_scientific_names_were_not_translated():
    """Phase ids and proper model names are nomenclature, not UI copy."""
    code = _app_js_code_only()
    # The "Semi-HMM" mention lived in the free-question panel's empty-context
    # hint, removed with that panel; no model name is written in the frontend
    # any more. The phase-id guarantee below is unaffected.
    # No phase id may have been rewritten into a French form anywhere.
    for source in (_index(), code):
        assert not re.search(r"\bp[0-9]+\b", source), "a phase id looks rewritten"


def test_time_unit_symbol_and_values_are_untouched():
    """Translation must not have touched the unit or introduced a conversion."""
    code = _app_js()
    assert 'symbol: "h"' in code
    assert 'label: "heures"' in code
    assert "toFixed(2)" in code
    for converter in ("* 60", "/ 60", "* 3600", "/ 3600"):
        assert converter not in code, f"a time conversion appeared: {converter!r}"
