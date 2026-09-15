"""Programmatic verification of points A-G of the pre-benchmark validation,
against the REAL rendered context for Q5/Q6/Q8/Q10 on Patient_319/val/156."""
import re, sys
sys.path.insert(0, "/path/to/embryonicDevelopmentSciMLExtension/Training")
sys.argv = [sys.argv[0]]
from orchestrator import prompt as prompt_module
from orchestrator.fixed_question_benchmark import FIXED_QUESTIONS
from orchestrator.orchestrator import answer_question

BAND = list(range(151, 162))
issues = []
for q in FIXED_QUESTIONS:
    if q.question_id not in ("Q5", "Q6", "Q8", "Q10"):
        continue
    ctx = answer_question(q.question, video_id="Patient_319", window=156, split="val")["context"]
    dyn = ctx["dynamic_context"]
    body = prompt_module._format_dynamic_context(dyn)
    lines = body.splitlines()
    print("=" * 70)
    print(f"{q.question_id}: {len(lines)} lines, dynamic keys = {list(dyn)}")

    # A. every value line inside a WINDOW block carries its window prefix
    current, bad = None, []
    for line in lines:
        m = re.match(r"^WINDOW (\d+)", line)
        if m:
            current = m.group(1); continue
        if line.startswith("===") or not line.strip():
            current = None; continue
        if current and line.startswith("    ") and not line.strip().startswith(f"W{current}."):
            bad.append(line)
    print(f"A. value lines missing their own window prefix : {len(bad)}")
    issues += [(q.question_id, "A", b) for b in bad]

    # A2. no window lost / invented
    found = sorted(int(m) for m in re.findall(r"^WINDOW (\d+)", body, re.M))
    print(f"A2. window blocks = {found}  (expected {BAND}) -> {'OK' if found == BAND else 'MISMATCH'}")
    if found != BAND: issues.append((q.question_id, "A2", found))

    # B. model_phase and observed_phase separated, per window
    okB = all(f"W{w}.model_phase = " in body and f"W{w}.observed_phase = " in body for w in BAND)
    labels = body.count("MODEL-DERIVED"), body.count("OBSERVED (annotation"), body.count("DERIVED (calcul deterministe")
    print(f"B. per-window model+observed present: {okB}; block labels (model, observed, derived) = {labels}")
    if not okB or labels != (11, 11, 11): issues.append((q.question_id, "B", labels))

    # C. entropy attached to the right window
    ents = dict(re.findall(r"W(\d+)\.entropy = ([0-9.eE+-]+)", body))
    truth = {str(e["window_start"]): repr(e["model_derived"]["entropy"])
             for e in dyn["get_inference_history"]["entries"]}
    okC = all(ents.get(w) == str(eval(v)) for w, v in truth.items())
    print(f"C. entropy on the correct window for all 11: {okC}")
    if not okC: issues.append((q.question_id, "C", ents))

    # D. distributions inside their window block (not deferred elsewhere)
    okD = all(f"W{w}.phase_probabilities :" in body and f"W{w}.next_phase_distribution :" in body
              for w in BAND)
    # and: are they physically inside that window's own block?
    inside = True
    for w in BAND:
        blk = body.split(f"WINDOW {w}")[1].split("WINDOW ")[0] if f"WINDOW {w}" in body else ""
        if f"W{w}.phase_probabilities :" not in blk: inside = False
    print(f"D. series distributions present: {okD}; inside their own block: {inside}")
    # residual: get_current_inference's own distributions are still deferred
    gci_deferred = [i for i, l in enumerate(lines) if l.startswith("get_current_inference.current_state.phase_probabilities.")]
    gci_header = [i for i, l in enumerate(lines) if l.startswith("get_current_inference.window.window_start ")]
    if gci_deferred and gci_header:
        print(f"D-residual. get_current_inference distributions deferred {gci_deferred[0]-gci_header[0]} lines "
              f"below its own window header (pre-existing single-window behaviour)")
    if not (okD and inside): issues.append((q.question_id, "D", (okD, inside)))

    # E. centre window not duplicated without an explicit note
    has_note = "RECONCILIATION" in body
    both = "get_current_inference" in dyn and "get_inference_history" in dyn
    print(f"E. both records present: {both}; reconciliation note: {has_note}")
    if both and not has_note: issues.append((q.question_id, "E", None))

    # F. DERIVED values labelled as Python-computed
    okF = ("calcul deterministe Python" in body and "derived_deterministic" in body
           and "ni sortie de modele, ni annotation" in body)
    print(f"F. derived values labelled as Python-computed: {okF}")
    if not okF: issues.append((q.question_id, "F", None))

    # G. no invented DERIVED quantity when the data is absent
    sa = dyn.get("series_analysis", {})
    lag_null = sa.get("timing", {}).get("model_vs_observed_lag_windows") is None
    lag_shown_null = "model_vs_observed_lag_windows = null" in body
    explained = "not computable here" in body
    print(f"G. lag is null: {lag_null}; rendered as null: {lag_shown_null}; explained: {explained}")
    if not (lag_null and lag_shown_null and explained): issues.append((q.question_id, "G", None))

print("=" * 70)
print("ISSUES:", issues if issues else "none")
