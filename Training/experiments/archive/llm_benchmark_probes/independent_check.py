"""Independent recomputation of the series quantities, deliberately NOT
importing temporal_context -- reads the raw get_inference_history payload
and recomputes everything with plain loops, then compares."""
import json, sys
sys.path.insert(0, "/path/to/embryonicDevelopmentSciMLExtension/Training")
sys.argv = [sys.argv[0]]
from orchestrator import tools
from orchestrator.temporal_context import derive_series_analysis

H = tools.get_inference_history("Patient_319", 156, 5, 5, "val", top_k_probabilities=4)
A = derive_series_analysis(H)

rows = []
for e in sorted(H["entries"], key=lambda e: e["window_start"]):
    m, o = e["model_derived"], e["observed"]
    pp = sorted(m["phase_probabilities"].values(), reverse=True)
    rows.append({
        "w": e["window_start"], "off": e["offset_from_center"],
        "phase": m["current_phase"], "idx": m["current_phase_index"],
        "p1": pp[0], "p2": pp[1] if len(pp) > 1 else None,
        "ent": m["entropy"], "obs": o["ground_truth_phase"],
    })

fail = []
def check(name, mine, theirs, tol=1e-6):
    ok = (mine == theirs) if not isinstance(mine, float) else abs(mine - theirs) <= tol
    print(f"{'OK ' if ok else 'MISMATCH'}  {name}: independent={mine!r}  module={theirs!r}")
    if not ok:
        fail.append(name)

# gap series
for r in rows:
    check(f"gap@W{r['w']}", round(r["p1"] - r["p2"], 6), A["probability_gap"]["series"][str(r["w"])])
gaps = {r["w"]: round(r["p1"] - r["p2"], 6) for r in rows}
gmin_w = min(gaps, key=lambda w: (gaps[w], w))
check("gap_minimum", gaps[gmin_w], A["probability_gap"]["minimum"])
check("gap_minimum_window", gmin_w, A["probability_gap"]["minimum_window"])
check("gap_at_center", gaps[156], A["probability_gap"]["at_center_window"])

# crossing
cross = None
for a, b in zip(rows, rows[1:]):
    if a["phase"] != b["phase"]:
        cross = b["w"]; break
check("crossing_window", cross, A["convergence"]["crossing_window"])

# model / observed changes
mchanges = [(a["w"], b["w"], a["phase"], b["phase"]) for a, b in zip(rows, rows[1:]) if a["phase"] != b["phase"]]
ochanges = [(a["w"], b["w"], a["obs"], b["obs"]) for a, b in zip(rows, rows[1:]) if a["obs"] != b["obs"]]
check("n_model_changes", len(mchanges), len(A["convergence"]["all_model_phase_changes"]))
check("n_observed_changes", len(ochanges), len(A["timing"]["all_observed_phase_changes"]))
print(f"      model changes    = {mchanges}")
print(f"      observed changes = {ochanges}")

# lag: only if a (from,to) pair matches
pairs_m = {(c[2], c[3]): c[1] for c in mchanges}
pairs_o = {(c[2], c[3]): c[1] for c in ochanges}
common = set(pairs_m) & set(pairs_o)
lag = (pairs_m[next(iter(common))] - pairs_o[next(iter(common))]) if common else None
check("lag", lag, A["timing"]["model_vs_observed_lag_windows"])
print(f"      matching (from,to) pairs = {sorted(common)}")

# stability / returns / skips / regressions
center_phase = next(r["phase"] for r in rows if r["w"] == 156)
after = [r for r in rows if r["off"] > 0]
diff = [r["w"] for r in after if r["phase"] != center_phase]
check("windows_after_center_differing", diff, A["stability"]["windows_after_center_differing_from_center_phase"])
check("is_stable_after_center", (bool(after) and not diff), A["stability"]["is_stable_after_center"])
idx = {r["phase"]: r["idx"] for r in rows}
skips = [(c[0], c[1]) for c in mchanges if idx[c[3]] - idx[c[2]] >= 2]
regs = [(c[0], c[1]) for c in mchanges if idx[c[3]] < idx[c[2]]]
left = set()
rets = []
for c in mchanges:
    if c[3] in left: rets.append((c[0], c[1]))
    left.add(c[2])
check("n_skips", len(skips), A["stability"]["n_phase_skips"])
check("n_regressions", len(regs), A["stability"]["n_phase_regressions"])
check("n_returns", len(rets), A["stability"]["n_phase_returns"])

# entropy
ents = {r["w"]: r["ent"] for r in rows}
emax_w = max(ents, key=lambda w: (ents[w], -w)); emin_w = min(ents, key=lambda w: (ents[w], w))
check("entropy_max", ents[emax_w], A["entropy"]["maximum"])
check("entropy_max_window", emax_w, A["entropy"]["maximum_window"])
check("entropy_min", ents[emin_w], A["entropy"]["minimum"])
check("entropy_at_center", ents[156], A["entropy"]["at_center_window"])
near = [r["ent"] for r in rows if abs(r["off"]) <= 1]
far = [r["ent"] for r in rows if abs(r["off"]) >= 3]
check("entropy_near_mean", round(sum(near)/len(near), 6), A["entropy"]["mean_near_center_offset_le_1"])
check("entropy_far_mean", round(sum(far)/len(far), 6), A["entropy"]["mean_far_from_center_offset_ge_3"])

print("\n--- LAG DETAIL (real data) ---")
print(f"annotation event        : {ochanges}")
print(f"annotation window       : {[c[1] for c in ochanges]}")
print(f"model transition        : {mchanges}")
print(f"model transition window : {[c[1] for c in mchanges]}")
print(f"lag (windows)           : {A['timing']['model_vs_observed_lag_windows']}")
print(f"sign convention         : {A['timing']['lag_definition']}")

print("\n--- EDGE CASES on real data ---")
one = tools.get_inference_history("Patient_319", 156, 0, 1, "val", top_k_probabilities=4)
a1 = derive_series_analysis(one)
print(f"2-window band: crossing={a1['convergence']['crossing_window']} lag={a1['timing']['model_vs_observed_lag_windows']}")
flat = tools.get_inference_history("Patient_319", 159, 1, 1, "val", top_k_probabilities=4)
a2 = derive_series_analysis(flat)
print(f"flat band @159: crossing={a2['convergence']['crossing_window']} lag={a2['timing']['model_vs_observed_lag_windows']}")
print(f"  warnings: {a2['warnings']}")
print(f"empty: {derive_series_analysis({'entries': []})}")

print("\nRESULT:", "ALL CONSISTENT" if not fail else f"{len(fail)} MISMATCH: {fail}")
