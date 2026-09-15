import json

r = json.load(open("Results/evaluation/e1_hmm_phase2_analysis/report.json"))
dist = r["section6_7_8_durations_distributions_censoring"]["distributions_by_phase"]
censoring = r["section6_7_8_durations_distributions_censoring"]["censoring_by_phase"]
phases = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB"]

print("phase, n, mean, median, std, cv, min, max, q10, q25, q75, q90, skewness")
for p in phases:
    e = dist.get(p, {})
    od = e.get("onset_duration_interior")
    if od is None:
        print(p, "NO onset_duration_interior key at all -- raw entry:", json.dumps(e))
        continue
    missing = [k for k in ("n", "mean", "median", "std", "cv_std_over_mean", "min", "max", "q10", "q25", "q75", "q90", "skewness") if od.get(k) is None]
    if missing:
        print(p, "onset_duration_interior present but missing/None fields:", missing, "-- raw:", json.dumps(od))
        continue
    print("{:6s} n={:4d} mean={:.3f} median={:.3f} std={:.3f} cv={:.4f} min={:.2f} max={:.2f} q10={:.2f} q25={:.2f} q75={:.2f} q90={:.2f} skew={:.2f}".format(
        p, od["n"], od["mean"], od["median"], od["std"], od["cv_std_over_mean"],
        od["min"], od["max"], od["q10"], od["q25"], od["q75"], od["q90"], od["skewness"]
    ))

print()
print("=== censoring_by_phase ===")
print(json.dumps(censoring, indent=2))
