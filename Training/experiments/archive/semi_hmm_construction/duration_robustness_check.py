from scipy import stats
data = {
'tPNa': (0.1094, 0.1797, 17.360, 6711),
'tPNf': (0.5718, 0.4084, 2.680, 991),
't2':   (0.1184, 0.2550, 11.729, 4605),
't3':   (0.3642, 1.1524, 3.657, 534),
't4':   (0.1193, 0.3936, 11.328, 4219),
't5':   (0.2639, 1.0027, 4.514, 1050),
't6':   (0.2416, 1.1823, 4.953, 856),
't7':   (0.1740, 1.2485, 5.839, 1509),
't8':   (0.0903, 0.5296, 13.491, 5254),
't9+':  (0.0819, 0.4505, 22.251, 7429),
'tM':   (0.1962, 0.6199, 7.690, 2579),
'tSB':  (0.1950, 0.5234, 8.523, 2841),
'tB':   (0.2032, 1.0264, 3.975, 1557),
}
names = list(data.keys())
brier = [data[n][0] for n in names]
cv = [data[n][1] for n in names]
dur = [data[n][2] for n in names]

def rep(drop, label, xvals, x_label):
    idx = [i for i, n in enumerate(names) if n not in drop]
    b = [brier[i] for i in idx]
    d = [xvals[i] for i in idx]
    r, p = stats.pearsonr(d, b)
    rho, ps = stats.spearmanr(d, b)
    print(f"{label} ({x_label}) n={len(idx)} Pearson r={r:.4f} p={p:.4f}  Spearman rho={rho:.4f} p={ps:.4f}")

print("=== Brier vs mean_duration robustness ===")
rep(set(), "A all", dur, "mean_duration")
rep({"tPNf"}, "B without tPNf", dur, "mean_duration")
rep({"t3"}, "C without t3", dur, "mean_duration")
rep({"tPNf", "t3"}, "D without both", dur, "mean_duration")

print()
print("=== n_val vs mean_duration (confound check) ===")
n_val = [data[n][3] for n in names]
r, p = stats.pearsonr(dur, n_val)
rho, ps = stats.spearmanr(dur, n_val)
print(f"n={len(names)} Pearson r={r:.4f} p={p:.4f}  Spearman rho={rho:.4f} p={ps:.4f}")
