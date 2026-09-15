import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from embeddings.dataset import EmbeddingDataset
from evaluation.models.hmm import DEFAULT_PHASE_NAMES
from evaluation.models.semi_hmm import CensoringType, extract_phase_durations
from evaluation.trajectory import group_into_trajectories

train_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "train"))
print("n_train_trajectories:", len(train_traj))

durations_by_phase = extract_phase_durations(train_traj)
report = {}
for i, name in enumerate(DEFAULT_PHASE_NAMES):
    durs = durations_by_phase.get(i, [])
    n_total = len(durs)
    n_observed = sum(1 for _, c in durs if c == CensoringType.OBSERVED)
    n_left = sum(1 for _, c in durs if c == CensoringType.LEFT_CENSORED)
    n_right = sum(1 for _, c in durs if c == CensoringType.RIGHT_CENSORED)
    n_both = sum(1 for _, c in durs if c == CensoringType.BOTH_CENSORED)
    report[name] = {"n_total": n_total, "n_observed": n_observed, "n_left_censored": n_left,
                     "n_right_censored": n_right, "n_both_censored": n_both}
    print(f"{name:6s} n_total={n_total:4d} n_observed={n_observed:4d} n_left={n_left:4d} "
          f"n_right={n_right:4d} n_both={n_both:4d}")

out_dir = Path("../Results/evaluation/semi_hmm_phase_a2_censoring")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "censoring_report.json").write_text(json.dumps(report, indent=2))
print("done")
