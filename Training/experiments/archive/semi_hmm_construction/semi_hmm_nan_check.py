import sys
import warnings

sys.path.insert(0, ".")
import numpy as np
from pathlib import Path

from embeddings.dataset import EmbeddingDataset
from evaluation.models.semi_hmm import SemiHMMModel, extract_phase_durations, determine_dmax
from evaluation.trajectory import group_into_trajectories

cache_root = Path("../Embeddings/resnet18")
train_traj = group_into_trajectories(EmbeddingDataset(cache_root, "train"))[:8]
val_traj = group_into_trajectories(EmbeddingDataset(cache_root, "val"))[:4]

durations_by_phase = extract_phase_durations(train_traj)
dmax = min(determine_dmax(durations_by_phase, strategy="observed_max"), 40)

model = SemiHMMModel(n_states=15, dmax=dmax, duration_family="negative_binomial")
model.fit(train_traj)

# Check log_b (emission) itself for -inf / nan on Val trajectories.
for traj in val_traj:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            log_b = model._log_emission(traj)
        except RuntimeWarning as w:
            print(f"{traj.video_name}: RuntimeWarning during _log_emission: {w}")
            log_b = model._log_emission(traj)  # recompute without raising, to inspect
    n_neg_inf = int(np.sum(~np.isfinite(log_b) & (log_b < 0)))
    n_nan = int(np.sum(np.isnan(log_b)))
    print(f"{traj.video_name}: log_b shape={log_b.shape} n_-inf_or_veryneg={n_neg_inf} n_nan={n_nan} "
          f"min={np.nanmin(log_b) if np.any(np.isfinite(log_b)) else 'all non-finite'}")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            log_alpha_end, log_age_posterior = model._explicit_duration_forward(traj)
            filt = model.filtering(traj)
            nan_in_filt = int(np.sum(np.isnan(filt)))
            print(f"  filtering(): nan_count={nan_in_filt}, all_finite={np.all(np.isfinite(filt))}, "
                  f"row_sums_range=({filt.sum(axis=1).min():.6f},{filt.sum(axis=1).max():.6f})")
        except RuntimeWarning as w:
            print(f"  RuntimeWarning during forward: {w}")
            log_alpha_end, log_age_posterior = model._explicit_duration_forward(traj)
            filt = model.filtering(traj)
            nan_in_filt = int(np.sum(np.isnan(filt)))
            print(f"  (after warning) filtering(): nan_count={nan_in_filt}, all_finite={np.all(np.isfinite(filt))}")
