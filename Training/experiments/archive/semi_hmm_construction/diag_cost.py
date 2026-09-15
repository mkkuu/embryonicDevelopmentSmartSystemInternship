import sys, time
sys.path.insert(0, ".")
from pathlib import Path
from embeddings.dataset import EmbeddingDataset
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

t0 = time.time()
model = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
t_model = time.time() - t0
print(f"load model: {t_model:.4f}s", flush=True)

t0 = time.time()
val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
t_traj_all = time.time() - t0
print(f"load ALL val trajectories (106 videos, one-time embeddings cache load): {t_traj_all:.4f}s", flush=True)

traj = val_traj[0]
print(f"video={traj.video_name} n_windows={len(traj)}", flush=True)

t0 = time.time()
posterior = model.filtering(traj)
t_filter = time.time() - t0
print(f"filtering: {t_filter:.4f}s", flush=True)

t0 = time.time()
dur_info = {"expected": model.expected_duration(0), "quantiles": model.duration_models[0].quantiles([0.5])}
t_dur = time.time() - t0
print(f"duration (single phase lookup): {t_dur:.6f}s", flush=True)

print("about to run next_phase_distribution (the slow one)...", flush=True)
