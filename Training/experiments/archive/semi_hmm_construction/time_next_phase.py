import sys, time
sys.path.insert(0, ".")
from pathlib import Path
from embeddings.dataset import EmbeddingDataset
from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import group_into_trajectories

val_traj = group_into_trajectories(EmbeddingDataset(Path("../Embeddings/resnet18"), "val"))
model = SemiHMMModel.load(Path("../Results/evaluation/semi_hmm_weekend_phaseF/model"))
traj = val_traj[0]
print("video:", traj.video_name, "n_windows:", len(traj), flush=True)
t0 = time.time()
_ = model.next_phase_distribution(traj)
print(f"next_phase_distribution time: {time.time()-t0:.2f}s for {len(traj)} windows", flush=True)
