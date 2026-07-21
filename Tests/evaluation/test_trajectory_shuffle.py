"""
Unit tests for evaluation.trajectory.shuffle_trajectory — the mechanism
behind the frame-shuffle sanity experiment.
"""

import numpy as np
import torch

from evaluation.trajectory import Trajectory, shuffle_trajectory


def make_trajectory(video_name, embeddings, consistency_flags):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    flags_t = torch.as_tensor(consistency_flags, dtype=torch.long)
    T = embeddings_t.shape[0]
    return Trajectory(
        video_name=video_name,
        window_starts=list(range(T)),
        embeddings=embeddings_t,
        consistency_flag=flags_t,
        first_frame_phase=torch.arange(T, dtype=torch.long),
        last_frame_phase=torch.arange(T, dtype=torch.long) + 100,
    )


def test_window_starts_are_unchanged():
    traj = make_trajectory("v1", embeddings=[[i] for i in range(10)], consistency_flags=list(range(10)))
    shuffled = shuffle_trajectory(traj, np.random.RandomState(0))
    assert shuffled.window_starts == traj.window_starts


def test_video_name_is_unchanged():
    traj = make_trajectory("v1", embeddings=[[0.0], [1.0]], consistency_flags=[0, 1])
    shuffled = shuffle_trajectory(traj, np.random.RandomState(0))
    assert shuffled.video_name == traj.video_name


def test_content_is_preserved_as_a_set_but_reordered():
    traj = make_trajectory("v1", embeddings=[[float(i)] for i in range(10)], consistency_flags=list(range(10)))
    shuffled = shuffle_trajectory(traj, np.random.RandomState(0))

    # Same multiset of embedding values, same multiset of flags — nothing invented or dropped.
    original_values = sorted(traj.embeddings.flatten().tolist())
    shuffled_values = sorted(shuffled.embeddings.flatten().tolist())
    assert original_values == shuffled_values
    assert sorted(traj.consistency_flag.tolist()) == sorted(shuffled.consistency_flag.tolist())

    # And genuinely reordered relative to the original, not a no-op —
    # checked probabilistically (a length-10 identity permutation from a
    # seeded RNG is astronomically unlikely, not just "unlikely").
    assert not torch.equal(shuffled.embeddings, traj.embeddings)


def test_embedding_flag_and_phase_stay_grouped_within_a_window():
    # The permutation must move (embedding, flag, phase, phase) together
    # as one unit per original window — not shuffle each field
    # independently, which would silently fabricate windows that never
    # existed.
    traj = make_trajectory("v1", embeddings=[[float(i)] for i in range(10)], consistency_flags=list(range(10)))
    shuffled = shuffle_trajectory(traj, np.random.RandomState(0))

    for i in range(len(shuffled)):
        emb_value = int(shuffled.embeddings[i].item())
        # By construction in make_trajectory: consistency_flag == embedding
        # value, first_frame_phase == embedding value, last_frame_phase ==
        # embedding value + 100 — so all four must still agree per row
        # after shuffling if grouping was preserved.
        assert shuffled.consistency_flag[i].item() == emb_value
        assert shuffled.first_frame_phase[i].item() == emb_value
        assert shuffled.last_frame_phase[i].item() == emb_value + 100


def test_different_rng_states_give_different_orderings():
    traj = make_trajectory("v1", embeddings=[[float(i)] for i in range(20)], consistency_flags=list(range(20)))
    shuffled_a = shuffle_trajectory(traj, np.random.RandomState(1))
    shuffled_b = shuffle_trajectory(traj, np.random.RandomState(2))
    assert not torch.equal(shuffled_a.embeddings, shuffled_b.embeddings)


def test_same_rng_seed_gives_reproducible_ordering():
    traj = make_trajectory("v1", embeddings=[[float(i)] for i in range(10)], consistency_flags=list(range(10)))
    shuffled_a = shuffle_trajectory(traj, np.random.RandomState(7))
    shuffled_b = shuffle_trajectory(traj, np.random.RandomState(7))
    torch.testing.assert_close(shuffled_a.embeddings, shuffled_b.embeddings)
