"""
CLI entry point for HMMModel (Training/evaluation/models/hmm.py) — Train/Val
only, structurally incapable of touching the Test split (no --final_eval
flag exists here at all, unlike run_gru_evaluation.py's guarded one; the
HMM/Semi-HMM branch's Phase 4 explicitly forbids any Test-touching script at
this stage, so the capability is simply absent rather than merely
discouraged).

Mirrors run_gru_evaluation.py's loading pattern (EmbeddingDataset +
evaluation.trajectory.group_into_trajectories) but is otherwise much
simpler: HMMModel.fit() is a single closed-form pass (counting + one
LogisticRegression fit, NO random_state/randomness anywhere in hmm.py), not
a multi-epoch training loop, so there is no --epochs/--patience/--resume/
--device here — none of that machinery applies to this model.

SELECTION PROTOCOL — k=7-aligned, revised 2026-08-21
------------------------------------------------------
The original version of this script (still reproducible from
Results/evaluation/e1_hmm_step1_train_val_sweep/, never deleted or
overwritten) selected alpha/rho by max val_auroc computed from
HMMModel.predict() — a k=1 (lag-1, inter-window) event. That was later
diagnosed (docs/PROJECT_STATE.md, "consistency_flag_prob calibration
DIAGNOSED" checkpoint) as the wrong event: `consistency_flag`
(Training/DataSet.py:310) compares a window's OWN first and last frame, an
intra-window, k=(window_size-1)=7 event, not an inter-window, k=1 one.
`HMMModel.predict_k_step_transition_probability(traj, k=7)` was built and
validated (128/128 tests, brute-force cross-checked) to compute the
correctly-aligned event; a subsequent diagnostic run
(run_hmm_k_step_alignment_check.py, Results/evaluation/
e1_hmm_step2_k_step_alignment_check/) confirmed on the frozen alpha=0.5/
rho=0.2 model that fixing the alignment improves AUROC (0.6096->0.6373) but
does NOT fix Brier calibration (0.1270->0.1410, both worse than the 0.1148
constant-rate baseline) — a second, distinct, still-open problem.

A dedicated methodology session (docs/PROJECT_STATE.md's "HMM
HYPERPARAMETER SELECTION PROTOCOL" checkpoint) analyzed AUROC/Brier/ECE/
log-likelihood/accuracy as candidate selection criteria against this
project's actual objective (a quantitatively reliable transition
probability for future reporting/RAG/web use, not just a good ranking) and
concluded:
  - Accuracy/precision/recall/f1 carry NO selection signal at all here —
    demonstrated, not assumed: on the original 9-config k=1 sweep, all four
    were bit-identical across every single grid point (threshold=0.5 is
    essentially never crossed given the ~13% base rate).
  - log-likelihood optimizes generative fit to ALL observations/
    transitions, not calibration of this one derived event — and on the
    SAME 9-config grid, its ranking (best: alpha=2.0/rho=0.5) is the exact
    OPPOSITE of the AUROC ranking (best: alpha=0.5/rho=0.2) — a real,
    measured disagreement on this project's own data, not a theoretical
    concern.
  - AUROC alone is insufficient (proven by the alignment-check finding
    above: AUROC improved while Brier got worse) but still useful as a
    non-degeneracy floor (a model near the constant-rate baseline gets a
    good Brier "for free" without discriminating at all).
  - Brier is the metric directly aligned with "this probability should mean
    what it says" — the project's actual objective — but needs the AUROC
    floor to avoid rewarding a near-constant, non-discriminating model.

SELECTED PROTOCOL (frozen; do not change silently — see the methodology
checkpoint in docs/PROJECT_STATE.md for the full justification):
  1. eligible := val_auroc_k7 >= MIN_K7_AUROC.
  2. Among eligible configs: minimum val_brier_k7 wins (compared at
     BRIER_TIE_DECIMALS decimals — "egalite pratique du Brier a la 4e
     decimale" per the protocol).
  3. Tie -> highest val_auroc_k7 (unrounded).
  4. Tie -> highest val_phase_decoding_accuracy.
  5. Tie -> first in grid order (an undocumented protocol gap, not a
     scientific decision — see select_best_config()'s docstring).
See select_best_config() below: a PURE function (no I/O, no model fitting),
deliberately factored out so this exact decision rule is unit-testable on
hand-built metric dicts (Tests/evaluation/models/test_hmm.py), never only
implicitly exercised by a real, slow sweep.

Usage (smoke test with a tiny synthetic-scale subset)
--------------------------------------------------------
    cd Training
    python -m experiments.hmm_semi_hmm.run_hmm_evaluation \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --alpha_grid 0.5 1.0 2.0 --rho_grid 0.2 0.3 0.5 \\
        --max_trajectories 5 \\
        --output_dir ../Results/evaluation/e1_hmm_smoke_selection

Not executed as part of writing this file — running it against the real
cache (default: full Train/Val, --output_dir
../Results/evaluation/e1_hmm_k7_sweep) is a separate, explicitly-authorized
next step, not something this script does on import or by accident.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from embeddings.dataset import EmbeddingDataset

from evaluation.calibration import flatten_predictions, score_probabilistic
from evaluation.metrics import compute_metrics
from evaluation.models.hmm import HMMModel
from evaluation.trajectory import Trajectory, group_into_trajectories

# The Val AUROC_k7 already measured for alpha=0.5/rho=0.2 in the k-step
# alignment-check run (Results/evaluation/e1_hmm_step2_k_step_alignment_check/
# k_step_alignment_report.json, "k_vs_consistency_flag_all_windows"."auroc",
# 2026-08-21) — used as a non-degeneracy floor: a real, already-observed
# number, not an arbitrary round threshold invented for this script. See
# this module's own docstring and docs/PROJECT_STATE.md's "HMM
# HYPERPARAMETER SELECTION PROTOCOL" checkpoint for the full justification.
# DO NOT change this silently — it is a frozen, documented protocol
# decision, not a tunable default.
MIN_K7_AUROC = 0.6373

# "egalite pratique du Brier a la 4e decimale" per the same protocol.
BRIER_TIE_DECIMALS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--n_states", type=int, default=15)
    p.add_argument("--alpha_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--rho_grid", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    p.add_argument("--logreg_C", type=float, default=1.0)
    p.add_argument(
        "--k7",
        type=int,
        default=7,
        help="k for predict_k_step_transition_probability's alignment with "
        "consistency_flag. 7 = window_size-1 at real Train/Val scale "
        "(DO NOT change for a real run; only a smoke test on tiny "
        "synthetic-scale trajectories might need a smaller value).",
    )
    p.add_argument(
        "--min_k7_auroc",
        type=float,
        default=MIN_K7_AUROC,
        help="Eligibility floor on val_auroc_k7 (default: the frozen "
        "MIN_K7_AUROC protocol constant — overriding this is a deliberate "
        "protocol deviation, not a routine option).",
    )
    p.add_argument("--n_bins", type=int, default=10, help="Reliability/ECE bin count for the k7 Brier/ECE computation.")
    p.add_argument(
        "--max_trajectories",
        type=int,
        default=0,
        help="0 (default) = use the entire split. >0 truncates TRAIN and VAL to "
        "the first N trajectories each, for smoke testing ONLY (mirrors "
        "run_gru_evaluation.py's own --max_trajectories convention and its "
        "documented caveat: a deterministic prefix, not a representative "
        "random sample). Never use for a real selection run.",
    )
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _load_split(cache_root: Path, split: str, max_trajectories: int) -> List[Trajectory]:
    trajectories = group_into_trajectories(EmbeddingDataset(cache_root, split))
    if max_trajectories > 0:
        trajectories = trajectories[:max_trajectories]
    return trajectories


def _phase_decoding_accuracy(model: HMMModel, trajectories: List[Trajectory]) -> float:
    correct, total = 0, 0
    for traj in trajectories:
        if len(traj) == 0:
            continue
        filt = model.filtering(traj)
        decoded = np.argmax(filt, axis=1)
        truth = traj.last_frame_phase.numpy()
        correct += int((decoded == truth).sum())
        total += len(traj)
    return correct / total if total else float("nan")


def _mean_pseudo_log_likelihood(model: HMMModel, trajectories: List[Trajectory]) -> float:
    values = [model.sequence_log_likelihood(t) for t in trajectories if len(t) > 0]
    return float(np.mean(values)) if values else float("nan")


def _k7_scores(model: HMMModel, trajectories: List[Trajectory], k: int, n_bins: int) -> Dict:
    """Brier/AUROC/ECE for the consistency_flag-aligned k-step event —
    explicitly via predict_k_step(..., k=k), NEVER model.predict() (which is
    the old, k=1, lag-1 quantity this whole pipeline revision exists to
    stop selecting on)."""
    predictions = model.predict_k_step(trajectories, k=k)
    y_true, y_prob = flatten_predictions(trajectories, predictions)
    scored = score_probabilistic(y_true, y_prob, n_bins=n_bins)
    return {
        "val_auroc_k7": scored["auroc"],
        "val_brier_k7": scored["brier"],
        "val_brier_baseline_k7": scored["brier_baseline_constant_rate"],
        "val_ece_k7": scored["ece"],
        "val_n_windows_k7": scored["n_windows"],
    }


def select_best_config(rows: List[Dict], min_k7_auroc: float = MIN_K7_AUROC) -> Dict:
    """Pure selection logic (no I/O, no model fitting) — factored out
    specifically so the protocol's decision rule is unit-testable against
    hand-built metric dicts, without ever training an HMM (matches
    Tests/evaluation/models/test_hmm.py's synthetic-data-only discipline).

    rows: each dict must have 'transition_smoothing_alpha',
    'transition_decay_rho', 'val_auroc_k7', 'val_brier_k7',
    'val_phase_decoding_accuracy'. Extra keys are passed through untouched
    (each row is copied, never mutated in place).

    Rule (frozen — see docs/PROJECT_STATE.md's "HMM HYPERPARAMETER
    SELECTION PROTOCOL" checkpoint for the justification of every step; do
    not change this silently):
      1. eligible := val_auroc_k7 >= min_k7_auroc.
      2. Among eligible rows: smallest val_brier_k7 wins, compared at
         BRIER_TIE_DECIMALS decimals (a bit-exact tie in the raw float is
         vanishingly unlikely for this deterministic model; a *practical*
         tie is expected, and the protocol explicitly calls for one, not
         for treating a 5th-decimal difference as meaningful).
      3. Tie -> highest val_auroc_k7 (unrounded).
      4. Tie -> highest val_phase_decoding_accuracy.
      5. Tie even after all three criteria: NOT specified by the protocol.
         Resolved here by taking the first such row in the input list's
         order — a deterministic, documented fallback, not a silent
         invention of new science. Flagged in 'selection_reason' if it is
         ever actually reached.

    Returns
    -------
    dict with:
      'selection_rule' : human-readable description of the rule above.
      'min_k7_auroc'   : the floor actually used.
      'rows'           : input rows, each with an added 'eligible' bool,
                         same order as the input.
      'n_eligible'     : int.
      'selected'       : the winning row dict, or None.
      'selection_status': 'selected' | 'no_eligible_configuration'.
      'selection_reason': human-readable trace of which tie-break step (if
                          any) decided the winner, or why nothing was
                          selected.

    If NO row is eligible, this function does NOT invent a fallback (e.g.
    silently reverting to a k=1 criterion, or picking the best AUROC
    anyway despite the floor) — the protocol never specified what to do in
    that case. 'selected' is None and 'selection_status' says so
    explicitly; resolving that situation is a human decision, not
    something this function should paper over.
    """
    if not rows:
        raise ValueError("select_best_config() called with an empty rows list.")

    annotated = []
    for row in rows:
        row = dict(row)
        row["eligible"] = bool(row["val_auroc_k7"] >= min_k7_auroc)
        annotated.append(row)

    eligible = [r for r in annotated if r["eligible"]]
    result: Dict = {
        "selection_rule": (
            f"eligible := val_auroc_k7 >= {min_k7_auroc} (MIN_K7_AUROC); among "
            f"eligible configs, min val_brier_k7 (rounded to {BRIER_TIE_DECIMALS} "
            f"decimals) wins; ties broken by max val_auroc_k7, then max "
            f"val_phase_decoding_accuracy, then first-in-grid-order."
        ),
        "min_k7_auroc": min_k7_auroc,
        "rows": annotated,
        "n_eligible": len(eligible),
    }

    if not eligible:
        result["selected"] = None
        result["selection_status"] = "no_eligible_configuration"
        result["selection_reason"] = (
            f"0/{len(annotated)} configurations reached val_auroc_k7 >= "
            f"{min_k7_auroc}. The protocol does not define a fallback for this "
            f"case — no configuration was selected. Resolving this requires a "
            f"human decision (e.g. lower the floor with an explicit new "
            f"justification, widen the alpha/rho grid, or investigate the "
            f"emission/transition model), not a silent default pick."
        )
        return result

    min_rounded_brier = min(round(r["val_brier_k7"], BRIER_TIE_DECIMALS) for r in eligible)
    tier1 = [r for r in eligible if round(r["val_brier_k7"], BRIER_TIE_DECIMALS) == min_rounded_brier]

    if len(tier1) == 1:
        winner = tier1[0]
        reason = (
            f"unique minimum val_brier_k7 (rounded to {BRIER_TIE_DECIMALS} "
            f"decimals) = {min_rounded_brier} among {len(eligible)} eligible configs."
        )
    else:
        max_auroc = max(r["val_auroc_k7"] for r in tier1)
        tier2 = [r for r in tier1 if r["val_auroc_k7"] == max_auroc]
        if len(tier2) == 1:
            winner = tier2[0]
            reason = (
                f"{len(tier1)} configs tied at val_brier_k7 (rounded) = "
                f"{min_rounded_brier}; broken by max val_auroc_k7 = {max_auroc}."
            )
        else:
            max_phase_acc = max(r["val_phase_decoding_accuracy"] for r in tier2)
            tier3 = [r for r in tier2 if r["val_phase_decoding_accuracy"] == max_phase_acc]
            if len(tier3) == 1:
                winner = tier3[0]
                reason = (
                    f"{len(tier2)} configs also tied at val_auroc_k7 = {max_auroc}; "
                    f"broken by max val_phase_decoding_accuracy = {max_phase_acc}."
                )
            else:
                winner = tier3[0]
                reason = (
                    f"{len(tier3)} configs tied on Brier, AUROC, AND phase-decoding "
                    f"accuracy — not anticipated by the protocol. Fell back to "
                    f"first-in-grid-order (deterministic, but an undocumented "
                    f"protocol gap, not a scientific decision): alpha="
                    f"{winner['transition_smoothing_alpha']}, rho="
                    f"{winner['transition_decay_rho']}."
                )

    result["selected"] = winner
    result["selection_status"] = "selected"
    result["selection_reason"] = reason
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_k7_auroc != MIN_K7_AUROC:
        print(
            f"WARNING: --min_k7_auroc={args.min_k7_auroc} overrides the frozen "
            f"protocol floor MIN_K7_AUROC={MIN_K7_AUROC} (docs/PROJECT_STATE.md's "
            f"HMM HYPERPARAMETER SELECTION PROTOCOL checkpoint). Only do this "
            f"deliberately, with the override recorded in model_selection.json "
            f"(it is, automatically, via 'min_k7_auroc')."
        )
    if args.k7 != 7:
        print(
            f"NOTE: --k7={args.k7} (not 7) does not match window_size-1 at real "
            f"Train/Val scale — only expected for a smoke test on tiny "
            f"synthetic-scale trajectories."
        )

    cache_root = Path(args.cache_root) / args.embedding_model_name
    train_trajectories = _load_split(cache_root, "train", args.max_trajectories)
    val_trajectories = _load_split(cache_root, "val", args.max_trajectories)

    grid = list(itertools.product(args.alpha_grid, args.rho_grid))
    rows: List[Dict] = []
    fitted_models: Dict[Tuple[float, float], HMMModel] = {}

    for alpha, rho in grid:
        model = HMMModel(
            n_states=args.n_states,
            transition_smoothing_alpha=alpha,
            transition_decay_rho=rho,
            logreg_C=args.logreg_C,
        )
        model.fit(train_trajectories)

        k1_predictions = model.predict(val_trajectories)  # kept for continuity
        # with the original sweep's metrics (accuracy/precision/recall/f1/
        # AUROC_k1) — NOT used anywhere in the new selection rule.
        k1_metrics = compute_metrics(val_trajectories, k1_predictions)
        phase_acc = _phase_decoding_accuracy(model, val_trajectories)
        mean_ll = _mean_pseudo_log_likelihood(model, val_trajectories)
        k7_metrics = _k7_scores(model, val_trajectories, args.k7, args.n_bins)

        config_dir_name = f"alpha_{alpha}_rho_{rho}"
        config_dir = output_dir / config_dir_name
        model.save(config_dir)  # every configuration's fitted model is saved,
        # not just the eventual winner — the original sweep only saved the
        # winner, which made any retroactive analysis of the other 8 configs
        # impossible without refitting (exactly the gap this revision closes).

        row = {
            "transition_smoothing_alpha": alpha,
            "transition_decay_rho": rho,
            "config_dir": config_dir_name,
            "val_accuracy": k1_metrics["accuracy"],
            "val_auroc_k1": k1_metrics.get("auroc"),
            "val_precision": k1_metrics["precision"],
            "val_recall": k1_metrics["recall"],
            "val_f1": k1_metrics["f1"],
            "val_phase_decoding_accuracy": phase_acc,
            "val_mean_pseudo_log_likelihood": mean_ll,
            **k7_metrics,
        }
        rows.append(row)
        fitted_models[(alpha, rho)] = model

        (config_dir / "config_metadata.json").write_text(json.dumps({
            "transition_smoothing_alpha": alpha,
            "transition_decay_rho": rho,
            # HMMModel.fit() is a deterministic closed-form fit (segment/
            # window-level counting + one LogisticRegression.fit(), no
            # random_state or np.random anywhere in hmm.py) — there is no
            # seed to record. Recorded explicitly as null rather than
            # omitted, so a future reader doesn't mistake this for an
            # oversight. Bit-exact reproducibility across independent runs
            # was already verified for this model family (see
            # Results/evaluation/e1_hmm_step1_reproducibility_check/).
            "seed": None,
            "n_states": args.n_states,
            "phase_mapping": model.state_names,
            "logreg_C": args.logreg_C,
            "val_metrics": row,
        }, indent=2))

        print(json.dumps(row))

    selection = select_best_config(rows, min_k7_auroc=args.min_k7_auroc)

    selection_report = {
        **selection,
        "k7": args.k7,
        "n_bins": args.n_bins,
        "n_states": args.n_states,
        "logreg_C": args.logreg_C,
        "max_trajectories": args.max_trajectories,
        "n_train_trajectories": len(train_trajectories),
        "n_val_trajectories": len(val_trajectories),
    }
    (output_dir / "model_selection.json").write_text(json.dumps(selection_report, indent=2))

    if selection["selected"] is not None:
        winner_row = selection["selected"]
        key = (winner_row["transition_smoothing_alpha"], winner_row["transition_decay_rho"])
        fitted_models[key].save(output_dir / "best_model")
        print("\nSelected config:", key)
        print("Reason:", selection["selection_reason"])
        print("Best model saved to", output_dir / "best_model")
    else:
        print("\nNO CONFIGURATION SELECTED:", selection["selection_reason"])
        print("best_model/ was NOT written — see selection_reason above and "
              "model_selection.json's 'rows' for the full per-config detail.")


if __name__ == "__main__":
    main()
