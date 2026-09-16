"""
Reporting API — the service layer between the frozen Semi-HMM scientific
model (Training/evaluation/models/semi_hmm.py) and future consumers (Web
App, RAG). Purely additive, like the rest of the SciML extension: never
edits Training/evaluation/, Training/embeddings/, or any of the two
original project halves (Training/'s classifier pipeline, WebApplication/).

Placed under Training/reporting/ (not a new top-level Reporting/) to stay
consistent with the existing Training/embeddings/, Training/evaluation/
naming convention -- this is one more layer of the same additive SciML
extension, not a third project "half".

Module map
----------
schemas.py            Typed, JSON-serializable dataclasses for every
                       reporting output, matching docs/INFERENCE_SCHEMA.md
                       field-by-field (AVAILABLE/DERIVED/UNAVAILABLE).
model_loader.py        Loads the frozen Semi-HMM once, validates its
                        configuration against the expected reference
                        (n_states=15, dmax=268, negative_binomial,
                        embedding_dim=512), never refits, never saves.
trajectory_service.py  Loads embeddings/metadata_with_time.csv, groups
                        windows into Trajectory objects, lists videos --
                        Val (and Train) only; Test is hard-refused.
inference_service.py   Composes model_loader + trajectory_service +
                        SemiHMMModel's own already-implemented methods
                        (filtering, next_phase_distribution,
                        duration_distribution, expected_duration,
                        quantiles, transition_chain) into one canonical
                        InferenceRecord per window. Never reimplements
                        any of the model's mathematics.
api.py                  Flask app (matches WebApplication/'s existing
                         stack; no new web framework dependency) exposing
                         the endpoints in docs/REPORTING_API.md.

Test = LOCKED: every function in this package that accepts a `split`
argument refuses split="test" explicitly and immediately -- not by
omission, by a hard, tested guard (see trajectory_service.py).
"""
