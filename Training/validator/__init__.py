"""
Scientific Validator v1 -- a QUALIFICATION layer, not a truth oracle.

WHAT THIS PACKAGE IS FOR
------------------------
The three-condition benchmark (FULL / OBSERVED_ONLY / PREDICTION_ONLY,
`Results/evaluation/prediction_only_experiment/analysis_three_conditions_v1.md`)
established two facts that this package exists to act on:

  1. The LLM turns a model output into an observation. On Q3, 3 replications
     out of 3 answered "la phase dominante juste avant la transition etait t7"
     -- t7 is the Semi-HMM's `current_phase`, the annotation says t4. No
     hallucination: t7 really is in the context. A pure presence check cannot
     see this.
  2. `grounded=True` is not correctness. In PREDICTION_ONLY, 21 of the 26
     grounded answers (81%) were not PASS, and two of them were grounded while
     contradicting the ground truth.

So this package checks the one thing a presence check structurally cannot:
**does the claim's grammatical framing match the provenance of the data that
supports it?**

WHAT THIS PACKAGE IS NOT
------------------------
It is NOT an oracle for the scientific model. It never decides whether the
Semi-HMM is right about an embryo. It cannot: the annotation is not available
at inference time, and it must not be -- `validate_answer()` takes the same
context the LLM saw and nothing else.

The hardest form of this rule, demonstrated by the benchmark itself: the
PREDICTED transition `t7 -> t8` is perfectly compatible with the Consensus's
chronological ordering -- MORE regular than the truth, since the real annotated
transition `t4 -> t6` SKIPS t5. A validator that reasoned "compatible with the
Consensus, therefore plausible" would validate the false prediction and reject
reality. Hence the central rule, enforced in `rules.py` and pinned by tests:

    "compatible avec le Consensus"  !=  "vrai pour cet embryon"

SCOPE DISCIPLINE
----------------
Additive only. Nothing here modifies the dataset, the annotations, the
scientific models, the Reporting API, the RAG corpus or its Chroma index, the
router, the system prompt, the renderers, the Web App, or the 15 frozen
questions. It IS, however, reached from production since the Validator was
wired in: `orchestrator.scientific_validation` imports `validator.validator`
and `orchestrator.scientific_reference` imports `validator.{corpus,extraction,
retrieval}`. In return `validator.validator` imports
`orchestrator.grounding_check`, so the two packages form a deliberate cycle.
It is benign and must stay that way: both packages' `__init__.py` are
docstring-only, so importing either one never triggers the other at package
level, and every module resolves. Do not add imports to either `__init__.py`.
The Istanbul corpus is read from its curated Markdown file --
`docs/corpus/ISTANBUL_CONSENSUS_2025.md` -- with NO embedding and NO vector
store, so the 432 existing embeddings are not touched either.
"""
