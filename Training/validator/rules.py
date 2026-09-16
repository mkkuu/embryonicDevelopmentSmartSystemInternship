"""
The Istanbul validation rules, one named function each.

Every rule is a pure function of (claim, evidence about the context, corpus)
and returns an `Outcome` or None. None means "this rule has nothing to say".
Rules never read the annotation: `ctx` is exactly the context the LLM saw.

Each docstring states the rule, and where it comes from -- either a corpus
block id (so it can be chased back to the PDF) or the three-condition
benchmark finding that made it necessary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from . import provenance as prov
from .corpus import Corpus
from .schema import (CAUSAL_CLAIM, CLINICAL_IMPLICATION, COMBINED,
                     COMPATIBILITY_VERDICT, CONTRADICTED, DERIVED_FROM_MODEL,
                     DERIVED_FROM_OBSERVED, EXTERNAL_DEFINITION,
                     KNOWLEDGE_GAP_DECLARED, MISSING_DATA, MODEL_DERIVED,
                     NOT_ASSESSABLE, NOT_SUPPORTED, OBSERVED, OUT_OF_SCOPE,
                     PARTIALLY_SUPPORTED, QUALIFY, RETRIEVAL_MISS, SCIENTIFIC,
                     SUPPORTED, SUPPRESS, SYSTEM, UNKNOWN_INSEMINATION_METHOD,
                     UNVERIFIED_TIME_BASIS)

# Severity order: a later status never downgrades an earlier, more severe one.
_SEVERITY = {SUPPORTED: 0, PARTIALLY_SUPPORTED: 1, NOT_ASSESSABLE: 2,
             NOT_SUPPORTED: 3, CONTRADICTED: 4}


@dataclass
class Outcome:
    rule: str
    status: str
    qualification: str
    reason_code: Optional[str] = None
    action: str = QUALIFY
    limitations: List[str] = None

    def __post_init__(self):
        if self.limitations is None:
            self.limitations = []


def _has(markers: List[str], *names: str) -> bool:
    return any(n in markers for n in names)


_MODEL_SIDE = {MODEL_DERIVED, DERIVED_FROM_MODEL}
_OBSERVED_SIDE = {OBSERVED, DERIVED_FROM_OBSERVED}


def _effective_framing(claim: Dict, info: Dict) -> Set[str]:
    """v1.3: the sentence's own framing, or -- when it has none -- the framing
    it inherits from its discourse position (list opened by an attributing
    sentence, pronoun subject). Own framing always wins."""
    own = set(info["framing_classes"])
    if own - {SYSTEM}:
        return own
    return own | set(claim.get("inherited_framing") or [])


# v1.1 -- R2: at least this many distinct phase names in ONE sentence is an
# enumeration of the taxonomy, never a claim about the embryo's phase.
_TAXONOMY_ENUMERATION_MIN = 5
# v1.3 -- R2: "t4 -> t5 -> t6 -> t7." (Q11 r2) is a chain of the reference
# order, not a claim about this embryo. A sentence that is nothing but >= 3
# phase names joined by arrows / commas / "puis" is such a chain.
_CHAIN_FILLER = re.compile(r"(?:->|→|<|>|,|;|\.|puis|ensuite|et|then|\s|\*|-)+", re.IGNORECASE)
_CHAIN_MIN = 3
# v1.3 -- R2: a sentence that speaks of the reference ORDER ("t6, qui se
# produit normalement apres t7") while the phase is carried by the SCIENTIFIC
# context (the canonical order in the retrieved documentation) refers to the
# taxonomy, not to this embryo's state.
_ORDER_WORDS = ("normalement", "ordre", "attendu", "attendue", "canonique", "sequence normale",
                "succede", "precede", "suit la phase", "apres la phase", "avant la phase",
                "chronologi", "referentiel")


def _is_phase_chain(text: str) -> bool:
    phases = prov.PHASE_TOKEN.findall(text or "")
    if len(phases) < _CHAIN_MIN:
        return False
    residue = _CHAIN_FILLER.sub("", prov.PHASE_TOKEN.sub("", prov._norm(text)))
    return residue.strip() == ""

# v1.1 -- R4: a denial ABOUT THE DATA ("il n'y a pas de donnees sur les
# reperes morphocinetiques") is the honest limitation R4 exists to protect,
# not a claim that an event did not happen. Q13 compact_v1 replay flagged the
# correct answer with R4.
_DATA_ABSENCE_PHRASES = ("pas de donnee", "pas d'information", "aucune donnee",
                         "aucune information", "non collecte", "ne contient pas",
                         "pas de repere", "aucun repere", "pas disponible",
                         "indisponible", "non annote", "ni annote", "pas defini",
                         "ni defini", "ne sont pas abordes", "n'est pas aborde",
                         # v1.3 -- Q15 r2: "le consensus n'a pas de definition
                         # operationnelle pour ..." denies a DEFINITION, not an event.
                         "pas de definition", "aucune definition", "definition operationnelle",
                         "sans aucune connaissance", "connaissance externe")


# ---------------------------------------------------------------------------
# R2 -- the rule the whole package exists for
# ---------------------------------------------------------------------------

def r2_prediction_is_not_observation(claim: Dict, ctx: Dict, corpus: Corpus,
                                     info: Dict) -> Optional[Outcome]:
    """A claim framed as an OBSERVATION, whose phase is carried ONLY by a
    MODEL-DERIVED field, is not supported by this context.

    Source: the three-condition benchmark, Q3, 3 replications out of 3 --
    `current_phase = t7` (Semi-HMM) restated as "la phase dominante juste avant
    la transition etait t7". No hallucination, `grounded=True`, and wrong: the
    annotation says t4. The validator never learns t4; it only reports that the
    context licenses a prediction, not an observation.
    """
    framing = _effective_framing(claim, info)
    if OBSERVED in framing or DERIVED_FROM_OBSERVED in framing or SCIENTIFIC in framing:
        return None            # the sentence attributes itself; nothing to fix
    if MODEL_DERIVED in framing or DERIVED_FROM_MODEL in framing:
        return None            # correctly framed as a prediction
    # v1.3: a sentence that declines ("pas de moyen de savoir si la phase
    # suivante est t7 ou non", Q15 r2) asserts no phase.
    if claim.get("hedged"):
        return None
    # v1.1: a sentence enumerating five or more distinct phase names is a
    # recitation of the taxonomy (`tPB2, tPNa, ..., tEB`), not a statement
    # that THIS embryo is in one of them. Such a list reached R2 in the
    # compact_v1 replay (Q11) only because every phase name is also a KEY of
    # `phase_probabilities`, i.e. reachable on the model side.
    if len(set(p.lower() for p in claim.get("phases", []))) >= _TAXONOMY_ENUMERATION_MIN:
        return None
    if _is_phase_chain(claim["text"]):
        return None
    # v1.3: a verbatim copy of a printed field ("* t8 : 0.2367...", "W157=t8")
    # carries that field's provenance and asserts nothing about the embryo.
    restated = info.get("restated") or {}
    low_text = prov._norm(claim["text"])
    speaks_of_order = any(w in low_text for w in _ORDER_WORDS)
    offenders = []
    for phase in claim.get("phases", []):
        if phase.lower() in restated:
            continue
        sources = set(prov.sources_for_phase(ctx, phase))
        if not sources:
            continue
        model_side = sources & _MODEL_SIDE
        observed_side = sources & _OBSERVED_SIDE
        if SCIENTIFIC in sources and speaks_of_order:
            continue           # v1.3: a statement about the reference order
        if model_side and not observed_side:
            offenders.append(phase)
    if not offenders:
        return None
    joined = ", ".join(offenders)
    # v1.3: the closing sentence states only what is STRUCTURALLY true of this
    # context -- never "aucune annotation" when one is present.
    if prov.has_observed_annotation(ctx):
        closing = (f"L'annotation presente dans ce contexte ne rapporte pas {joined} ; "
                   f"la valeur observee n'est donc ni confirmee ni infirmee par cette phrase.")
    else:
        closing = ("Aucune annotation n'est presente dans ce contexte, donc la valeur "
                   "reelle n'est ni confirmee ni infirmee ici.")
    return Outcome(
        rule="R2_prediction_is_not_observation",
        status=NOT_SUPPORTED,
        qualification=(
            f"Le contexte disponible etablit une PREDICTION du modele pour {joined}, "
            f"pas une observation embryologique. La formulation presente "
            f"{joined} comme un fait constate ; elle doit etre attribuee "
            f"(« le modele predit {joined} »). {closing}"),
        action=QUALIFY)


def r2b_transition_provenance(claim: Dict, ctx: Dict, corpus: Corpus,
                              info: Dict) -> Optional[Outcome]:
    """A transition pair must keep the provenance of the block it comes from.
    `t7 -> t8` is never an observed transition just because it is printed in
    MODEL-DERIVED; `t4 -> t6` is never "predicted" because it comes from the
    annotation.

    v1.3: fires only when the sentence frames itself on ONE side and that side
    is the wrong one. A sentence carrying both framings ("le modele n'a pas
    predit la transition qui a ete annotee (t4 -> t6 au lieu de t7 -> t8)",
    Q6 r2) attributes each pair explicitly; the lexical detector cannot tell
    which pair goes with which verb, so it must not guess."""
    framing = _effective_framing(claim, info)
    frames_observed = bool(framing & _OBSERVED_SIDE)
    frames_model = bool(framing & _MODEL_SIDE)
    if frames_observed == frames_model:
        return None            # neither side, or both sides: nothing decidable
    for pair in info.get("transition_pairs", []):
        sources = set(prov.sources_for_transition(ctx, pair))
        if not sources:
            continue
        text = f"{pair[0]} -> {pair[1]}"
        if frames_observed and sources == {MODEL_DERIVED}:
            return Outcome(
                rule="R2b_transition_provenance",
                status=NOT_SUPPORTED,
                qualification=(f"La transition {text} est une sortie du MODELE dans ce "
                               f"contexte ; la presenter comme annotee/observee inverse "
                               f"sa provenance."))
        if frames_model and sources == {OBSERVED}:
            return Outcome(
                rule="R2b_transition_provenance",
                status=NOT_SUPPORTED,
                qualification=(f"La transition {text} provient de l'ANNOTATION dans ce "
                               f"contexte ; la presenter comme predite inverse sa provenance."))
    return None


# ---------------------------------------------------------------------------
# R11 -- offset_from_center is not a lag (added 2026-09-12)
# ---------------------------------------------------------------------------

_LAG_WORDS = ("decalage", "lag", "en avance", "en retard", "avant l'annotation",
              "apres l'annotation", "avant ou apres l'annotation",
              # v1.3 -- Q7 r3: "le modele a predit la transition une fenetre
              # avant qu'elle ne soit observee" is the same lag claim in other words.
              # ("avant/apres la transition annotee" is NOT listed: "les
              # probabilites evoluent avant et apres la transition annotee"
              # describes windows, not a lag -- Q6 r3.)
              "avant qu'elle ne soit observ", "apres qu'elle ne soit observ",
              "avant l'observation", "apres l'observation")
_NUMBER_WORDS = ("une fenetre", "un fenetre", "deux fenetres", "trois fenetres",
                 "quatre fenetres", "cinq fenetres")
_DIRECTION_WORDS = ("avant l'annotation", "apres l'annotation", "en avance", "en retard",
                    "avant qu'elle ne soit observ", "apres qu'elle ne soit observ",
                    "avant l'observation", "apres l'observation")


def _context_lag(ctx: Dict) -> tuple:
    """(present, value) of model_vs_observed_lag_windows in whichever series
    analysis this context carries. present=False when no series is there."""
    dyn = (ctx or {}).get("dynamic_context") or {}
    for key in ("series_analysis", "model_series_analysis", "observed_series_analysis"):
        series = dyn.get(key)
        if isinstance(series, dict):
            timing = series.get("timing") or {}
            if isinstance(timing, dict) and "model_vs_observed_lag_windows" in timing:
                return True, timing.get("model_vs_observed_lag_windows")
    return False, None


def r11_offset_is_not_a_lag(claim: Dict, ctx: Dict, corpus: Corpus,
                            info: Dict) -> Optional[Outcome]:
    """A signed lag or a detection direction stated while the context prints
    `model_vs_observed_lag_windows = null` is not licensed by the context: the
    only lag the pipeline computes is unavailable, and an `offset_from_center`
    (a window's position in the band) or an instant is never that lag.
    Source: 4/4 replications of the 07c benchmark and Q7 of compact_v1 (V2
    "2 fenetres", COMPACT "-1 fenetre" -- the -1 is W155's offset). A hedged
    sentence ("non calculable") is the correct answer and is exempt."""
    low = prov._norm(claim["text"])
    if not any(w in low for w in _LAG_WORDS) or claim.get("hedged"):
        return None
    present, lag = _context_lag(ctx)
    if not present or lag is not None:
        return None
    asserts_value = bool(re.search(r"[-+\u2212]?\d+\s*fenetre", low)) or any(
        w in low for w in _NUMBER_WORDS)
    asserts_direction = any(w in low for w in _DIRECTION_WORDS)
    if not (asserts_value or asserts_direction):
        return None
    return Outcome(
        rule="R11_offset_is_not_a_lag",
        status=NOT_SUPPORTED,
        qualification=("Le contexte imprime model_vs_observed_lag_windows = null : aucun decalage "
                       "modele/annotation n'est calculable ici, ni en valeur ni en direction. Un "
                       "offset_from_center (position d'une fenetre dans la bande) ou un instant "
                       "n'est pas ce decalage ; la valeur ou la direction avancee n'est pas "
                       "licenciee par le contexte."))


# ---------------------------------------------------------------------------
# R1, R3, R4
# ---------------------------------------------------------------------------

# v1.3 -- R1 needs an INTERPRETATION, not merely the word "embryon". In the
# OFF/ON benchmark R1 fired PARTIALLY_SUPPORTED on "la phase actuelle de
# l'embryon, telle que predite par le modele, est t7" (Q2 r2), on "ce genre
# d'incoherence est un domaine actif de recherche" (Q6 r2) and on restated
# questions -- none of them draws a biological conclusion from the data.
_INTERPRETATION_WORDS = (
    "viabilite", "viable", "implantation", "aneuploid", "euploid", "qualite de l'embryon",
    "bonne qualite", "mauvaise qualite", "competence", "pronostic", "prognostic",
    "sain", "pathologi", "anomalie", "developpement normal", "developpement anormal",
    "bon developpement", "mauvais developpement", "retard de developpement",
    "risque", "indique que l'embryon", "suggere que l'embryon", "signifie que l'embryon",
    "l'embryon est normal", "l'embryon est anormal", "potentiel", "bon signe", "mauvais signe",
)


def r1_observation_is_not_interpretation(claim: Dict, ctx: Dict, corpus: Corpus,
                                         info: Dict) -> Optional[Outcome]:
    """A data statement does not become a biological interpretation. Source:
    IC2025-STATUS-01 (GPR, "should not be interpreted as setting a standard of
    care") and IC2025-LIM-01."""
    markers = claim["markers"]
    if not _has(markers, "biological") or claim["hedged"]:
        return None
    if not (_has(markers, "phase", "transition", "probability", "stability")):
        return None
    if SCIENTIFIC in info["framing_classes"]:
        return None
    if not any(w in prov._norm(claim["text"]) for w in _INTERPRETATION_WORDS):
        return None            # v1.3: no interpretation is drawn
    return Outcome(
        rule="R1_observation_is_not_interpretation",
        status=PARTIALLY_SUPPORTED,
        qualification=("La donnee peut etre citee, mais l'interpretation biologique "
                       "qui en est tiree n'est pas etablie par le contexte."),
        limitations=["IC2025-LIM-01", "IC2025-STATUS-01"])


def r3_annotated_skip_is_not_predicted_skip(claim: Dict, ctx: Dict, corpus: Corpus,
                                            info: Dict) -> Optional[Outcome]:
    """An annotated skip and a predicted skip are different objects. Source:
    the project's own standing rule, and `observed_only.py`'s design note."""
    if not _has(claim["markers"], "skip"):
        return None
    framing = _effective_framing(claim, info)
    if bool(framing & _MODEL_SIDE) == bool(framing & _OBSERVED_SIDE):
        return None            # v1.3: same single-side discipline as R2b
    table = prov.transitions_by_provenance(ctx)
    observed_pairs, model_pairs = table[OBSERVED], table[MODEL_DERIVED]
    if MODEL_DERIVED in framing and observed_pairs and not model_pairs:
        return Outcome(
            rule="R3_annotated_skip_is_not_predicted_skip",
            status=NOT_SUPPORTED,
            qualification=("Le saut disponible dans ce contexte appartient a l'ANNOTATION ; "
                           "il ne peut pas etre presente comme un saut de la sequence predite."))
    if OBSERVED in framing and model_pairs and not observed_pairs:
        return Outcome(
            rule="R3_annotated_skip_is_not_predicted_skip",
            status=NOT_SUPPORTED,
            qualification=("Le saut disponible dans ce contexte appartient a la sequence "
                           "PREDITE ; il ne peut pas etre presente comme un saut annote."))
    return None


def r4_absence_of_evidence(claim: Dict, ctx: Dict, corpus: Corpus,
                           info: Dict) -> Optional[Outcome]:
    """"Not collected" is not "did not happen". Source: IC2025-ABSENT-06/07 and
    the corpus's own "Not covered by this source" section."""
    low = prov._norm(claim["text"])
    denial = any(p in low for p in ("il n'y a pas eu", "n'a pas eu lieu", "aucun",
                                    "ne correspond pas", "ne peut correspondre",
                                    "il n'existe pas", "pas de "))
    if not denial:
        return None
    if any(p in low for p in _DATA_ABSENCE_PHRASES):
        return None            # v1.1: the sentence denies DATA, not an event
    if not _has(claim["markers"], "direct_cleavage", "reverse_cleavage",
                "chaotic_division", "biological"):
        return None
    return Outcome(
        rule="R4_absence_of_evidence_is_not_evidence_of_absence",
        status=NOT_ASSESSABLE,
        reason_code=MISSING_DATA,
        qualification=("L'information n'est pas collectee dans ce systeme : « non observe » "
                       "ne peut pas etre enonce comme « n'a pas eu lieu »."),
        limitations=["IC2025-ABSENT-06", "IC2025-ABSENT-07"])


# ---------------------------------------------------------------------------
# R5, R6, R14 -- strength of claim
# ---------------------------------------------------------------------------

_PROVEN_WORDS = ("prouve", "demontre", "confirme", "atteste", "etabli que",
                 "signifie que l'embryon", "donc normal", "est normal",
                 "est anormal", "certifie")


def r5_compatible_is_not_proven(claim: Dict, ctx: Dict, corpus: Corpus,
                                info: Dict) -> Optional[Outcome]:
    """"Compatible with" never becomes "demonstrated"/"normal". Source:
    IC2025-LIM-01 (GPR, not a standard of care) and IC2025-STATUS-04."""
    low = prov._norm(claim["text"])
    if not _has(claim["markers"], "compatibility"):
        return None
    if not any(w in low for w in _PROVEN_WORDS):
        return None
    return Outcome(
        rule="R5_compatible_is_not_proven",
        status=NOT_SUPPORTED,
        qualification=("« compatible avec » n'autorise pas « prouve », « normal » ou "
                       "« anormal » : la compatibilite avec un referentiel n'est pas "
                       "une demonstration."),
        limitations=["IC2025-LIM-01"])


def r6_association_is_not_causation(claim: Dict, ctx: Dict, corpus: Corpus,
                                    info: Dict) -> Optional[Outcome]:
    """The Consensus reports associations ("associated with", "may be associated
    with"), never causation. Source: IC2025-TIME-03, IC2025-AC-C01/C02, all
    graded Low or Very low."""
    if claim["claim_kind"] != CAUSAL_CLAIM and not _has(claim["markers"], "causal"):
        return None
    return Outcome(
        rule="R6_association_is_not_causation",
        status=NOT_SUPPORTED,
        qualification=("Le Consensus ne rapporte que des ASSOCIATIONS (« associated with »), "
                       "de niveau Very low a Low ; une formulation causale depasse le niveau "
                       "de preuve disponible."),
        limitations=["IC2025-LIM-06", "IC2025-TIME-03"])


def r14_consensus_never_validates_a_prediction(claim: Dict, ctx: Dict, corpus: Corpus,
                                               info: Dict) -> Optional[Outcome]:
    """THE central rule. A MODEL-DERIVED claim never becomes SUPPORTED because
    the Consensus describes a compatible general ordering.

    Demonstrated by this project's own data: the PREDICTED transition
    `t7 -> t8` is perfectly compatible with the Consensus ordering -- more
    regular than the truth, since the annotated transition `t4 -> t6` SKIPS t5
    (IC2025-T5, IC2025-AC-A03). "Compatible with the Consensus" and "true for
    this embryo" are different statements, and only the first is obtainable
    here.
    """
    framing = _effective_framing(claim, info)
    # v1.3: a sentence that frames itself as SCIENTIFIC and names no model
    # ("la prochaine etape attendue selon le referentiel est t7", Q12) states
    # the reference order -- it validates no prediction. The model side must
    # be EXPLICIT, or the phase must be model-only in a sentence that does not
    # claim the reference for itself.
    explicit_model = bool(framing & _MODEL_SIDE)
    implicit_model = bool(info.get("model_only_phases")) and SCIENTIFIC not in framing
    if not (explicit_model or implicit_model):
        return None
    if not (_has(claim["markers"], "compatibility", "expected_development")
            or SCIENTIFIC in framing):
        return None
    return Outcome(
        rule="R14_consensus_never_validates_a_prediction",
        status=PARTIALLY_SUPPORTED,
        qualification=("Cette affirmation peut au mieux etre SCIENTIFIQUEMENT COMPATIBLE : "
                       "le Consensus decrit un ordre general, il ne peut pas etablir qu'une "
                       "prediction est vraie pour cet embryon. « compatible avec le Consensus » "
                       "n'est pas « vrai pour cet embryon » -- verifier la prediction exigerait "
                       "l'annotation ou une nouvelle observation."),
        limitations=["IC2025-LIM-01", "IC2025-STATUS-01"])


# ---------------------------------------------------------------------------
# R7, R8 -- what the corpus does not carry
# ---------------------------------------------------------------------------

def r8_definition_absent_from_corpus(claim: Dict, ctx: Dict, corpus: Corpus,
                                     info: Dict) -> Optional[Outcome]:
    """direct cleavage / reverse cleavage / irregular chaotic division are
    NAMED by the Consensus and never DEFINED: the source defers its
    nomenclature to Ciray et al. (2014), which is not in this corpus.
    Source: IC2025-AC-B01, IC2025-TERM-08, IC2025-ABSENT-01.

    A phase skip is not a direct cleavage; a phase return is not a reverse
    cleavage; an irregular phase sequence is not a chaotic division."""
    if not _has(claim["markers"], "direct_cleavage", "reverse_cleavage",
                "chaotic_division"):
        return None
    return Outcome(
        rule="R8_definition_absent_from_corpus",
        status=NOT_ASSESSABLE,
        reason_code=EXTERNAL_DEFINITION,
        qualification=("Le Consensus NOMME direct cleavage / reverse cleavage / division "
                       "chaotique mais n'en donne AUCUNE definition operationnelle : il renvoie "
                       "sa nomenclature a Ciray et al. (2014), absent de ce corpus. Ces motifs "
                       "ne sont pas non plus annotes dans les donnees. Un saut de phase n'est "
                       "pas un direct cleavage, un retour de phase n'est pas un reverse "
                       "cleavage, une sequence irreguliere n'est pas une division chaotique."),
        limitations=["IC2025-AC-B01", "IC2025-TERM-08", "IC2025-ABSENT-01"])


def r7_consensus_silent(claim: Dict, ctx: Dict, corpus: Corpus,
                        info: Dict) -> Optional[Outcome]:
    """When a claim needs the corpus and retrieval returns nothing relevant,
    the answer is NOT_ASSESSABLE -- never a silent guess."""
    if SCIENTIFIC not in info["framing_classes"] and not _has(
            claim["markers"], "expected_development", "compatibility"):
        return None
    if info.get("retrieved"):
        return None
    if not corpus.available:
        return Outcome(rule="R7_consensus_silent", status=NOT_ASSESSABLE,
                       reason_code=RETRIEVAL_MISS,
                       qualification="Le corpus Istanbul n'est pas disponible pour ce controle.")
    return Outcome(rule="R7_consensus_silent", status=NOT_ASSESSABLE,
                   reason_code=OUT_OF_SCOPE,
                   qualification=("Aucun bloc du Consensus ne traite cette affirmation ; "
                                  "le Consensus ne permet pas de conclure."))


# ---------------------------------------------------------------------------
# R9, R10 -- timing safety
# ---------------------------------------------------------------------------

_UNIT_WORDS = ("heure", "heures", "minute", "minutes", "seconde", "secondes",
               "jour", "jours", "hpi")
_SOURCED_UNITS = ("fenetre", "fenetres", "window", "windows")


def r9_timing_requires_unit_and_origin(claim: Dict, ctx: Dict, corpus: Corpus,
                                       info: Dict) -> Optional[Outcome]:
    """No timing without a sourced unit AND a verified time origin.

    The dataset prints `time_unit = "unknown/unverified"` and
    `duration_unit = "windows"`; the Consensus requires hours post-insemination
    (IC2025-CP1-01) and gives medians only, with NO dispersion (IC2025-T1,
    IC2025-LIM-09). The three-condition run produced "43.1 heures",
    "22,75 secondes", "-3,8 minutes" and "4,739 jours" from exactly these
    fields."""
    low = prov._norm(claim["text"])
    named = [u for u in _UNIT_WORDS if re.search(rf"\d[\d\s.,]*\s*{u}\b", low)]
    if not named:
        return None
    unit = info.get("context_time_unit")
    if unit and unit.lower() not in ("unknown/unverified", "unknown", "unverified"):
        return None
    return Outcome(
        rule="R9_timing_requires_unit_and_origin",
        status=NOT_ASSESSABLE,
        reason_code=UNVERIFIED_TIME_BASIS,
        qualification=(f"Unite temporelle non sourcee : la reponse emploie « {named[0]} » "
                       f"alors que le contexte imprime time_unit = "
                       f"« {unit or 'non renseigne'} ». Le Consensus exige des heures "
                       f"post-insemination (hpi) et une origine verifiee ; ni l'unite ni "
                       f"l'origine ne sont etablies ici."),
        limitations=["IC2025-CP1-01", "IC2025-LIM-09"])


def r9b_median_is_not_a_range(claim: Dict, ctx: Dict, corpus: Corpus,
                              info: Dict) -> Optional[Outcome]:
    """Table 1 gives MEDIANS and prints no dispersion; a timing may not be
    called typical or atypical on that basis. Source: IC2025-T1,
    IC2025-LIM-09, IC2025-LIM-11, IC2025-KG-04."""
    low = prov._norm(claim["text"])
    if not _has(claim["markers"], "timing"):
        return None
    verdict = any(w in low for w in ("atypique", "typique", "normal", "anormal",
                                     "dans la norme", "hors norme"))
    # v1.3 -- Q13 r1/r2, Q14 r3: "dans l'intervalle de confiance", "dans la
    # plage attendue", "dans les limites attendues de la variabilite biologique
    # connue" all invoke a dispersion Table 1 does not print.
    dispersion = any(w in low for w in _DISPERSION_WORDS)
    if not (verdict or dispersion):
        return None
    if claim.get("hedged"):
        return None            # v1.1: "ne peut pas etre considere comme atypique" declines the verdict
    if dispersion and not verdict:
        qualification = ("Le Consensus ne publie que des MEDIANES (Table 1), sans aucune "
                         "dispersion (ni ecart-type, ni intervalle, ni plage) : aucun "
                         "intervalle de confiance, plage attendue ou limite de variabilite "
                         "ne peut etre tire de cette source.")
    else:
        qualification = ("Le Consensus ne publie que des MEDIANES (Table 1), sans aucune "
                         "dispersion, et ne fournit aucun seuil d'atypie : qualifier un timing "
                         "de typique ou d'atypique depasse ce que la source autorise.")
    return Outcome(
        rule="R9b_median_is_not_a_range",
        status=NOT_ASSESSABLE,
        reason_code=UNVERIFIED_TIME_BASIS,
        qualification=qualification,
        limitations=["IC2025-T1", "IC2025-LIM-09", "IC2025-KG-04"])


_DISPERSION_WORDS = ("intervalle de confiance", "plage attendue", "dans la plage",
                     "ecart-type", "ecart type", "interquartile", "fourchette",
                     "limites attendues", "dans les limites", "variabilite biologique connue",
                     "variabilite connue")

# v1.3 -- R9c: a MORPHOKINETIC compatibility verdict is a timing comparison.
_MORPHOKINETIC_WORDS = ("morphocinetique", "morphokinetic", "repere", "reperes", "hpi",
                        "timing", "moment observe", "temps observe", "table 1",
                        "duree attendue", "temps attendu", "heures post")


def r9c_morphokinetic_compatibility_requires_time_basis(
        claim: Dict, ctx: Dict, corpus: Corpus, info: Dict) -> Optional[Outcome]:
    """"Compatible avec les reperes morphocinetiques" compares an observed
    time with reference timings in hpi (IC2025-CP1-01). When the context
    prints `time_unit = unknown/unverified`, no such comparison has a basis:
    the verdict is not assessable, whichever way it goes. Source: the OFF/ON
    benchmark, Q13, 3/3 replications declared compatibility from a context
    whose time unit is unverified; the corpus holds no hpi reference for
    t4 -> t6 either. A hedged sentence ("ne peut pas etre evaluee") is the
    correct answer and is exempt. Silent when the context prints no time unit
    at all: nothing structured to rest on."""
    if not _has(claim["markers"], "compatibility") or claim.get("hedged"):
        return None
    low = prov._norm(claim["text"])
    if not any(w in low for w in _MORPHOKINETIC_WORDS):
        return None
    unit = info.get("context_time_unit")
    if not unit or unit.lower() not in ("unknown/unverified", "unknown", "unverified"):
        return None
    return Outcome(
        rule="R9c_morphokinetic_compatibility_requires_time_basis",
        status=NOT_ASSESSABLE,
        reason_code=UNVERIFIED_TIME_BASIS,
        qualification=(f"Un verdict de compatibilite morphocinetique compare un instant observe "
                       f"a des reperes exprimes en heures post-insemination ; le contexte imprime "
                       f"time_unit = « {unit} », donc aucune comparaison n'a de base ici, "
                       f"ni dans le sens compatible ni dans le sens incompatible."),
        limitations=["IC2025-CP1-01", "IC2025-LIM-09"])


def r10_no_insemination_assumption(claim: Dict, ctx: Dict, corpus: Corpus,
                                   info: Dict) -> Optional[Outcome]:
    """Table 1's reference timings are given SEPARATELY for ICSI and IVF; the
    dataset does not record which applies. Source: IC2025-T1, IC2025-TIME-01."""
    low = prov._norm(claim["text"])
    if not any(w in low for w in ("icsi", "ivf", "fiv ", "insemination")):
        return None
    return Outcome(
        rule="R10_no_insemination_assumption",
        status=NOT_ASSESSABLE,
        reason_code=UNKNOWN_INSEMINATION_METHOD,
        qualification=("Les repères de la Table 1 sont distincts pour ICSI et IVF ; le mode "
                       "d'insemination n'est pas enregistre dans ce jeu de donnees, donc "
                       "aucun des deux jeux de valeurs ne peut etre retenu."),
        limitations=["IC2025-T1", "IC2025-TIME-01"])


# ---------------------------------------------------------------------------
# R12 -- a phase-return claim is checked against the printed stability block
# (added v1.3, 2026-09-14)
# ---------------------------------------------------------------------------

# Existence assertions only. A sentence that merely MENTIONS returns ("une
# analyse plus approfondie pourrait etre necessaire pour determiner les raisons
# de ces retours observes", Q8 r2) or DEFINES them ("une regression impliquerait
# un retour en arriere", Q9 r1) is not an assertion that they occurred.
_RETURN_ASSERTIONS = ("il y a des retours", "il y a eu des retours", "il y a un retour",
                      "il y a eu un retour", "on observe des retours", "on observe un retour",
                      "presente des retours", "presente un retour", "montre des retours",
                      "montre un retour", "retours sont observes", "est revenue a",
                      "est revenu a", "revient a la phase", "il y a une regression",
                      "il y a eu une regression", "on observe une regression",
                      "presente une regression", "montre une regression")
_RETURN_NEGATIONS = ("pas de retour", "aucun retour", "aucune regression", "pas de regression",
                     "0 retour", "zero retour", "sans retour", "ni retour", "n'y a pas de retour",
                     "ni saut de phase ni regression", "ni regression", "pas eu de regression",
                     "pas eu de retour")


def _context_returns(ctx: Dict) -> Optional[bool]:
    """None when no stability block is printed; otherwise whether ANY block of
    the context records a backward move -- a model phase_return / regression
    in `stability`, or an annotated transition with phase_distance < 0."""
    dyn = (ctx or {}).get("dynamic_context") or {}
    stability = None
    for key in ("series_analysis", "model_series_analysis"):
        series = dyn.get(key)
        if isinstance(series, dict) and isinstance(series.get("stability"), dict):
            stability = series["stability"]
            break
    if stability is None:
        return None
    if (stability.get("n_phase_returns") or 0) > 0 or stability.get("phase_returns"):
        return True
    if (stability.get("n_phase_regressions") or 0) > 0 or stability.get("phase_regressions"):
        return True
    events = dyn.get("get_transition_events") or {}
    for t in (events.get("transitions") or []) if isinstance(events, dict) else []:
        if isinstance(t, dict) and isinstance(t.get("phase_distance"), (int, float)) \
                and t["phase_distance"] < 0:
            return True
    return False


def r12_phase_return_claim_vs_stability(claim: Dict, ctx: Dict, corpus: Corpus,
                                        info: Dict) -> Optional[Outcome]:
    """A sentence asserting that phase RETURNS (or a regression) occur, while
    every block of the context records none -- `stability.n_phase_returns = 0`,
    `n_phase_regressions = 0`, and no annotated transition goes backward -- is
    not supported by this context. Source: the OFF/ON benchmark, Q8 r2 and r3
    ("il y a des retours vers la phase precedente", "la phase observee est
    revenue a t4") against a stability block printing 0/0. The rule checks the
    claim against the ALREADY-DERIVED block; it recomputes nothing, and it is
    silent whenever the block is absent or records a return."""
    if not _has(claim["markers"], "stability", "regression") or claim.get("hedged"):
        return None
    if _has(claim["markers"], "direct_cleavage", "reverse_cleavage", "chaotic_division"):
        return None            # a cleavage-pattern sentence is R8's, whose text already
        #                        says "un retour de phase n'est pas un reverse cleavage"
    low = prov._norm(claim["text"])
    if any(n in low for n in _RETURN_NEGATIONS):
        return None
    if not any(a in low for a in _RETURN_ASSERTIONS):
        return None
    if _context_returns(ctx) is not False:
        return None
    return Outcome(
        rule="R12_phase_return_claim_vs_stability",
        status=NOT_SUPPORTED,
        qualification=("Le contexte imprime stability.n_phase_returns = 0 et "
                       "n_phase_regressions = 0 pour la sequence predite, et aucune transition "
                       "annotee de ce contexte ne va vers une phase anterieure : aucun retour "
                       "de phase n'est etabli par les donnees disponibles."))


# ---------------------------------------------------------------------------
# R13 -- clinical caution
# ---------------------------------------------------------------------------

def r13_clinical_claims_extra_caution(claim: Dict, ctx: Dict, corpus: Corpus,
                                      info: Dict) -> Optional[Outcome]:
    """A clinical statement carries the GPR disclaimer and the
    not-a-diagnostic-approach limitation. Source: IC2025-STATUS-01,
    IC2025-LIM-01, IC2025-BLA-03/IC2025-LIM-16."""
    if claim["claim_kind"] != CLINICAL_IMPLICATION and not _has(claim["markers"], "clinical"):
        return None
    return Outcome(
        rule="R13_clinical_claims_extra_caution",
        status=NOT_ASSESSABLE,
        reason_code=OUT_OF_SCOPE,
        qualification=("Affirmation de portee clinique. Le Consensus est un document de "
                       "bonnes pratiques (GPR) : « should not be interpreted as setting a "
                       "standard of care », et l'identification d'un risque par la morphologie "
                       "« is not a diagnostic approach ». Ce systeme ne produit pas de "
                       "conclusion clinique."),
        limitations=["IC2025-STATUS-01", "IC2025-LIM-01", "IC2025-LIM-16"])


ALL_RULES = (
    r2_prediction_is_not_observation,
    r2b_transition_provenance,
    r3_annotated_skip_is_not_predicted_skip,
    r11_offset_is_not_a_lag,
    r8_definition_absent_from_corpus,
    r4_absence_of_evidence,
    # r10 before r9: when an answer names ICSI/IVF, the insemination assumption
    # is the primary defect -- no reference column can be selected at all --
    # and the unit problem is downstream of it. `most_severe()` keeps the first
    # rule among equals, so order encodes specificity here.
    r10_no_insemination_assumption,
    r9_timing_requires_unit_and_origin,
    r9b_median_is_not_a_range,
    r9c_morphokinetic_compatibility_requires_time_basis,
    r12_phase_return_claim_vs_stability,
    r13_clinical_claims_extra_caution,
    r6_association_is_not_causation,
    r5_compatible_is_not_proven,
    r14_consensus_never_validates_a_prediction,
    r1_observation_is_not_interpretation,
    r7_consensus_silent,
)


def most_severe(outcomes: List[Outcome]) -> Optional[Outcome]:
    if not outcomes:
        return None
    return max(outcomes, key=lambda o: _SEVERITY[o.status])
