"""
Claim extraction, v1: pragmatic, auditable, NOT an NLP model.

Same discipline as `router.py`: sentence splitting plus an explicit marker
table a reader can check line by line. The goal is not perfect claim
segmentation -- it is to find the sentences that ASSERT something checkable,
and to hand them on WITH THEIR ORIGINAL TEXT. A claim is never rewritten before
validation; `Claim.claim` is a verbatim slice of the answer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .provenance import PHASE_TOKEN, _norm, framing_classes
from .schema import (ASSOCIATION, CAUSAL_CLAIM, CLINICAL_IMPLICATION,
                     COMPATIBILITY_VERDICT, DATA_STATEMENT, DEFINITION,
                     PREDICTION_STATEMENT, TIMING_REFERENCE)

# v1.3 (2026-09-14): ":" is no longer a sentence boundary. In the VALIDATOR
# OFF/ON benchmark (Q4 r2, Q6 r3) a distribution listing such as
# "* t8 : 0.2367988 (la deuxieme phase la plus probable)" was cut into the
# fragment "* t8 :" -- a bare phase token stripped of the value and of the
# vocabulary that attributed it to the model -- and R2 fired 12 times on one
# correct answer. A colon joins a label to its value; the two are one claim.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")

# v1.3 -- a line that continues a list opened by the previous sentence.
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)]|W\d+\b|t(?:PB2|PNa|PNf|M|SB|EB|B|\d\+?)\s*[:=])",
                        re.IGNORECASE)
# v1.3 -- a sentence whose grammatical subject is the previous sentence's
# ("La sequence predite ... . Elle se compose de t7." -- Q9 r1). Narrow and
# explicit: pronoun subjects only, never a general coreference model.
_PRONOUN_START = re.compile(
    r"^\s*(?:elle|elles|il|ils|celle-ci|celui-ci|celles-ci|ceux-ci|cette derniere|"
    r"ce dernier|cela|ceci|c'est)\b", re.IGNORECASE)

# Topic markers -> the thing the sentence is talking about. A sentence with no
# marker at all carries no checkable claim and is skipped.
MARKERS: Dict[str, Tuple[str, ...]] = {
    "phase": ("phase", "etape", "stade"),
    "transition": ("transition", "passage", "changement de phase", "->", "→"),
    "timing": ("instant", "heure", "minute", "seconde", "jour", "hpi", "duree",
               "depuis", "temps", "fenetre", "moment", "timing", "delai", "decalage"),
    "probability": ("probabilite", "probability", "posterior", "distribution",
                    "confiance", "%"),
    "uncertainty": ("entropie", "entropy", "incertitude", "incertain"),
    "stability": ("stable", "stabilite", "retour", "retours"),
    "skip": ("saut", "skip", "saute"),
    "regression": ("regression", "regresse", "retour en arriere"),
    "expected_development": ("ordre attendu", "developpement attendu", "referentiel",
                             "chronologique", "prochaine etape", "sequence attendue"),
    "compatibility": ("compatible", "incompatible", "coherent", "conforme",
                      "en ligne avec"),
    "direct_cleavage": ("direct cleavage", "clivage direct", "division directe"),
    "reverse_cleavage": ("reverse cleavage", "clivage inverse", "division inverse"),
    "chaotic_division": ("division chaotique", "chaotic division", "clivage chaotique",
                         "division irreguliere", "irregular chaotic"),
    "biological": ("embryon", "biologique", "morphocinetique", "morphokinetic",
                   "cellulaire", "blastomere", "clivage", "cleavage", "aneuploidie",
                   "implantation", "naissance", "viabilite"),
    # v1.2: "transfer" stem covers transfert / transfere / transferer (the
    # rules-matrix test "doit etre transfere" was missed by the noun only).
    "clinical": ("clinique", "transfer", "selection", "patient", "diagnostic",
                 "prognostic", "pronostic"),
    # v1.1: "entraine" alone matched "entraine sur" (= trained on, Q7 compact_v1
    # replay: "le modele n'a pas ete entraine sur des transitions annotees" was
    # flagged as a causal claim). The causal verb is kept through its
    # determiner forms only.
    "causal": ("cause", "causes", "provoque", "entraine un", "entraine une",
               "entraine des", "entraine le", "entraine la", "entraine l'",
               "parce que le modele", "donc l'embryon", "ce qui prouve", "demontre"),
}

_ASSERTIVE_HEDGES = ("non calculable", "pas possible de determiner", "indisponible",
                     "ne permet pas", "pas suffisamment d'informations",
                     "n'est pas disponible", "impossible de",
                     # v1.1: "ne peut pas etre considere comme atypique" is a
                     # declined verdict, not a verdict (Q14 compact_v1 replay).
                     "ne peut pas etre", "ne peut etre", "n'est pas possible",
                     "pas possible de",
                     # v1.3: "il n'y a pas de moyen de savoir si la phase suivante
                     # est t7 ou non" (Q15 r2) asserts nothing about t7 -- it
                     # declines. R2 fired NOT_SUPPORTED on it.
                     "pas de moyen de savoir", "aucun moyen de savoir", "impossible de savoir",
                     "on ne sait pas", "n'est pas certain", "pas possible de savoir",
                     "sans plus d'informations", "sans davantage d'informations",
                     "ne permet pas de savoir", "ne permet pas de determiner",
                     # v1.3: a conditional or modal sentence is not a verdict --
                     # "si le decalage se situe a l'interieur..., alors il pourrait
                     # etre considere comme..." / "il est possible que ce decalage
                     # releve de la variabilite biologique connue" (Q14 r1, r2).
                     "il est possible que", "il se peut que", "pourrait etre considere",
                     "pourrait etre consideree", "peut etre une explication",
                     "pourrait relever", "pourrait correspondre", "peut correspondre",
                     "peut-etre")

# v1.1: sample identifiers such as `Patient_319#val` are not clinical vocabulary
# (Q12 compact_v1 replay: R13 fired on "pour le patient Patient_319#val").
_IDENTIFIER = re.compile(r"(?:patient\s+)?patient_\d+\S*")


def _kind_for(markers: List[str], text_low: str) -> str:
    if "causal" in markers:
        return CAUSAL_CLAIM
    if "clinical" in markers:
        return CLINICAL_IMPLICATION
    if "compatibility" in markers or "expected_development" in markers:
        return COMPATIBILITY_VERDICT
    if any(m in markers for m in ("direct_cleavage", "reverse_cleavage", "chaotic_division")):
        return DEFINITION
    if "timing" in markers and ("phase" in markers or "transition" in markers):
        return TIMING_REFERENCE
    if any(k in text_low for k in ("predit", "predite", "prevoit", "prediction")):
        return PREDICTION_STATEMENT
    if "probability" in markers or "uncertainty" in markers:
        return DATA_STATEMENT
    if "biological" in markers and "phase" not in markers:
        return ASSOCIATION
    return DATA_STATEMENT


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """(sentence, start, end) triples over the ORIGINAL string, so every claim
    keeps a span back into the answer."""
    out: List[Tuple[str, int, int]] = []
    pos = 0
    for piece in _SENTENCE_SPLIT.split(text or ""):
        if not piece:
            continue
        start = (text or "").find(piece, pos)
        if start < 0:
            start = pos
        end = start + len(piece)
        pos = end
        stripped = piece.strip()
        if stripped:
            out.append((stripped, start, end))
    return out


def markers_in(sentence: str) -> List[str]:
    low = _IDENTIFIER.sub(" ", _norm(sentence))
    found = [topic for topic, words in MARKERS.items() if any(w in low for w in words)]
    if PHASE_TOKEN.search(sentence) and "phase" not in found:
        found.append("phase")
    return sorted(found)


def is_question(sentence: str) -> bool:
    """v1.3: a sentence ending in "?" asserts nothing. The benchmark answers
    routinely restate the question first ("Le modele predit-il la transition
    avant ou apres l'annotation ?", Q7 r1; Q2, Q8, Q14), and the restated
    question was validated as if it were a claim."""
    return (sentence or "").rstrip().endswith("?")


def extract_claims(answer_text: str) -> List[Dict]:
    """Every sentence that asserts something checkable, verbatim, with its
    span, its topic markers, its claim_kind, and whether it hedges.

    v1.3 adds `inherited_framing`: the provenance classes the sentence
    inherits from its discourse position when it carries none of its own --
    (a) a list item inherits from the sentence that opened the list with ":"
    ("...la quantite `phase_probabilities` du modele ... est la suivante :"
    followed by "* t8 : 0.2367..."), (b) a sentence whose subject is a
    pronoun inherits from the sentence before it in the same paragraph. Own
    framing, when present, is never overridden."""
    claims: List[Dict] = []
    text = answer_text or ""
    introducer: Set[str] = set()       # framing of the sentence that opened a list
    in_list = False
    prev_framing: Set[str] = set()
    prev_end = 0
    for sentence, start, end in split_sentences(text):
        own = framing_classes(sentence)
        same_paragraph = "\n\n" not in text[prev_end:start]
        inherited: Set[str] = set()
        if _LIST_ITEM.match(sentence):
            if introducer and not own:
                inherited = set(introducer)
            in_list = bool(introducer)
        else:
            if in_list:
                introducer, in_list = set(), False
            if _PRONOUN_START.match(sentence) and same_paragraph and not own:
                inherited = set(prev_framing)
            if sentence.rstrip().endswith(":"):
                introducer = set(own)
        prev_framing = set(own) if own else set(inherited)
        prev_end = end

        markers = markers_in(sentence)
        if not markers or is_question(sentence):
            continue
        low = _norm(sentence)
        claims.append({
            "text": sentence,
            "span": {"start": start, "end": end},
            "markers": markers,
            "claim_kind": _kind_for(markers, low),
            "hedged": any(h in low for h in _ASSERTIVE_HEDGES),
            "phases": sorted({m.group(0) for m in PHASE_TOKEN.finditer(sentence)}),
            "inherited_framing": sorted(inherited),
        })
    return claims
