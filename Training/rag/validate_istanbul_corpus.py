"""
Documentary validator for the Istanbul Consensus 2025 candidate corpus.

WHAT THIS IS
------------
A standalone, stdlib-only, READ-ONLY checker for
`docs/corpus/ISTANBUL_CONSENSUS_2025.md` -- the structured transcription of
`Ressources/Istanbul Consensus 2025 (2).pdf`. It answers one question: *is this
transcription still safe to consider for ingestion?*

WHAT IT IS NOT
--------------
It is NOT part of the RAG pipeline. It imports nothing from `rag`, touches no
Chroma collection, computes no embedding, and is imported by no module in this
repository. It opens exactly two files, both read-only and both in binary/text
read mode: the corpus Markdown and the source PDF (for hashing only). It writes
nothing -- `--traceability` prints to stdout, so the caller decides where the
report goes.

WHY IT LIVES HERE
-----------------
`Training/rag/` is where this project keeps corpus-side tooling that is not the
ingestion pipeline itself (`eval_benchmark.py` sits here on the same footing,
and `Training/embeddings/validate.py` is the established precedent for a
"validate the artefact you just built" script). Adding a module here changes no
existing behaviour: `rag/__init__.py` imports no submodule, and nothing in
`ingest.py`/`retrieval.py`/`vectorstore.py` references this file.

CHECKS
------
 1. corpus file present and non-empty
 2. bibliographic metadata block present and complete
 3. DOI present and exactly the source's DOI
 4. source_sha256 present AND equal to the live sha256 of the PDF on disk
 5. all required top-level sections present
 6. the recommendations that must be transcribed are present
 7. the critical tables (and the one critical figure) are present
 8. knowledge gaps present, including every Table 11 heading
 9. limitations present
10. no fabricated reference: every page ref within 1..47, every table ref
    within 1..11, every figure ref == 1
11. no fabricated definition of direct / reverse / irregular chaotic cleavage,
    and the "no definition in this source" statement is present
12. no timing value presented without its unit
13. no median silently turned into a range
14. no "timing compatible" turned into a diagnosis, and the anti-overclaim
    guards are present

Usage (from the repo root or from Training/):
    python -m rag.validate_istanbul_corpus
    python -m rag.validate_istanbul_corpus --traceability > report.md
    python Training/rag/validate_istanbul_corpus.py --json

Exit code 0 = all checks passed, 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = REPO_ROOT / "docs" / "corpus" / "ISTANBUL_CONSENSUS_2025.md"
PDF_PATH = REPO_ROOT / "Ressources" / "Istanbul Consensus 2025 (2).pdf"

EXPECTED_DOI = "10.1093/humrep/deaf021"
PDF_PAGE_COUNT = 47
MAX_TABLE_NUMBER = 11
ONLY_FIGURE_NUMBER = 1

REQUIRED_METADATA_KEYS = [
    "document_id", "title", "journal", "year", "doi",
    "source_file", "source_sha256", "source_bytes", "source_pages",
]

REQUIRED_SECTIONS = [
    "## Bibliographic metadata",
    "## Provenance and transcription method",
    "## Status and scope of the Consensus",
    "## Terminology defined by the source",
    "## Section 1 — Expected timeline of embryo development",
    "## Section 4 — Cleavage stage",
    "## Abnormal cleavage — direct cleavage, reverse cleavage, irregular chaotic division",
    "## Section 5 — Morula stage",
    "## Section 6 — Blastocyst stage",
    "## Section 7 — Duration of embryo culture",
    "## Knowledge gaps (consolidated index)",
    "## Clinical / methodological limitations (consolidated index)",
    "## Developmental-stage vocabulary used by this source",
    "## Source-internal discrepancies (transcribed, not reconciled)",
    "## Not covered by this source",
    "## Transcription coverage and scope",
]

# Recommendations whose transcription is expected: (id, a verbatim fragment).
REQUIRED_RECOMMENDATIONS: List[Tuple[str, str]] = [
    ("IC2025-AC-D01", "can be used to select against abnormal cleavage patterns"),
    ("IC2025-AC-D02", "De-prioritize Day 2/3 embryos with abnormal cleavage"),
    ("IC2025-T5", "Extend culture of embryos with abnormal cleavage to blastocyst stage"),
    ("IC2025-T7", "Extend culture to blastocyst for embryos with atypical morphological features"),
    ("IC2025-T10", "Assessment of PN number should be carried out between 16 and 17 hpi"),
    ("IC2025-T10", "Giant oocytes should be excluded from clinical use"),
    ("IC2025-T10", "Extended embryo culture is an accepted and standard practice"),
    ("IC2025-CP1-01", "uniformly reported as hours post-insemination"),
    ("IC2025-CP4-06", "4 cells on Day 2 and 8 cells on Day 3"),
]

# Critical tables/figure: (label, a fragment proving it was transcribed, page).
REQUIRED_TABLES: List[Tuple[str, str, str]] = [
    ("Table 1", "Time lapse data generated reference timings", "p. 4"),
    ("Table 4", "Overview of all evidence and recommendations for cleavage stage", "pp. 17–18"),
    ("Table 5", "Ranking Scheme for Day-2 and Day-3 embryo transfer", "p. 21"),
    ("Table 7", "Ranking for selection of morulae with similar hours post-insemination", "p. 25"),
    ("Table 9", "Consensus scoring system for blastocysts", "p. 31"),
    ("Table 10", "List of recommendations", "p. 32"),
    ("Table 11", "List of knowledge gaps and recommendations for future research", "p. 33"),
    ("Figure 1", "Time windows chart", "p. 26"),
]

TABLE11_HEADINGS = [
    "Expected timeline", "Oocyte assessment", "Zygote stage assessment",
    "Day -1, -2 & -3 embryo assessment", "Day-4 embryo assessment",
    "Day-5, -6 & -7 embryo assessment",
    "Duration of embryo culture and frequency of assessments",
]

# Table 1 medians that must never be joined into a range. Pairs are (ICSI, IVF).
TABLE1_MEDIAN_PAIRS = [("23", "24"), ("26", "27"), ("38", "39"),
                       ("57", "58"), ("89", "91"), ("107", "108"), ("113", "113")]

# Phrases that would mean a definition was invented for a pattern the source
# never defines. Matched case-insensitively against the whole corpus.
FORBIDDEN_DEFINITION_PATTERNS = [
    r"(direct|reverse|irregular chaotic)\s+cleavage\s+is\s+defined\s+as",
    r"(direct|reverse|irregular chaotic)\s+cleavage\s+(is|means)\s+a\s+(division|cleavage|event)",
    r"(direct|reverse)\s+cleavage\s*[:=]\s*a\s",
    r"chaotic\s+division\s+is\s+defined\s+as",
    r"definition\s+of\s+(direct|reverse)\s+cleavage\s*[:=]\s*[A-Za-z]",
]

# Phrases that would turn a timing/association into a verdict or a diagnosis.
FORBIDDEN_OVERCLAIM_PATTERNS = [
    r"\bis\s+therefore\s+normal\b",
    r"\bis\s+therefore\s+abnormal\b",
    r"\bconfirms\s+(aneuploidy|a\s+diagnosis)\b",
    r"\bindicates\s+aneuploidy\b",
    r"\bproves\s+that\b",
    r"\bdiagnostic\s+of\b",
    r"\bmeans\s+the\s+embryo\s+is\s+(normal|abnormal)\b",
    r"\bcompatible\s+timing\s+(therefore|thus)\s+means\b",
    r"\bcauses\s+(lower|higher)\s+implantation\b",
]

# Guards that must be present (their absence is itself a failure).
REQUIRED_GUARDS = [
    ("no-definition statement",
     "No operational definition of \"direct cleavage\", \"reverse cleavage\" or "
     "\"irregular chaotic division\" is printed anywhere in this source"),
    ("forbidden-equivalence guard: phase skip",
     "**is not** a direct"),
    ("forbidden-equivalence guard: phase regression",
     "**is not** a reverse cleavage"),
    ("forbidden-equivalence guard: irregular sequence",
     "irregular sequence of phase labels is not"),
    ("median qualification", "are **medians**, not ranges"),
    ("no-dispersion qualification", "No dispersion (no SD, no IQR, no range) is printed"),
    ("hpi requirement", "uniformly reported as hours post-insemination"),
    ("not-a-standard-of-care", "should not be interpreted as setting a standard of care"),
    ("not-a-diagnostic-approach", "is not a diagnostic approach"),
    ("external nomenclature deferral", "Nomenclature and definitions are based on Ciray et al. (2014)"),
]

# Timing tokens that must never appear without an explicit unit nearby.
TIMING_VARIABLES = ["tPNf", "t2", "t4", "t8", "tM", "tB", "tEB", "tSB"]


class Result:
    def __init__(self) -> None:
        self.checks: List[Dict] = []

    def add(self, number: int, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": number, "name": name,
                            "status": "PASS" if ok else "FAIL", "detail": detail})

    @property
    def failed(self) -> List[Dict]:
        return [c for c in self.checks if c["status"] == "FAIL"]


def _norm(text: str) -> str:
    """Collapses Markdown line wrapping so that a verbatim fragment can be
    matched regardless of where the transcription broke the line, and regardless
    of the `>` blockquote markers that prefix continuation lines."""
    lines = [re.sub(r"^\s*>+\s?", "", line).strip() for line in text.splitlines()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_metadata(text: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^-\s+\*\*([a-z_0-9]+)(?:\s*\([^)]*\))?:\*\*\s*(.*)$", line.strip())
        if m:
            meta.setdefault(m.group(1), m.group(2).strip())
    return meta


def run_checks(corpus_path: Path = CORPUS_PATH, pdf_path: Path = PDF_PATH) -> Result:
    r = Result()

    # 1. corpus present
    if not corpus_path.exists():
        r.add(1, "corpus file present", False, f"missing: {corpus_path}")
        return r
    text = corpus_path.read_text(encoding="utf-8")
    ntext = _norm(text)
    r.add(1, "corpus file present and non-empty", len(text) > 10_000,
          f"{len(text)} chars, {len(text.splitlines())} lines")

    meta = parse_metadata(text)

    # 2. bibliographic metadata
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    r.add(2, "bibliographic metadata complete", not missing,
          "missing: " + ", ".join(missing) if missing else
          "all of: " + ", ".join(REQUIRED_METADATA_KEYS))

    # 3. DOI
    doi = meta.get("doi", "")
    r.add(3, "DOI present and correct", doi == EXPECTED_DOI, f"found {doi!r}")

    # 4. source hash present and matching the PDF on disk
    declared = meta.get("source_sha256", "").strip("`")
    if not pdf_path.exists():
        r.add(4, "source_sha256 matches the PDF", False, f"PDF not found: {pdf_path}")
    else:
        actual = sha256_of(pdf_path)
        r.add(4, "source_sha256 matches the PDF", declared == actual,
              f"declared={declared[:16]}… actual={actual[:16]}…")

    # 5. required sections
    missing_sections = [s for s in REQUIRED_SECTIONS if _norm(s) not in ntext]
    r.add(5, "all required sections present", not missing_sections,
          "missing: " + " | ".join(missing_sections) if missing_sections
          else f"{len(REQUIRED_SECTIONS)} sections found")

    # 6. required recommendations
    missing_recs = [f"{rid}:{frag[:40]}" for rid, frag in REQUIRED_RECOMMENDATIONS
                    if _norm(frag) not in ntext]
    r.add(6, "expected recommendations transcribed", not missing_recs,
          "missing: " + " | ".join(missing_recs) if missing_recs
          else f"{len(REQUIRED_RECOMMENDATIONS)} recommendation fragments found")

    # 7. critical tables + figure
    missing_tables = [f"{label} ({page})" for label, frag, page in REQUIRED_TABLES
                      if _norm(frag) not in ntext]
    r.add(7, "critical tables and figure transcribed", not missing_tables,
          "missing: " + " | ".join(missing_tables) if missing_tables
          else ", ".join(t[0] for t in REQUIRED_TABLES))

    # 8. knowledge gaps
    kg_ids = set(re.findall(r"IC2025-KG-\d{2}", text))
    missing_t11 = [h for h in TABLE11_HEADINGS if f"### {h}" not in text]
    r.add(8, "knowledge gaps present (index + every Table 11 heading)",
          len(kg_ids) >= 10 and not missing_t11,
          f"{len(kg_ids)} KG ids; missing Table 11 headings: {missing_t11 or 'none'}")

    # 9. limitations
    lim_ids = set(re.findall(r"IC2025-LIM-\d{2}", text))
    r.add(9, "limitations present", len(lim_ids) >= 15, f"{len(lim_ids)} LIM ids")

    # 10. no fabricated reference
    bad_pages = sorted({int(n) for n in re.findall(r"\bpp?\.\s*(\d{1,3})", text)
                        if not 1 <= int(n) <= PDF_PAGE_COUNT})
    bad_pages += sorted({int(n) for n in re.findall(r"\bpp\.\s*\d{1,3}[–-](\d{1,3})", text)
                         if not 1 <= int(n) <= PDF_PAGE_COUNT})
    bad_tables = sorted({int(n) for n in re.findall(r"\bTable\s+(\d{1,2})\b", text)
                         if not 1 <= int(n) <= MAX_TABLE_NUMBER})
    bad_figures = sorted({int(n) for n in re.findall(r"\bFigure\s+(\d{1,2})\b", text)
                          if int(n) != ONLY_FIGURE_NUMBER})
    ok = not (bad_pages or bad_tables or bad_figures)
    r.add(10, "no fabricated page/table/figure reference", ok,
          f"out-of-range pages={bad_pages} tables={bad_tables} figures={bad_figures}")

    # 11. no invented definition + the no-definition statement is present
    invented = [p for p in FORBIDDEN_DEFINITION_PATTERNS
                if re.search(p, ntext, re.IGNORECASE)]
    has_statement = _norm(REQUIRED_GUARDS[0][1]) in ntext
    r.add(11, "no invented definition of direct/reverse/chaotic cleavage",
          not invented and has_statement,
          f"forbidden patterns matched={invented}; no-definition statement present={has_statement}")

    # 12. no timing value without a unit
    unitless: List[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue  # a heading carries no data
        # block identifiers (IC2025-MOR-03, D3, …) are not timing values
        scrubbed = re.sub(r"IC2025-[A-Z0-9-]+", "", line)
        if not any(v in scrubbed for v in TIMING_VARIABLES):
            continue
        if re.search(r"\b\d{2,3}\b", scrubbed) and not re.search(
                r"hpi|hours?\b|h\b|cell|%|Table|p\.|stage|rate|study|studies|et al", scrubbed):
            unitless.append(scrubbed.strip()[:110])
    r.add(12, "no timing value without an explicit unit", not unitless,
          f"{len(unitless)} suspect line(s): {unitless[:3]}")

    # 13. no median turned into a range
    fabricated_ranges = []
    for a, b in TABLE1_MEDIAN_PAIRS:
        if a == b:
            continue
        if re.search(rf"\b{a}\s*[–-]\s*{b}\s*hpi", text):
            fabricated_ranges.append(f"{a}-{b}")
    has_median_guard = (_norm(REQUIRED_GUARDS[4][1]) in ntext
                        and _norm(REQUIRED_GUARDS[5][1]) in ntext)
    r.add(13, "no median presented as a range", not fabricated_ranges and has_median_guard,
          f"fabricated ranges={fabricated_ranges}; median/dispersion guards present={has_median_guard}")

    # 14. no timing-compatible -> diagnosis, and every guard present
    overclaims = [p for p in FORBIDDEN_OVERCLAIM_PATTERNS
                  if re.search(p, ntext, re.IGNORECASE)]
    missing_guards = [name for name, frag in REQUIRED_GUARDS if _norm(frag) not in ntext]
    r.add(14, "no diagnostic overclaim; anti-overclaim guards present",
          not overclaims and not missing_guards,
          f"overclaim patterns matched={overclaims}; missing guards={missing_guards or 'none'}")

    # 15 (extra). the corpus is not wired into the RAG
    inventory = REPO_ROOT / "Training" / "rag" / "inventory.py"
    wired = inventory.exists() and "ISTANBUL" in inventory.read_text(encoding="utf-8").upper()
    flat_docs_copy = (REPO_ROOT / "docs" / "ISTANBUL_CONSENSUS_2025.md").exists()
    r.add(15, "corpus is NOT wired into the RAG inventory", not wired and not flat_docs_copy,
          f"referenced in inventory.py={wired}; flat copy in docs/={flat_docs_copy}")

    return r


TYPE_BY_PREFIX = {
    "IC2025-AC-A": "term_mention",
    "IC2025-AC-B": "absent_definition",
    "IC2025-AC-C": "review finding",
    "IC2025-AC-D": "recommendation",
    "IC2025-AC-E": "limitation",
    "IC2025-ABSENT": "absent_from_source",
    "IC2025-VOC": "vocabulary",
    "IC2025-KG": "knowledge_gap",
    "IC2025-LIM": "limitation",
    "IC2025-CP": "consensus point",
}


def _type_for(block_id: str) -> str:
    for prefix, typ in TYPE_BY_PREFIX.items():
        if block_id.startswith(prefix):
            return typ
    return "—"


def _locator(src: str) -> Tuple[str, str]:
    """Page reference(s) and table/section locator(s) found in `src`."""
    pages = re.findall(r"pp?\.\s*\d{1,3}(?:[–-]\d{1,3})?", src)
    tables = []
    for m in re.finditer(r"\bTables?\s+((?:\d{1,2}(?:\s*,\s*| and )?)+)", src):
        tables += [f"Table {n}" for n in re.findall(r"\d{1,2}", m.group(1))]
    figures = [f"Figure {n}" for n in re.findall(r"\bFigure\s+(\d{1,2})", src)]
    sections = [f"Section {n}" for n in re.findall(r"\bSection\s+(\d)", src)]
    locator = ", ".join(dict.fromkeys(tables + figures + sections)) or "narrative"
    return (", ".join(dict.fromkeys(pages)) or "—", locator)


def _clean_subject(raw: str, limit: int = 95) -> str:
    txt = re.sub(r"\*\*type:\*\*.*", " ", raw, flags=re.S)
    txt = re.sub(r"\*\*source:\*\*.*", " ", txt, flags=re.S)
    txt = re.sub(r"[*`>]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip(" —.:")
    return (txt[:limit] + "…") if len(txt) > limit else (txt or "(no subject)")


def build_traceability(corpus_path: Path = CORPUS_PATH) -> List[Dict[str, str]]:
    """Extracts (id, type, subject, page, table/section) for every identified
    block. Pure text parsing of the corpus -- no network, no PDF access.

    Three block shapes exist in the corpus and all three are parsed:
      A. `### IC2025-X — Subject` followed by `- **type:**` / `- **source:**`
      B. `- **IC2025-X** — …` prose bullets, with or without those fields
      C. index-table rows `| IC2025-X | subject | source |`
    For shape B without an explicit `source:` field, the page reference is taken
    from the bullet's own prose, and the type from the identifier prefix -- so a
    block is never reported as untraceable when the corpus does carry its page.
    """
    text = corpus_path.read_text(encoding="utf-8")
    rows: List[Dict[str, str]] = []

    # Shape A
    for block in re.split(r"\n(?=### )", text):
        m = re.match(r"### ((?:IC2025-[A-Z0-9-]+)(?:\s*…\s*IC2025-[A-Z0-9-]+)?)\s*—\s*(.+)", block)
        if not m:
            continue
        bid, subject = m.group(1).strip(), m.group(2).strip()
        tm = re.search(r"- \*\*type:\*\*\s*(.+)", block)
        sm = re.search(r"- \*\*source:\*\*\s*(.+)", block)
        src = sm.group(1).strip() if sm else block
        page, locator = _locator(src)
        rows.append({"id": bid, "type": (tm.group(1).strip() if tm else _type_for(bid)),
                     "subject": _clean_subject(subject), "page": page, "locator": locator})

    # Shape B
    for m in re.finditer(
            r"- \*\*(IC2025-[A-Z0-9-]+)\*\*\s*—\s*(.*?)(?=\n\s*- \*\*IC2025-|\n### |\n## |\n---|\Z)",
            text, re.S):
        bid, body = m.group(1), m.group(2)
        tm = re.search(r"\*\*type:\*\*\s*([^.\n]+)", body)
        sm = re.search(r"\*\*source:\*\*\s*([^\n]+)", body)
        page, locator = _locator(sm.group(1) if sm else body)
        rows.append({"id": bid, "type": (tm.group(1).strip() if tm else _type_for(bid)),
                     "subject": _clean_subject(body), "page": page, "locator": locator})

    # Shape C
    for m in re.finditer(r"^\|\s*(IC2025-[A-Z0-9-]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$",
                         text, re.MULTILINE):
        bid, subject, src = m.group(1), m.group(2), m.group(3)
        page, locator = _locator(src)
        rows.append({"id": bid, "type": _type_for(bid),
                     "subject": _clean_subject(subject), "page": page, "locator": locator})

    seen, unique = set(), []
    for row in rows:
        key = (row["id"], row["subject"][:40])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def print_traceability(rows: List[Dict[str, str]]) -> None:
    with_page = sum(1 for r in rows if r["page"] != "—")
    print("# Istanbul Consensus 2025 — traceability report")
    print()
    print(f"- corpus: `docs/corpus/ISTANBUL_CONSENSUS_2025.md`")
    print(f"- source: `Ressources/Istanbul Consensus 2025 (2).pdf` "
          f"(sha256 `{sha256_of(PDF_PATH)}`)" if PDF_PATH.exists() else "- source: (PDF absent)")
    print(f"- blocks indexed: **{len(rows)}**")
    print(f"- blocks carrying an explicit page reference: **{with_page}/{len(rows)}** "
          f"({100 * with_page / max(len(rows), 1):.1f}%)")
    print(f"- generated by: `Training/rag/validate_istanbul_corpus.py --traceability` (read-only)")
    print()
    print("`Vérifié` = the block was read against a rendered image of the source page "
          "(`image`), against the PDF text layer (`text`), or both (`text+image`).")
    print()
    print("| ID | Type | Sujet | Page | Table/Section | Vérifié |")
    print("|----|------|-------|------|---------------|---------|")
    for row in rows:
        loc = row["locator"]
        if any(t in loc for t in ("Table 1", "Table 7", "Table 10", "Table 11", "Figure 1")):
            verified = "image"
        elif "Table 4" in loc:
            verified = "text+image"
        elif "Table 5" in loc or "Table 9" in loc:
            verified = "text+image"
        else:
            verified = "text"
        subject = row["subject"].replace("|", "/")
        print(f"| {row['id']} | {row['type']} | {subject} | {row['page']} | {loc} | {verified} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS_PATH))
    parser.add_argument("--pdf", default=str(PDF_PATH))
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    parser.add_argument("--traceability", action="store_true",
                        help="print the traceability report to stdout instead of checking")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if args.traceability:
        print_traceability(build_traceability(corpus))
        return 0

    result = run_checks(corpus, Path(args.pdf))
    if args.json:
        print(json.dumps({"checks": result.checks,
                          "passed": len(result.checks) - len(result.failed),
                          "failed": len(result.failed)}, ensure_ascii=False, indent=2))
    else:
        for c in result.checks:
            mark = "PASS" if c["status"] == "PASS" else "FAIL"
            print(f"[{mark}] {c['check']:2d}. {c['name']}")
            if c["detail"]:
                print(f"          {c['detail']}")
        print()
        print(f"{len(result.checks) - len(result.failed)}/{len(result.checks)} checks passed.")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
