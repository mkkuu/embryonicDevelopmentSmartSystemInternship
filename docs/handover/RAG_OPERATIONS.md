# RAG operations — corpus, index, retrieval, provenance

Written 2026-09-17 from the code (`Training/rag/`, `Training/orchestrator/`) and the verified
state of the server. Design rationale lives in the RAG documents `docs/RAG_ARCHITECTURE.md`
and `docs/RAG_DATA_MODEL.md` (frozen, partly historical) and in
`docs/archive/reports/RAG_PHASE_3_REPORT.md`.

## 1. What is in the corpus

| Corpus | Location | Read by | In the vector index? |
|---|---|---|---|
| The 30 project documents | `docs/*.md`, listed by path in `Training/rag/inventory.py::INCLUDED` | `rag/ingest.py` (index), `orchestrator/compact_context.py` (17 of them named in curation rules) | **yes** |
| Istanbul Consensus 2025 | `docs/corpus/ISTANBUL_CONSENSUS_2025.md` (138 page-referenced blocks) + `_TRACEABILITY.md`; source PDF `Ressources/Istanbul Consensus 2025 (2).pdf` | `validator/corpus.py` (lexical search), `orchestrator/scientific_reference.py` (`[SCIENTIFIC]` context block) | **no** (read directly) |

### INCLUDED / EXCLUDED semantics

`inventory.py` is a hand-reviewed list, not a directory scan:

- `INCLUDED` — 30 entries, each with `document_type` (scientific / project / methodological),
  `status` (authoritative / current / historical / experimental), `authoritative` flag,
  optional `supersedes`, and a written `reason`. Only these are ingested.
- `EXCLUDED` — a dict `path → reason` for every other top-level `docs/*.md` (currently the
  six successor guides and `docs/README.md`). Nothing reads it except
  `Tests/rag/test_rag_inventory.py`, which fails if a top-level `docs/*.md` is in neither list
  or if a listed file is missing. That is why new documentation goes in `docs/handover/` (a
  subdirectory, outside the `docs/*.md` glob) rather than at the top level.
- Structurally, no code path in `rag/` can list `Results/`, `Embeddings/`, `Cache/` or any
  per-window artefact: dynamic data is never in the vector DB.

Several INCLUDED documents are historical (roadmaps, implementation reports, an early web-app
plan marked "not implemented"). They are indexed **on purpose** (retrieval-scope decision of
2026-08-25) and can only be re-classified together with a code change, a re-ingestion and a
new benchmark identity.

## 2. Embedding model, store, index

| Element | Value | Where |
|---|---|---|
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, CPU, downloaded from Hugging Face on first use | `rag/embeddings.py` (`EMBEDDING_VERSION = "<model>@dim384"` stored with every chunk) |
| Why multilingual | the corpus is English, the users' questions are French; an English-only model failed a French query in a measured test | `docs/archive/reports/RAG_PHASE_3_REPORT.md` |
| Chunking | Markdown-aware: never across a heading, code block or table; target ≈ 1 800 characters (max 2 600), ≈ 200 overlap within a section | `rag/chunker.py` |
| Store | Chroma, persistent embedded, `RagIndex/` (gitignored, regenerable), collection `docs_v1` | `rag/vectorstore.py` (the only module that talks to Chroma) |
| Manifest | `RagIndex/manifest.json`: content hash per document, `embedding_version`, `last_run_at`, `vector_db` | written by `rag/ingest.py` |
| Size at last ingestion | 30 documents, 432 chunks (2026-08-25) | manifest, report |

Ingestion is idempotent: a re-run re-embeds only documents whose content hash changed, drops
chunks of documents no longer INCLUDED, and re-embeds everything (logged) on an
`EMBEDDING_VERSION` bump. Chroma's SQLite file changes on reads too, so do not use the
directory's mtime as evidence of re-ingestion; use the manifest.

## 3. Retrieval and query enrichment

- `rag/retrieval.py::retrieve(question, top_k=5)` embeds the query, queries `docs_v1`, returns
  the top 5 chunks with their stored metadata (document path, type, status, authoritative flag);
  optional filters on type / status / authoritative. No re-ranking, no multi-collection merge.
- `orchestrator/tools.py::retrieve_documents()` / `get_phase_information()` wrap it with typed
  errors for the orchestrator.
- `orchestrator.build_retrieval_query()` appends to the question the phase names already
  present in that turn's fetched tool outputs (string values and the keys of a
  phase-distribution dict) **only** when the question matches a small trigger vocabulary
  (collision-checked to fire on Q11/Q12). It invents nothing and hard-codes no phase list.
  Measured effect on the real index: the canonical phase-order chunk moved from rank 6 to 1 for
  Q11 and from outside the top 25 to rank 4 for Q12, every other question byte-identical.
- In the `compact_v2` context format, 15 candidates are retrieved and
  `compact_context.select_relevant_chunks()` keeps 5 using `SOURCE_RULES` (D1/D2) and section
  rules; the decisions are written to `context["retrieval_audit"]`.

Commands (from `Training/`, CPU):

```bash
CUDA_VISIBLE_DEVICES="" python -m rag.ingest                                   # build / update the index
CUDA_VISIBLE_DEVICES="" python -m rag.retrieval "Qu'est-ce que le Semi-HMM ?" --top-k 5
CUDA_VISIBLE_DEVICES="" python -m experiments.rag_llm_diagnostics.measure_retrieval_rank   # read-only rank measurement
```

## 4. Provenance in the answer

Retrieved chunks enter the context as `document_context`, physically separate from
`dynamic_context` (Reporting API) and `scientific_context` (Istanbul), each with its source path
and section. The rendered `compact_v2` block is `[DOCUMENTATION PROJET — …]`. The Validator's
provenance step (`validator/provenance.py`) resolves every claim against the context blocks,
`document_context` included, with the classes OBSERVED / MODEL-DERIVED / DERIVED-FROM-OBSERVED /
DERIVED-FROM-MODEL / SCIENTIFIC (the Istanbul block counts as SCIENTIFIC); the grounding check
treats the `[SCIENTIFIC]` block as presence only. Every benchmark artefact records `istanbul_corpus_sha256`, the RAG index
identity (`capture_run_identity`: manifest hashes, `embedding_version`, `last_run_at`,
collection) and, per question, `document_context_sha256`.

## 5. The server index is older than the repository corpus — known and intentional

Verified 2026-09-17: the server's `RagIndex/manifest.json` says `last_run_at =
2026-08-25T09:54:55Z`, 30 documents, and the server's `docs/` is the flat 39-file tree of that
date (30 RAG documents + 9 session files, now `EXCLUDED`-irrelevant). The repository, since
2026-09-16, versions the same 30 documents (content unchanged — the 2026-09-15 restructure moved
other files, never these) plus guides and `corpus/`. Every valid benchmark artefact (including
the v1.2 Validator reference and the preflight of the pending v1.3b) was produced against that
2026-08-25 index, and its `document_context` hashes match it.

Consequences:

- **Do not rebuild the index on the server before the v1.3b comparison**; a rebuild with the
  same 30 documents should give the same chunks, but `last_run_at` and the index identity
  change, and any content drift would change `document_context` and end comparability.
- **Do not modify the corpus** (any of the 30 files, `inventory.py`, `docs/corpus/`) in a
  documentation pass. Editing `inventory.py` on 2026-09-16 (EXCLUDED only, INCLUDED intact)
  already changed that module's hash in `capture_run_identity`'s list; the validator runner's
  own `_HASHED_MODULES` does not include it, so the artefact identity is unaffected.
- **Do not re-classify documents** (moving a historical RAG document to `archive/`, ingesting
  a guide) without: a commit that changes `inventory.py`, an explicit re-ingestion, a new
  `capture_run_identity`, and a new baseline run. Corpus and index changes alter experiment
  identity and must be versioned like code.
- A fresh clone can rebuild an equivalent index (`python -m rag.ingest`) from the versioned 30
  documents; verify the resulting `manifest.json` content hashes against the server's before
  treating retrieval results as comparable.

## 6. Checks

```bash
# from the repository root
pytest Tests/rag/                       # inventory ⇄ disk, chunker, hashing, schemas, ingest (mocked), retrieval (mocked)
python - <<'PY'
import sys; sys.path.insert(0, "Training")
from rag import inventory; print(inventory.verify_inventory_matches_disk())
PY
# from Training/: traceability of the Istanbul transcription against the PDF (read-only)
python -m rag.validate_istanbul_corpus
```
