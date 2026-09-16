"""
RAG ingestion foundation (Product Roadmap Phase 3) -- ADDITIVE only.

Never imports from, modifies, or is imported by Training/evaluation/,
Training/embeddings/, or Training/reporting/. The RAG corpus is static
documentary knowledge (docs/*.md); it never stores current predictions,
Semi-HMM outputs, or GRU scores -- see docs/RAG_ARCHITECTURE.md's
"dynamic data: never in the vector DB" rule, which this package enforces
structurally (the document inventory in inventory.py only ever lists
docs/*.md paths, never Results/evaluation/*/analysis.json or any
per-window artifact).
"""
