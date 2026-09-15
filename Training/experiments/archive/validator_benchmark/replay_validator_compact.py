"""Read-only replay of the Scientific Validator v1 over the 90 answers of the
paired compact_v1 artefact. No LLM, no GPU, nothing written into Results/.
Run from Training/ on the GPU server."""
import json, sys, collections
sys.argv = sys.argv[:1]
try:
    from validator import validate_answer
except ImportError:
    from validator.validator import validate_answer
from validator.corpus import Corpus

ART = "../Results/evaluation/compact_context_experiment/paired_mistral-nemo_12b_compact_v1.json"
d = json.load(open(ART, encoding="utf-8"))
corpus = Corpus()
print("corpus_available =", corpus.available, "blocks =", len(getattr(corpus, "blocks", []) or []))
status_counts = {c: collections.Counter() for c in ("V2", "COMPACT")}
rule_counts = {c: collections.Counter() for c in ("V2", "COMPACT")}
r2_hits = {c: [] for c in ("V2", "COMPACT")}
out = []
for row in d["rows"]:
    ctx = {"question": row["question"], "dynamic_context": row["dynamic_context"],
           "document_context": row["document_context"], "warnings": row["context_warnings"]}
    for rep in row["replications"]:
        for cond in ("V2", "COMPACT"):
            ans = rep[cond].get("answer")
            if ans is None:
                continue
            rep_ = validate_answer(ans, ctx, corpus)
            statuses = [c.status for c in rep_.claims]
            rules = [r for c in rep_.claims for r in c.rules_fired]
            for s in statuses: status_counts[cond][s] += 1
            for r in rules: rule_counts[cond][r] += 1
            worst = max(statuses, key=lambda s: {"SUPPORTED":0,"PARTIALLY_SUPPORTED":1,"NOT_ASSESSABLE":2,"NOT_SUPPORTED":3,"CONTRADICTED":4}[s]) if statuses else None
            flagged = [(c.claim[:90], c.status, c.reason_code, c.rules_fired) for c in rep_.claims if c.status in ("NOT_SUPPORTED","CONTRADICTED")]
            na = [(c.claim[:90], c.reason_code, c.rules_fired) for c in rep_.claims if c.status == "NOT_ASSESSABLE" and c.reason_code != "out_of_scope"]
            if any("R2_prediction_is_not_observation" in c.rules_fired for c in rep_.claims):
                r2_hits[cond].append((row["question_id"], rep["replication"]))
            out.append({"q": row["question_id"], "rep": rep["replication"], "cond": cond,
                        "n_claims": len(rep_.claims), "worst": worst, "has_violation": rep_.has_violation,
                        "flagged": flagged, "not_assessable": na})
            serial = json.dumps(rep_.to_dict() if hasattr(rep_, "to_dict") else {}, ensure_ascii=False)
            if row["question_id"] == "Q3" and "t4" in serial and "t4" not in ans:
                print("!!! SAFETY: report mentions t4 not present in answer", row["question_id"], rep["replication"], cond)
print("\n== status counts (claims)")
for c in ("V2","COMPACT"): print(c, dict(status_counts[c]))
print("\n== rules fired")
for c in ("V2","COMPACT"): print(c, dict(rule_counts[c]))
print("\n== R2 hits (prediction framed as observation)")
for c in ("V2","COMPACT"): print(c, r2_hits[c])
print("\n== per answer, has_violation")
for o in out:
    if o["has_violation"] or o["not_assessable"]:
        print(f"{o['q']} r{o['rep']} {o['cond']}: worst={o['worst']} viol={o['has_violation']}")
        for f in o["flagged"]: print("    NS:", f)
        for f in o["not_assessable"]: print("    NA:", f)
json.dump(out, open("/tmp/validator_replay_compact_v1.json","w"), ensure_ascii=False, indent=1)
print("\nwrote /tmp/validator_replay_compact_v1.json")
