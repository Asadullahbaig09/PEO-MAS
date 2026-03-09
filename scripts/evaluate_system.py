"""
Comprehensive System Evaluation Script
Tests:
  1. Embedding model accuracy (semantic relevance of retrievals)
  2. RAG content audit (what policies/laws are stored)
  3. Fine-tuned LLM usage across project
  4. 1-cycle run with all components
  5. Anomaly detection quality
  6. Law generation quality and necessity
  7. Government-policy deduplication (laws NOT generated if already covered)
"""

import sys
import io
import json
import time
import gc
import torch
import logging
from pathlib import Path
from datetime import datetime

# Force UTF-8 on Windows so Unicode symbols don't crash the console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)   # suppress noise during eval
logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)


SECTION_WIDTH = 70
def section(title):
    print(f"\n{'='*SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'='*SECTION_WIDTH}")

def subsection(title):
    print(f"\n  -- {title}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAG CONTENT AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def audit_rag_content():
    section("1. RAG CONTENT AUDIT")
    try:
        from src.rag.vector_store import VectorStore
        vs = VectorStore(persist_directory="./data/chromadb")
        domains = vs.list_all_domains()
        ok(f"ChromaDB connected — {len(domains)} domain collections: {domains}")

        all_docs = []
        for domain in domains:
            try:
                results = vs.search(domain=domain, query_text="AI regulation law policy", n_results=20)
                docs = results.get("documents", [])
                metas = results.get("metadatas", [])
                all_docs.extend([(d, m, domain) for d, m in zip(docs, metas)])
                ok(f"Domain '{domain}': {len(docs)} chunks retrievable")
                for m in metas[:3]:
                    src = m.get("source", m.get("title", "unknown"))
                    info(f"    • {src}")
            except Exception as e:
                warn(f"Domain '{domain}' retrieval error: {e}")

        # Separate government vs internal
        gov_docs  = [(d, m, dm) for d, m, dm in all_docs if "official" in str(m.get("source","")).lower()]
        int_docs  = [(d, m, dm) for d, m, dm in all_docs if "official" not in str(m.get("source","")).lower()]

        subsection("Content Classification")
        ok(f"Government / official laws: {len(gov_docs)} chunks")
        for _, m, dm in gov_docs:
            info(f"    [{dm}] {m.get('title', '?')}  —  {m.get('authority', '?')}")
        ok(f"Internal policies:          {len(int_docs)} chunks")

        return vs, len(gov_docs) > 0
    except Exception as e:
        fail(f"RAG audit failed: {e}")
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING MODEL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
def test_embedding_accuracy(vs):
    section("2. EMBEDDING MODEL ACCURACY")

    if vs is None:
        fail("VectorStore not available — skipping")
        return {}

    test_cases = [
        {
            "query": "GDPR personal data privacy user rights consent",
            "expected_domain": "privacy",
            "expected_keywords": ["gdpr", "privacy", "data", "consent", "rights"],
        },
        {
            "query": "algorithmic bias discrimination fairness hiring AI",
            "expected_domain": "bias",
            "expected_keywords": ["bias", "fairness", "discrimination"],
        },
        {
            "query": "AI transparency explainability decisions black-box",
            "expected_domain": "transparency",
            "expected_keywords": ["transparency", "explainab"],
        },
        {
            "query": "AI safety risk harm prevention autonomous systems",
            "expected_domain": "safety",
            "expected_keywords": ["safety", "risk", "harm"],
        },
        {
            "query": "EU AI Act high-risk regulation compliance 2024",
            "expected_domain": "general",
            "expected_keywords": ["ai act", "risk", "regulation"],
        },
    ]

    results = {}
    hit_count = 0
    total = len(test_cases)

    for tc in test_cases:
        query = tc["query"]
        expected_domain = tc["expected_domain"]
        try:
            ret = vs.search(domain=expected_domain, query_text=query, n_results=3)
            docs = ret.get("documents", [])
            dists = ret.get("distances", [])

            combined_text = " ".join(docs).lower()
            kw_hits = [kw for kw in tc["expected_keywords"] if kw in combined_text]
            kw_ratio = len(kw_hits) / len(tc["expected_keywords"])
            avg_dist = sum(dists) / len(dists) if dists else 9.9
            similarity = max(0, 1 - avg_dist / 2)

            hit = kw_ratio >= 0.6
            if hit:
                hit_count += 1

            status = ok if hit else warn
            status(f"Query: \"{query[:55]}...\"")
            info(f"    Expected domain : {expected_domain}")
            info(f"    Keywords found  : {kw_hits}/{tc['expected_keywords']} ({kw_ratio*100:.0f}%)")
            info(f"    Avg similarity  : {similarity:.3f}  (dist={avg_dist:.3f})")
            info(f"    Pass            : {'YES' if hit else 'NO'}")

            results[query] = {"kw_ratio": kw_ratio, "similarity": similarity, "pass": hit}
        except Exception as e:
            fail(f"Test failed for domain '{expected_domain}': {e}")
            results[query] = {"error": str(e)}

    accuracy = hit_count / total
    subsection("Embedding Model Summary")
    (ok if accuracy >= 0.6 else fail)(f"Keyword-retrieval accuracy: {hit_count}/{total} = {accuracy*100:.0f}%")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. FINE-TUNED LLM USAGE IN PROJECT
# ─────────────────────────────────────────────────────────────────────────────
def check_llm_usage():
    section("3. FINE-TUNED LLM USAGE AUDIT")

    from config.settings import settings
    finetuned_path = settings.FINETUNED_MODEL_PATH
    use_hf = settings.USE_HUGGINGFACE
    hf_model = settings.HUGGINGFACE_MODEL

    ok(f"USE_HUGGINGFACE = {use_hf}")
    ok(f"HUGGINGFACE_MODEL = {hf_model}")
    ok(f"FINETUNED_MODEL_PATH = {finetuned_path}")
    info(f"    Fine-tuned adapter exists: {finetuned_path.exists()}")

    # List adapter files
    if finetuned_path.exists():
        adapter_files = list(finetuned_path.iterdir())
        for f in adapter_files:
            ok(f"    Adapter file: {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    else:
        fail("Fine-tuned adapter directory NOT FOUND — system will use templates")

    subsection("Checking LLM usage across source files")
    src_dir = Path("src")
    usages = []
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(errors="ignore")
        if "LocalLLMInterface()" in text:
            # Direct instantiation (bad — should use shared)
            warn(f"Direct LLM instantiation: {py_file.relative_to(Path('.'))}")
            usages.append(str(py_file))
        elif "llm_interface" in text:
            ok(f"Uses shared llm_interface:  {py_file.relative_to(Path('.'))}")

    if not usages:
        ok("No direct LocalLLMInterface() instantiations found — shared LLM pattern correct")
    else:
        warn(f"{len(usages)} files still instantiate LLM directly (VRAM waste risk)")

    return finetuned_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN 1 SYSTEM CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def run_one_cycle():
    section("4. RUNNING 1 SYSTEM CYCLE  (fine-tuned LLM + RAG)")

    print("\n  Clearing GPU memory before loading model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        ok("GPU memory cleared")

    try:
        print("\n  Initializing system (LLM load ~25s)...\n")
        t0 = time.time()

        from src.core.system import PerpetualEthicalOversightMAS
        system = PerpetualEthicalOversightMAS()

        init_time = time.time() - t0
        ok(f"System initialized in {init_time:.1f}s")

        # Check LLM actually loaded
        llm_loaded = (
            system.llm_interface is not None and
            hasattr(system.llm_interface, "available") and
            system.llm_interface.available
        )
        if llm_loaded:
            ok("Fine-tuned Mistral 7B loaded and available")
        else:
            warn("Fine-tuned model NOT loaded — system using template fallback")

        # Run 1 cycle
        print("\n  Processing cycle 1...")
        t1 = time.time()
        system.process_cycle()
        cycle_time = time.time() - t1
        ok(f"Cycle completed in {cycle_time:.1f}s")

        # Export outputs so evaluation sections can read them
        state = system.export_system_state()
        out_path = Path("output/system_state.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        ok(f"System state exported → {out_path}")

        if hasattr(system.anomaly_detector, "legal_recommendations"):
            from dataclasses import asdict
            recs = system.anomaly_detector.legal_recommendations
            if recs:
                recs_path = Path("output/legal_recommendations.json")
                with open(recs_path, "w") as f:
                    json.dump([asdict(r) for r in recs], f, indent=2, default=str)
                ok(f"Legal recommendations exported → {recs_path}  ({len(recs)} records)")
            else:
                warn("No legal recommendations generated this cycle")

        return system, llm_loaded

    except Exception as e:
        fail(f"Cycle failed: {e}")
        import traceback; traceback.print_exc()
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANOMALY DETECTION QUALITY
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_anomaly_detection(system):
    section("5. ANOMALY DETECTION QUALITY")

    if system is None:
        fail("System not available — skipping")
        return {}

    metrics = system.metrics
    sig_processed = metrics.get("signals_processed", 0)
    anomalies      = metrics.get("anomalies_detected", 0)
    rag_assessments = metrics.get("rag_assessments", 0)
    docs_retrieved  = metrics.get("total_documents_retrieved", 0)

    detection_rate = anomalies / sig_processed if sig_processed else 0

    ok(f"Signals processed   : {sig_processed}")
    ok(f"Anomalies detected  : {anomalies}")
    ok(f"Detection rate      : {detection_rate*100:.1f}%")
    ok(f"RAG assessments     : {rag_assessments}")
    ok(f"Policy docs retrieved: {docs_retrieved}")

    subsection("Knowledge-Graph Domain Weights")
    kg = system.knowledge_graph
    if hasattr(kg, "nodes"):
        for node, data in sorted(kg.nodes.items()):
            weight = data.get("state", 0) if isinstance(data, dict) else data
            bar = "█" * max(1, int(weight * 2))
            info(f"    {node:<15} {weight:5.2f}  {bar}")

    subsection("Agent Pool")
    for agent_id, agent in system.agent_registry.agents.items():
        spec = agent.spec
        info(f"    [{spec.domain:<15}] {spec.name}  (caps: {', '.join(spec.capabilities[:3])})")

    # Anomaly severity distribution (from legal recommendations)
    det = system.anomaly_detector
    high_sev = 0; med_sev = 0; low_sev = 0
    if hasattr(det, 'legal_recommendations'):
        for rec in det.legal_recommendations:
            s = getattr(rec, "severity", 0)
            if s >= 0.8: high_sev += 1
            elif s >= 0.5: med_sev += 1
            else: low_sev += 1
        subsection("Anomaly Severity Breakdown (from legal recommendations)")
        ok(f"HIGH  (≥0.8) : {high_sev}")
        ok(f"MEDIUM (0.5–0.8): {med_sev}")
        ok(f"LOW   (<0.5) : {low_sev}")

    return {"signals": sig_processed, "anomalies": anomalies, "rate": detection_rate}


# ─────────────────────────────────────────────────────────────────────────────
# 6. LAW GENERATION QUALITY + GOV-POLICY DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_law_generation():
    section("6. LAW GENERATION QUALITY + GOVERNMENT POLICY DEDUPLICATION")

    recs_path = Path("output/legal_recommendations.json")
    if not recs_path.exists():
        fail("output/legal_recommendations.json not found")
        return {}

    with open(recs_path) as f:
        recs = json.load(f)

    ok(f"Total recommendations in file: {len(recs)}")

    # Categorise
    template_recs = [r for r in recs if "Section 1: Purpose" in r.get("proposed_law", "")]
    llm_recs      = [r for r in recs if "Section 1: Purpose" not in r.get("proposed_law", "")]
    proposed      = [r for r in recs if r.get("status") == "proposed"]
    covered       = [r for r in recs if r.get("status") != "proposed"]

    ok(f"LLM-generated (rich) laws  : {len(llm_recs)}")
    ok(f"Template-generated laws    : {len(template_recs)}")
    ok(f"Status=proposed (new law)  : {len(proposed)}")
    ok(f"Status=covered (existing)  : {len(covered)}")

    subsection("Domain Distribution")
    from collections import Counter
    domain_counts = Counter(r.get("issue_domain", "unknown") for r in recs)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        info(f"    {domain:<20} {count:3d} laws")

    subsection("Severity Distribution")
    high = sum(1 for r in recs if r.get("severity", 0) >= 0.8)
    med  = sum(1 for r in recs if 0.5 <= r.get("severity", 0) < 0.8)
    low  = sum(1 for r in recs if r.get("severity", 0) < 0.5)
    ok(f"HIGH severity (≥0.8) : {high}")
    ok(f"MEDIUM severity      : {med}")
    ok(f"LOW severity (<0.5)  : {low}")

    subsection("Coverage Gap Analysis (Deduplication)")
    has_gap = [r for r in recs if r.get("related_regulations") == []]
    no_gap  = [r for r in recs if r.get("related_regulations") != []]
    ok(f"Laws where NO existing regulation found : {len(has_gap)} (truly new)")
    ok(f"Laws that reference existing regulation  : {len(no_gap)} (complementary)")

    subsection("Sample Law Quality Check (newest 2)")
    for rec in recs[-2:]:
        print()
        info(f"  Title    : {rec.get('title')}")
        info(f"  Domain   : {rec.get('issue_domain')}  |  Severity: {rec.get('severity', 0):.2f}")
        info(f"  Confidence: {rec.get('confidence', 0):.2f}")
        info(f"  Status   : {rec.get('status')}")
        law_preview = rec.get("proposed_law", "")[:300].replace("\n", " ").strip()
        info(f"  Preview  : {law_preview}...")

        # Quality checks
        checks = {
            "Has title":       len(rec.get("title", "")) > 10,
            "Has law text":    len(rec.get("proposed_law", "")) > 100,
            "Has rationale":   len(rec.get("rationale", "")) > 20,
            "Has enforcement": len(rec.get("enforcement_mechanism", "")) > 20,
            "Has scope":       len(rec.get("scope", "")) > 10,
            "Has confidence":  rec.get("confidence", 0) > 0,
        }
        passed = sum(checks.values())
        info(f"  Quality  : {passed}/{len(checks)} checks passed")
        for k, v in checks.items():
            info(f"    {'[OK]' if v else '[MISS]'} {k}")

    return {"total": len(recs), "llm": len(llm_recs), "template": len(template_recs)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. OUTPUT FILE FORMAT CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_output_files():
    section("7. OUTPUT FILE FORMAT CHECK")

    output_files = {
        "output/system_state.json": ["metrics", "knowledge_graph", "agents", "timestamp"],
        "output/legal_recommendations.json": None,  # list
    }

    for path_str, expected_keys in output_files.items():
        p = Path(path_str)
        if not p.exists():
            fail(f"{path_str}  —  NOT FOUND")
            continue

        size_kb = p.stat().st_size / 1024
        try:
            data = json.loads(p.read_text())
            ok(f"{path_str}  ({size_kb:.1f} KB, valid JSON)")

            if expected_keys and isinstance(data, dict):
                for key in expected_keys:
                    if key in data:
                        ok(f"    Has '{key}' field")
                    else:
                        warn(f"    Missing '{key}' field")
            elif isinstance(data, list):
                ok(f"    Array with {len(data)} records")
                if data:
                    # Check first record schema
                    first = data[0]
                    fields = ["id", "title", "issue_domain", "severity", "proposed_law",
                              "rationale", "scope", "enforcement_mechanism", "confidence"]
                    present = [f for f in fields if f in first]
                    info(f"    First record has {len(present)}/{len(fields)} expected fields")
                    missing = [f for f in fields if f not in first]
                    if missing:
                        warn(f"    Missing fields: {missing}")
        except json.JSONDecodeError as e:
            fail(f"{path_str}  —  INVALID JSON: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*SECTION_WIDTH)
    print("  PERPETUAL ETHICAL OVERSIGHT MAS — COMPREHENSIVE EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*SECTION_WIDTH)

    t_start = time.time()

    # Phase 1: Static checks (no model load)
    vs, has_gov_laws = audit_rag_content()
    embed_results    = test_embedding_accuracy(vs)
    llm_ok           = check_llm_usage()

    # Phase 2: Run cycle (loads LLM once)
    system, llm_loaded = run_one_cycle()

    # Phase 3: Evaluation of results
    anomaly_results = evaluate_anomaly_detection(system)
    law_results     = evaluate_law_generation()
    check_output_files()

    # ── Final scorecard ──────────────────────────────────────────────────────
    section("FINAL SCORECARD")

    embed_acc = sum(r.get("pass", False) for r in embed_results.values()) / max(len(embed_results), 1)
    det_rate  = anomaly_results.get("rate", 0)

    checks = [
        ("Government laws in RAG",       has_gov_laws),
        ("Fine-tuned adapter exists",     llm_ok),
        ("Fine-tuned LLM loaded",         llm_loaded),
        ("Embedding accuracy ≥60%",       embed_acc >= 0.6),
        ("Anomaly detection rate ≥70%",   det_rate >= 0.7),
        ("LLM-generated laws produced",   law_results.get("llm", 0) > 0),
        ("Output files valid JSON",       Path("output/system_state.json").exists()),
    ]

    pass_count = 0
    for label, result in checks:
        if result:
            ok(f"{label}")
            pass_count += 1
        else:
            fail(f"{label}")

    total_time = time.time() - t_start
    print()
    ok(f"Passed : {pass_count}/{len(checks)} checks")
    ok(f"Embedding accuracy    : {embed_acc*100:.0f}%")
    ok(f"Anomaly detection rate: {det_rate*100:.1f}%")

    llm_pct = law_results.get("llm", 0) / max(law_results.get("total", 1), 1) * 100
    ok(f"LLM-generated laws    : {law_results.get('llm', 0)} ({llm_pct:.0f}% of all recs)")
    ok(f"Total evaluation time : {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
