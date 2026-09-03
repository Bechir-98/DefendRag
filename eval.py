from graph import app
import time

GOLDEN=[
    {
        "query":"What is prompt injection?",
        "expected_source":"OWASP-GenAI-LLM-Top-10-2026-v1.0.pdf",
        "required_terms":["injection"]
    },
    {
        "query":"How does OWASP classify LLM security risks?",
        "expected_source":"OWASP-GenAI-LLM-Top-10-2026-v1.0.pdf",
        "required_terms":["LLM","risk"]
    },
    {
        "query":"How to mitigate jailbreak attacks on LLMs?",
        "expected_source":"OWASP-GenAI-LLM-Top-10-2026-v1.0.pdf",
        "required_terms":["jailbreak","GPT-4"]
    },
    {
        "query":"What are MITRE ATLAS techniques for AI systems?",
        "expected_source":"SAFEAI_Full_Report.pdf",
        "required_terms":["MITRE","ATLAS"]
    },
    {
        "query":"What is data poisoning in adversarial machine learning?",
        "expected_source":"NIST.AI.100-2e2025.pdf",
        "required_terms":["poisoning"]
    },
    {
        "query":"How does NIST define evasion attacks on ML models?",
        "expected_source":"NIST.AI.100-2e2025.pdf",
        "required_terms":["evasion"]
    },
]


def evaluate(case):
    r=app.invoke({"query":case["query"]})
    sources=[s["source"] for s in r["sources"]]

    recall=case["expected_source"] in sources

    answer=r.get("answer") or ""
    text_term=any(t in answer for t in case["required_terms"])
    cited_expected=any(case["expected_source"] in f"[Source: {s['source']}" for s in r["sources"])

    return recall,text_term,cited_expected,answer,sources


def run_eval():
    results=[]
    summary={"pass":0,"recall":0,"cite":0,"total":len(GOLDEN)}

    for i,case in enumerate(GOLDEN):
        recall,text_term,cited,answer,sources=evaluate(case)
        ok=all([recall,text_term,cited])

        summary["pass"]+=ok
        summary["recall"]+=recall
        summary["cite"]+=cited

        results.append({
            "query":case["query"],
            "status":"PASS" if ok else "FAIL",
            "recall":recall,
            "term":text_term,
            "cite":cited,
            "n_sources":len(sources),
            "expected_source":case["expected_source"],
            "sources":sources,
            "answer":answer,
        })

        if i<len(GOLDEN)-1:
            time.sleep(3)

    return results,summary


if __name__=="__main__":
    results,summary=run_eval()

    for r in results:
        print(f"\n=== {r['query']} [{r['status']} {int(r['recall'])}/1 recall, {int(r['term'])}/1 term, {int(r['cite'])}/1 cite]")
        print(f"  expected source: {r['expected_source']}")
        print(f"  sources: {r['sources']}")
        print(f"  answer: {r['answer'][:200] if r['answer'] else '(empty)'}")

    n=summary["total"]
    print(f"\n--- Summary: {summary['pass']}/{n} fully pass | recall {summary['recall']}/{n} | citation {summary['cite']}/{n}")
    if summary["pass"]<n:
        raise SystemExit(1)