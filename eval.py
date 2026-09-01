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


if __name__=="__main__":
    pass_count=recall_count=cite_count=0
    n=len(GOLDEN)

    for i,case in enumerate(GOLDEN):
        recall,text_term,cited,answer,sources=evaluate(case)
        ok=all([recall,text_term,cited])
        total=recall+int(text_term)+int(cited)

        recall_count+=recall
        cite_count+=cited
        pass_count+=ok

        status="PASS" if ok else "FAIL"
        print(f"\n=== {case['query']} [{status} {int(recall)}/1 recall, {int(text_term)}/1 term, {int(cited)}/1 cite]")
        print(f"  expected source: {case['expected_source']}")
        print(f"  sources: {sources}")
        print(f"  answer: {answer[:200] if answer else '(empty)'}")

        if i<n-1:
            time.sleep(3)

    print(f"\n--- Summary: {pass_count}/{n} fully pass | recall {recall_count}/{n} | citation {cite_count}/{n}")
    if pass_count<n:
        raise SystemExit(1)