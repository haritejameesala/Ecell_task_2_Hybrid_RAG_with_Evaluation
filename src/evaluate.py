import os
import time
import json
from dataclasses import dataclass
from dotenv import load_dotenv

from groq import Groq
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase

from src.train import load_rag_pipeline
from src.utils import is_refusal

load_dotenv()


class GroqLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name="llama-3.3-70b-versatile",
        api_key=None
    ):
        self.model_name = model_name

        self.client = Groq(
            api_key=api_key or os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON. "
                        "Do not include markdown. "
                        "Do not include explanations outside JSON. "
                        "Escape all strings correctly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        output = (
            response.choices[0]
            .message.content.strip()
        )


        json.loads(output)

        return output

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


@dataclass
class EvalResult:
    query: str
    answer: str
    sources: list
    confidence: float
    cr: float
    f: float
    ar: float
    latency: float
    resolved: bool


def evaluate_pipeline(
    test_queries,
    faiss_dir,
    top_k=8,
    output_path=None,
    mode="api",
    use_rerank=False,
    groq_api_key=None  

    key = groq_api_key or os.getenv("GROQ_API_KEY")

    rag = load_rag_pipeline(
        faiss_dir,
        mode=mode,
        use_rerank=use_rerank,
        groq_api_key=key
    )

    judge_model = GroqLLM(api_key=key)

    results = []

    for i, query in enumerate(test_queries, 1):
        print(
            f"[EVAL] Query {i}/{len(test_queries)}: {query}"
        )

        start = time.time()

        result = rag.invoke(query, top_k)

        latency = round(time.time() - start, 3)

        answer = result["answer"]
        docs = result["documents"]

        contexts = (
            [doc.page_content for doc in docs]
            if docs
            else ["No relevant context retrieved."]
        )


        cr_metric = ContextualRelevancyMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True
        )

        cr_test = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=contexts
        )

        cr_metric.measure(cr_test)


        f_metric = FaithfulnessMetric(
            threshold=0.75,
            model=judge_model,
            include_reason=True
        )

        f_test = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=contexts
        )

        f_metric.measure(f_test)


        ar_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=judge_model,
            include_reason=True
        )

        ar_test = LLMTestCase(
            input=query,
            actual_output=answer
        )

        ar_metric.measure(ar_test)

        results.append(
            EvalResult(
                query=query,
                answer=answer,
                sources=result["sources"],
                confidence=result["confidence"],
                cr=round(cr_metric.score, 4),
                f=round(f_metric.score, 4),
                ar=round(ar_metric.score, 4),
                latency=latency,
                resolved=not is_refusal(answer)
            )
        )

    n = len(results)

    report = {
        "config": {
            "mode": mode,
            "rerank": use_rerank,
            "top_k": top_k
        },
        "aggregate": {
            "CR": round(
                sum(r.cr for r in results) / n,
                4
            ),
            "F": round(
                sum(r.f for r in results) / n,
                4
            ),
            "AR": round(
                sum(r.ar for r in results) / n,
                4
            ),
            "L": round(
                sum(r.latency for r in results) / n,
                3
            ),
            "QR": round(
                sum(
                    1 for r in results
                    if r.resolved
                ) / n,
                4
            ),
            "refusals": sum(
                1 for r in results
                if not r.resolved
            ),
            "total_queries": n
        },
        "per_query": [
            {
                "query": r.query,
                "answer": r.answer,
                "sources": r.sources,
                "confidence": r.confidence,
                "CR": r.cr,
                "F": r.f,
                "AR": r.ar,
                "latency": r.latency,
                "resolved": r.resolved
            }
            for r in results
        ]
    }

    print("\n===== EVALUATION REPORT =====")

    for k, v in report["aggregate"].items():
        print(f"{k}: {v}")

    if output_path:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    import sys

    faiss_dir = (
	    sys.argv[1]
	    if len(sys.argv) > 1
	    else os.getenv("FAISS_DIR", "models/faiss_store")
	)

    notebook = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "default"
    )


    groq_api_key_1 = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    groq_api_key_2 = os.getenv("GROQ_API_KEY_2") or os.getenv("GROQ_API_KEY")
    groq_api_key_3 = os.getenv("GROQ_API_KEY_3") or os.getenv("GROQ_API_KEY")

    sample_queries = [
        "What are the phases of incident response?",
        "What is the minimum password length?",
        "What are remote access security requirements?",
        "Which access control family enforces least privilege?",
        "How do you troubleshoot Kubernetes clusters?"
    ]

    api_report = evaluate_pipeline(
        sample_queries,
        faiss_dir,
        mode="api",
        use_rerank=False,
        groq_api_key=groq_api_key_1
    )

    local_report = evaluate_pipeline(
        sample_queries,
        faiss_dir,
        mode="local",
        use_rerank=False,
        groq_api_key=groq_api_key_2
    )

    api_rerank_report = evaluate_pipeline(
        sample_queries,
        faiss_dir,
        mode="api",
        use_rerank=True,
        groq_api_key=groq_api_key_3
    )

    final_report = {
        "api": api_report,
        "local": local_report,
        "api_rerank": api_rerank_report
    }

    output_path = os.getenv(
	    "EVAL_REPORT_PATH",
	    "models/eval_report.json"
	)

    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(
        f"[INFO] Saved benchmark results to {output_path}"
    )