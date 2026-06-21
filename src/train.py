import os
import re
import torch
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder

from src.features import EmbeddingManager, VectorStoreManager
from src.utils import REFUSAL_PHRASES

load_dotenv()

class RetrieverManager:
    def __init__(self, vectorstore, bm25):
        self.vectorstore = vectorstore
        self.bm25 = bm25

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            default_activation_function=torch.nn.Sigmoid()
        )

    def rewrite_query(self, query):
        query = query.lower()

        expansions = {
            "incident": (
                "incident response lifecycle preparation "
                "detection containment eradication recovery"
            ),
            "password": (
                "password policy minimum length "
                "passphrase complexity"
            ),
            "remote access": (
                "vpn remote access authentication "
                "mfa secure access"
            ),
            "kubernetes": (
                "kubectl cluster nodes pods troubleshooting"
            ),
            "cluster": (
                "kubectl cluster nodes pods troubleshooting "
                "failure modes shutdown network partition crash"
            ),
            "least privilege": (
                "access control family AC-6 least privilege "
                "minimum necessary authorization"
            ),
            "access control": (
                "access control family least privilege "
                "authorization permissions"
            ),
            "authentication": (
                "IA-2 IA-3 IA-4 IA-5 IA-7 IA-8 "
                "identification authentication authenticator"
            )
        }

        additions = [
            value
            for key, value in expansions.items()
            if key in query
        ]

        if additions:
            return f"{query} {' '.join(additions)}"

        return query

    def metadata_filter(self, query, docs):
        query = query.lower()

        relevant_categories = []

        if "password" in query:
            relevant_categories = ["POLICY"]

        elif "remote" in query or "vpn" in query:
            relevant_categories = ["POLICY", "COMPLIANCE"]

        elif "incident" in query:
            relevant_categories = ["SOP", "COMPLIANCE"]

        elif "kubernetes" in query or "cluster" in query:
            relevant_categories = ["TROUBLESHOOTING"]

        elif (
            "access control" in query
            or "privilege" in query
            or "authentication" in query
        ):
            relevant_categories = ["COMPLIANCE"]

        if not relevant_categories:
            return docs

        boosted = [
            d for d in docs
            if d.metadata.get("doc_category") in relevant_categories
        ]

        rest = [
            d for d in docs
            if d.metadata.get("doc_category") not in relevant_categories
        ]

        return boosted + rest

    def _doc_id(self, doc):
        return (
            doc.metadata.get("source_file", "")
            + "_"
            + str(doc.metadata.get("page", ""))
            + "_"
            + doc.page_content[:50]
        )

    def retrieve(self, query, top_k=5, use_rerank=False):
        query = self.rewrite_query(query)

        dense_results = self.vectorstore.similarity_search_with_score(
            query,
            k=20
        )

        sparse_results = self.bm25.invoke(query)

        docs = []
        seen = set()

        passed_threshold = [
            (doc, score)
            for doc, score in dense_results
            if score <= 1.15
        ]

        candidates = (
            passed_threshold
            if len(passed_threshold) >= 3
            else dense_results[:5]
        )

        for doc, score in candidates:
            doc_id = self._doc_id(doc)

            if doc_id not in seen:
                docs.append(doc)
                seen.add(doc_id)

        for doc in sparse_results:
            doc_id = self._doc_id(doc)

            if doc_id not in seen:
                docs.append(doc)
                seen.add(doc_id)

        docs = self.metadata_filter(query, docs)

        if use_rerank and docs:
            docs = self.rerank(query, docs)

        return docs[:top_k]

    def rerank(self, query, docs):
        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        filtered = [
            doc
            for doc, score in ranked
            if score >= 0.25
        ]

        if len(filtered) < 3:
            filtered = [
                doc
                for doc, _ in ranked[:5]
            ]

        return filtered

class LocalLLMManager:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3:latest",
            temperature=0
        )

    def get_llm(self):
        return self.llm

class LLMManager:
    def __init__(self, groq_api_key=None):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=(
                groq_api_key or os.getenv("GROQ_API_KEY")
            ),
            temperature=0
        )

    def get_llm(self):
        return self.llm

class RAGChain:
    def __init__(self, retriever, llm, use_rerank=False):
        self.retriever = retriever
        self.llm = llm
        self.use_rerank = use_rerank

    def format_context(self, docs):
        parts = []

        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            category = doc.metadata.get("doc_category", "UNKNOWN")
            section = doc.metadata.get("section_title", "UNKNOWN")

            parts.append(
                f"Source: {source}\n"
                f"Category: {category}\n"
                f"Section: {section}\n"
                f"Page: {page}\n\n"
                f"{doc.page_content[:1500]}"
            )

        return "\n\n---\n\n".join(parts)

    def compute_confidence(self, docs, answer):
        if not docs or not answer.strip():
            return 0.0

        clean_answer = re.sub(
            r"\(Source:.*?\)",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer_words = set(
            re.findall(r"\w+", clean_answer.lower())
        )

        if not answer_words:
            return 0.0

        best_ratio = 0.0

        for doc in docs:
            chunk_words = set(
                re.findall(
                    r"\w+",
                    doc.page_content.lower()
                )
            )

            overlap = len(answer_words & chunk_words)

            ratio = overlap / len(answer_words)

            best_ratio = max(best_ratio, ratio)

        score = best_ratio

        if any(
            p in answer.lower()
            for p in REFUSAL_PHRASES
        ):
            score = min(score, 0.3)

        return round(min(score, 1.0), 2)

    def invoke(self, query, top_k=8):
        docs = self.retriever.retrieve(
            query,
            top_k=top_k,
            use_rerank=self.use_rerank
        )

        if not docs:
            return {
                "answer": (
                    "The provided documents do not contain "
                    "information to answer this question."
                ),
                "documents": [],
                "sources": [],
                "confidence": 0.0
            }

        context = self.format_context(docs)

        prompt = f"""
You are a grounded retrieval assistant.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. Answer ONLY if explicitly or clearly supported by context.
4. If partially supported, answer only the supported part.
5. Never infer missing facts.
6. Use refusal ONLY if no relevant information exists.
7. Cite source file names.
8. When listing codes, include only those explicitly stated in context as related to the asked concept.
9. Do not include nearby or cross-referenced codes unless the context directly ties them to the concept.
10. Never mix IDs, titles, or definitions.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)
        answer = response.content.strip()

        return {
            "answer": answer,
            "documents": docs,
            "sources": list(
                set(
                    doc.metadata.get("source_file", "Unknown")
                    for doc in docs
                )
            ),
            "confidence": self.compute_confidence(docs, answer)
        }

def load_rag_pipeline(
    faiss_dir,
    mode="api",
    use_rerank=False,
    groq_api_key=None
):
    embedding_model = EmbeddingManager().get_model()

    vectorstore, bm25 = VectorStoreManager(
        embedding_model,
        faiss_dir
    ).load()

    retriever = RetrieverManager(vectorstore, bm25)

    if mode == "api":
        llm = LLMManager(groq_api_key=groq_api_key).get_llm()
    else:
        llm = LocalLLMManager().get_llm()

    rag_chain = RAGChain(
        retriever,
        llm,
        use_rerank=use_rerank
    )

    rag_chain.embedding_model = embedding_model

    return rag_chain