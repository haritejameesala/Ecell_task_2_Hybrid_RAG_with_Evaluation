import os
import pickle

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from src.preprocess import load_all_docs, chunking
from src.utils import compute_index_stats, save_stats


load_dotenv()

class EmbeddingManager:
    def __init__(
        self,
        model_name="BAAI/bge-large-en-v1.5"
    ):
        self.model_name = model_name

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def get_model(self):
        return self.embedding_model

class VectorStoreManager:
    def __init__(
        self,
        embedding_model,
        persist_dir="models/notebooks/default/faiss_store"
    ):
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir

        self.vectorstore = None
        self.bm25 = None
        self.chunk_metadata = None

        os.makedirs(
            self.persist_dir,
            exist_ok=True
        )

    def create(self, chunks):
        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embedding_model
        )

        self.chunk_metadata = [
            doc.metadata
            for doc in chunks
        ]

        self.bm25 = BM25Retriever.from_documents(
            chunks
        )
        self.bm25.k = 10

    def save(self):
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore not created"
            )

        self.vectorstore.save_local(
            self.persist_dir
        )

        bm25_path = os.path.join(
            self.persist_dir,
            "bm25.pkl"
        )

        with open(
            bm25_path,
            "wb"
        ) as f:
            pickle.dump(
                self.bm25,
                f
            )

        metadata_path = os.path.join(
            self.persist_dir,
            "chunk_metadata.pkl"
        )

        with open(
            metadata_path,
            "wb"
        ) as f:
            pickle.dump(
                self.chunk_metadata,
                f
            )

    def load(self):
        self.vectorstore = FAISS.load_local(
            self.persist_dir,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        bm25_path = os.path.join(
            self.persist_dir,
            "bm25.pkl"
        )

        if os.path.exists(bm25_path):
            with open(
                bm25_path,
                "rb"
            ) as f:
                self.bm25 = pickle.load(f)

        return self.vectorstore, self.bm25

def build_index(
    docs_folder,
    persist_dir,
    max_chunk_size=1000,
    min_chunk_size=200,
    breakpoint_threshold_amount=80
):
    docs = load_all_docs(
        docs_folder
    )

    if not docs:
        raise ValueError(
            f"No documents found in '{docs_folder}'. "
            "Ensure the folder exists and contains PDF or TXT files."
        )

    chunks = chunking(
        docs,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )

    if not chunks:
        raise ValueError(
            "Chunking produced no usable chunks. "
            "Check that the documents contain extractable text content."
        )

    stats = compute_index_stats(
        docs,
        chunks
    )

    stats["doc_categories"] = list(
        set(
            doc.metadata.get(
                "doc_category",
                "UNKNOWN"
            )
            for doc in docs
        )
    )

    chunk_lengths = [
        len(c.page_content)
        for c in chunks
    ]

    stats["chunk_size_min"] = (
        min(chunk_lengths)
        if chunk_lengths else 0
    )

    stats["chunk_size_max"] = (
        max(chunk_lengths)
        if chunk_lengths else 0
    )

    stats["chunk_size_avg"] = (
        round(
            sum(chunk_lengths) / len(chunk_lengths),
            1
        )
        if chunk_lengths else 0
    )

    embedding_manager = EmbeddingManager()

    embedding_model = (
        embedding_manager.get_model()
    )

    vectorstore = VectorStoreManager(
        embedding_model,
        persist_dir
    )

    vectorstore.create(
        chunks
    )

    vectorstore.save()

    stats["embedding_model"] = (
        embedding_manager.model_name
    )

    stats["vector_store"] = (
        "FAISS + BM25"
    )

    stats["retrieval_strategy"] = (
        "Hybrid Search"
    )

    stats["chunking_strategy"] = (
        f"Structure-aware + Semantic + Recursive Hybrid "
        f"(max={max_chunk_size}, min={min_chunk_size}, "
        f"breakpoint_pct={breakpoint_threshold_amount})"
    )

    stats_path = os.path.join(
        os.path.dirname(persist_dir),
        "stats.json"
    )

    save_stats(
        stats_path,
        stats
    )

if __name__ == "__main__":
    import sys

    notebook = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.getenv("DEFAULT_NOTEBOOK", "default")
    )

    data_base = os.getenv("DATA_DIR", "data/notebooks")
    model_base = os.getenv("MODEL_DIR", "models/notebooks")

    docs_folder = os.path.join(data_base, notebook, "docs")
    persist_dir = os.path.join(model_base, notebook, "faiss_store")

    build_index(
        docs_folder,
        persist_dir
    )