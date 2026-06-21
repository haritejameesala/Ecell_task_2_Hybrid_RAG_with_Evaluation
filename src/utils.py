import os
import json

REFUSAL_PHRASES = [
    "not contain",
    "no information",
    "not found",
    "i don't know",
    "cannot answer",
    "insufficient context"
]

def is_refusal(answer):
    if not answer:
        return False

    normalized = answer.strip().lower()

    return (
        any(
            phrase in normalized
            for phrase in REFUSAL_PHRASES
        )
        and len(normalized) < 150
    )

def compute_index_stats(documents, chunks):
    total_tokens = sum(
        len(chunk.page_content.split())
        for chunk in chunks
    )

    source_files = set(
        doc.metadata.get(
            "source_file",
            "unknown"
        )
        for doc in documents
    )

    file_types = {}

    for doc in documents:
        file_type = doc.metadata.get(
            "file_type",
            "unknown"
        )

        file_types[file_type] = (
            file_types.get(file_type, 0) + 1
        )

    return {
        "papers": len(source_files),
        "chunks": len(chunks),
        "tokens": total_tokens,
        "queries": 0,
        "file_types": file_types
    }

def save_stats(stats_path, stats):
    os.makedirs(
        os.path.dirname(stats_path) or ".",
        exist_ok=True
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )

def load_stats(stats_path):
    default_stats = {
        "papers": 0,
        "chunks": 0,
        "tokens": 0,
        "queries": 0,
        "file_types": {}
    }

    if not os.path.exists(stats_path):
        return default_stats

    try:
        with open(
            stats_path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception:
        return default_stats

def increment_query_stats(stats_path):
    stats = load_stats(stats_path)

    stats["queries"] = stats.get(
        "queries",
        0
    ) + 1

    save_stats(
        stats_path,
        stats
    )

def save_benchmark_report(output_path, report):
    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )