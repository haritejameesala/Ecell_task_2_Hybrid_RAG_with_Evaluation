import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

TECH_COMMAND_PATTERN = re.compile(
    r"^\$?\s*(kubectl|docker|systemctl|grep|curl|ssh|scp|ls|cd|cat|ps|top|df|du|netstat|chmod|chown|sudo|iptables|tail|head|awk|sed|python|pip|git|npm|yarn|terraform|ansible-playbook|make)\b",
    re.IGNORECASE
)

HEADER_PATTERN = re.compile(
    r"^(\d+(\.\d+)*\s+\S.*|[A-Z]{2}-\d+\s+\S.*|[A-Z][A-Z0-9 \-/]{4,40})$"
)

def detect_doc_category(text):
    text = text[:3000].lower()

    if any(term in text for term in [
        "incident response",
        "incident handling",
        "containment",
        "eradication",
        "recovery"
    ]):
        return "SOP"

    if any(term in text for term in [
        "password policy",
        "access control",
        "remote access",
        "authentication"
    ]):
        return "POLICY"

    if any(term in text for term in [
        "control family",
        "security controls",
        "compliance",
        "nist"
    ]):
        return "COMPLIANCE"

    if any(term in text for term in [
        "kubernetes",
        "cluster",
        "deployment",
        "pod",
        "container"
    ]):
        return "TROUBLESHOOTING"

    return "UNKNOWN"

def extract_section_title(text):
    lines = text.split("\n")

    for line in lines[:15]:
        line = line.strip()

        if not line:
            continue

        if len(line.split()) > 15:
            continue

        if re.match(r"^\d+(\.\d+)*\s", line):
            return line

        if line.isupper():
            return line

        if line.istitle():
            return line

        if line.endswith(":"):
            return line

    return "UNKNOWN"

def load_all_docs(data_directory):
    data_path = Path(data_directory).resolve()

    pdf_files = list(data_path.glob("**/*.pdf"))
    txt_files = list(data_path.glob("**/*.txt"))

    documents = []

    for pdf_file in pdf_files:
        try:
            pdf_docs = PyPDFLoader(str(pdf_file)).load()

            category = detect_doc_category(
                "\n".join(page.page_content for page in pdf_docs[:3])
            )

            for page in pdf_docs:
                page.metadata["source_file"] = pdf_file.name
                page.metadata["file_type"] = "pdf"
                page.metadata["doc_category"] = category
                page.metadata["section_title"] = extract_section_title(
                    page.page_content
                )

            documents.extend(pdf_docs)

        except Exception as e:
            print(f"[ERROR] {pdf_file.name}: {e}")

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(
                page_content=content,
                metadata={
                    "source_file": txt_file.name,
                    "file_type": "txt",
                    "doc_category": detect_doc_category(content),
                    "section_title": extract_section_title(content)
                }
            )

            documents.append(doc)

        except Exception as e:
            print(f"[ERROR] {txt_file.name}: {e}")

    return documents

def find_repeated_lines(documents):
    line_counts = {}

    for doc in documents:
        for line in doc.page_content.split("\n"):
            stripped = line.strip().lower()

            if stripped:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

    return {
        line for line, count in line_counts.items()
        if count >= 3
    }

def clean_text(text, repeated_headers=None):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        if TECH_COMMAND_PATTERN.match(stripped):
            cleaned.append(stripped)
            continue

        if re.fullmatch(r"\d{1,4}", stripped):
            continue

        if repeated_headers and stripped.lower() in repeated_headers:
            continue

        if len(stripped.split()) <= 3 and stripped.isupper():
            continue

        if len(re.findall(r"[a-zA-Z]", stripped)) < 3:
            continue

        cleaned.append(stripped)

    text = "\n".join(cleaned)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    return text.strip()

def split_by_structure(text):
    lines = text.split("\n")
    sections = []
    current = []

    for line in lines:
        stripped = line.strip()

        is_header = (
            stripped
            and len(stripped.split()) <= 12
            and HEADER_PATTERN.match(stripped)
        )

        if is_header and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return [s for s in sections if s.strip()]

def chunking(
    documents,
    max_chunk_size=1000,
    min_chunk_size=200,
    breakpoint_threshold_amount=80
):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )

    repeated_headers = find_repeated_lines(documents)

    structural_docs = []

    for doc in documents:
        content = clean_text(doc.page_content, repeated_headers)

        if not content:
            continue

        for section_text in split_by_structure(content):
            new_metadata = doc.metadata.copy()
            new_metadata["parent_section"] = extract_section_title(
                section_text
            )

            structural_docs.append(
                Document(
                    page_content=section_text,
                    metadata=new_metadata
                )
            )

    if not structural_docs:
        print("[WARNING] No structural sections produced after cleaning. Documents may be empty or unreadable.")
        return []

    raw_chunks = splitter.split_documents(structural_docs)

    capper = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    capped_chunks = []

    for chunk in raw_chunks:
        if len(chunk.page_content) > max_chunk_size:
            capped_chunks.extend(capper.split_documents([chunk]))
        else:
            capped_chunks.append(chunk)

    final_chunks = []
    buffer = None

    for chunk in capped_chunks:
        if buffer is None:
            buffer = chunk
            continue

        same_source = (
            buffer.metadata.get("source_file") ==
            chunk.metadata.get("source_file")
        )

        if len(buffer.page_content) < min_chunk_size and same_source:
            buffer.page_content += "\n\n" + chunk.page_content
            continue

        final_chunks.append(buffer)
        buffer = chunk

    if buffer is not None:
        final_chunks.append(buffer)

    return final_chunks