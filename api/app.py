import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.features import build_index
from src.train import load_rag_pipeline

app = FastAPI(
    title="RAG Knowledge Retrieval API",
    description="Semantic retrieval over technical document corpus",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
DEFAULT_NOTEBOOK = os.getenv("DEFAULT_NOTEBOOK", "default")

rag_pipelines = {}


class QueryRequest(BaseModel):
    query: str


def get_pipeline(notebook=DEFAULT_NOTEBOOK):
    if notebook not in rag_pipelines:
        faiss_dir = os.path.join(
            MODEL_DIR,
            "faiss_store"
        )

        rag_pipelines[notebook] = load_rag_pipeline(
            faiss_dir,
            mode="api",
            use_rerank=False,
            groq_api_key=os.getenv("GROQ_API_KEY_1")
        )

    return rag_pipelines[notebook]


@app.get("/")
def root():
    return {
        "message": "RAG API running"
    }


@app.post("/upload/{notebook}")
async def upload_files(
    notebook: str,
    files: list[UploadFile] = File(...)
):
    notebook_dir = os.path.join(
        DATA_DIR,
        notebook,
        "docs"
    )

    os.makedirs(
        notebook_dir,
        exist_ok=True
    )

    saved_files = []

    for file in files:
        file_path = os.path.join(
            notebook_dir,
            file.filename
        )

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        saved_files.append(file.filename)

    return {
        "status": "uploaded",
        "files": saved_files
    }


@app.post("/build-index/{notebook}")
def build(notebook: str):
    docs_dir = os.path.join(
        DATA_DIR,
        notebook,
        "docs"
    )

    faiss_dir = os.path.join(
        MODEL_DIR,
        "faiss_store"
    )

    build_index(
        docs_dir,
        faiss_dir
    )

    if notebook in rag_pipelines:
        del rag_pipelines[notebook]

    return {
        "status": "index_built",
        "notebook": notebook
    }


@app.post("/query/{notebook}")
def query(
    notebook: str,
    request: QueryRequest
):
    pipeline = get_pipeline(notebook)

    result = pipeline.invoke(request.query)

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"]
    }