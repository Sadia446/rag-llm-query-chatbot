import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json

from langchain_chroma import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function
from populate_database import load_documents, split_documents, add_to_chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"
DEFAULT_MODEL = "mistral"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

app = FastAPI(title="RAG Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup — reused across requests.
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


# ---------- Schemas ----------

class QueryRequest(BaseModel):
    query_text: str
    model: str = DEFAULT_MODEL
    k: int = 5
    source: str | None = None  # optional filename to restrict search to


class ChunkResult(BaseModel):
    id: str
    content: str
    score: float
    source: str


class QueryResponse(BaseModel):
    response: str
    chunks: list[ChunkResult]


# ---------- Helpers ----------

def list_ollama_models() -> list[str]:
    """Return locally installed Ollama models, falling back to a default list."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().splitlines()[1:]  # skip header row
        models = [line.split()[0] for line in lines if line.strip()]
        return models or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def list_sources() -> list[str]:
    """Return unique source PDF filenames currently indexed in Chroma."""
    items = db.get(include=["metadatas"])
    sources = set()
    for meta in items["metadatas"]:
        if meta and meta.get("source"):
            sources.add(Path(meta["source"]).name)
    return sorted(sources)


# ---------- Routes ----------

@app.get("/models")
def get_models():
    return {"models": list_ollama_models()}


@app.get("/sources")
def get_sources():
    return {"sources": list_sources()}


@app.get("/health")
def health():
    count = db.get(include=[])
    return {"status": "ok", "documents_indexed": len(count["ids"])}


@app.post("/query")
async def query(request_data: QueryRequest, request: Request):
    if not request_data.query_text.strip():
        raise HTTPException(status_code=400, detail="query_text cannot be empty")

    search_filter = None
    if request_data.source:
        matches = [
            Path(m["source"]).name
            for m in db.get(include=["metadatas"])["metadatas"]
            if m and m.get("source")
        ]
        full_path = next(
            (
                m["source"]
                for m in db.get(include=["metadatas"])["metadatas"]
                if m and Path(m["source"]).name == request_data.source
            ),
            None,
        )
        if full_path:
            search_filter = {"source": full_path}

    results = db.similarity_search_with_score(
        request_data.query_text, k=request_data.k, filter=search_filter
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Try a different question or upload more PDFs.",
        )

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request_data.query_text)

    chunks = [
        {
            "id": doc.metadata.get("id", "unknown"),
            "content": doc.page_content,
            "score": float(score),
            "source": Path(doc.metadata.get("source", "unknown")).name,
        }
        for doc, score in results
    ]

    async def generate_response():
        # First send the chunks as a JSON event
        yield f"data: {json.dumps({'type': 'sources', 'chunks': chunks})}\n\n"

        try:
            model = OllamaLLM(model=request_data.model)
            # Use stream() to get tokens in real-time
            for chunk in model.stream(prompt):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
            # Send an explicit done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            
    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data_path = Path(DATA_PATH)
    data_path.mkdir(exist_ok=True)
    dest = data_path / file.filename

    contents = await file.read()
    dest.write_bytes(contents)

    # Re-run the same load -> split -> add pipeline used by populate_database.py.
    # add_to_chroma only embeds chunks with new IDs, so this is safe to rerun.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    return {"status": "indexed", "filename": file.filename}


# Serve the frontend last, so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)