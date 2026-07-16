#  RAG Query System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/Mistral_7B-7B2D8B?style=for-the-badge&logo=ai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Chroma-00D084?style=for-the-badge&logo=database&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-FF6B35?style=for-the-badge&logo=ai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A full-stack Retrieval-Augmented Generation (RAG) application built with <b>FastAPI, LangChain, Chroma & Ollama</b> — enabling intelligent document search and context-aware question answering..
</p>

---

##  Table of Contents.

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [API Endpoints](#-api-endpoints)
- [Author](#-author)

---

## Overview

**RAG Query** is a full-stack document intelligence system that combines semantic search with large language models. Upload PDF documents, index them into a vector database, and get accurate answers grounded in your document collection. Built with **Mistral-7B via Ollama**, **Chroma vector database**, and a modern **FastAPI** backend with an interactive web interface.

---

## Features

| Feature | Description |
|:---|:---|
|  PDF Upload & Indexing | Dynamically load and chunk PDFs into vector database |
|  Semantic Search | Find relevant document chunks using embeddings |
|  Context-Aware QA | Generate answers grounded only in indexed documents |
|  Real-time Streaming | Stream LLM responses token-by-token to the UI |
|  Model Selection | Switch between available Ollama models on-the-fly |
|  Source Filtering | Query specific documents or all indexed PDFs |
|  Conversation History | Track query history within a session |
|  Web UI | Modern, responsive interface |
|  REST API | Full API for programmatic access |

---

##  Tech Stack

| Tool | Purpose |
|:---|:---|
| Python | Core language |
| FastAPI | Backend API framework |
| Chroma | Vector database for embeddings |
| LangChain | LLM orchestration & prompt management |
| Ollama | Local LLM runtime |
| Mistral-7B Instruct | Default language model |
| nomic-embed-text | Embedding model |
| pypdf | PDF text extraction |
| HTML/CSS/JavaScript | Frontend UI |

---

##  Project Structure

```
rag-tutorial-v2-main/
│
├──  api.py                        # FastAPI application & routes
├──  query_data.py                 # Query execution logic
├──  populate_database.py           # PDF processing & indexing
├──  get_embedding_function.py     # Embedding model initialization
├──  _verify_db.py                 # Database verification utility
├──  test_rag.py                   # Unit tests
├──  requirements.txt              # Python dependencies
├──  README.md                     # Project documentation
│
├──  chroma/                       # Vector database storage
│   ├── chroma.sqlite3
│   └── [collection directories]
│
├──  data/                         # PDF input directory
│   └── (your PDF files here)
│
└──  static/                       # Frontend assets
    └── index.html                 # Web UI
```

---

##  Setup & Installation

**1. Install Ollama**

Download and install [Ollama](https://ollama.ai) for your operating system. This provides the local LLM runtime.

**2. Pull the required models**

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

**3. Clone the repository**

```bash
git clone https://github.com/your-username/rag-query.git
cd rag-query
```

**4. Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Add your PDF files**

Place your PDF documents in the `data/` directory:

```bash
mkdir -p data
# Copy your PDFs here
```

**7. Populate the database**

```bash
python populate_database.py
```

Or reset and repopulate:

```bash
python populate_database.py --reset
```

**8. Run the application**

```bash
python api.py
```

The API will start at `http://localhost:8000` and automatically serve the web UI.

---

##  Configuration

No configuration file needed! The system uses sensible defaults:

- **Vector DB**: Stored in `./chroma/` (auto-created)
- **Embeddings**: `nomic-embed-text` (auto-downloaded by Ollama)
- **LLM Model**: `mistral` (switchable in the UI)
- **Default Chunk Size**: 800 characters with 80-char overlap
- **Default Retrieval**: Top-5 most relevant chunks

To use a different embedding or LLM, edit the corresponding functions in `get_embedding_function.py` or select a different model in the web UI.

---

##  How It Works

### 1. **Document Ingestion**

```bash
python populate_database.py
```

- Loads all PDF files from `data/` directory
- Extracts text from each page
- Splits documents into 800-character chunks with 80-char overlap
- Generates embeddings using `nomic-embed-text`
- Stores chunks + embeddings in Chroma vector database

### 2. **Query Processing**

When you ask a question:

1. **Semantic Search**: User query is embedded and top-K chunks retrieved from Chroma
2. **Context Building**: Retrieved chunks formatted into LLM prompt
3. **LLM Response**: Mistral-7B generates an answer based only on the context
4. **Streaming**: Response is streamed token-by-token to the UI

### 3. **Components**

| Component | Role |
|:---|:---|
| `api.py` | FastAPI routes, CORS, streaming response handling |
| `query_data.py` | Query execution, prompt templating, LLM invocation |
| `populate_database.py` | PDF loading, text splitting, database population |
| `get_embedding_function.py` | Embedding model initialization (Ollama) |
| `static/index.html` | Modern web UI with real-time updates |

---

##  API Endpoints

### **GET /health**
```bash
curl http://localhost:8000/health
```
Returns system status and indexed document count.

### **GET /models**
```bash
curl http://localhost:8000/models
```
Returns list of available Ollama models.

### **GET /sources**
```bash
curl http://localhost:8000/sources
```
Returns list of indexed PDF filenames.

### **POST /query**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is the main topic?",
    "model": "mistral",
    "k": 5,
    "source": null
  }'
```

**Request Parameters:**
- `query_text` (required): Your question
- `model` (optional): LLM model name (default: `mistral`)
- `k` (optional): Number of chunks to retrieve (default: `5`)
- `source` (optional): Restrict search to specific PDF filename

**Response**: Streaming JSON events with sources and token-by-token response

### **POST /upload**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

Uploads a new PDF and indexes it immediately.

---

##  Usage Examples

### Command-Line Query

```bash
python query_data.py "What are the main topics covered?"
```

### Verify Database

```bash
python _verify_db.py
```

Checks database integrity and performs a sample search.

### Run Tests

```bash
pytest test_rag.py -v
```

Validates RAG responses against expected answers.

---

##  Troubleshooting

| Issue | Solution |
|:---|:---|
| Ollama not found | Ensure Ollama is installed and running (`ollama serve`) |
| Model not found | Pull models with `ollama pull mistral nomic-embed-text` |
| Database error | Reset with `python populate_database.py --reset` |
| Port 8000 in use | Change port in `api.py` or kill the existing process |
| Empty search results | Add more PDFs to `data/` and rerun `populate_database.py` |

---

##  Performance Tips

- **Chunk Size**: Adjust `chunk_size` in `populate_database.py` (larger = more context, fewer results)
- **Retrieval Count**: Increase `k` for broader context, decrease for precision
- **Model Selection**: Use lighter models (7B) for speed, larger for quality
- **Embedding Model**: `nomic-embed-text` is fast; alternatives exist in Ollama

---

## Author

**Sadia Noreen**
*Software Engineering Graduate | AI & ML Enthusiast*

---

<p align="center"> If you found this helpful, consider giving it a star!</p>
