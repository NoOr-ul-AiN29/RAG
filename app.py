# app.py
# ---------------------------------------------------------------------------
# PURPOSE:
#   The main FastAPI application. This is the entry point you run with:
#       uvicorn app:app --reload
#
#   It exposes:
#     GET  /           — serves the HTML frontend (templates/index.html)
#     POST /upload     — accepts a PDF, extracts text, builds FAISS index
#     POST /ask        — accepts a question, returns a contextual AI answer
#
#   All heavy lifting is delegated to document_loader.py, embeddings.py,
#   and rag.py so this file stays clean and readable.
#
#   API KEYS REQUIRED:
#     - GROQ_API_KEY only. Embeddings are generated locally for free.
# ---------------------------------------------------------------------------

import os
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from dotenv import load_dotenv

from document_loader import extract_text_from_pdf, split_text_into_chunks
from embeddings import build_faiss_index, save_index
from rag import answer_question

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
load_dotenv()

# Only GROQ_API_KEY is needed — embeddings run locally for free
if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Please add it to your .env file. "
        "Get a free key at https://console.groq.com"
    )

# ---------------------------------------------------------------------------
# Directory setup — create folders if they don't already exist
# ---------------------------------------------------------------------------
UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Paths for the persisted FAISS index and chunks
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.json")

# ---------------------------------------------------------------------------
# FastAPI app and Jinja2 template configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Upload a PDF and ask questions about its content.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Body for the /ask endpoint."""
    question: str


class UploadResponse(BaseModel):
    """Response body for the /upload endpoint."""
    message: str
    chunks_created: int
    filename: str


class AnswerResponse(BaseModel):
    """Response body for the /ask endpoint."""
    question: str
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """
    Serve the HTML frontend from templates/index.html.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload and build a searchable FAISS vector index from it.

    Steps:
      1. Validate that the uploaded file is a PDF.
      2. Save the file to the uploads/ directory.
      3. Extract text using pypdf.
      4. Split text into overlapping chunks.
      5. Generate LOCAL embeddings (free, no API) for every chunk.
      6. Build and save a FAISS index to vectorstore/.

    Returns:
        UploadResponse with a success message and the number of chunks.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Validate file type
    # ------------------------------------------------------------------ #
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # ------------------------------------------------------------------ #
    # Step 2: Save the uploaded file to disk
    # ------------------------------------------------------------------ #
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[app] PDF saved to '{save_path}'.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # ------------------------------------------------------------------ #
    # Step 3: Extract text from the PDF
    # ------------------------------------------------------------------ #
    try:
        raw_text = extract_text_from_pdf(save_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read PDF: {str(e)}"
        )

    # ------------------------------------------------------------------ #
    # Step 4: Split text into chunks
    # ------------------------------------------------------------------ #
    chunks = split_text_into_chunks(raw_text, chunk_size=500, overlap=50)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text chunks could be created from this PDF."
        )

    # ------------------------------------------------------------------ #
    # Step 5 & 6: Build FAISS index and save to disk
    # ------------------------------------------------------------------ #
    try:
        index = build_faiss_index(chunks)
        save_index(index, chunks, FAISS_INDEX_PATH, CHUNKS_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build vector index: {str(e)}"
        )

    return UploadResponse(
        message="PDF uploaded and indexed successfully! You can now ask questions.",
        chunks_created=len(chunks),
        filename=file.filename
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(body: QuestionRequest):
    """
    Accept a question and return an AI-generated answer grounded in the
    uploaded document.

    Steps:
      1. Validate the question is not empty.
      2. Call the RAG pipeline (rag.py) which retrieves context + generates answer.
      3. Return the answer and the source chunks used.

    Returns:
        AnswerResponse with the question, answer, and sources.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Validate input
    # ------------------------------------------------------------------ #
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # ------------------------------------------------------------------ #
    # Step 2: Run the RAG pipeline
    # ------------------------------------------------------------------ #
    try:
        result = answer_question(question)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e) + " Please upload a PDF document first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )

    # ------------------------------------------------------------------ #
    # Step 3: Return structured response
    # ------------------------------------------------------------------ #
    return AnswerResponse(
        question=question,
        answer=result["answer"],
        sources=result["sources"]
    )


# ---------------------------------------------------------------------------
# Health check endpoint — useful to verify the server is running
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Simple health check — returns OK if the server is running."""
    return JSONResponse(content={"status": "ok", "message": "Server is running."})