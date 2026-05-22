# src/serving/app.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.serving.inference import generator


# --- Request and Response Models ---

class GenerateRequest(BaseModel):
    """
    What the API caller must send.

    question: Natural language question about the data
    context:  CREATE TABLE SQL describing the database schema
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
    )
    context: str = Field(
        ...,
        min_length=10,
        max_length=2000,
    )


class GenerateResponse(BaseModel):
    """What the API returns."""
    sql:      str
    question: str
    context:  str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


# --- Lifespan: runs on startup and shutdown ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model when the server starts.
    FastAPI's lifespan replaces the old @app.on_event("startup").
    Code before yield runs on startup, code after yield on shutdown.
    """
    print("Server starting — loading model...")
    generator.load()
    print("Server ready")
    yield
    print("Server shutting down")


# --- FastAPI App ---

app = FastAPI(
    title="QueryCraft",
    description=(
        "Fine-tuned Phi-3 Mini for Text-to-SQL. "
        "Send a natural language question and a database schema, "
        "get back a valid SQL query."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.
    Used by Docker and load balancers to verify the server is alive.
    """
    return HealthResponse(
        status="ok",
        model_ready=generator.ready,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """
    Generate a SQL query from a natural language question.

    Send a POST request with:
    - question: what you want to know in plain English
    - context:  the CREATE TABLE schema of your database

    Returns the generated SQL query.
    """
    if not generator.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not ready yet. Try again in a few seconds."
        )

    sql = generator.generate(
        question=request.question,
        context=request.context,
    )

    return GenerateResponse(
        sql=sql,
        question=request.question,
        context=request.context,
    )


@app.get("/")
def root():
    return {
        "message": "QueryCraft API",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate",
    }