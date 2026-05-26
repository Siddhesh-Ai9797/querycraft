# QueryCraft — Fine-Tuned LLM for Text-to-SQL

![CI](https://github.com/Siddhesh-Ai9797/querycraft/actions/workflows/ci.yml/badge.svg)

Fine-tuned **Phi-3 Mini 3.8B** using **QLoRA** on 76,000 Text-to-SQL examples.
Exact match accuracy improved from **0% → 82%** on a held-out validation set.
Served via a **FastAPI** inference endpoint, containerised with **Docker**,
and deployed with a **GitHub Actions** CI/CD pipeline.

---

## Results

| Metric | Base Model | Fine-Tuned | Improvement |
|---|---|---|---|
| Exact Match | 0.0% | 82.0% | +82 pts |
| Execution Accuracy | 84.0% | 96.0% | +12 pts |
| BLEU Score | 55.79 | 96.42 | +40.63 pts |

Evaluated on 50 held-out validation examples not seen during training.

---

## What It Does

Send a natural language question and a database schema — get back a valid SQL query.

**Input:**
```json
{
  "question": "How many employees are in the sales department?",
  "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, department VARCHAR)"
}
```

**Output:**
```json
{
  "sql": "SELECT COUNT(*) FROM employees WHERE department = 'sales'",
  "question": "How many employees are in the sales department?",
  "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, department VARCHAR)"
}
```

---

## Model

| Property | Value |
|---|---|
| Base model | microsoft/Phi-3-mini-4k-instruct (3.8B) |
| Fine-tuning method | QLoRA (4-bit NF4 quantization + LoRA r=16) |
| Trainable parameters | 8,912,896 (0.23% of total) |
| Training dataset | b-mc2/sql-create-context (76,577 examples) |
| Training hardware | NVIDIA RTX 5060 Ti 8GB (Blackwell) |
| Training time | 3 hours 2 minutes |
| Final train loss | 0.5677 |

Adapter weights on HuggingFace Hub:
**[Sid9797/querycraft-phi3-sql](https://huggingface.co/Sid9797/querycraft-phi3-sql)**

---

## Architecture
Dataset (76K examples)
↓
Prompt Formatter (Alpaca template)
↓
Tokenizer (Phi-3 Mini, MAX_LENGTH=256, packing enabled)
↓
QLoRA Fine-Tuning (bitsandbytes 4-bit + PEFT LoRA + TRL SFTTrainer)
↓
LoRA Adapter (saved to HuggingFace Hub)
↓
Evaluation Harness (Exact Match / Execution Accuracy / BLEU)
↓
FastAPI Inference Server → Docker Container → GitHub Actions CI/CD

---

## Tech Stack

**ML/Training**
- PyTorch 2.11 + CUDA 12.8
- HuggingFace Transformers, PEFT, TRL, datasets
- bitsandbytes (4-bit quantization)

**Serving**
- FastAPI + Uvicorn
- Pydantic v2
- Docker (multi-stage CUDA build)

**MLOps**
- GitHub Actions CI/CD (lint → test → build → health check)
- HuggingFace Hub (model hosting)
- Ruff (linting)
- Pytest (unit tests with mock model)

---

## Run Locally

**Prerequisites:** Python 3.11, CUDA-capable GPU, conda

```bash
git clone https://github.com/Siddhesh-Ai9797/querycraft.git
cd querycraft
conda activate ai-env
pip install -r requirements.txt
```

**Start the API server:**
```bash
python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

**Run with Docker (CPU/mock mode):**
```bash
docker build -t querycraft:latest .
docker run -e MOCK_MODEL=true -p 8000:8000 querycraft:latest
```

**Run the Streamlit demo:**
```bash
streamlit run demo/streamlit_app.py
```

---

## API Reference

### `GET /health`
Returns server status and model readiness.

```bash
curl http://localhost:8000/health
```

### `POST /generate`
Generate SQL from a natural language question.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the average salary by department?",
    "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, department VARCHAR, salary FLOAT)"
  }'
```

---

## Run Tests

```bash
MOCK_MODEL=true PYTHONPATH=. pytest tests/ -v
```

All 5 tests run without a GPU using the mock model flag.

---

## Links

- **HuggingFace Model:** https://huggingface.co/Sid9797/querycraft-phi3-sql
- **GitHub:** https://github.com/Siddhesh-Ai9797/querycraft
- **Training Dataset:** https://huggingface.co/datasets/b-mc2/sql-create-context
- **Base Model:** https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

---

*Built as part of an ML Engineer portfolio project.*
*MS in Artificial Intelligence, DePaul University, Chicago.*
