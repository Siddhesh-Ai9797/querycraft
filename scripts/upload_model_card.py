# scripts/upload_model_card.py

"""
Upload a model card to HuggingFace Hub.
Run from querycraft root:
    python scripts/upload_model_card.py
"""

from huggingface_hub import HfApi

MODEL_CARD = """---
language:
- en
license: mit
tags:
- text-to-sql
- sql
- nlp
- fine-tuning
- qlora
- lora
- phi-3
- peft
base_model: microsoft/Phi-3-mini-4k-instruct
datasets:
- b-mc2/sql-create-context
metrics:
- bleu
pipeline_tag: text-generation
---

# QueryCraft — Phi-3 Mini Fine-Tuned for Text-to-SQL

Fine-tuned **Phi-3 Mini 3.8B** using **QLoRA** on 76,000 Text-to-SQL examples.
Converts natural language questions into valid SQL queries.

## Evaluation Results

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| Exact Match | 0.0% | 82.0% |
| Execution Accuracy | 84.0% | 96.0% |
| BLEU Score | 55.79 | 96.42 |

Evaluated on 50 held-out validation examples not seen during training.

## Model Details

| Property | Value |
|---|---|
| Base model | microsoft/Phi-3-mini-4k-instruct (3.8B params) |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA) |
| LoRA rank | r=16, alpha=32 |
| Trainable parameters | 8,912,896 (0.23%) |
| Training examples | 76,577 |
| Training hardware | NVIDIA RTX 5060 Ti 8GB |
| Training time | 3 hours 2 minutes |
| Final train loss | 0.5677 |
| Max sequence length | 256 tokens |

## How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "microsoft/Phi-3-mini-4k-instruct"
adapter    = "Sid9797/querycraft-phi3-sql"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

prompt = '''### System:
You are a SQL expert. Given a database schema and a natural language question, generate a valid SQL query that answers the question. Output only the SQL query with no explanation.

### Schema:
CREATE TABLE employees (id INTEGER, name VARCHAR, department VARCHAR, salary FLOAT)

### Question:
What is the average salary by department?

### SQL:
'''

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

sql = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(sql.strip().split("\\n")[0])
# SELECT AVG(salary) FROM employees GROUP BY department
```

## Prompt Format

The model was trained on the Alpaca instruction format:
System:
You are a SQL expert. Given a database schema and a natural language question,
generate a valid SQL query that answers the question.
Output only the SQL query with no explanation.
Schema:
{CREATE TABLE statements}
Question:
{natural language question}
SQL:
{model generates SQL here}
## Training Details

- **Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
  — 78,577 examples with inline CREATE TABLE schemas
- **Train/Val split:** 76,577 train / 2,000 validation (seeded shuffle)
- **Quantization:** 4-bit NF4 with double quantization (bitsandbytes)
- **LoRA target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Optimizer:** AdamW with cosine LR schedule, warmup_ratio=0.05
- **Effective batch size:** 16 (batch_size=4, gradient_accumulation=4)
- **Packing:** Enabled — short examples concatenated to fill 256-token sequences

## Why the Base Model Scored 0% Exact Match

The base Phi-3 Mini, without fine-tuning, consistently wrapped SQL output
in markdown code fences (`` ```sql ... ``` ``) and appended semicolons.
This formatting breaks exact match evaluation even when the SQL logic is correct.
Fine-tuning on consistently formatted examples eliminated this entirely.

## Limitations

- Optimised for single-table and simple multi-table queries
- Schema must be provided as CREATE TABLE SQL statements
- Best results on English-language questions
- May struggle with highly complex nested subqueries

## Links

- **GitHub:** https://github.com/Siddhesh-Ai9797/querycraft
- **Base Model:** https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **Training Dataset:** https://huggingface.co/datasets/b-mc2/sql-create-context
"""


def main():
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/querycraft-phi3-sql"

    print(f"Uploading model card to {repo_id}...")

    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model card uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()