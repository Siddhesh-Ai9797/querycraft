# src/data/prompt_formatter.py

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language "
    "question, generate a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)

PROMPT_TEMPLATE = """\
### System:
{system}

### Schema:
{context}

### Question:
{question}

### SQL:
"""


def format_training_example(example: dict) -> str:
    """
    Format one example into a complete training string.

    The model sees prompt + answer during training so it learns
    to complete the '### SQL:' section with valid SQL.

    Args:
        example: dict with keys 'question', 'context', 'answer'

    Returns:
        Full string: prompt + gold SQL answer
    """
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=example["context"],
        question=example["question"],
    )
    return prompt + example["answer"]


def format_inference_prompt(question: str, context: str) -> str:
    """
    Format a prompt for inference only — no answer appended.
    The model generates the SQL as its completion.

    Args:
        question: Natural language question
        context:  CREATE TABLE schema string

    Returns:
        Prompt string ending at '### SQL:\\n'
    """
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )