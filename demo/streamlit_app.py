# demo/streamlit_app.py

"""
Streamlit demo for QueryCraft.

Run locally:
    streamlit run demo/streamlit_app.py

Or deploy free on Streamlit Community Cloud.
"""

import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(
    page_title="QueryCraft",
    page_icon="🔍",
    layout="centered",
)

st.title("QueryCraft")
st.markdown(
    "Fine-tuned Phi-3 Mini for Text-to-SQL · "
    "[GitHub](https://github.com/Siddhesh-Ai9797/querycraft) · "
    "[Model](https://huggingface.co/Sid9797/querycraft-phi3-sql)"
)
st.divider()

# --- API URL ---
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL of your running QueryCraft FastAPI server"
)

# --- Example Schemas ---
EXAMPLES = {
    "Employees": {
        "schema": "CREATE TABLE employees (id INTEGER, name VARCHAR, department VARCHAR, salary FLOAT, hire_date DATE)",
        "questions": [
            "How many employees are in the sales department?",
            "What is the average salary by department?",
            "Who was hired after 2020?",
        ]
    },
    "Students": {
        "schema": "CREATE TABLE students (id INTEGER, name VARCHAR, gpa FLOAT, major VARCHAR, graduation_year INTEGER)",
        "questions": [
            "List all students with a GPA above 3.5",
            "How many students are in each major?",
            "What is the highest GPA in computer science?",
        ]
    },
    "Orders": {
        "schema": "CREATE TABLE orders (id INTEGER, customer VARCHAR, product VARCHAR, quantity INTEGER, price FLOAT, order_date DATE)",
        "questions": [
            "What is the total revenue per product?",
            "Which customer placed the most orders?",
            "Show all orders from last month",
        ]
    },
}

# --- Input Section ---
st.subheader("Database Schema")
example_choice = st.selectbox(
    "Load an example schema",
    ["Custom"] + list(EXAMPLES.keys())
)

if example_choice != "Custom":
    default_schema = EXAMPLES[example_choice]["schema"]
    default_question = EXAMPLES[example_choice]["questions"][0]
else:
    default_schema = "CREATE TABLE your_table (column1 INTEGER, column2 VARCHAR)"
    default_question = "Write your question here"

schema = st.text_area(
    "CREATE TABLE schema",
    value=default_schema,
    height=100,
)

st.subheader("Your Question")

if example_choice != "Custom":
    question = st.selectbox(
        "Try an example question",
        EXAMPLES[example_choice]["questions"]
    )
else:
    question = st.text_input("Question", value=default_question)

# --- Generate Button ---
if st.button("Generate SQL", type="primary", use_container_width=True):
    if not question or not schema:
        st.error("Please provide both a schema and a question.")
    else:
        with st.spinner("Generating SQL..."):
            try:
                response = requests.post(
                    f"{API_URL}/generate",
                    json={"question": question, "context": schema},
                    timeout=30,
                )
                if response.status_code == 200:
                    sql = response.json()["sql"]
                    st.success("Generated SQL:")
                    st.code(sql, language="sql")
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to API at {API_URL}. "
                    "Make sure the server is running."
                )

# --- Footer ---
st.divider()
st.caption(
    "QueryCraft · Fine-tuned Phi-3 Mini 3.8B with QLoRA · "
    "82% exact match on validation set"
)