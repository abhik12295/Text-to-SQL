import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, inspect, text
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer, util
import numpy as np

st.set_page_config(page_title="UniQL Natural Language-to-SQL", layout="wide")


@st.cache_resource
def load_model():
    with st.spinner("Loading local model..."):
        return load("./models/qwen2.5-coder-7b")

model, tokenizer = load_model()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

def connect_db(url: str):
    try:
        engine = create_engine(url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        columns = {t: [c["name"] for c in inspector.get_columns(t)] for t in tables}
        return engine, tables, columns, None
    except Exception as e:
        return None, None, None, str(e)


def clean_llm_output(output: str) -> str:
    sql = output.strip()
    sql = re.sub(r"<\|.*?\|>", " ", sql)
    sql = sql.replace("assistant", "").replace("user", "")
    sql = sql.replace("```sql", "").replace("```", "")
    up = sql.upper()
    if "SELECT" in up:
        sql = sql[up.index("SELECT"):]
    return sql.strip()

def is_safe_sql(sql: str) -> bool:
    sql_clean = sql.strip().upper()
    if not sql_clean.startswith("SELECT"):
        return False
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", 
                 "TRUNCATE", "CREATE", "REPLACE", "EXEC", "GRANT", "REVOKE"]
    return not any(f" {kw} " in f" {sql_clean} " for kw in forbidden)

# symentic shema search
def build_schema_text(tables, columns):
    schema_text = []
    for t in tables:
        cols = ", ".join(columns[t])
        schema_text.append(f"Table {t} with columns: {cols}")
    return schema_text

def compute_schema_embeddings(schema_text):
    return embed_model.encode(schema_text, convert_to_tensor=True)

def find_relevant_schema(user_query, schema_text, schema_embeddings, top_k=7):
    query_emb = embed_model.encode(user_query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, schema_embeddings)[0]
    top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
    return [schema_text[i] for i in top_results if cos_scores[i] > 0.25]

# chat history
if "chat" not in st.session_state:
    st.session_state.chat = []


st.title("UniQL Natural Language-to-SQL (Semantic & Schema Safe)")

st.subheader("Database Connection")
db_url = st.text_input("Enter SQLAlchemy Connection URL", placeholder="sqlite:///chinook.db")

if "engine" not in st.session_state:
    st.session_state.engine = None
    st.session_state.tables = None
    st.session_state.columns = None
    st.session_state.schema_text = None
    st.session_state.schema_embeds = None

if st.button("Connect"):
    engine, tables, columns, err = connect_db(db_url)
    if err:
        st.error(f"Connection failed: {err}")
    else:
        st.success(f"Connected! Found tables: {tables[:5]}...")
        st.session_state.engine = engine
        st.session_state.tables = tables
        st.session_state.columns = columns
        st.session_state.schema_text = build_schema_text(tables, columns)
        st.session_state.schema_embeds = compute_schema_embeddings(st.session_state.schema_text)

# Stop if not connected
if not st.session_state.engine:
    st.stop()

# Show chat history
st.subheader("Query History")
for msg in st.session_state.chat:
    with st.container():
        st.markdown(f"**🧑 User:** {msg['question']}")
        with st.expander("Generated SQL"):
            st.code(msg["sql"], language="sql")
        st.markdown("**🔍 Result:**")
        st.dataframe(msg["df"], width='content')

# Chat Input
st.subheader("Ask a Question")
user_question = st.chat_input("Ask anything about your data...")

if user_question:
    # Find relevant tables/columns
    relevant_schema = find_relevant_schema(
        user_question,
        st.session_state.schema_text,
        st.session_state.schema_embeds,
        top_k=7
    )
    schema_text = "\n".join(relevant_schema)

    # Build prompt with strict schema 

    llm_prompt = f"""<|im_start|>system
You are an expert SQL analyst. Return ONLY a valid SQLite SELECT query.
You MUST only use tables and columns from the schema provided below.
You may map common words using synonyms provided.
Do not invent any table or column names.
<|im_end|>
<|im_start|>user
Database schema (relevant tables & columns):
{schema_text}

Question: {user_question}

SQL:<|im_end|>
<|im_start|>assistant
"""

    # Generate SQL
    with st.spinner("Generating SQL..."):
        raw = generate(model, tokenizer, prompt=llm_prompt, max_tokens=300)
        sql = clean_llm_output(raw)

    if not is_safe_sql(sql):
        st.error("Generated SQL is unsafe and was blocked.")
        st.code(sql)
        st.stop()

    # Execute SQL
    try:
        with st.spinner("Running query..."):
            df = pd.read_sql(text(sql), st.session_state.engine.connect())
        st.success(f"Query OK — returned {len(df)} rows")

        st.session_state.chat.append({
            "question": user_question,
            "sql": sql,
            "df": df
        })

        st.rerun()
    except Exception as e:
        st.error(f"SQL Error: {e}")
        st.code(sql)
