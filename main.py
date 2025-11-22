# import streamlit as st
# import pandas as pd
# from sqlalchemy import create_engine, text, inspect
# from mlx_lm import load, generate


# st.set_page_config(page_title="UniSQL - Universal DB to SQL AI", layout="wide")
# st.title("UniSQL - Universal DB to SQL AI")
# st.caption("• Any SQL Database • Basic SQL Executor")

# MODEL_PATH = "./models/qwen2.5-coder-7b"

# @st.cache_resource
# def load_model():
#     with st.spinner("Loading model into memory..."):
#         model, tokenizer = load(MODEL_PATH)
#     return model, tokenizer

# model, tokenizer = load_model()


# st.subheader("1. Database Connection")
# db_url = st.text_input("Enter SQLAlchemy connection URL", placeholder="sqlite:///chinook.db or mysql+pymysql://user:pass@localhost/dbname")
# if not db_url:
#     st.stop()

# try:
#     engine = create_engine(db_url)
#     inspector = inspect(engine)
#     tables = inspector.get_table_names()[:2]
#     if tables:
#         st.success(f"Connected! Found tables: {tables}")
#     else:
#         st.warning("Connected, but no tables found in the database.")
# except Exception as e:
#     st.error(f"Connection failed: {e}")
#     st.stop()


# # SCHEMA LOADER
# @st.cache_data(ttl=3600)
# def get_schema(_engine):
#     inspector = inspect(_engine)
#     schema_text = ""
#     for table in inspector.get_table_names():
#         schema_text += f"-- Table: {table}\n"
#         for col in inspector.get_columns(table):
#             schema_text += f"  - {col['name']} ({col['type']})\n"
#         try:
#             df = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 2', _engine)
#             schema_text += f"-- Sample rows:\n{df.to_string(index=False)}\n\n"
#         except:
#             schema_text += "-- Sample rows unavailable\n\n"
#     return schema_text

# schema = get_schema(engine)


# # SQL SAFETY
# def is_safe_sql(sql: str) -> bool:
#     sql_clean = "\n".join([line for line in sql.splitlines() if not line.strip().startswith("--")]).strip().upper()
#     if "SELECT" not in sql_clean:
#         return False
#     sql_clean = sql_clean[sql_clean.index("SELECT"):]  # everything after first SELECT
#     forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC", "GRANT"]
#     return not any(kw in sql_clean for kw in forbidden)

# # CHAT HISTORY
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "sql" in msg:
#             with st.expander("Show SQL"):
#                 st.code(msg["sql"], language="sql")
#         if "df" in msg:
#             st.dataframe(msg["df"], width='stretch')

# # USER INPUT → LLM → SQL
# prompt = st.chat_input("Ask anything about your database")
# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Generating SQL..."):
#             full_prompt = f"""
# <|im_start|>system
# You are a helpful SQL assistant. Generate only a valid SQL SELECT query. No explanation.
# <|im_end|>

# <|im_start|>user
# Database schema:
# {schema}

# Question: {prompt}

# SQL:
# <|im_end|>
# <|im_start|>assistant
# """
#             response = generate(model, tokenizer, prompt=full_prompt, max_tokens=300)
#             sql = response.strip()
#             if "```" in sql:
#                 sql = sql.split("```")[0].strip()
#             if "SELECT" in sql.upper():
#                 sql = sql[sql.upper().index("SELECT"):].strip()

#             if not is_safe_sql(sql):
#                 st.error("Generated SQL is unsafe and blocked.")
#                 st.code(sql, language="sql")
#                 st.stop()

#             with st.expander("Generated SQL"):
#                 st.code(sql, language="sql")

#             try:
#                 df = pd.read_sql(text(sql), engine)
#                 st.success(f"Found **{len(df):,}** rows")
#                 st.dataframe(df.head(100), width='stretch')

#                 csv = df.to_csv(index=False).encode()
#                 st.download_button("Download CSV", csv, "result.csv", "text/csv")

#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "content": f"Query successful! Found **{len(df):,}** rows.",
#                     "sql": sql,
#                     "df": df.head(100)
#                 })

#             except Exception as e:
#                 st.error(f"SQL Error: {e}")
#                 st.code(sql, language="sql")


import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

st.set_page_config(page_title="UniSQL - Universal DB to SQL AI", layout="wide")
st.title("UniSQL - Universal DB to SQL AI")
st.caption("• Any SQL Database • Basic SQL Executor")

# -------------------------
# MODEL LOADING (CPU)
# -------------------------
MODEL_PATH = "./models/qwen2.5-coder-7b"

@st.cache_resource
def load_model():
    with st.spinner("Loading model into memory..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU
    return generator

generator = load_model()

# -------------------------
# DATABASE CONNECTION
# -------------------------
st.subheader("1. Database Connection")
db_url = st.text_input(
    "Enter SQLAlchemy connection URL",
    placeholder="sqlite:///chinook.db or mysql+pymysql://user:pass@localhost/dbname"
)
if not db_url:
    st.stop()

try:
    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = inspector.get_table_names()[:2]
    if tables:
        st.success(f"Connected! Found tables: {tables}")
    else:
        st.warning("Connected, but no tables found in the database.")
except Exception as e:
    st.error(f"Connection failed: {e}")
    st.stop()

# -------------------------
# SCHEMA LOADER
# -------------------------
@st.cache_data(ttl=3600)
def get_schema(_engine):
    inspector = inspect(_engine)
    schema_text = ""
    for table in inspector.get_table_names():
        schema_text += f"-- Table: {table}\n"
        for col in inspector.get_columns(table):
            schema_text += f"  - {col['name']} ({col['type']})\n"
        try:
            df = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 2', _engine)
            schema_text += f"-- Sample rows:\n{df.to_string(index=False)}\n\n"
        except:
            schema_text += "-- Sample rows unavailable\n\n"
    return schema_text

schema = get_schema(engine)

# -------------------------
# SQL SAFETY CHECK
# -------------------------
def is_safe_sql(sql: str) -> bool:
    sql_clean = "\n".join([line for line in sql.splitlines() if not line.strip().startswith("--")]).strip().upper()
    if "SELECT" not in sql_clean:
        return False
    sql_clean = sql_clean[sql_clean.index("SELECT"):]  # everything after first SELECT
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC", "GRANT"]
    return not any(kw in sql_clean for kw in forbidden)

# -------------------------
# CHAT HISTORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg:
            with st.expander("Show SQL"):
                st.code(msg["sql"], language="sql")
        if "df" in msg:
            st.dataframe(msg["df"], width='stretch')

# -------------------------
# USER INPUT → LLM → SQL
# -------------------------
prompt = st.chat_input("Ask anything about your database")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            # Construct prompt for the model
            full_prompt = f"""
You are a helpful SQL assistant. Generate only a valid SQL SELECT query. No explanation.
Database schema:
{schema}

Question: {prompt}

SQL:
"""
            # Generate SQL
            response = generator(full_prompt, max_length=300, do_sample=False)[0]["generated_text"]
            sql = response.strip()

            # Extract SELECT query
            if "SELECT" in sql.upper():
                sql = sql[sql.upper().index("SELECT"):].strip()

            if not is_safe_sql(sql):
                st.error("Generated SQL is unsafe and blocked.")
                st.code(sql, language="sql")
                st.stop()

            with st.expander("Generated SQL"):
                st.code(sql, language="sql")

            # Execute SQL
            try:
                df = pd.read_sql(text(sql), engine)
                st.success(f"Found **{len(df):,}** rows")
                st.dataframe(df.head(100), width='stretch')

                csv = df.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "result.csv", "text/csv")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Query successful! Found **{len(df):,}** rows.",
                    "sql": sql,
                    "df": df.head(100)
                })

            except Exception as e:
                st.error(f"SQL Error: {e}")
                st.code(sql, language="sql")
