import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    layout="wide",
    page_title="iPhone RAG (Groq)",
    page_icon="📱"
)

st.title("📱 iPhone Q&A – Tiny RAG")
st.caption("LangChain + FAISS + Groq")

# loading
DATA_DIR = Path("data")
FILES = [
    DATA_DIR / "iphone_history.txt",
    DATA_DIR / "iphone_specs.txt",
    DATA_DIR / "iphone_care.txt",
]

TOP_K = st.slider("Top K documents", 1, 8, 5)

PROMPT_VERSION = st.radio(
    "Prompt style",
    ["v1 (hallucinate)", "v2 (loose RAG)", "v3 (strict RAG)"],
    horizontal=True,
)

# vecstore
@st.cache_resource(show_spinner=False)
def build_store():
    texts = []
    for p in FILES:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        texts.append(p.read_text(encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40
    )

    docs = []
    for p, t in zip(FILES, texts):
        for chunk in splitter.split_text(t):
            docs.append({
                "page_content": chunk,
                "metadata": {"source": p.name}
            })

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = FAISS.from_texts(
        [d["page_content"] for d in docs],
        embeddings,
        metadatas=[d["metadata"] for d in docs],
    )
    return store

# llm
@st.cache_resource(show_spinner=False)
def load_llm():
    return init_chat_model(
        model="llama-3.1-8b-instant",
        model_provider="groq",
    )

# prompt
def make_prompt(version, context, question):
    if version.startswith("v1"):
        return f"""
You are an imaginative storyteller.
Ignore accuracy and answer creatively.

Question: {question}
Answer:
"""
    if version.startswith("v2"):
        return f"""
You are a helpful iPhone expert.
Use the context if relevant. If missing, make reasonable inferences.

Context:
{context}

Question: {question}
Answer:
"""
    return f"""
You are precise and factual.
Use ONLY the context below.
If the answer is not present, say:
"I don't know based on the documents."

Context:
{context}

Question: {question}
Answer with brief citations like [source]:
"""

# ui
col1, col2 = st.columns(2)

with col1:
    if st.button("Run Indexing (Vector Store)"):
        t0 = time.time()
        st.session_state["store"] = build_store()
        st.success(f"Indexing completed in {time.time() - t0:.1f}s")

with col2:
    st.info("Groq API key must be set in .env")

# test
question = st.text_input("Ask a question about iPhone")
ask = st.button("Ask")

if ask:
    if "store" not in st.session_state:
        st.warning("Please run indexing first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        store = st.session_state["store"]
        docs = store.similarity_search(question, k=TOP_K)

        context = "\n\n".join(
            f"{d.page_content} [{d.metadata['source']}]"
            for d in docs
        )

        prompt = make_prompt(PROMPT_VERSION, context, question)

        llm = load_llm()

        with st.spinner("Generating answer..."):
            answer = llm.invoke(prompt)

        st.markdown("### 📌 Answer")
        st.write(answer.content)
