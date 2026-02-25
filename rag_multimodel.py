import streamlit as st
import ollama
import re
import tempfile
import os
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===================== Utility Functions
def clean_answer(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def combine_docs(docs, max_chars=900):
    return "\n\n".join(doc.page_content[:max_chars] for doc in docs)

# ===================== Fast PDF Loader
def fast_load_pdf(path):
    pdf = fitz.open(path)
    texts = []
    for page in pdf:
        text = page.get_text("text")
        if text and len(text.strip()) > 200:
            texts.append(text)
    return texts

# ===================== Build Retriever (Cached)
@st.cache_resource(show_spinner=False)
def build_retriever(pdf_path):
    texts = fast_load_pdf(pdf_path)
    docs = [Document(page_content=t) for t in texts]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        encode_kwargs={"batch_size": 32}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ===================== LLM Caller (MODEL-AGNOSTIC)
def call_llm(model_name, question, context):
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 300
        }
    )

    return clean_answer(response["message"]["content"])

# ===================== RAG Pipeline
def rag_chain(question, retriever, model_name, mode):
    docs = retriever.invoke(question)
    max_chars = 600 if mode == "⚡ Fast" else 1000
    context = combine_docs(docs, max_chars)
    return call_llm(model_name, question, context)

# ===================== Streamlit Config
st.set_page_config(
    page_title="Multi-Model PDF RAG",
    layout="centered"
)

# ===================== Session State
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===================== Sidebar Controls
st.sidebar.title("⚙️ Settings")

MODEL_MAP = {
    "LLaMA 3 (8B)": "llama3:8b",
    "Mistral (7B)": "mistral:7b",
    "Qwen 2.5 (7B)": "qwen2.5:7b",
    "Granite 3.3 (8B)": "granite3.3:8b",
    "DeepSeek R1 (8B)": "deepseek-r1:8b"
}

selected_model_label = st.sidebar.selectbox(
    "Select LLM Model",
    list(MODEL_MAP.keys())
)

mode = st.sidebar.radio(
    "Answer Mode",
    ["⚡ Fast", "🎯 Accurate"]
)

selected_model = MODEL_MAP[selected_model_label]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active model:** `{selected_model}`")

# ===================== Header
st.title("📘 DD-RAG")
st.markdown("""
🚀 **Features**
- Multi-Model PDF RAG System"
- Single FAISS retriever (cached)
- Model-agnostic RAG architecture
- Runtime LLM switching
- Closed-domain hallucination control
""")

# ===================== PDF Upload
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file and st.session_state.retriever is None:
    with st.spinner("Indexing PDF (first time only)..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        try:
            st.session_state.retriever = build_retriever(tmp_path)
            st.success("PDF indexed successfully ✅")
        finally:
            os.remove(tmp_path)

# ===================== Tabs
tab1, tab2 = st.tabs(["💬 Chat", "🕘 History"])

# ===================== Chat Tab
with tab1:
    if st.session_state.retriever:
        question = st.text_input(
            "Ask a question about the PDF:",
            placeholder="What is Retrieval-Augmented Generation?"
        )

        if st.button("Submit") and question.strip():
            with st.spinner(f"Generating answer using {selected_model_label}..."):
                answer = rag_chain(
                    question,
                    st.session_state.retriever,
                    selected_model,
                    mode
                )
                st.session_state.chat_history.append(
                    (selected_model_label, question, answer)
                )

            st.subheader("🧠 Answer")
            st.write(answer)
    else:
        st.info("Upload a PDF to begin.")

# ===================== History Tab
with tab2:
    if st.session_state.chat_history:
        for i, (model, q, a) in enumerate(
            reversed(st.session_state.chat_history), 1
        ):
            st.markdown(f"**Q{i} ({model})**: {q}")
            st.markdown(f"**A{i}:** {a}")
            st.markdown("---")
    else:
        st.info("No questions asked yet.")
