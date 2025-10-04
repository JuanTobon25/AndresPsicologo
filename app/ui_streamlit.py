import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.rag_pipeline import build_chain, load_vectorstore_from_disk
from app.docx_analysis import load_interview_docx
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

st.set_page_config(page_title="Asistente Psicología", layout="centered")
st.title("🤖 Análisis de Entrevistas - Psy Bot")

VECTOR_DIR = Path(__file__).parent / "vectorstore"
DATA_DIR = Path(__file__).parent.parent / "data"
PROMPT_DIR = Path(__file__).parent / "prompts"

# Inicializar historial
st.session_state.setdefault("analysis_history", [])

# --- PDF / Vectorstore ---
pdf_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"]
if not pdf_files:
    st.error("No hay PDFs en la carpeta data/")
    st.stop()
PDF_FILE = pdf_files[0]

try:
    vectordb = load_vectorstore_from_disk()
    st.success("Vectorstore cargado correctamente.")
except FileNotFoundError:
    st.warning("Vectorstore no encontrado, crea uno desde PDF primero.")

# --- DOCX ---
st.header("📄 Análisis de entrevistas (Word)")

uploaded_file = st.file_uploader("Sube un DOCX con la entrevista", type=["docx"])
prompt_files = sorted([f.name for f in PROMPT_DIR.iterdir() if f.suffix in [".txt", ".md"]]) if PROMPT_DIR.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt", prompt_files)

if uploaded_file and prompt_choice:
    fragments = load_interview_docx(uploaded_file)
    st.write("Primeros 5 fragmentos extraídos:")
    st.write(fragments[:5])

    if st.button("🔍 Analizar entrevista"):
        st.info("Analizando con IA, esto puede tardar...")
        full_text = "\n".join(fragments)
        prompt_path = PROMPT_DIR / prompt_choice
        chain = build_chain(None, prompt_path=str(prompt_path))

        try:
            result = chain.run(interview_text=full_text)
            st.markdown("### Resultado del análisis")
            st.json(result)
        except Exception as e:
            st.error("Error ejecutando el análisis: " + repr(e))
