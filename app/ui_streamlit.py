# app/ui_streamlit.py
import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ No se encontró la variable OPENAI_API_KEY. Configúrala en un archivo .env")
    st.stop()

# --- Imports LangChain recomendados ---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from app.rag_pipeline import build_chain
from app.docx_analysis import load_interview_docx

st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")
st.title("🤖 Asistente de Psicología - Andrés")

# --- Inicializar historial (solo para chat si se quisiera usar) ---
st.session_state.setdefault("chat_history", [])

# --- Paths ---
ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "app" / "prompts"

# --- PDF a vectorstore ---
pdf_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"]
if not pdf_files:
    st.error("No se encontró ningún PDF en la carpeta 'data/'. Agrega al menos un PDF.")
    st.stop()

PDF_FILE = pdf_files[0]  # Tomamos el primer PDF disponible

def crear_vectorstore_desde_pdf(pdf_path: Path, vector_dir: Path, chunk_size=500, chunk_overlap=50):
    st.info(f"Creando vectorstore desde PDF: {pdf_path.name} (esto puede tardar)...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    st.success("Vectorstore creado y guardado en " + str(vector_dir))
    return vectordb

# --- Cargar o crear vectorstore ---
vectordb = None
index_file = VECTOR_DIR / "index.faiss"
pickle_file = VECTOR_DIR / "index.pkl"

if not VECTOR_DIR.exists() or not index_file.exists() or not pickle_file.exists():
    try:
        vectordb = crear_vectorstore_desde_pdf(PDF_FILE, VECTOR_DIR)
    except Exception as e:
        st.error("Error creando vectorstore desde PDF: " + repr(e))
        st.stop()
else:
    st.warning(
        "Se detectó un vectorstore existente. Solo cargar si confías en su origen."
    )
    allow = st.checkbox("✅ Confirmo que confío en este vectorstore y deseo cargarlo")
    if allow:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectordb = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
            st.success("Vectorstore cargado correctamente.")
        except Exception as e:
            st.error("No fue posible cargar el vectorstore existente: " + repr(e))
            st.stop()
    else:
        st.info("El vectorstore no se cargará. Elimina la carpeta 'vectorstore/' para regenerarlo.")
        st.stop()

# --- Construir chain ---
try:
    chain = build_chain(vectordb)
except Exception as e:
    st.error("Error al construir la cadena (chain) con el vectorstore: " + repr(e))
    st.stop()

# --- Análisis de entrevistas DOCX ---
st.header("📄 Análisis de entrevistas (Word)")
uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])

prompt_files = sorted([p.name for p in PROMPTS_DIR.iterdir() if p.suffix in {".txt", ".md"}]) if PROMPTS_DIR.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt", prompt_files)

if uploaded_file and prompt_choice:
    qa_pairs = load_interview_docx(uploaded_file)
    st.write("Preguntas y respuestas detectadas (primeros 5 fragmentos):")
    st.write(qa_pairs[:5])

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            joined_text = "\n".join(qa_pairs)
            prompt_path = PROMPTS_DIR / prompt_choice
            try:
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error("No se pudo leer el prompt seleccionado: " + repr(e))
                custom_prompt = ""

            # --- Entrada para el chain ---
            chain_input = {
                "interview_text": joined_text,
                "prompt_text": custom_prompt,
                "categoria_analisis": None  # El chain generará las categorías automáticamente
            }

            try:
                analysis = chain.invoke(chain_input)
                st.markdown("### 📊 Resultado del análisis (JSON)")
                st.json(analysis)
            except Exception as e:
                st.error("Error al ejecutar el análisis con el chain: " + repr(e))
