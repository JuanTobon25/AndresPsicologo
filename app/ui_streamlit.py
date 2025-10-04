# app/ui_streamlit.py
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.docx_analysis import load_interview_docx

# Imports recomendados por LangChain Community
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from app.rag_pipeline import crear_vectorstore_desde_pdf

# --- Configuración de la página ---
st.set_page_config(page_title="Chatbot GenAI Psy", layout="centered")
st.title("🤖 Asistente de Psicología - Andrés")

# --- Sidebar: API Key ---
st.sidebar.header("🔑 Configuración API")
api_key = st.sidebar.text_input("Pega tu OpenAI API Key:", type="password")
if not api_key:
    st.warning("Debes ingresar tu OpenAI API Key para continuar")
    st.stop()

# --- Inicializar historial ---
st.session_state.setdefault("chat_history", [])

# --- Paths ---
ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "vectorstore"
DATA_DIR = ROOT / "data"
prompts_dir = ROOT / "app" / "prompts"

# --- Verificar PDF(s) ---
if not DATA_DIR.exists():
    st.error("La carpeta 'data/' no existe. Coloca tus PDFs allí y reinicia.")
    st.stop()

pdf_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"]
if not pdf_files:
    st.error("No se encontró ningún PDF en 'data/'. Agrega al menos uno.")
    st.stop()

PDF_FILE = pdf_files[0]

# --- Cargar o crear vectorstore ---
vectordb = None
index_file = VECTOR_DIR / "index.faiss"
pickle_file = VECTOR_DIR / "index.pkl"

if not VECTOR_DIR.exists() or not index_file.exists() or not pickle_file.exists():
    try:
        vectordb = crear_vectorstore_desde_pdf(PDF_FILE, VECTOR_DIR, api_key=api_key)
    except Exception as e:
        st.error("Error creando vectorstore desde PDF: " + repr(e))
        st.stop()
else:
    st.warning(
        "Se detectó un vectorstore existente. Solo carga si confías en su origen."
    )
    allow = st.checkbox("Cargar vectorstore existente (permitir deserialización peligrosa)")
    if allow:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectordb = FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
            st.success("Vectorstore cargado correctamente.")
        except Exception as e:
            st.error("No fue posible cargar el vectorstore: " + repr(e))
            st.stop()
    else:
        st.info("Elimina 'vectorstore/' si deseas regenerarlo desde PDF.")
        st.stop()

# --- Construir chain ---
def build_chain(vectordb: FAISS, prompt_path: Path = None, api_key: str = None) -> LLMChain:
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    prompt_text = "Analiza el texto de la entrevista y devuelve JSON de categorías y conceptos."
    if prompt_path:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt = PromptTemplate(input_variables=["interview_text"], template=prompt_text)
    return LLMChain(llm=llm, prompt=prompt)

chain = build_chain(vectordb, api_key=api_key)

# --- Análisis de entrevistas DOCX ---
st.header("📄 Análisis de entrevistas (Word)")
uploaded_file = st.file_uploader("Sube un documento Word con la entrevista", type=["docx"])
prompt_files = sorted([p.name for p in prompts_dir.iterdir() if p.suffix in {".txt", ".md"}]) if prompts_dir.exists() else []
prompt_choice = st.selectbox("Selecciona un prompt", prompt_files)

if uploaded_file and prompt_choice:
    qa_pairs = load_interview_docx(uploaded_file)
    st.write("Fragmentos detectados (primeros 5):")
    st.write(qa_pairs[:5])

    if st.button("🔍 Analizar entrevista"):
        with st.spinner("Analizando con IA..."):
            joined_text = "\n".join(qa_pairs)
            prompt_path = prompts_dir / prompt_choice
            try:
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error("No se pudo leer el prompt seleccionado: " + repr(e))
                custom_prompt = ""

            full_input = {"interview_text": joined_text}
            try:
                # Ejecutar chain con la clave API ingresada
                analysis = chain(full_input)
                st.markdown("### 📊 Resultado del análisis")
                st.write(analysis.get("text", str(analysis)))
            except Exception as e:
                st.error("Error al ejecutar el análisis con el chain: " + repr(e))

