# app/rag_pipeline.py
import os
from pathlib import Path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

VECTOR_DIR = Path(__file__).parent / "vectorstore"

def crear_vectorstore_desde_pdf(pdf_path: Path, vector_dir: Path, api_key: str,
                                chunk_size: int = 500, chunk_overlap: int = 50) -> FAISS:
    """
    Crea un vectorstore desde un PDF y lo guarda localmente.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Cargar PDF
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    # Dividir en fragmentos
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = splitter.split_documents(docs)
    
    # Crear vectorstore
    vectordb = FAISS.from_documents(docs_split, embeddings)
    
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    
    return vectordb


def load_vectorstore_from_disk(api_key: str) -> FAISS:
    """
    Carga un vectorstore existente desde disco usando la API key.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if VECTOR_DIR.exists() and (VECTOR_DIR / "index.faiss").exists():
        return FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError("Vectorstore no encontrado. Debes crearlo desde PDF primero.")


def build_chain(vectordb: FAISS, prompt_path: Path = None, api_key: str = None) -> LLMChain:
    """
    Construye un chain LLM para análisis de entrevistas.
    """
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    
    prompt_text = "Analiza el texto de la entrevista y devuelve JSON de categorías y conceptos."
    if prompt_path:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    
    prompt = PromptTemplate(input_variables=["interview_text"], template=prompt_text)
    return LLMChain(llm=llm, prompt=prompt)
