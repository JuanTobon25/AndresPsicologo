# app/rag_pipeline.py
import os
from pathlib import Path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

VECTOR_DIR = Path(__file__).parent / "vectorstore"

def load_vectorstore_from_disk(api_key: str) -> FAISS:
    """Carga el vectorstore existente desde disco usando la API key proporcionada"""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    index_file = VECTOR_DIR / "index.faiss"
    pickle_file = VECTOR_DIR / "index.pkl"
    if VECTOR_DIR.exists() and index_file.exists() and pickle_file.exists():
        return FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError("Vectorstore no encontrado. Debes crearlo desde PDF primero.")

def build_chain(vectordb: FAISS, prompt_path: str = None, api_key: str = None) -> LLMChain:
    """Construye el chain para análisis de entrevistas"""
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    
    # Prompt por defecto si no se proporciona archivo
    prompt_text = (
        "Analiza el texto de la entrevista y devuelve JSON de categorías y conceptos.\n"
        "Llaves esperadas: 'interview_text', 'categoria_analisis'"
    )
    if prompt_path:
        prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    
    prompt = PromptTemplate(
        input_variables=["interview_text", "categoria_analisis"],
        template=prompt_text
    )
    return LLMChain(llm=llm, prompt=prompt)
