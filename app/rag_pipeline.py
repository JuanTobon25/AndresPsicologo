import os
from pathlib import Path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

VECTOR_DIR = Path(__file__).parent / "vectorstore"

def load_vectorstore_from_disk() -> FAISS:
    embeddings = OpenAIEmbeddings()
    if VECTOR_DIR.exists() and (VECTOR_DIR / "index.faiss").exists():
        return FAISS.load_local(str(VECTOR_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError("Vectorstore no encontrado. Debes crearlo desde PDF primero.")

def build_chain(vectordb: FAISS, prompt_path: str = None) -> LLMChain:
    llm = ChatOpenAI(temperature=0)
    prompt_text = "Analiza el texto de la entrevista y devuelve JSON de categorías y conceptos."
    if prompt_path:
        prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    prompt = PromptTemplate(
        input_variables=["interview_text"],
        template=prompt_text
    )
    return LLMChain(llm=llm, prompt=prompt)
