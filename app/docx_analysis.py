from typing import List
from docx import Document

def load_interview_docx(file) -> List[str]:
    """
    Extrae fragmentos de texto de un documento DOCX como lista de strings.
    """
    doc = Document(file)
    fragments = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            fragments.append(text)
    return fragments
