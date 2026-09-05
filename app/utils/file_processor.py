# app/utils/file_processor.py
from pypdf import PdfReader
from typing import List, Dict
import os

def process_pdf(file_path: str) -> List[Dict]:
    """
    Извлекает текст из PDF с метаданными.
    """
    reader = PdfReader(file_path)
    documents = []
    filename = os.path.basename(file_path)
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            documents.append({
                'page_content': text,
                'metadata': {
                    'source': filename,
                    'page': page_num,
                    'total_pages': len(reader.pages),
                }
            })
    
    return documents