from typing import List, Dict

class SmartChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        chunks_with_metadata = []
        
        for doc in documents:
            text = doc.get('page_content', '')
            metadata = doc.get('metadata', {})
            
            if not text.strip():
                continue
            
            paragraphs = text.split('\n\n')
            current_chunk = ""
            current_length = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if current_length + len(para) > self.chunk_size and current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': f"{metadata.get('source', 'unknown')}_{len(chunks_with_metadata)}",
                    })
                    chunks_with_metadata.append({
                        'text': current_chunk,
                        'metadata': chunk_metadata,
                        'embedding': None,
                    })
                    current_chunk = ""
                    current_length = 0
                
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_length += len(para) + 2
            
            if current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': f"{metadata.get('source', 'unknown')}_{len(chunks_with_metadata)}",
                })
                chunks_with_metadata.append({
                    'text': current_chunk,
                    'metadata': chunk_metadata,
                    'embedding': None,
                })
        
        return chunks_with_metadata