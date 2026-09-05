# app/core/rag_chain.py
from typing import List, Dict

class RAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def ask(self, question: str) -> Dict:
        retrieved_results = self.retriever.retrieve(question)
        
        sources = []
        context_parts = []
        
        for res in retrieved_results:
            chunk = res['chunk']
            source = chunk['metadata'].get('source', 'unknown')
            context_parts.append(f"[Source: {source}]\n{chunk['text']}")
            sources.append({
                'text': chunk['text'],
                'source': source,
                'score': res['score'],
            })
        
        context = "\n\n---\n\n".join(context_parts)
        answer = f"Найдено {len(sources)} релевантных фрагментов:\n\n{context}"
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
        }