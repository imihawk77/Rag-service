import hashlib
import json
from datetime import datetime, timedelta

class QueryCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get_key(self, question: str) -> str:
        return hashlib.md5(question.encode()).hexdigest()
    
    def get(self, question: str):
        key = self.get_key(question)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return data
        return None
    
    def set(self, question: str, answer: dict):
        key = self.get_key(question)
        self.cache[key] = (answer, datetime.now())