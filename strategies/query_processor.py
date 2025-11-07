import re
from typing import Dict, List
from utils.helpers import STOPWORDS

class QueryProcessor:
    def __init__(self):
        self.stopwords = STOPWORDS
        self.synonyms = self._load_synonyms()
    
    def process(self, query: str) -> Dict[str, any]:
        cleaned = self._clean_query(query)
        keywords = self._extract_keywords(cleaned)
        expanded = self._expand_query(cleaned)
        query_type = self._detect_query_type(query)
        
        return {
            'original': query,
            'cleaned': cleaned,
            'expanded': expanded,
            'keywords': keywords,
            'type': query_type
        }
    
    def _clean_query(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query)
        
        words = query.split()
        words = [w for w in words if w not in self.stopwords]
        query = ' '.join(words)
        
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        words = query.split()
        return [w for w in words if len(w) > 2]
    
    def _expand_query(self, query: str) -> str:
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in self.synonyms:
                expanded_words.extend(self.synonyms[word])
        
        return ' '.join(expanded_words)
    
    def _detect_query_type(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how to', 'how do', 'how can']):
            return 'how_to'
        elif any(word in query_lower for word in ['what is', 'define', 'definition']):
            return 'definition'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return 'comparison'
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return 'explanation'
        else:
            return 'factual'
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        return {
            'car': ['vehicle', 'automobile'],
            'fix': ['repair', 'solve', 'correct'],
            'computer': ['pc', 'machine', 'system'],
            'phone': ['mobile', 'smartphone', 'device'],
            'fast': ['quick', 'rapid', 'speedy'],
            'slow': ['sluggish', 'gradual'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'compact'],
            'good': ['great', 'excellent', 'quality'],
            'bad': ['poor', 'terrible', 'awful'],
            'help': ['assist', 'support', 'aid'],
            'issue': ['problem', 'error', 'bug'],
            'start': ['begin', 'launch', 'initiate'],
            'stop': ['halt', 'cease', 'end'],
            'buy': ['purchase', 'acquire', 'get'],
            'sell': ['market', 'trade'],
            'work': ['function', 'operate', 'run'],
            'break': ['damage', 'malfunction', 'fail']
        }