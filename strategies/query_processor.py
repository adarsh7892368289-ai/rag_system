import re
from typing import Dict, List

class QueryProcessor:
    """Process and enhance search queries"""
    
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
    
    def process(self, query: str) -> Dict:
        """Process query and return cleaned and expanded versions"""
        cleaned = self._clean_query(query)
        expanded = self._expand_query(cleaned)
        keywords = self._extract_keywords(cleaned)
        
        return {
            'original': query,
            'cleaned': cleaned,
            'expanded': expanded,
            'keywords': keywords
        }
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query"""
        query = query.strip().lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        query = ' '.join(query.split())
        return query
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms (basic implementation)"""
        expansions = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nn': 'neural network',
            'nlp': 'natural language processing',
            'cv': 'computer vision'
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            if word in expansions:
                expanded_words.extend([word, expansions[word]])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        words = query.split()
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
        return keywords


class QueryRouter:
    """Automatically route queries to optimal search strategies"""
    
    @staticmethod
    def detect_query_type(query: str) -> str:
        """
        Detect query type based on patterns
        
        Returns:
            Query type: 'how-to', 'definition', 'comparison', 'list', 'keyword', 'general'
        """
        query_lower = query.lower().strip()
        
        # How-to questions
        if any(query_lower.startswith(phrase) for phrase in [
            'how to', 'how do', 'how can', 'how does', 'how should'
        ]):
            return 'how-to'
        
        # Definition questions
        if any(query_lower.startswith(phrase) for phrase in [
            'what is', 'what are', 'define', 'explain', 'describe'
        ]):
            return 'definition'
        
        # Comparison questions
        if any(word in query_lower for word in [
            'compare', 'difference', 'versus', 'vs', 'better', 'differ'
        ]):
            return 'comparison'
        
        # List/enumeration questions
        if any(query_lower.startswith(phrase) for phrase in [
            'list', 'enumerate', 'what are the', 'name the'
        ]):
            return 'list'
        
        # Short keyword queries (3 words or less)
        if len(query.split()) <= 3:
            return 'keyword'
        
        return 'general'
    
    @staticmethod
    def route_query(query: str) -> str:
        """
        Route query to optimal search strategy based on type
        
        Returns:
            Search mode: 'semantic', 'hybrid', 'parallel', 'rerank', 'mmr', 'bm25'
        """
        query_type = QueryRouter.detect_query_type(query)
        
        # Routing map: query type -> search strategy
        routing_map = {
            'how-to': 'rerank',        # Complex understanding needed
            'definition': 'semantic',   # Conceptual similarity
            'comparison': 'parallel',   # Multiple perspectives helpful
            'list': 'mmr',             # Diversity important
            'keyword': 'hybrid',        # Balance keyword + semantic
            'general': 'parallel'       # Best overall performance
        }
        
        mode = routing_map.get(query_type, 'parallel')
        print(f"🧭 Query type: '{query_type}' → Routing to: '{mode}'")
        
        return mode