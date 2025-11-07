import re
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime
from utils.helpers import clean_text, STOPWORDS

class DocumentProcessor:
    def __init__(self):
        self.stopwords = STOPWORDS

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []

        for doc in documents:
            cleaned = clean_text(doc['content'])

            if len(cleaned.split()) < 20:
                continue

            quality = self._calculate_quality(cleaned)
            keywords = self._extract_keywords(cleaned)

            doc['content'] = cleaned
            doc['metadata'].update({
                'quality_score': quality,
                'keywords': keywords,
                'processed_at': datetime.now().isoformat()
            })
            processed.append(doc)

        return processed

    def _calculate_quality(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0

        word_count = len(words)
        unique_words = len(set(words))
        diversity = unique_words / word_count if word_count > 0 else 0

        length_score = min(word_count / 500, 1.0)
        diversity_score = diversity

        quality = (length_score * 0.4 + diversity_score * 0.6)
        return round(quality, 2)

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self.stopwords and len(w) > 2]

        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(10)]
        return keywords
