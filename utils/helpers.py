import re
from typing import List, Dict, Any

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    flattened = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[key] = value
        elif isinstance(value, list):
            flattened[key] = ', '.join(map(str, value))
        else:
            flattened[key] = str(value)
    return flattened

def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|vs|e\.g|i\.e)\.',
                 lambda m: m.group(0).replace('.', '<DOT>'), text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]

def calculate_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

STOPWORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
    'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
    'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them'
}