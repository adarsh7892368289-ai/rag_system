import re
from typing import List, Dict
from dataclasses import dataclass
from collections import Counter
from config.settings import CHUNKING
from utils.helpers import split_sentences, calculate_similarity

@dataclass
class Chunk:
    text: str
    index: int
    char_count: int
    word_count: int
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'index': self.index,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'metadata': self.metadata
        }

class DocumentChunker:
    def __init__(self):
        self.config = CHUNKING
    
    def chunk(self, text: str, method: str = None) -> List[Chunk]:
        if not text or len(text.strip()) < 10:
            return [self._create_chunk(text, 0)]
        
        method = method or self.config.method
        
        if method == 'semantic':
            return self._semantic_chunk(text)
        elif method == 'paragraph':
            return self._paragraph_chunk(text)
        elif method == 'fixed_size':
            return self._fixed_size_chunk(text)
        else:
            return self._sentence_aware_chunk(text)
    
    def _sentence_aware_chunk(self, text: str) -> List[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return [self._create_chunk(text, 0)]
        
        chunks = []
        current = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if sentence_words < 3:
                continue
            
            if current_words + sentence_words > self.config.target_words and current:
                chunk_text = ' '.join(current)
                chunks.append(self._create_chunk(chunk_text, len(chunks)))
                
                overlap = self._get_overlap_sentences(current, self.config.overlap_words)
                current = overlap + [sentence]
                current_words = sum(len(s.split()) for s in current)
            else:
                current.append(sentence)
                current_words += sentence_words
        
        if current and current_words >= self.config.min_chunk_words:
            chunks.append(self._create_chunk(' '.join(current), len(chunks)))
        
        return chunks if chunks else [self._create_chunk(text, 0)]
    
    def _semantic_chunk(self, text: str) -> List[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return [self._create_chunk(text, 0)]
        
        chunks = []
        current = []
        
        for sentence in sentences:
            if not current:
                current.append(sentence)
                continue
            
            similarity = calculate_similarity(current[-1], sentence)
            should_split = (
                similarity < self.config.similarity_threshold or
                len(current) >= 5 or
                sum(len(s.split()) for s in current) > self.config.target_words
            )
            
            if should_split and current:
                chunks.append(self._create_chunk(' '.join(current), len(chunks)))
                current = [sentence]
            else:
                current.append(sentence)
        
        if current:
            chunks.append(self._create_chunk(' '.join(current), len(chunks)))
        
        return chunks if chunks else [self._create_chunk(text, 0)]
    
    def _paragraph_chunk(self, text: str) -> List[Chunk]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return [self._create_chunk(text, 0)]
        
        chunks = []
        current = []
        current_words = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_words + para_words > self.config.target_words and current:
                chunks.append(self._create_chunk('\n\n'.join(current), len(chunks)))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words
        
        if current:
            chunks.append(self._create_chunk('\n\n'.join(current), len(chunks)))
        
        return chunks if chunks else [self._create_chunk(text, 0)]
    
    def _fixed_size_chunk(self, text: str) -> List[Chunk]:
        words = text.split()
        if not words:
            return []
        
        chunks = []
        target = self.config.target_words
        overlap = self.config.overlap_words
        
        for i in range(0, len(words), target - overlap):
            chunk_words = words[i:i + target]
            if len(chunk_words) >= self.config.min_chunk_words:
                chunks.append(self._create_chunk(' '.join(chunk_words), len(chunks)))
        
        return chunks if chunks else [self._create_chunk(text, 0)]
    
    def _create_chunk(self, text: str, index: int) -> Chunk:
        words = [w.lower() for w in text.split() if len(w) > 3]
        keywords = [word for word, _ in Counter(words).most_common(5)]
        
        return Chunk(
            text=text,
            index=index,
            char_count=len(text),
            word_count=len(text.split()),
            metadata={'keywords': keywords}
        )
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_words: int) -> List[str]:
        overlap = []
        word_count = 0
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= overlap_words:
                overlap.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        return overlap