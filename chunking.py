"""
 Chunking Module

Provides multiple chunking strategies for optimal retrieval performance.
Each strategy optimized for different document types and use cases.
"""

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for chunking"""
    target_words: int = 150
    overlap_words: int = 30
    min_chunk_words: int = 50


class Chunker:
    """chunking strategies"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    def sentence_aware_chunk(self, text: str) -> List[str]:
        """
        Sentence-aware chunking - respects sentence boundaries
        
        Best for: General documents, articles, documentation
        Performance: High accuracy, maintains context
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if sentence_words < 3:
                continue
            
            if current_words + sentence_words > self.config.target_words and current:
                chunks.append(' '.join(current))
                overlap = self._get_overlap(current, self.config.overlap_words)
                current = overlap + [sentence]
                current_words = sum(len(s.split()) for s in current)
            else:
                current.append(sentence)
                current_words += sentence_words
        
        if current and current_words >= self.config.min_chunk_words:
            chunks.append(' '.join(current))
        
        return chunks if chunks else [text]
    
    def semantic_chunk(self, text: str, similarity_threshold: float = 0.6) -> List[str]:
        """
        Semantic chunking - splits on topic changes
        
        Best for: Long documents with distinct sections
        Performance: Groups related content together
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current = []
        
        for i, sentence in enumerate(sentences):
            if not current:
                current.append(sentence)
                continue
            
            prev_words = set(current[-1].lower().split())
            curr_words = set(sentence.lower().split())
            
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words)
                similarity = overlap / (len(prev_words) + len(curr_words) - overlap)
            else:
                similarity = 0
            
            if similarity < similarity_threshold or len(current) >= 5:
                if current:
                    chunks.append(' '.join(current))
                current = [sentence]
            else:
                current.append(sentence)
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks if chunks else [text]
    
    def fixed_size_chunk(self, text: str) -> List[str]:
        """
        Fixed-size chunking with overlap
        
        Best for: Speed-critical applications, uniform chunks needed
        Performance: Fastest, predictable sizes
        """
        words = text.split()
        
        if not words:
            return []
        
        chunks = []
        target = self.config.target_words
        overlap = self.config.overlap_words
        
        for i in range(0, len(words), target - overlap):
            chunk = ' '.join(words[i:i + target])
            if len(chunk.split()) >= self.config.min_chunk_words:
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def paragraph_chunk(self, text: str, max_paragraphs: int = 3) -> List[str]:
        """
        Paragraph-based chunking
        
        Best for: Documents with clear paragraph structure
        Performance: Preserves document structure
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return [text] if text.strip() else []
        
        chunks = []
        current = []
        current_words = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_words + para_words > self.config.target_words and current:
                chunks.append('\n\n'.join(current))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words
        
        if current:
            chunks.append('\n\n'.join(current))
        
        return chunks if chunks else [text]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, sentences: List[str], overlap_words: int) -> List[str]:
        """Get overlap sentences from end of list"""
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


def chunk_text(text: str, 
               method: str = 'sentence_aware',
               target_words: int = 150,
               overlap_words: int = 30) -> List[str]:
    """
    Main chunking function - production interface
    
    Args:
        text: Text to chunk
        method: 'sentence_aware', 'semantic', 'fixed_size', 'paragraph'
        target_words: Target chunk size
        overlap_words: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    config = ChunkConfig(
        target_words=target_words,
        overlap_words=overlap_words,
        min_chunk_words=max(50, target_words // 3)
    )
    
    chunker = Chunker(config)
    
    methods = {
        'sentence_aware': chunker.sentence_aware_chunk,
        'semantic': chunker.semantic_chunk,
        'fixed_size': chunker.fixed_size_chunk,
        'paragraph': chunker.paragraph_chunk
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
    
    return methods[method](text)