"""
Chunking Module - Clean Production Version

Provides multiple chunking strategies optimized for RAG systems.
"""

import re
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import Counter

from config.settings import ChunkingConfig, CHUNKING
from utils.logger import get_logger, PerformanceLogger
from utils.validators import TextValidator, ConfigValidator

logger = get_logger("chunking")


@dataclass
class Chunk:
    """Enhanced chunk with metadata"""
    text: str
    index: int
    char_count: int
    word_count: int
    sentence_count: int
    start_pos: int
    end_pos: int
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'index': self.index,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'metadata': self.metadata
        }


class Chunker:
    """
    Advanced text chunking with multiple strategies
    
    Strategies:
    1. sentence_aware: Respects sentence boundaries (best for general docs)
    2. semantic: Groups by topic similarity (best for long documents)
    3. fixed_size: Fast, uniform chunks (best for speed)
    4. paragraph: Preserves structure (best for structured docs)
    5. recursive: Hierarchical chunking (best for complex documents)
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker
        
        Args:
            config: Chunking configuration (uses default if None)
        """
        self.config = config or CHUNKING
        
        ConfigValidator.validate_chunk_config(
            self.config.target_words,
            self.config.overlap_words,
            self.config.min_chunk_words
        )
        
        logger.info(f"✂️  Chunker initialized: {self.config.method}, "
                   f"{self.config.target_words}w target")
    
    def chunk(self, text: str, method: Optional[str] = None) -> List[Chunk]:
        """
        Chunk text using specified method
        
        Args:
            text: Text to chunk
            method: Override configured method
        
        Returns:
            List of Chunk objects
        """
        text = TextValidator.validate_text(text, min_length=10, field_name="Text")
        method = method or self.config.method
        
        with PerformanceLogger(f"Chunking ({method})"):
            if method == 'sentence_aware':
                chunks = self._sentence_aware_chunk(text)
            elif method == 'semantic':
                chunks = self._semantic_chunk(text)
            elif method == 'fixed_size':
                chunks = self._fixed_size_chunk(text)
            elif method == 'paragraph':
                chunks = self._paragraph_chunk(text)
            elif method == 'recursive':
                chunks = self._recursive_chunk(text)
            else:
                logger.warning(f"Unknown method '{method}', using sentence_aware")
                chunks = self._sentence_aware_chunk(text)
        
        logger.info(f"Created {len(chunks)} chunks from {len(text):,} chars")
        return chunks
    
    def _sentence_aware_chunk(self, text: str) -> List[Chunk]:
        """Sentence-aware chunking - best for general documents"""
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [self._create_chunk(text, 0, 0, len(text))]
        
        chunks = []
        current = []
        current_words = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if sentence_words < 3:
                continue
            
            if current_words + sentence_words > self.config.target_words and current:
                chunk_text = ' '.join(current)
                chunks.append(self._create_chunk(chunk_text, len(chunks), start_pos, 
                                                start_pos + len(chunk_text)))
                
                overlap = self._get_overlap_sentences(current, self.config.overlap_words)
                current = overlap + [sentence]
                current_words = sum(len(s.split()) for s in current)
                start_pos += len(chunk_text) - len(' '.join(overlap))
            else:
                current.append(sentence)
                current_words += sentence_words
        
        if current and current_words >= self.config.min_chunk_words:
            chunk_text = ' '.join(current)
            chunks.append(self._create_chunk(chunk_text, len(chunks), start_pos, 
                                            start_pos + len(chunk_text)))
        
        return chunks if chunks else [self._create_chunk(text, 0, 0, len(text))]
    
    def _semantic_chunk(self, text: str) -> List[Chunk]:
        """Semantic chunking - groups related content"""
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [self._create_chunk(text, 0, 0, len(text))]
        
        chunks = []
        current = []
        start_pos = 0
        
        for sentence in sentences:
            if not current:
                current.append(sentence)
                continue
            
            similarity = self._calculate_similarity(current[-1], sentence)
            
            should_split = (
                similarity < self.config.similarity_threshold or
                len(current) >= 5 or
                sum(len(s.split()) for s in current) > self.config.target_words
            )
            
            if should_split and current:
                chunk_text = ' '.join(current)
                chunks.append(self._create_chunk(chunk_text, len(chunks), start_pos,
                                                start_pos + len(chunk_text)))
                start_pos += len(chunk_text)
                current = [sentence]
            else:
                current.append(sentence)
        
        if current:
            chunk_text = ' '.join(current)
            chunks.append(self._create_chunk(chunk_text, len(chunks), start_pos,
                                            start_pos + len(chunk_text)))
        
        return chunks if chunks else [self._create_chunk(text, 0, 0, len(text))]
    
    def _fixed_size_chunk(self, text: str) -> List[Chunk]:
        """Fixed-size chunking with overlap"""
        words = text.split()
        
        if not words:
            return []
        
        chunks = []
        target = self.config.target_words
        overlap = self.config.overlap_words
        
        for i in range(0, len(words), target - overlap):
            chunk_words = words[i:i + target]
            
            if len(chunk_words) >= self.config.min_chunk_words:
                chunk_text = ' '.join(chunk_words)
                chunks.append(self._create_chunk(chunk_text, len(chunks), i, i + len(chunk_words)))
        
        return chunks if chunks else [self._create_chunk(text, 0, 0, len(text))]
    
    def _paragraph_chunk(self, text: str) -> List[Chunk]:
        """Paragraph-based chunking"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return [self._create_chunk(text, 0, 0, len(text))]
        
        chunks = []
        current = []
        current_words = 0
        start_pos = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_words + para_words > self.config.target_words and current:
                chunk_text = '\n\n'.join(current)
                chunks.append(self._create_chunk(chunk_text, len(chunks), start_pos,
                                                start_pos + len(chunk_text)))
                start_pos += len(chunk_text)
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words
        
        if current:
            chunk_text = '\n\n'.join(current)
            chunks.append(self._create_chunk(chunk_text, len(chunks), start_pos,
                                            start_pos + len(chunk_text)))
        
        return chunks if chunks else [self._create_chunk(text, 0, 0, len(text))]
    
    def _recursive_chunk(self, text: str, depth: int = 0, max_depth: int = 2) -> List[Chunk]:
        """Recursive chunking - hierarchical splitting"""
        word_count = len(text.split())
        
        if word_count <= self.config.target_words or depth >= max_depth:
            return [self._create_chunk(text, 0, 0, len(text))]
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 1:
            mid = len(paragraphs) // 2
            left_text = '\n\n'.join(paragraphs[:mid])
            right_text = '\n\n'.join(paragraphs[mid:])
        else:
            sentences = self._split_sentences(text)
            if len(sentences) > 1:
                mid = len(sentences) // 2
                left_text = ' '.join(sentences[:mid])
                right_text = ' '.join(sentences[mid:])
            else:
                words = text.split()
                mid = len(words) // 2
                left_text = ' '.join(words[:mid])
                right_text = ' '.join(words[mid:])
        
        left_chunks = self._recursive_chunk(left_text, depth + 1, max_depth)
        right_chunks = self._recursive_chunk(right_text, depth + 1, max_depth)
        
        all_chunks = left_chunks + right_chunks
        for i, chunk in enumerate(all_chunks):
            chunk.index = i
        
        return all_chunks
    
    def _create_chunk(self, text: str, index: int, start_pos: int, end_pos: int) -> Chunk:
        """Create Chunk object with metadata"""
        sentences = self._split_sentences(text)
        words = [w.lower() for w in text.split() if len(w) > 3]
        keywords = [word for word, _ in Counter(words).most_common(5)]
        
        return Chunk(
            text=text,
            index=index,
            char_count=len(text),
            word_count=len(text.split()),
            sentence_count=len(sentences),
            start_pos=start_pos,
            end_pos=end_pos,
            metadata={
                'keywords': keywords,
                'avg_sentence_length': len(text.split()) / len(sentences) if sentences else 0,
                'chunk_method': self.config.method
            }
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Replace common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|vs|e\.g|i\.e)\.',
                     lambda m: m.group(0).replace('.', '<DOT>'), text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_words: int) -> List[str]:
        """Get overlap sentences from end"""
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
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        word_counts = [c.word_count for c in chunks]
        char_counts = [c.char_count for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'avg_char_count': sum(char_counts) / len(char_counts),
            'total_words': sum(word_counts),
            'total_chars': sum(char_counts)
        }


if __name__ == "__main__":
    # Demo
    sample_text = """
    Machine learning is a subset of artificial intelligence. It focuses on teaching computers to learn from data.
    
    There are three main types of machine learning. Supervised learning uses labeled data. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns through trial and error.
    
    Deep learning is a subset of machine learning. It uses neural networks with multiple layers. These networks can learn complex patterns from large amounts of data.
    """
    
    chunker = Chunker()
    
    print("Testing Chunking Methods:\n")
    
    for method in ['sentence_aware', 'semantic', 'fixed_size', 'paragraph']:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print('='*60)
        
        chunks = chunker.chunk(sample_text, method=method)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i} ({chunk.word_count} words):")
            print(f"  {chunk.text[:100]}...")
        
        stats = chunker.get_chunk_statistics(chunks)
        print(f"\nStats: {stats['total_chunks']} chunks, "
              f"avg {stats['avg_word_count']:.1f} words")