import re
import math
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import CHUNKING


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


class AdaptiveKeywordExtractor:
    """Extract keywords using TF-IDF-like scoring (no hardcoded stopwords)"""
    
    def __init__(self):
        self.common_patterns = [
            r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(is|are|was|were|be|been|being|have|has|had)\b',
            r'\b(this|that|these|those|what|which|who|when|where|why|how)\b',
        ]
        self.common_regex = re.compile('|'.join(self.common_patterns), re.IGNORECASE)
    
    def extract(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords using statistical scoring - FIXED VERSION"""
        # CRITICAL FIX: Extract words properly, not characters
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # Words with 4+ letters
        
        if not words:
            return []
        
        word_freq = Counter(words)
        total_words = len(words)
        
        scored_words = []
        for word, count in word_freq.items():
            # Skip common function words
            if self.common_regex.match(word):
                continue
            
            # TF-IDF-like scoring
            tf = count / total_words
            idf = math.log(total_words / count + 1)
            length_bonus = min(len(word) / 15, 1.0)
            
            score = 0.4 * tf * 100 + 0.4 * idf + 0.2 * length_bonus
            scored_words.append((word, score))
        
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored_words[:top_k]]


class ChunkQualityScorer:
    """Adaptive quality scoring"""
    
    def __init__(self):
        self.keyword_extractor = AdaptiveKeywordExtractor()
    
    def score_chunk(self, chunk: Chunk, document_stats: Dict = None) -> float:
        """Calculate adaptive quality score"""
        if document_stats is None:
            document_stats = self._default_stats()
        
        length_score = self._score_length_adaptive(
            chunk.word_count, 
            document_stats.get('avg_chunk_words', 300),
            document_stats.get('std_chunk_words', 100)
        )
        
        keyword_score = self._score_keywords_adaptive(chunk)
        coherence_score = self._score_coherence_adaptive(chunk.text)
        info_score = self._score_information_adaptive(chunk.text)
        
        if self._is_boilerplate(chunk.text):
            return 0.0
        
        quality = (
            0.30 * length_score +
            0.25 * keyword_score +
            0.25 * coherence_score +
            0.20 * info_score
        )
        
        return round(quality, 3)
    
    def _score_length_adaptive(self, word_count: int, avg: float, std: float) -> float:
        """Score based on z-score distance from average"""
        if std == 0:
            return 1.0 if word_count > 50 else 0.3
        
        z_score = abs((word_count - avg) / std)
        
        if z_score <= 1:
            return 1.0 - (z_score * 0.2)
        elif z_score <= 2:
            return 0.8 - ((z_score - 1) * 0.3)
        else:
            return max(0.3, 0.5 - ((z_score - 2) * 0.1))
    
    def _score_keywords_adaptive(self, chunk: Chunk) -> float:
        """Score based on keyword density"""
        keywords = chunk.metadata.get('keywords', [])
        
        if chunk.word_count == 0:
            return 0.0
        
        density = len(keywords) / chunk.word_count
        
        if chunk.word_count < 100:
            optimal_density = 0.08
        elif chunk.word_count < 300:
            optimal_density = 0.05
        else:
            optimal_density = 0.03
        
        ratio = density / optimal_density if optimal_density > 0 else 0
        
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif ratio < 0.5:
            return 0.3
        else:
            return max(0.5, 1.0 - abs(1.0 - ratio) * 0.5)
    
    def _score_coherence_adaptive(self, text: str) -> float:
        """Score sentence structure consistency"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not sentences or len(sentences) < 2:
            return 0.6
        
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        consistency_score = 1.0 / (1.0 + std_dev / avg_length) if avg_length > 0 else 0.5
        fragment_ratio = sum(1 for l in lengths if l < 5) / len(lengths)
        
        return max(0.3, consistency_score - fragment_ratio * 0.3)
    
    def _score_information_adaptive(self, text: str) -> float:
        """Score information density using universal patterns"""
        words = text.split()
        
        if not words:
            return 0.0
        
        unique_ratio = len(set(words)) / len(words)
        diversity_score = min(unique_ratio * 1.5, 1.0)
        
        numeric_count = len(re.findall(r'\b\d+\b', text))
        numeric_score = min(numeric_count / len(words), 0.2) * 5
        
        entities = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
        entity_score = min(len(entities) / len(words), 0.15) * 6.67
        
        return 0.5 * diversity_score + 0.25 * numeric_score + 0.25 * entity_score
    
    def _is_boilerplate(self, text: str) -> bool:
        """Detect boilerplate using universal patterns"""
        text_lower = text.lower()
        
        nav_patterns = [
            r'jump to (content|navigation|main)',
            r'skip to',
            r'(menu|navigation|footer|header)\s*$',
            r'^\s*(home|about|contact|search|login|sign up)\s*$',
            r'copyright.*all rights reserved',
            r'(privacy policy|terms of service|cookie)',
            r'^\s*\d+\s*$',
        ]
        
        for pattern in nav_patterns:
            if re.search(pattern, text_lower):
                return True
        
        words = text.split()
        if len(words) < 20:
            unique_ratio = len(set(words)) / len(words) if words else 0
            if unique_ratio > 0.9:
                return True
        
        return False
    
    def _default_stats(self) -> Dict:
        return {
            'avg_chunk_words': 300,
            'std_chunk_words': 100
        }


class EnsembleChunker:
    """
    Apply multiple chunking strategies and select best chunks
    
    Modern RAG approach:
    1. Run all 4 strategies in parallel
    2. Deduplicate similar chunks
    3. Score all chunks for quality
    4. Select top N chunks globally
    """
    
    def __init__(self):
        self.config = CHUNKING
        self.quality_scorer = ChunkQualityScorer()
        self.keyword_extractor = AdaptiveKeywordExtractor()
        self.embedding_model = None
    
    def chunk_with_ensemble(self, text: str, max_chunks: int = 20) -> List[Chunk]:
        """
        Apply all chunking strategies and return best chunks
        
        Args:
            text: Document text
            max_chunks: Maximum chunks to keep per document
            
        Returns:
            Best quality chunks from all strategies
        """
        if not text or len(text.strip()) < 10:
            return []
        
        print(f"   📊 Running ensemble chunking (4 strategies)...")
        
        # Step 1: Run all 4 strategies in parallel
        all_chunks = self._run_all_strategies_parallel(text)
        
        if not all_chunks:
            return []
        
        print(f"   ├─ Generated {len(all_chunks)} candidate chunks")
        
        # Step 2: Calculate document statistics
        doc_stats = self._calculate_document_stats(all_chunks)
        
        # Step 3: Score all chunks
        for chunk in all_chunks:
            quality = self.quality_scorer.score_chunk(chunk, doc_stats)
            chunk.metadata['quality_score'] = quality
        
        # Step 4: Deduplicate similar chunks
        unique_chunks = self._deduplicate_chunks(all_chunks)
        print(f"   ├─ Deduplicated to {len(unique_chunks)} unique chunks")
        
        # Step 5: Select best N chunks
        best_chunks = self._select_best_chunks(unique_chunks, max_chunks)
        print(f"   └─ Selected top {len(best_chunks)} chunks")
        
        return best_chunks
    
    def _run_all_strategies_parallel(self, text: str) -> List[Chunk]:
        """Run all 4 chunking strategies in parallel"""
        strategies = {
            'sentence_aware': self._sentence_aware_chunk,
            'semantic': self._semantic_chunk,
            'paragraph': self._paragraph_chunk,
            'fixed_size': self._fixed_size_chunk
        }
        
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_strategy = {
                executor.submit(chunker, text): name 
                for name, chunker in strategies.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    chunks = future.result()
                    # Tag each chunk with its strategy
                    for chunk in chunks:
                        chunk.metadata['chunking_strategy'] = strategy_name
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"   ⚠️  {strategy_name} failed: {e}")
        
        return all_chunks
    
    def _deduplicate_chunks(self, chunks: List[Chunk], threshold: float = 0.90) -> List[Chunk]:
        """
        Remove near-duplicate chunks using text similarity
        
        Uses Jaccard similarity (word overlap) - fast and effective
        """
        if not chunks:
            return []
        
        unique_chunks = []
        seen_texts = []
        
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            is_duplicate = False
            
            for seen_text in seen_texts:
                seen_words = set(seen_text.lower().split())
                
                # Jaccard similarity
                intersection = len(chunk_words & seen_words)
                union = len(chunk_words | seen_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_texts.append(chunk.text)
        
        return unique_chunks
    
    def _select_best_chunks(self, chunks: List[Chunk], max_chunks: int) -> List[Chunk]:
        """
        Select best N chunks ensuring document coverage
        
        Strategy:
        1. Sort by quality score
        2. Ensure coverage of document sections (beginning, middle, end)
        """
        if len(chunks) <= max_chunks:
            return sorted(chunks, key=lambda c: c.metadata['quality_score'], reverse=True)
        
        # Sort by quality
        sorted_chunks = sorted(chunks, key=lambda c: c.metadata['quality_score'], reverse=True)
        
        # Divide into sections for coverage
        total_chars = sum(c.char_count for c in chunks)
        section_size = total_chars / 3
        
        beginning = [c for c in sorted_chunks if c.index * 1000 < section_size]
        middle = [c for c in sorted_chunks if section_size <= c.index * 1000 < 2 * section_size]
        end = [c for c in sorted_chunks if c.index * 1000 >= 2 * section_size]
        
        # Take proportionally from each section
        chunks_per_section = max_chunks // 3
        
        selected = (
            beginning[:chunks_per_section] +
            middle[:chunks_per_section] +
            end[:max_chunks - 2 * chunks_per_section]
        )
        
        # If not enough, fill from sorted list
        if len(selected) < max_chunks:
            for chunk in sorted_chunks:
                if chunk not in selected:
                    selected.append(chunk)
                    if len(selected) >= max_chunks:
                        break
        
        return selected[:max_chunks]
    
    def _calculate_document_stats(self, chunks: List[Chunk]) -> Dict:
        """Calculate document-level statistics"""
        if not chunks:
            return {'avg_chunk_words': 300, 'std_chunk_words': 100}
        
        word_counts = [c.word_count for c in chunks]
        avg = sum(word_counts) / len(word_counts)
        variance = sum((wc - avg) ** 2 for wc in word_counts) / len(word_counts)
        std = math.sqrt(variance)
        
        return {
            'avg_chunk_words': avg,
            'std_chunk_words': max(std, 50),
            'total_chunks': len(chunks)
        }
    
    def _sentence_aware_chunk(self, text: str) -> List[Chunk]:
        """Chunk respecting sentence boundaries"""
        from utils.helpers import split_sentences
        
        sentences = split_sentences(text)
        if not sentences:
            return []
        
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
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[Chunk]:
        """Chunk based on semantic similarity"""
        from utils.helpers import split_sentences, calculate_similarity
        
        sentences = split_sentences(text)
        if not sentences:
            return []
        
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
        
        return chunks
    
    def _paragraph_chunk(self, text: str) -> List[Chunk]:
        """Chunk based on paragraph boundaries"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return []
        
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
        
        return chunks
    
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
                chunks.append(self._create_chunk(' '.join(chunk_words), len(chunks)))
        
        return chunks
    
    def _create_chunk(self, text: str, index: int) -> Chunk:
        """Create chunk with keywords"""
        keywords = self.keyword_extractor.extract(text, top_k=5)
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        return Chunk(
            text=text,
            index=index,
            char_count=len(text),
            word_count=len(text.split()),
            metadata={
                'keywords': keywords,
                'sentence_count': sentence_count
            }
        )
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_words: int) -> List[str]:
        """Get sentences for overlap"""
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


# Legacy single-strategy chunker (kept for backward compatibility)
class DocumentChunker:
    """Single-strategy chunker - use EnsembleChunker for better results"""
    
    def __init__(self):
        self.config = CHUNKING
        self.quality_scorer = ChunkQualityScorer()
        self.keyword_extractor = AdaptiveKeywordExtractor()
    
    def chunk(self, text: str, method: str = None, filter_quality: bool = True) -> List[Chunk]:
        """Chunk with single strategy (legacy mode)"""
        ensemble = EnsembleChunker()
        
        # For single strategy, just run that one
        if method == 'semantic':
            chunks = ensemble._semantic_chunk(text)
        elif method == 'paragraph':
            chunks = ensemble._paragraph_chunk(text)
        elif method == 'fixed_size':
            chunks = ensemble._fixed_size_chunk(text)
        else:
            chunks = ensemble._sentence_aware_chunk(text)
        
        if not chunks:
            return []
        
        # Score and filter
        doc_stats = ensemble._calculate_document_stats(chunks)
        
        for chunk in chunks:
            quality = self.quality_scorer.score_chunk(chunk, doc_stats)
            chunk.metadata['quality_score'] = quality
            chunk.metadata['chunking_method'] = method or 'sentence_aware'
        
        if filter_quality and len(chunks) > 5:
            quality_scores = [c.metadata['quality_score'] for c in chunks]
            threshold = self._calculate_adaptive_threshold(quality_scores)
            
            original_count = len(chunks)
            chunks = [c for c in chunks if c.metadata['quality_score'] >= threshold]
            
            if len(chunks) < original_count:
                print(f"   ⚡ Filtered {original_count - len(chunks)} low-quality chunks "
                      f"({original_count} → {len(chunks)}, threshold: {threshold:.2f})")
        
        return chunks
    
    def _calculate_adaptive_threshold(self, quality_scores: List[float]) -> float:
        """Calculate adaptive threshold using percentile"""
        if not quality_scores:
            return 0.0
        
        sorted_scores = sorted(quality_scores)
        percentile_25 = sorted_scores[len(sorted_scores) // 4]
        return max(0.3, min(percentile_25, 0.7))