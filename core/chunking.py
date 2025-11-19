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
    
    def __init__(self):
        self.common_patterns = [
            r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(is|are|was|were|be|been|being|have|has|had)\b',
            r'\b(this|that|these|those|what|which|who|when|where|why|how)\b',
        ]
        self.common_regex = re.compile('|'.join(self.common_patterns), re.IGNORECASE)
    
    def extract(self, text: str, top_k: int = 5) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        if not words:
            return []
        
        word_freq = Counter(words)
        total_words = len(words)
        
        scored_words = []
        for word, count in word_freq.items():
            if self.common_regex.match(word):
                continue
            
            tf = count / total_words
            idf = math.log(total_words / count + 1)
            length_bonus = min(len(word) / 15, 1.0)
            
            score = 0.4 * tf * 100 + 0.4 * idf + 0.2 * length_bonus
            scored_words.append((word, score))
        
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored_words[:top_k]]


class ChunkQualityScorer:
    
    def __init__(self):
        self.keyword_extractor = AdaptiveKeywordExtractor()
    
    def score_chunk(self, chunk: Chunk, document_stats: Dict = None) -> float:
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
    
    def __init__(self):
        self.config = CHUNKING
        self.quality_scorer = ChunkQualityScorer()
        self.keyword_extractor = AdaptiveKeywordExtractor()
    
    def chunk_all_strategies(self, text: str) -> Dict[str, List[Chunk]]:
        if not text or len(text.strip()) < 10:
            return {}
        
        strategies = {
            'sentence_aware': self._sentence_aware_chunk,
            'semantic': self._semantic_chunk,
            'paragraph': self._paragraph_chunk,
            'fixed_size': self._fixed_size_chunk
        }
        
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_strategy = {
                executor.submit(chunker, text): name 
                for name, chunker in strategies.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    chunks = future.result()
                    for chunk in chunks:
                        chunk.metadata['chunking_strategy'] = strategy_name
                    all_results[strategy_name] = chunks
                except Exception as e:
                    print(f"   ⚠️  {strategy_name} failed: {e}")
                    all_results[strategy_name] = []
        
        for strategy_name, chunks in all_results.items():
            if chunks:
                doc_stats = self._calculate_document_stats(chunks)
                for chunk in chunks:
                    quality = self.quality_scorer.score_chunk(chunk, doc_stats)
                    chunk.metadata['quality_score'] = quality
        
        return all_results
    
    def _calculate_document_stats(self, chunks: List[Chunk]) -> Dict:
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