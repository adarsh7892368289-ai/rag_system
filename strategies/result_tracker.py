"""Persists query results to disk for offline analysis and reproducibility.

Writes intermediate results (per chunking strategy, before fusion) and final
results (post-fusion) to separate folders. Filenames are unique even for
concurrent batched queries, so two queries running in the same wall-clock
second don't overwrite each other.
"""

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import BASE_DIR


_QUERY_SLUG_MAX = 50

# Unique-suffix counter so concurrent saves can't collide on the same timestamp.
_save_counter = 0
_save_counter_lock = threading.Lock()


def _next_save_token() -> int:
    global _save_counter
    with _save_counter_lock:
        _save_counter += 1
        return _save_counter


class ResultTracker:

    def __init__(self, base_dir: Optional[str] = None):
        # Anchor to project root by default so save location doesn't depend on CWD.
        root = Path(base_dir) if base_dir else Path(BASE_DIR) / 'data' / 'search_results'
        self.base_dir = root
        self.intermediate_dir = root / 'intermediate'
        self.final_dir = root / 'final'

        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def save_intermediate_results(
        self,
        query: str,
        chunking_results: Dict[str, List[Dict]],
        mode: str,
        timestamp: Optional[str] = None,
    ) -> str:
        timestamp = timestamp or self._timestamp()
        filepath = self.intermediate_dir / self._build_filename(query, mode, timestamp)

        total_results = sum(len(results) for results in chunking_results.values())
        data = {
            'query': query,
            'mode': mode,
            'timestamp': timestamp,
            'total_results': total_results,
            'chunking_strategies': {
                strategy: {
                    'count': len(results),
                    'results': [self._serialize_result(r) for r in results],
                }
                for strategy, results in chunking_results.items()
            },
        }
        self._write_json(filepath, data)
        return str(filepath)

    def save_final_results(
        self,
        query: str,
        final_results: List[Dict],
        mode: str,
        timestamp: Optional[str] = None,
    ) -> str:
        timestamp = timestamp or self._timestamp()
        filepath = self.final_dir / self._build_filename(query, mode, timestamp)

        chunking_distribution: Dict[str, int] = {}
        search_distribution: Dict[str, int] = {}

        # `search_strategy` is the per-result tag set by each primitive; it
        # survives both fusion passes. `strategies_used` after the outer
        # fusion holds chunking-strategy provenance (already counted below).
        for result in final_results:
            chunking = result.get('chunking_strategy', 'unknown')
            chunking_distribution[chunking] = chunking_distribution.get(chunking, 0) + 1

            search = result.get('search_strategy', 'unknown')
            search_distribution[search] = search_distribution.get(search, 0) + 1

        data = {
            'query': query,
            'mode': mode,
            'timestamp': timestamp,
            'total_results': len(final_results),
            'chunking_distribution': chunking_distribution,
            'search_strategy_distribution': search_distribution,
            'results': [self._serialize_result(r) for r in final_results],
        }
        self._write_json(filepath, data)
        return str(filepath)

    def get_recent_results(
        self, result_type: str = 'final', limit: int = 10
    ) -> List[str]:
        directory = self.final_dir if result_type == 'final' else self.intermediate_dir
        files = sorted(
            directory.glob('*.json'), key=lambda f: f.stat().st_mtime, reverse=True
        )
        return [str(f) for f in files[:limit]]

    # ------------------------------------------------------------------ private

    def _build_filename(self, query: str, mode: str, timestamp: str) -> str:
        slug = self._slugify(query)[:_QUERY_SLUG_MAX] or 'query'
        return f"{mode}_{slug}_{timestamp}_{_next_save_token():04d}.json"

    @staticmethod
    def _serialize_result(result: Dict) -> Dict:
        metadata = result.get('metadata', {}) or {}
        serialized = {
            'content': result.get('content', ''),
            'chunking_strategy': result.get('chunking_strategy', 'unknown'),
            'search_strategy': result.get('search_strategy', 'unknown'),
            'final_score': result.get('final_score', 0.0),
            'metadata': {
                'source': metadata.get('source', 'unknown'),
                'title': metadata.get('title', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'quality_score': metadata.get('quality_score', 0.0),
                'keywords': metadata.get('keywords', []),
            },
        }
        for optional_key in ('fusion_score', 'confidence', 'strategies_used', 'strategy_ranks'):
            if optional_key in result:
                serialized[optional_key] = result[optional_key]
        return serialized

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_-]+', '_', text)
        return text.strip('_')

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    @staticmethod
    def _write_json(filepath: Path, data: Dict):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
