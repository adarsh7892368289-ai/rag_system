import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class ResultTracker:
    
    def __init__(self, base_dir: str = 'data/search_results'):
        self.base_dir = Path(base_dir)
        self.intermediate_dir = self.base_dir / 'intermediate'
        self.final_dir = self.base_dir / 'final'
        
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
    
    def save_intermediate_results(self, query: str, chunking_results: Dict[str, List[Dict]], 
                                 mode: str, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        query_slug = self._slugify(query)[:50]
        filename = f"{mode}_{query_slug}_{timestamp}.json"
        filepath = self.intermediate_dir / filename
        
        total_results = sum(len(results) for results in chunking_results.values())
        
        data = {
            'query': query,
            'mode': mode,
            'timestamp': timestamp,
            'total_results': total_results,
            'chunking_strategies': {
                strategy: {
                    'count': len(results),
                    'results': [self._serialize_result(r) for r in results]
                }
                for strategy, results in chunking_results.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_final_results(self, query: str, final_results: List[Dict], 
                          mode: str, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        query_slug = self._slugify(query)[:50]
        filename = f"{mode}_{query_slug}_{timestamp}.json"
        filepath = self.final_dir / filename
        
        chunking_distribution = {}
        search_distribution = {}
        
        for result in final_results:
            chunking_strat = result.get('chunking_strategy', 'unknown')
            search_strat = result.get('search_strategy', 'unknown')
            
            chunking_distribution[chunking_strat] = chunking_distribution.get(chunking_strat, 0) + 1
            search_strat_list = result.get('strategies_used', [search_strat])
            for strat in search_strat_list:
                search_distribution[strat] = search_distribution.get(strat, 0) + 1
        
        data = {
            'query': query,
            'mode': mode,
            'timestamp': timestamp,
            'total_results': len(final_results),
            'chunking_distribution': chunking_distribution,
            'search_strategy_distribution': search_distribution,
            'results': [self._serialize_result(r) for r in final_results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _serialize_result(self, result: Dict) -> Dict:
        serialized = {
            'content': result.get('content', ''),
            'chunking_strategy': result.get('chunking_strategy', 'unknown'),
            'search_strategy': result.get('search_strategy', 'unknown'),
            'final_score': result.get('final_score', 0.0),
            'metadata': {
                'source': result.get('metadata', {}).get('source', 'unknown'),
                'title': result.get('metadata', {}).get('title', ''),
                'chunk_index': result.get('metadata', {}).get('chunk_index', 0),
                'quality_score': result.get('metadata', {}).get('quality_score', 0.0),
                'keywords': result.get('metadata', {}).get('keywords', [])
            }
        }
        
        if 'fusion_score' in result:
            serialized['fusion_score'] = result['fusion_score']
        if 'confidence' in result:
            serialized['confidence'] = result['confidence']
        if 'strategies_used' in result:
            serialized['strategies_used'] = result['strategies_used']
        if 'strategy_ranks' in result:
            serialized['strategy_ranks'] = result['strategy_ranks']
        
        return serialized
    
    def _slugify(self, text: str) -> str:
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_-]+', '_', text)
        return text
    
    def get_recent_results(self, result_type: str = 'final', limit: int = 10) -> List[str]:
        directory = self.final_dir if result_type == 'final' else self.intermediate_dir
        files = sorted(directory.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in files[:limit]]