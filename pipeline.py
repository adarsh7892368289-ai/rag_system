"""High-level RAG pipeline.

Glues together extraction → multi-strategy chunking → vector storage →
multi-strategy retrieval → fusion. Most callers only need:

    pipeline = RAGPipeline()
    pipeline.ingest(sources=[...])
    results = pipeline.query("...")
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional


# Ensure progress prints don't crash on legacy Windows code pages (cp1252 etc.)
# where the status emojis used throughout the codebase aren't representable.
# `errors='replace'` is preferred over a hard switch so any unexpected output
# degrades gracefully instead of throwing.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, 'reconfigure', None)
    if reconfigure is not None:
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

from core.database import MultiCollectionManager
from core.extraction import DocumentExtractor
from strategies.fusion import ResultFusion
from strategies.query_router import QueryRouter
from strategies.result_tracker import ResultTracker
from strategies.search_strategies import AdvancedSearchStrategies


# Approximate characters per token for rough context-budget calculations.
# 1 token ≈ 4 chars for English text in most BPE tokenizers.
_CHARS_PER_TOKEN = 4

# Per-query timeout in batch mode.
_BATCH_QUERY_TIMEOUT = 120


class RAGPipeline:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.extractor = DocumentExtractor()
        self.db_manager = MultiCollectionManager()
        self.db_manager.initialize()
        self.search_strategies = AdvancedSearchStrategies(self.db_manager)
        self.query_router = QueryRouter()
        self.result_tracker = ResultTracker()
        self.fusion = ResultFusion(self.db_manager.embedding_model)

    # ------------------------------------------------------------------ ingest

    def ingest(
        self,
        sources: List[str],
        reset: bool = False,
        save_extracted: bool = True,
        update_mode: str = 'skip',
    ):
        """Extract sources, chunk them four ways, and index into Chroma."""
        self._log(f"\n📥 Ingesting {len(sources)} source(s)...")

        strategy_documents = self.extractor.extract_multiple(sources)
        if not any(strategy_documents.values()):
            print("⚠️  No documents extracted")
            return

        if save_extracted:
            self.extractor.save_documents(strategy_documents)

        if reset:
            self.db_manager.initialize(reset=True)

        self.db_manager.add_documents_by_strategy(
            strategy_documents, update_mode=update_mode
        )
        # BM25 indexes are content-aware; new docs invalidate the cache.
        self.search_strategies.invalidate_caches()

        if self.verbose:
            stats = self.db_manager.get_stats()
            print(f"✅ Ingestion complete: {stats['total']} total chunks\n")

    def load_from_json(
        self, folder_path: str, reset: bool = False, update_mode: str = 'skip'
    ):
        """Re-index previously-extracted chunks from disk (skips re-extraction)."""
        self._log(f"\n📂 Loading from {folder_path}...")

        strategy_documents = self.extractor.load_from_folder(folder_path)
        if not any(strategy_documents.values()):
            print("⚠️  No documents loaded")
            return

        if reset:
            self.db_manager.initialize(reset=True)

        self.db_manager.add_documents_by_strategy(
            strategy_documents, update_mode=update_mode
        )
        self.search_strategies.invalidate_caches()

        if self.verbose:
            stats = self.db_manager.get_stats()
            print(f"✅ Loaded {stats['total']} total chunks\n")

    # ------------------------------------------------------------------- query

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        auto_route: bool = True,
        min_confidence: float = 0.0,
        save_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run a search and return the top-k fused results.

        mode:
          - None + auto_route=True: classify the query and pick a mode.
          - 'parallel': fan out across all 16 search × chunking combinations.
          - 'semantic' | 'bm25' | 'hybrid' | 'mmr' | 'rerank': run the named
            search strategy across all four chunking strategies.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if mode is None:
            mode = self.query_router.route(query_text) if auto_route else 'parallel'

        if mode == 'parallel':
            self._log(
                "\n🔍 Running PARALLEL mode "
                "(16 searches: 4 chunking × 4 search strategies)"
            )
            chunking_results = self.search_strategies.parallel_search_all(
                query_text, top_k
            )
        else:
            self._log(
                f"\n🔍 Running {mode.upper()} mode "
                "(4 searches: 4 chunking × 1 search strategy)"
            )
            chunking_results = self.search_strategies.single_strategy_all_chunking(
                query_text, mode, top_k
            )

        if save_results:
            self.result_tracker.save_intermediate_results(
                query_text, chunking_results, mode, timestamp
            )

        final_results = self.fusion.reciprocal_rank_fusion(
            chunking_results, top_n=top_k
        )

        if min_confidence > 0.0:
            final_results = [
                r for r in final_results
                if r.get('confidence', 1.0) >= min_confidence
            ]

        if save_results:
            saved_path = self.result_tracker.save_final_results(
                query_text, final_results, mode, timestamp
            )
            self._log(f"\n💾 Results saved to: {saved_path}")

        if self.verbose:
            self._print_results_summary(final_results)

        return final_results

    def query_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        mode: Optional[str] = None,
        auto_route: bool = True,
        max_workers: int = 4,
        save_results: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run multiple queries in parallel. Errors are isolated per query."""
        self._log(f"\n🔄 Processing {len(queries)} queries in parallel...")

        results_dict: Dict[str, List[Dict[str, Any]]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(
                    self.query, query, top_k, mode, auto_route, 0.0, save_results
                ): query
                for query in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results_dict[query] = future.result(timeout=_BATCH_QUERY_TIMEOUT)
                except Exception as e:
                    self._log(f"⚠️  Query '{query}' failed: {e}")
                    results_dict[query] = []

        self._log("✅ Batch complete\n")
        return results_dict

    # ------------------------------------------------------------- LLM context

    def build_llm_context(
        self,
        results: List[Dict[str, Any]],
        max_tokens: int = 4000,
        include_metadata: bool = True,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """Pack results into a context blob that fits the given token budget.

        Returns context plus accounting (chars, est. tokens, sources, etc.).
        Stops adding chunks as soon as the next one would exceed the budget.
        """
        filtered_results = [
            r for r in results if r.get('confidence', 1.0) >= min_confidence
        ]

        if not filtered_results:
            return {
                'context': '',
                'chunks_used': 0,
                'chunks_filtered': len(results),
                'total_chars': 0,
                'estimated_tokens': 0,
                'sources': [],
                'avg_confidence': 0.0,
            }

        char_limit = max_tokens * _CHARS_PER_TOKEN
        context_parts: List[str] = []
        total_chars = 0
        sources: set = set()
        chunks_used = 0

        for i, result in enumerate(filtered_results, start=1):
            chunk_text = self._format_chunk(result, i, include_metadata)
            if total_chars + len(chunk_text) > char_limit:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            chunks_used += 1
            source = result.get('metadata', {}).get('source')
            if source and source != 'Unknown':
                sources.add(source)

        avg_confidence = (
            sum(r.get('confidence', 1.0) for r in filtered_results[:chunks_used])
            / chunks_used
            if chunks_used > 0
            else 0.0
        )

        return {
            'context': '\n---\n'.join(context_parts),
            'chunks_used': chunks_used,
            'chunks_filtered': len(results) - len(filtered_results),
            'total_chars': total_chars,
            'estimated_tokens': total_chars // _CHARS_PER_TOKEN,
            'sources': list(sources),
            'avg_confidence': avg_confidence,
        }

    def format_for_llm(
        self,
        query: str,
        context_dict: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format a complete prompt from query + context for an LLM call."""
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer based on the provided context."
            )

        return (
            f"{system_prompt}\n\n"
            f"Context:\n{context_dict['context']}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    # ----------------------------------------------------------------- maintenance

    def clear_database(self):
        self.db_manager.clear_all()
        self.db_manager.initialize()
        self.search_strategies.invalidate_caches()
        self._log("✅ All databases cleared\n")

    def get_stats(self) -> Dict[str, Any]:
        return self.db_manager.get_stats()

    # ------------------------------------------------------------------ private

    def _log(self, message: str):
        if self.verbose:
            print(message)

    @staticmethod
    def _format_chunk(
        result: Dict[str, Any], index: int, include_metadata: bool
    ) -> str:
        content = result['content']
        if not include_metadata:
            return f"{content}\n"

        confidence = result.get('confidence', 1.0)
        source = result.get('metadata', {}).get('source', 'Unknown')
        chunking = result.get('chunking_strategy', 'unknown')
        search = result.get('search_strategy', 'unknown')

        header = (
            f"[Result {index} | Confidence: {confidence:.2f} | "
            f"Chunking: {chunking} | Search: {search}]"
        )
        if source != 'Unknown':
            header += f"\n[Source: {source}]"
        return f"{header}\n{content}\n"

    def _print_results_summary(self, final_results: List[Dict]):
        print("\n📊 Results Summary:")
        print(f"   • Final results: {len(final_results)}")

        chunking_dist: Dict[str, int] = {}
        search_dist: Dict[str, int] = {}

        # `search_strategy` is set per-result by each search primitive and survives
        # both fusion passes via "first occurrence wins". `strategies_used` holds
        # the chunking-strategy provenance after the outer fusion, so we don't
        # use it here — that breakdown is captured in `chunking_dist`.
        for result in final_results:
            chunking = result.get('chunking_strategy', 'unknown')
            chunking_dist[chunking] = chunking_dist.get(chunking, 0) + 1

            search = result.get('search_strategy', 'unknown')
            search_dist[search] = search_dist.get(search, 0) + 1

        print(f"   • Chunking distribution: {chunking_dist}")
        print(f"   • Search strategy distribution: {search_dist}")

        if final_results:
            avg_confidence = (
                sum(r.get('confidence', 0) for r in final_results) / len(final_results)
            )
            print(f"   • Average confidence: {avg_confidence:.3f}")
