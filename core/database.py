"""Multi-collection ChromaDB wrapper.

We keep one collection per chunking strategy so each retriever can search
its own embedding space without cross-strategy noise. Result fusion happens
above this layer (see strategies/fusion.py).
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import chromadb

from config.settings import DATABASE
from core.embedding import EmbeddingGenerator


logger = logging.getLogger(__name__)


# Metadata fields stored as JSON-encoded strings in ChromaDB (which only
# accepts scalar values). These get parsed back on read.
_JSON_ENCODED_METADATA_FIELDS = frozenset({'keywords', 'strategies_used'})

_CHUNKING_STRATEGIES = ('sentence_aware', 'semantic', 'paragraph', 'fixed_size')


class MultiCollectionManager:

    def __init__(self):
        self.persist_directory = DATABASE.persist_directory
        self.batch_size = DATABASE.batch_size

        self.client: Optional[chromadb.PersistentClient] = None
        self.embedding_model: Optional[EmbeddingGenerator] = None
        self.collections: Dict[str, Any] = {}

        # In-memory mirror of each collection's documents. BM25 needs the raw
        # text, so we cache it rather than re-fetching from Chroma per query.
        self.documents_cache: Dict[str, List[Dict]] = {}

        self.collection_names = {
            strategy: f'rag_{strategy}' for strategy in _CHUNKING_STRATEGIES
        }

    def initialize(self, reset: bool = False):
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = EmbeddingGenerator()

        for strategy, collection_name in self.collection_names.items():
            if reset:
                self._safe_delete_collection(collection_name)

            self.collections[strategy] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={'hnsw:space': 'cosine'},
            )
            self.documents_cache[strategy] = []

        self._refresh_all_caches()
        total = sum(col.count() for col in self.collections.values())
        print(f"✅ Multi-collection ChromaDB ready ({total} total documents)")

    def add_documents_by_strategy(
        self,
        strategy_documents: Dict[str, List[Dict]],
        update_mode: str = 'skip',
    ):
        """Add documents to the appropriate collection per chunking strategy.

        update_mode:
          - 'skip': drop docs whose IDs already exist (idempotent re-ingest).
          - 'replace': delete existing IDs first, then insert.
          - 'merge': drop docs whose *content* hash already exists.
        """
        if not self.collections:
            self.initialize()

        if update_mode not in ('skip', 'replace', 'merge'):
            raise ValueError(f"Unknown update_mode: {update_mode}")

        total_added = 0
        for strategy, documents in strategy_documents.items():
            if strategy not in self.collections:
                logger.warning("Unknown chunking strategy: %s", strategy)
                continue
            if not documents:
                continue

            collection = self.collections[strategy]
            doc_dicts = self._to_dict_list(documents)

            if update_mode == 'skip':
                doc_dicts = self._skip_existing(doc_dicts, collection)
            elif update_mode == 'replace':
                ids_to_replace = [d['id'] for d in doc_dicts]
                self._delete_by_ids(ids_to_replace, collection)
                # Drop replaced ids from the in-memory cache so we don't keep
                # stale duplicates after re-adding below.
                self._evict_cached_ids(strategy, ids_to_replace)
            elif update_mode == 'merge':
                doc_dicts = self._skip_duplicate_content(doc_dicts, strategy)

            if not doc_dicts:
                continue

            self._add_in_batches(collection, doc_dicts)
            self.documents_cache[strategy].extend(doc_dicts)
            total_added += len(doc_dicts)
            print(f"   ✓ {strategy}: {len(doc_dicts)} chunks")

        total = sum(col.count() for col in self.collections.values())
        print(f"✅ Added {total_added} documents (Total: {total})")

    def search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if strategy not in self.collections:
            raise ValueError(f"Unknown strategy: {strategy}")

        collection = self.collections[strategy]
        query_embedding = self.embedding_model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
        )
        return self._format_results(results, query, strategy)

    def clear_all(self):
        if not self.client:
            return
        for collection_name in self.collection_names.values():
            self._safe_delete_collection(collection_name)
        self.collections = {}
        self.documents_cache = {}

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for strategy, collection in self.collections.items():
            stats[strategy] = {
                'count': collection.count(),
                'cached': len(self.documents_cache.get(strategy, [])),
            }
        stats['total'] = sum(col.count() for col in self.collections.values())
        return stats

    # ------------------------------------------------------------------ private

    def _add_in_batches(self, collection, doc_dicts: List[Dict]):
        texts = [d['content'] for d in doc_dicts]
        ids = [d['id'] for d in doc_dicts]
        metadatas = [self._flatten_metadata(d.get('metadata', {})) for d in doc_dicts]

        embeddings = self.embedding_model.encode_batch(texts, show_progress=False).tolist()

        for i in range(0, len(doc_dicts), self.batch_size):
            end = i + self.batch_size
            collection.add(
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce metadata to ChromaDB-compatible scalars.

        Lists and dicts are JSON-encoded and decoded on read.
        """
        flattened: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, bool) or isinstance(value, (str, int, float)):
                flattened[key] = value
            elif isinstance(value, (list, dict)):
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = str(value)
        return flattened

    def _unflatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not metadata:
            return {}
        result: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key in _JSON_ENCODED_METADATA_FIELDS and isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
            result[key] = value
        return result

    def _refresh_all_caches(self):
        for strategy, collection in self.collections.items():
            try:
                result = collection.get()
            except Exception as e:
                logger.warning("Failed to refresh cache for %s: %s", strategy, e)
                self.documents_cache[strategy] = []
                continue

            ids = result.get('ids') or []
            documents = result.get('documents') or []
            metadatas = result.get('metadatas') or []

            self.documents_cache[strategy] = [
                {
                    'id': ids[i],
                    'content': documents[i] if i < len(documents) else '',
                    'metadata': self._unflatten_metadata(
                        metadatas[i] if i < len(metadatas) else {}
                    ),
                }
                for i in range(len(ids))
            ]

    def _to_dict_list(self, documents: List[Any]) -> List[Dict]:
        result: List[Dict] = []
        for doc in documents:
            if isinstance(doc, dict):
                result.append(doc)
            else:
                result.append({
                    'id': getattr(doc, 'id', str(doc)),
                    'content': getattr(doc, 'content', str(doc)),
                    'metadata': getattr(doc, 'metadata', {}),
                })
        return result

    def _skip_existing(self, documents: List[Dict], collection) -> List[Dict]:
        existing_ids = self._collection_ids(collection)
        new_docs = [d for d in documents if d['id'] not in existing_ids]
        skipped = len(documents) - len(new_docs)
        if skipped:
            print(f"      ⚠️  Skipped {skipped} existing")
        return new_docs

    def _skip_duplicate_content(self, documents: List[Dict], strategy: str) -> List[Dict]:
        existing_hashes = {
            self._content_hash(d['content'])
            for d in self.documents_cache.get(strategy, [])
        }

        new_docs: List[Dict] = []
        for doc in documents:
            doc_hash = self._content_hash(doc['content'])
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                existing_hashes.add(doc_hash)

        skipped = len(documents) - len(new_docs)
        if skipped:
            print(f"      ⚠️  Skipped {skipped} duplicates")
        return new_docs

    def _delete_by_ids(self, ids: List[str], collection):
        existing_ids = self._collection_ids(collection)
        to_delete = [i for i in ids if i in existing_ids]
        if to_delete:
            collection.delete(ids=to_delete)

    def _evict_cached_ids(self, strategy: str, ids: List[str]):
        """Remove the given ids from the in-memory mirror for `strategy`."""
        if not ids:
            return
        id_set = set(ids)
        cache = self.documents_cache.get(strategy, [])
        self.documents_cache[strategy] = [d for d in cache if d['id'] not in id_set]

    def _collection_ids(self, collection) -> set:
        if collection.count() == 0:
            return set()
        return set(collection.get().get('ids') or [])

    def _format_results(
        self, results: Dict, query: str, strategy: str
    ) -> List[Dict[str, Any]]:
        if not results or not results.get('ids') or not results['ids'][0]:
            return []

        ids = results['ids'][0]
        documents = (results.get('documents') or [[]])[0]
        metadatas = (results.get('metadatas') or [[]])[0]
        distances = (results.get('distances') or [[]])[0]

        formatted: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            distance = float(distances[i]) if i < len(distances) else 2.0
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            metadata = self._unflatten_metadata(
                metadatas[i] if i < len(metadatas) else {}
            )

            formatted.append({
                'id': doc_id,
                'content': documents[i] if i < len(documents) else '',
                'metadata': metadata,
                'similarity_score': similarity,
                'distance': distance,
                'rank': i + 1,
                'query': query,
                'chunking_strategy': strategy,
            })
        return formatted

    def _safe_delete_collection(self, name: str):
        try:
            self.client.delete_collection(name)
        except Exception as e:
            # Already absent or never existed — both are no-ops for us.
            logger.debug("delete_collection(%s): %s", name, e)

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
