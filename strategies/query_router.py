"""Routes a natural-language query to the best-fit search strategy.

Routing is heuristic: we classify the query by surface form (how-to,
definition, comparison, list, short keyword, general) and pick the strategy
that empirically performs best for that class. Output of `route()` is one of
the modes accepted by `RAGPipeline.query`.
"""

from typing import Tuple


# (prefix-match phrases, query-type)
_PREFIX_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (('how to', 'how do', 'how can', 'how does', 'how should'), 'how-to'),
    (('what is', 'what are', 'define', 'explain', 'describe'), 'definition'),
    (('list', 'enumerate', 'what are the', 'name the'), 'list'),
)

# Substring-match keywords for comparison queries (anywhere in the query).
_COMPARISON_TERMS = ('compare', 'difference', 'versus', 'vs', 'better', 'differ')

# Queries this short are treated as keyword search.
_KEYWORD_QUERY_MAX_WORDS = 3

# Routing table: query type -> recommended search mode.
_ROUTING_MAP = {
    'how-to': 'rerank',        # Cross-encoder shines on intent matching.
    'definition': 'semantic',  # Conceptual similarity is the right signal.
    'comparison': 'parallel',  # Multiple perspectives improve recall.
    'list': 'mmr',             # Diversity matters — MMR enforces it.
    'keyword': 'hybrid',       # Mix BM25 (lexical) + vector for short queries.
    'general': 'parallel',     # Best overall when intent is unclear.
}


class QueryRouter:

    @staticmethod
    def detect_query_type(query: str) -> str:
        query_lower = query.lower().strip()
        if not query_lower:
            return 'general'

        for prefixes, query_type in _PREFIX_RULES:
            if any(query_lower.startswith(prefix) for prefix in prefixes):
                return query_type

        if any(term in query_lower for term in _COMPARISON_TERMS):
            return 'comparison'

        if len(query_lower.split()) <= _KEYWORD_QUERY_MAX_WORDS:
            return 'keyword'

        return 'general'

    @staticmethod
    def route(query: str) -> str:
        query_type = QueryRouter.detect_query_type(query)
        mode = _ROUTING_MAP.get(query_type, 'parallel')
        print(f"🧭 Query type: '{query_type}' → Routing to: '{mode}'")
        return mode
