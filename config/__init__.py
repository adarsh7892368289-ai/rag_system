"""
Config Package - Phase 1+2 Complete
"""

from .settings import (
    # Configurations
    METADATA,
    CHUNKING,
    EMBEDDING,
    DATABASE,
    SEARCH,
    EXTRACTION,
    
    # Classes
    MetadataConfig,
    ChunkingConfig,
    EmbeddingConfig,
    DatabaseConfig,
    SearchConfig,
    ExtractionConfig,
    
    # Routing rules
    QUERY_ROUTING_RULES,
    SEARCH_STRATEGY_MAP,
    
    # Utilities
    print_config_summary,
    
    # Directories
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR
)

__all__ = [
    # Configurations
    'METADATA',
    'CHUNKING',
    'EMBEDDING',
    'DATABASE',
    'SEARCH',
    'EXTRACTION',
    
    # Classes
    'MetadataConfig',
    'ChunkingConfig',
    'EmbeddingConfig',
    'DatabaseConfig',
    'SearchConfig',
    'ExtractionConfig',
    
    # Routing
    'QUERY_ROUTING_RULES',
    'SEARCH_STRATEGY_MAP',
    
    # Utilities
    'print_config_summary',
    
    # Directories
    'BASE_DIR',
    'DATA_DIR',
    'LOGS_DIR'
]