"""
PPuRI-AI Ultimate - Embeddings Module
BGE-M3 기반 임베딩 서비스
"""

from .bge_m3_service import (
    BGEM3Service,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingResult,
    get_embedding_service
)

__all__ = [
    "BGEM3Service",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingResult",
    "get_embedding_service"
]
