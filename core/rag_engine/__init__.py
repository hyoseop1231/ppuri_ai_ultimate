"""
PPuRI-AI Ultimate RAG Engine - 고성능 검색 증강 생성 시스템

Firestarter + LightRAG에서 영감을 받은 고성능 RAG 엔진으로
뿌리산업 문서의 정밀 검색과 맥락적 생성을 지원하는 시스템.

Features:
- 네임스페이스 기반 문서 조직
- LightRAG 스타일 지식그래프 기반 검색
- 고급 검색 전략 (하이브리드, 의미적, 키워드)
- 한국어 최적화 검색
- 엔티티/관계 추출 및 정규화
- 청크 단위 정밀 검색
- 실시간 성능 최적화
"""

# Core components (with graceful fallback)
try:
    from .document_processor import DocumentProcessor
except ImportError:
    DocumentProcessor = None

try:
    from .namespace_manager import NamespaceManager
except ImportError:
    NamespaceManager = None

try:
    from .retrieval_engine import RetrievalEngine
except ImportError:
    RetrievalEngine = None

try:
    from .chunk_manager import ChunkManager
except ImportError:
    ChunkManager = None

try:
    from .search_strategies import SearchStrategyManager
except ImportError:
    SearchStrategyManager = None

try:
    from .rag_orchestrator import RAGOrchestrator
except ImportError:
    RAGOrchestrator = None

# LightRAG components (main feature)
from .lightrag_engine import (
    LightRAGEngine,
    LightRAGConfig,
    Entity,
    Relationship,
    RetrievalMode,
    RetrievalResult,
    get_lightrag_engine
)

__all__ = [
    # Firestarter-style components
    "DocumentProcessor",
    "NamespaceManager",
    "RetrievalEngine",
    "ChunkManager",
    "SearchStrategyManager",
    "RAGOrchestrator",
    # LightRAG components
    "LightRAGEngine",
    "LightRAGConfig",
    "Entity",
    "Relationship",
    "RetrievalMode",
    "RetrievalResult",
    "get_lightrag_engine"
]

__version__ = "2.0.0"
__author__ = "PPuRI-AI Team"
__firestarter_inspired__ = True
__lightrag_inspired__ = True