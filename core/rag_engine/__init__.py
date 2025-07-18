"""
Firestarter RAG Engine - 고성능 검색 증강 생성 시스템

Firestarter에서 영감을 받은 네임스페이스 기반 고성능 RAG 엔진으로
뿌리산업 문서의 정밀 검색과 맥락적 생성을 지원하는 시스템.

Features:
- 네임스페이스 기반 문서 조직
- 고급 검색 전략 (하이브리드, 의미적, 키워드)
- 한국어 최적화 검색
- 지식 그래프 연동
- 청크 단위 정밀 검색
- 실시간 성능 최적화
"""

from .document_processor import DocumentProcessor
from .namespace_manager import NamespaceManager  
from .retrieval_engine import RetrievalEngine
from .chunk_manager import ChunkManager
from .search_strategies import SearchStrategyManager
from .rag_orchestrator import RAGOrchestrator

__all__ = [
    "DocumentProcessor",
    "NamespaceManager",
    "RetrievalEngine", 
    "ChunkManager",
    "SearchStrategyManager",
    "RAGOrchestrator"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__firestarter_inspired__ = True