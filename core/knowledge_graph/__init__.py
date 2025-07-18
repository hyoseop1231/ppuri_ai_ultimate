"""
Knowledge Graph System - RedPlanet Core 영감 지식 그래프

Neo4j 기반 포터블 메모리 그래프 시스템으로 대화, 지식, 관계를 
모델링하여 컨텍스트 유지와 지식 발견을 지원하는 시스템.

Features:
- Neo4j 그래프 데이터베이스 통합
- 대화-지식 관계 모델링
- 의미적 검색 및 지식 발견
- 뿌리산업 특화 온톨로지
- 실시간 지식 그래프 업데이트
"""

from .graph_manager import GraphManager
from .knowledge_extractor import KnowledgeExtractor
from .relationship_builder import RelationshipBuilder
from .semantic_search import SemanticSearchEngine
from .ontology_manager import IndustryOntologyManager

__all__ = [
    "GraphManager",
    "KnowledgeExtractor", 
    "RelationshipBuilder",
    "SemanticSearchEngine",
    "IndustryOntologyManager"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__redplanet_inspired__ = True