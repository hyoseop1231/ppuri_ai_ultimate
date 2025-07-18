"""
KITECH Base Integration - 검증된 패턴 통합 시스템

KITECH RAG 챗봇에서 검증된 패턴들을 PPuRI-AI Ultimate와 통합하여
안정성과 혁신성을 모두 확보하는 하이브리드 시스템.

Features:
- 5초 빠른 시작 최적화
- 한국어 특화 처리
- THINK 블록 UI 시스템
- 검증된 FastAPI 패턴
- AdalFlow 엔진 통합
"""

from .fast_startup import FastStartupManager
from .korean_optimizer import KoreanLanguageOptimizer
from .think_ui import ThinkBlockManager
from .config_manager import KitechConfigManager
from .conversational_engine import ConversationalEngine

__all__ = [
    "FastStartupManager",
    "KoreanLanguageOptimizer", 
    "ThinkBlockManager",
    "KitechConfigManager",
    "ConversationalEngine"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__kitech_verified__ = True