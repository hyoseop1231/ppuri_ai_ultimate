"""
Config Package - PPuRI-AI Ultimate 설정 관리 패키지

환경 변수, 설정 파일, 동적 설정을 통합 관리하는 패키지.

Features:
- .env 파일 기반 환경 설정
- 계층적 설정 구조
- 타입 안전성
- 설정 검증
- 동적 설정 변경 감지
"""

from .config_manager import (
    ConfigManager,
    DatabaseConfig,
    EmbeddingConfig,
    RAGConfig,
    OllamaConfig,
    ServerConfig,
    LoggingConfig,
    KoreanConfig,
    SecurityConfig,
    CacheConfig,
    EnvironmentType
)

__all__ = [
    "ConfigManager",
    "DatabaseConfig",
    "EmbeddingConfig", 
    "RAGConfig",
    "OllamaConfig",
    "ServerConfig",
    "LoggingConfig",
    "KoreanConfig",
    "SecurityConfig",
    "CacheConfig",
    "EnvironmentType"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__description__ = "통합 설정 관리 시스템"