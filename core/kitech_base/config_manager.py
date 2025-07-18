"""
KITECH Config Manager - 검증된 설정 관리 시스템

KITECH RAG 챗봇에서 검증된 설정들을 관리하고
새로운 AdalFlow 기능과 통합하는 설정 관리자.

Features:
- KITECH 검증된 기본 설정
- 환경별 설정 자동 전환
- 동적 설정 최적화
- 호환성 보장
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class KitechConfig:
    """KITECH 검증된 설정 구조"""
    
    # === 5초 빠른 시작 최적화 ===
    preload_embedding_model: bool = False
    enable_external_access: bool = True
    cors_origins: str = "*"
    lazy_loading: bool = True
    cache_embeddings: bool = True
    
    # === 한국어 최적화 ===
    default_language: str = "ko"
    korean_tokenizer: str = "mecab"
    use_korean_stopwords: bool = True
    korean_normalization: bool = True
    
    # === Ollama 설정 ===
    ollama_api_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen3:30b-a3b"
    ollama_timeout: int = 120
    ollama_max_retries: int = 3
    
    # === FastAPI 최적화 ===
    uvicorn_workers: int = 1
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000
    reload_on_change: bool = False
    
    # === RAG 설정 ===
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_db: str = "qdrant"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # === THINK 블록 UI ===
    enable_think_blocks: bool = True
    think_block_styles: Dict[str, str] = field(default_factory=lambda: {
        "think": "🧠 **THINK**",
        "megathink": "🚀 **MEGATHINK**", 
        "ultrathink": "⚡ **ULTRATHINK**"
    })
    
    # === 성능 최적화 ===
    max_memory_usage: float = 0.8  # 80%
    gc_threshold: int = 1000
    connection_pool_size: int = 20
    
    # === 로깅 설정 ===
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "1 day"
    
    # === 보안 설정 ===
    enable_auth: bool = False
    api_rate_limit: int = 100  # requests per minute
    max_request_size: int = 10_000_000  # 10MB


class KitechConfigManager:
    """
    KITECH 검증된 설정 관리자
    
    환경별 설정 자동 관리와 동적 최적화를 담당하며
    AdalFlow 엔진과의 호환성을 보장.
    """
    
    def __init__(
        self,
        config_dir: str = "/app/config",
        environment: str = "production"
    ):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.environment = environment
        
        # 기본 설정 로드
        self.config = self._load_config()
        self.runtime_overrides: Dict[str, Any] = {}
        
        # 환경 변수 적용
        self._apply_environment_variables()
        
        logger.info(f"KITECH 설정 관리자 초기화: {environment} 환경")
    
    def _load_config(self) -> KitechConfig:
        """설정 파일 로드 또는 기본값 생성"""
        
        config_files = [
            self.config_dir / f"kitech_{self.environment}.yaml",
            self.config_dir / "kitech_default.yaml",
            self.config_dir / "kitech.yaml"
        ]
        
        # 설정 파일 순서대로 시도
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    logger.info(f"설정 파일 로드: {config_file}")
                    return KitechConfig(**config_data)
                    
                except Exception as e:
                    logger.warning(f"설정 파일 로드 실패 ({config_file}): {e}")
        
        # 기본 설정 사용
        logger.info("기본 KITECH 설정 사용")
        return KitechConfig()
    
    def _apply_environment_variables(self):
        """환경 변수 기반 설정 오버라이드"""
        
        env_mappings = {
            "PRELOAD_EMBEDDING_MODEL": ("preload_embedding_model", bool),
            "ENABLE_EXTERNAL_ACCESS": ("enable_external_access", bool),
            "CORS_ORIGINS": ("cors_origins", str),
            "OLLAMA_API_URL": ("ollama_api_url", str),
            "OLLAMA_MODEL": ("ollama_model", str),
            "UVICORN_WORKERS": ("uvicorn_workers", int),
            "UVICORN_PORT": ("uvicorn_port", int),
            "LOG_LEVEL": ("log_level", str),
            "ENABLE_AUTH": ("enable_auth", bool)
        }
        
        applied_overrides = []
        
        for env_var, (config_attr, config_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # 타입 변환
                    if config_type == bool:
                        parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif config_type == int:
                        parsed_value = int(env_value)
                    else:
                        parsed_value = env_value
                    
                    # 설정 적용
                    setattr(self.config, config_attr, parsed_value)
                    applied_overrides.append(f"{config_attr}={parsed_value}")
                    
                except ValueError as e:
                    logger.warning(f"환경 변수 파싱 실패 ({env_var}): {e}")
        
        if applied_overrides:
            logger.info(f"환경 변수 오버라이드 적용: {', '.join(applied_overrides)}")
    
    def get_config(self) -> KitechConfig:
        """현재 설정 반환"""
        return self.config
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """특정 설정값 조회"""
        
        # 런타임 오버라이드 우선 확인
        if key in self.runtime_overrides:
            return self.runtime_overrides[key]
        
        # 기본 설정에서 조회
        return getattr(self.config, key, default)
    
    def set_value(self, key: str, value: Any, persistent: bool = False):
        """설정값 동적 변경"""
        
        if persistent and hasattr(self.config, key):
            # 영구 설정 변경
            setattr(self.config, key, value)
        else:
            # 런타임 오버라이드
            self.runtime_overrides[key] = value
        
        logger.debug(f"설정 변경: {key} = {value} (persistent: {persistent})")
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Ollama 전용 설정 반환"""
        return {
            "api_url": self.config.ollama_api_url,
            "model": self.config.ollama_model,
            "timeout": self.config.ollama_timeout,
            "max_retries": self.config.ollama_max_retries,
            "options": {
                "temperature": 0.3,  # KITECH 검증된 기본값
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
    
    def get_fastapi_config(self) -> Dict[str, Any]:
        """FastAPI 전용 설정 반환"""
        return {
            "host": self.config.uvicorn_host,
            "port": self.config.uvicorn_port,
            "workers": self.config.uvicorn_workers,
            "reload": self.config.reload_on_change,
            "access_log": self.config.log_level == "DEBUG"
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """RAG 전용 설정 반환"""
        return {
            "embedding_model": self.config.embedding_model,
            "vector_db": self.config.vector_db,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "preload_embeddings": self.config.preload_embedding_model
        }
    
    def get_korean_config(self) -> Dict[str, Any]:
        """한국어 처리 전용 설정 반환"""
        return {
            "language": self.config.default_language,
            "tokenizer": self.config.korean_tokenizer,
            "use_stopwords": self.config.use_korean_stopwords,
            "normalization": self.config.korean_normalization
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """성능 최적화 설정 반환"""
        return {
            "max_memory_usage": self.config.max_memory_usage,
            "gc_threshold": self.config.gc_threshold,
            "connection_pool_size": self.config.connection_pool_size,
            "lazy_loading": self.config.lazy_loading,
            "cache_embeddings": self.config.cache_embeddings
        }
    
    def optimize_for_environment(self, target_env: str = None):
        """환경별 최적화 설정 적용"""
        
        if target_env is None:
            target_env = self.environment
        
        optimizations = []
        
        if target_env == "development":
            # 개발 환경 최적화
            self.set_value("reload_on_change", True)
            self.set_value("log_level", "DEBUG")
            self.set_value("preload_embedding_model", False)
            self.set_value("uvicorn_workers", 1)
            optimizations.append("개발 환경 최적화")
        
        elif target_env == "production":
            # 프로덕션 최적화
            self.set_value("reload_on_change", False)
            self.set_value("log_level", "INFO")
            self.set_value("preload_embedding_model", True)
            self.set_value("cache_embeddings", True)
            optimizations.append("프로덕션 최적화")
        
        elif target_env == "testing":
            # 테스트 환경 최적화
            self.set_value("ollama_timeout", 30)
            self.set_value("max_request_size", 1_000_000)
            self.set_value("api_rate_limit", 1000)
            optimizations.append("테스트 환경 최적화")
        
        if optimizations:
            logger.info(f"환경 최적화 적용: {', '.join(optimizations)}")
    
    def validate_config(self) -> List[str]:
        """설정 유효성 검증"""
        
        warnings = []
        
        # Ollama 연결 확인 (간접적)
        if not self.config.ollama_api_url.startswith(('http://', 'https://')):
            warnings.append("Ollama API URL 형식이 올바르지 않음")
        
        # 포트 범위 확인
        if not 1024 <= self.config.uvicorn_port <= 65535:
            warnings.append(f"Uvicorn 포트 범위 이상: {self.config.uvicorn_port}")
        
        # 메모리 사용량 확인
        if not 0.1 <= self.config.max_memory_usage <= 1.0:
            warnings.append(f"메모리 사용량 설정 이상: {self.config.max_memory_usage}")
        
        # Workers 수 확인
        if self.config.uvicorn_workers < 1:
            warnings.append("Uvicorn workers 수가 너무 적음")
        
        # 청크 크기 확인
        if self.config.chunk_size < 100 or self.config.chunk_size > 4000:
            warnings.append(f"RAG 청크 크기 비권장: {self.config.chunk_size}")
        
        return warnings
    
    def export_config(self, file_path: Optional[str] = None) -> str:
        """현재 설정을 파일로 내보내기"""
        
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config_dir / f"kitech_export_{timestamp}.yaml"
        
        # 설정을 딕셔너리로 변환
        config_dict = {
            key: getattr(self.config, key)
            for key in dir(self.config)
            if not key.startswith('_')
        }
        
        # 런타임 오버라이드 포함
        if self.runtime_overrides:
            config_dict["runtime_overrides"] = self.runtime_overrides
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict, 
                    f, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    sort_keys=True
                )
            
            logger.info(f"설정 내보내기 완료: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"설정 내보내기 실패: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 및 설정 요약"""
        
        warnings = self.validate_config()
        
        return {
            "environment": self.environment,
            "config_source": "환경 변수 + 기본값",
            "kitech_verified": True,
            "fast_startup": not self.config.preload_embedding_model,
            "korean_optimized": self.config.default_language == "ko",
            "think_blocks_enabled": self.config.enable_think_blocks,
            "performance_optimized": self.config.lazy_loading and self.config.cache_embeddings,
            "warnings": warnings,
            "runtime_overrides": len(self.runtime_overrides),
            "last_updated": datetime.now().isoformat()
        }