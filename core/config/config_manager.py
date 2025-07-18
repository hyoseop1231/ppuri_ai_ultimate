"""
Config Manager - 통합 설정 관리자

환경 변수, 설정 파일, 동적 설정을 통합 관리하는
실제 동작하는 설정 시스템.

Features:
- .env 파일 기반 환경 설정
- 계층적 설정 구조
- 동적 설정 변경 감지
- 타입 안전성
- 설정 검증
- 기본값 관리
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import yaml
from enum import Enum

# Python-dotenv 임포트
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EnvironmentType(Enum):
    """환경 타입"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    type: str = "chromadb"
    host: str = "localhost"
    port: int = 8000
    database: str = "ppuri_ai"
    username: str = ""
    password: str = ""
    ssl: bool = False
    pool_size: int = 10
    timeout: int = 30
    persist_directory: str = "./vector_db_data"


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    cache_dir: str = "./models"


@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_results: int = 10
    similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    adaptive_sizing: bool = True
    use_semantic_chunking: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass
class OllamaConfig:
    """Ollama 설정"""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True


@dataclass
class ServerConfig:
    """서버 설정"""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 100
    api_prefix: str = "/api"


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class KoreanConfig:
    """한국어 처리 설정"""
    normalize_text: bool = True
    extract_entities: bool = True
    confidence_threshold: float = 0.6
    industry_terms_only: bool = False
    use_morphological_analysis: bool = True


@dataclass
class SecurityConfig:
    """보안 설정"""
    secret_key: str = ""
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"
    enable_cors: bool = True
    enable_api_key: bool = False
    api_key: str = ""


@dataclass
class CacheConfig:
    """캐시 설정"""
    redis_url: Optional[str] = None
    ttl_seconds: int = 3600
    max_size: int = 1000
    enable_file_cache: bool = True
    cache_dir: str = "./cache"


class ConfigManager:
    """
    통합 설정 관리자
    
    환경 변수, 설정 파일, 기본값을 통합하여 관리하고
    타입 안전성과 검증을 제공하는 설정 시스템.
    """
    
    def __init__(
        self,
        env_file: Optional[str] = None,
        config_dir: Optional[str] = None,
        environment: Optional[str] = None
    ):
        self.env_file = env_file or ".env"
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self.environment = EnvironmentType(environment or os.getenv("ENVIRONMENT", "development"))
        
        # 설정 저장소
        self._configs: Dict[str, Any] = {}
        self._watchers: Dict[str, List[callable]] = {}
        self._loaded_files: Dict[str, datetime] = {}
        
        # 설정 스키마 정의
        self._config_schemas = {
            "database": DatabaseConfig,
            "embedding": EmbeddingConfig,
            "rag": RAGConfig,
            "ollama": OllamaConfig,
            "server": ServerConfig,
            "logging": LoggingConfig,
            "korean": KoreanConfig,
            "security": SecurityConfig,
            "cache": CacheConfig
        }
        
        logger.info(f"Config Manager 초기화: {self.environment.value} 환경")
    
    def initialize(self) -> bool:
        """설정 시스템 초기화"""
        
        try:
            # 1. .env 파일 로드
            if not self._load_env_file():
                logger.warning("환경 파일 로드 실패, 기본값 사용")
            
            # 2. 설정 디렉토리 생성
            self.config_dir.mkdir(exist_ok=True)
            
            # 3. 기본 설정 로드
            self._load_default_configs()
            
            # 4. 환경별 설정 로드
            self._load_environment_configs()
            
            # 5. 환경 변수 오버라이드
            self._apply_environment_overrides()
            
            # 6. 설정 검증
            validation_errors = self._validate_configs()
            if validation_errors:
                logger.error(f"설정 검증 실패: {validation_errors}")
                return False
            
            # 7. 로깅 설정 적용
            self._setup_logging()
            
            logger.info("✅ 설정 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"설정 초기화 실패: {e}")
            return False
    
    def _load_env_file(self) -> bool:
        """환경 파일 로드"""
        
        if not DOTENV_AVAILABLE:
            logger.warning("python-dotenv 미설치, 환경 변수만 사용")
            return True
        
        env_paths = [
            self.env_file,
            f".env.{self.environment.value}",
            ".env.local"
        ]
        
        loaded_any = False
        for env_path in env_paths:
            if Path(env_path).exists():
                load_dotenv(env_path, override=False)
                logger.info(f"환경 파일 로드: {env_path}")
                loaded_any = True
        
        return loaded_any
    
    def _load_default_configs(self):
        """기본 설정 로드"""
        
        for config_name, config_class in self._config_schemas.items():
            self._configs[config_name] = config_class()
    
    def _load_environment_configs(self):
        """환경별 설정 파일 로드"""
        
        config_files = [
            self.config_dir / "config.json",
            self.config_dir / "config.yaml",
            self.config_dir / f"config.{self.environment.value}.json",
            self.config_dir / f"config.{self.environment.value}.yaml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                self._load_config_file(config_file)
    
    def _load_config_file(self, config_file: Path):
        """설정 파일 로드"""
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.json':
                    data = json.load(f)
                elif config_file.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    logger.warning(f"지원하지 않는 설정 파일 형식: {config_file}")
                    return
            
            # 설정 병합
            self._merge_config_data(data)
            self._loaded_files[str(config_file)] = datetime.now()
            
            logger.info(f"설정 파일 로드: {config_file}")
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패 ({config_file}): {e}")
    
    def _merge_config_data(self, data: Dict[str, Any]):
        """설정 데이터 병합"""
        
        for section, values in data.items():
            if section in self._config_schemas:
                config = self._configs[section]
                
                # 각 필드 업데이트
                for key, value in values.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        logger.warning(f"알 수 없는 설정 키: {section}.{key}")
            else:
                logger.warning(f"알 수 없는 설정 섹션: {section}")
    
    def _apply_environment_overrides(self):
        """환경 변수 오버라이드"""
        
        # 환경 변수 매핑
        env_mappings = {
            # Database
            "DB_TYPE": ("database", "type"),
            "DB_HOST": ("database", "host"),
            "DB_PORT": ("database", "port"),
            "DB_NAME": ("database", "database"),
            "DB_USER": ("database", "username"),
            "DB_PASSWORD": ("database", "password"),
            "VECTOR_DB_PATH": ("database", "persist_directory"),
            
            # Embedding
            "EMBEDDING_MODEL": ("embedding", "model_name"),
            "EMBEDDING_DEVICE": ("embedding", "device"),
            "EMBEDDING_BATCH_SIZE": ("embedding", "batch_size"),
            
            # RAG
            "RAG_CHUNK_SIZE": ("rag", "chunk_size"),
            "RAG_CHUNK_OVERLAP": ("rag", "chunk_overlap"),
            "RAG_MAX_RESULTS": ("rag", "max_results"),
            "RAG_SIMILARITY_THRESHOLD": ("rag", "similarity_threshold"),
            
            # Ollama
            "OLLAMA_BASE_URL": ("ollama", "base_url"),
            "OLLAMA_MODEL": ("ollama", "model"),
            "OLLAMA_TIMEOUT": ("ollama", "timeout"),
            "OLLAMA_TEMPERATURE": ("ollama", "temperature"),
            
            # Server
            "SERVER_HOST": ("server", "host"),
            "SERVER_PORT": ("server", "port"),
            "SERVER_DEBUG": ("server", "debug"),
            "CORS_ORIGINS": ("server", "cors_origins"),
            
            # Security
            "SECRET_KEY": ("security", "secret_key"),
            "API_KEY": ("security", "api_key"),
            
            # Logging
            "LOG_LEVEL": ("logging", "level"),
            "LOG_FILE": ("logging", "file_path"),
            
            # Korean
            "KOREAN_NORMALIZE": ("korean", "normalize_text"),
            "KOREAN_CONFIDENCE": ("korean", "confidence_threshold"),
            
            # Cache
            "REDIS_URL": ("cache", "redis_url"),
            "CACHE_TTL": ("cache", "ttl_seconds")
        }
        
        for env_key, (section, field) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                config = self._configs[section]
                
                # 타입 변환
                field_type = type(getattr(config, field))
                try:
                    if field_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int:
                        value = int(env_value)
                    elif field_type == float:
                        value = float(env_value)
                    elif field_type == list:
                        value = env_value.split(',')
                    else:
                        value = env_value
                    
                    setattr(config, field, value)
                    logger.debug(f"환경 변수 적용: {env_key} -> {section}.{field}")
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"환경 변수 타입 변환 실패 ({env_key}): {e}")
    
    def _validate_configs(self) -> List[str]:
        """설정 검증"""
        
        errors = []
        
        # Database 검증
        db_config = self.get_database_config()
        if db_config.type == "chromadb" and not db_config.persist_directory:
            errors.append("ChromaDB persist_directory 필수")
        
        # Embedding 검증
        embedding_config = self.get_embedding_config()
        if not embedding_config.model_name:
            errors.append("임베딩 모델명 필수")
        
        if embedding_config.dimension <= 0:
            errors.append("임베딩 차원은 양수여야 함")
        
        # RAG 검증
        rag_config = self.get_rag_config()
        if rag_config.chunk_size <= 0:
            errors.append("청크 크기는 양수여야 함")
        
        if rag_config.chunk_overlap >= rag_config.chunk_size:
            errors.append("청크 오버랩은 청크 크기보다 작아야 함")
        
        if not (0 <= rag_config.similarity_threshold <= 1):
            errors.append("유사도 임계값은 0과 1 사이여야 함")
        
        # Server 검증
        server_config = self.get_server_config()
        if not (1 <= server_config.port <= 65535):
            errors.append("서버 포트는 1-65535 범위여야 함")
        
        # Security 검증
        security_config = self.get_security_config()
        if not security_config.secret_key and self.environment != EnvironmentType.DEVELOPMENT:
            errors.append("프로덕션 환경에서는 SECRET_KEY 필수")
        
        return errors
    
    def _setup_logging(self):
        """로깅 설정"""
        
        logging_config = self.get_logging_config()
        
        # 로그 레벨 설정
        log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
        
        # 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 포매터 설정
        formatter = logging.Formatter(logging_config.format)
        
        # 콘솔 핸들러
        if logging_config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 파일 핸들러
        if logging_config.file_path:
            try:
                from logging.handlers import RotatingFileHandler
                
                log_dir = Path(logging_config.file_path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    logging_config.file_path,
                    maxBytes=logging_config.max_file_size_mb * 1024 * 1024,
                    backupCount=logging_config.backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                logger.error(f"파일 로깅 설정 실패: {e}")
    
    # 설정 조회 메소드들
    def get_database_config(self) -> DatabaseConfig:
        """데이터베이스 설정 조회"""
        return self._configs["database"]
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """임베딩 설정 조회"""
        return self._configs["embedding"]
    
    def get_rag_config(self) -> RAGConfig:
        """RAG 설정 조회"""
        return self._configs["rag"]
    
    def get_ollama_config(self) -> OllamaConfig:
        """Ollama 설정 조회"""
        return self._configs["ollama"]
    
    def get_server_config(self) -> ServerConfig:
        """서버 설정 조회"""
        return self._configs["server"]
    
    def get_logging_config(self) -> LoggingConfig:
        """로깅 설정 조회"""
        return self._configs["logging"]
    
    def get_korean_config(self) -> KoreanConfig:
        """한국어 설정 조회"""
        return self._configs["korean"]
    
    def get_security_config(self) -> SecurityConfig:
        """보안 설정 조회"""
        return self._configs["security"]
    
    def get_cache_config(self) -> CacheConfig:
        """캐시 설정 조회"""
        return self._configs["cache"]
    
    def get_config(self, section: str, key: str, default: Any = None) -> Any:
        """특정 설정값 조회"""
        
        if section not in self._configs:
            return default
        
        config = self._configs[section]
        return getattr(config, key, default)
    
    def set_config(self, section: str, key: str, value: Any) -> bool:
        """설정값 동적 변경"""
        
        try:
            if section not in self._configs:
                logger.error(f"알 수 없는 설정 섹션: {section}")
                return False
            
            config = self._configs[section]
            if not hasattr(config, key):
                logger.error(f"알 수 없는 설정 키: {section}.{key}")
                return False
            
            # 타입 검증
            current_value = getattr(config, key)
            if type(value) != type(current_value) and current_value is not None:
                logger.error(f"설정 타입 불일치: {section}.{key}")
                return False
            
            setattr(config, key, value)
            
            # 변경 알림
            self._notify_config_change(section, key, value)
            
            logger.info(f"설정 변경: {section}.{key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"설정 변경 실패: {e}")
            return False
    
    def watch_config(self, section: str, callback: callable):
        """설정 변경 감시"""
        
        if section not in self._watchers:
            self._watchers[section] = []
        
        self._watchers[section].append(callback)
        logger.debug(f"설정 감시 등록: {section}")
    
    def _notify_config_change(self, section: str, key: str, value: Any):
        """설정 변경 알림"""
        
        if section in self._watchers:
            for callback in self._watchers[section]:
                try:
                    callback(section, key, value)
                except Exception as e:
                    logger.error(f"설정 변경 콜백 실행 실패: {e}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """환경 정보 조회"""
        
        return {
            "environment": self.environment.value,
            "config_dir": str(self.config_dir),
            "env_file": self.env_file,
            "loaded_files": list(self._loaded_files.keys()),
            "available_sections": list(self._config_schemas.keys()),
            "watchers_count": {
                section: len(watchers) 
                for section, watchers in self._watchers.items()
            }
        }
    
    def export_config(self, format: str = "json") -> str:
        """설정 내보내기"""
        
        config_data = {}
        
        for section, config in self._configs.items():
            config_dict = {}
            for field in config.__dataclass_fields__:
                value = getattr(config, field)
                # 민감한 정보 마스킹
                if field in ["password", "secret_key", "api_key"] and value:
                    value = "*" * len(str(value))
                config_dict[field] = value
            
            config_data[section] = config_dict
        
        if format == "json":
            return json.dumps(config_data, indent=2, ensure_ascii=False)
        elif format == "yaml":
            return yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def reload_config(self) -> bool:
        """설정 다시 로드"""
        
        try:
            # 기존 설정 백업
            backup_configs = self._configs.copy()
            
            # 설정 초기화
            self._configs.clear()
            self._load_default_configs()
            self._load_environment_configs()
            self._apply_environment_overrides()
            
            # 검증
            validation_errors = self._validate_configs()
            if validation_errors:
                # 실패시 백업 복원
                self._configs = backup_configs
                logger.error(f"설정 리로드 검증 실패: {validation_errors}")
                return False
            
            logger.info("설정 리로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"설정 리로드 실패: {e}")
            return False
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """의존성 검증"""
        
        dependencies = {}
        
        # ChromaDB 검증
        try:
            import chromadb
            dependencies["chromadb"] = True
        except ImportError:
            dependencies["chromadb"] = False
        
        # Sentence Transformers 검증
        try:
            import sentence_transformers
            dependencies["sentence_transformers"] = True
        except ImportError:
            dependencies["sentence_transformers"] = False
        
        # FastAPI 검증
        try:
            import fastapi
            dependencies["fastapi"] = True
        except ImportError:
            dependencies["fastapi"] = False
        
        # Redis 검증 (선택사항)
        try:
            import redis
            dependencies["redis"] = True
        except ImportError:
            dependencies["redis"] = False
        
        # YAML 검증
        try:
            import yaml
            dependencies["yaml"] = True
        except ImportError:
            dependencies["yaml"] = False
        
        return dependencies
    
    def cleanup(self):
        """Config Manager 정리"""
        
        self._configs.clear()
        self._watchers.clear()
        self._loaded_files.clear()
        
        logger.info("Config Manager 정리 완료")