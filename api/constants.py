"""
Constants - PPuRI-AI Ultimate API 공통 상수
"""

from enum import Enum
import os


class APIErrors(Enum):
    """API 에러 코드"""
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    INVALID_SESSION = "INVALID_SESSION"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"
    FILE_UPLOAD_ERROR = "FILE_UPLOAD_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class MessageTypes(Enum):
    """WebSocket 메시지 타입"""
    CHAT_MESSAGE = "chat_message"
    THINK_UPDATE = "think_update"
    PERFORMANCE_UPDATE = "performance_update"
    MCP_UPDATE = "mcp_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SYSTEM_NOTIFICATION = "system_notification"
    SESSION_STATUS = "session_status"
    KNOWLEDGE_GRAPH_UPDATE = "knowledge_graph_update"


class HTTPStatus(Enum):
    """HTTP 상태 코드"""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class CacheKeys(Enum):
    """Redis 캐시 키"""
    SESSION_PREFIX = "session:"
    USER_PREFIX = "user:"
    PERFORMANCE_PREFIX = "performance:"
    MCP_PREFIX = "mcp:"
    KNOWLEDGE_GRAPH_PREFIX = "kg:"
    RATE_LIMIT_PREFIX = "rate_limit:"
    AUTH_TOKEN_PREFIX = "auth_token:"


class SecurityConstants:
    """보안 관련 상수 (환경 변수 기반)"""
    
    # JWT 설정
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-here')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_EXPIRATION_MINUTES = int(os.getenv('JWT_EXPIRATION_MINUTES', 60))
    JWT_REFRESH_EXPIRATION_DAYS = int(os.getenv('JWT_REFRESH_EXPIRATION_DAYS', 30))
    
    # Session 설정
    SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', 60))
    SESSION_CLEANUP_INTERVAL_MINUTES = int(os.getenv('SESSION_CLEANUP_INTERVAL_MINUTES', 15))
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
    API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', 60))
    WEBSOCKET_RATE_LIMIT = int(os.getenv('WEBSOCKET_RATE_LIMIT', 200))
    
    # 파일 업로드 제한
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
    ALLOWED_FILE_TYPES = set(os.getenv('ALLOWED_FILE_TYPES', 
        'text/plain,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/csv,application/json,image/png,image/jpeg,image/gif'
    ).split(','))
    
    # 입력 검증 제한
    MAX_MESSAGE_LENGTH = int(os.getenv('MAX_MESSAGE_LENGTH', 4000))
    MAX_SEARCH_QUERY_LENGTH = int(os.getenv('MAX_SEARCH_QUERY_LENGTH', 500))
    MAX_FILENAME_LENGTH = int(os.getenv('MAX_FILENAME_LENGTH', 255))
    MAX_ATTACHMENTS_PER_MESSAGE = int(os.getenv('MAX_ATTACHMENTS_PER_MESSAGE', 10))
    
    # 보안 패턴
    SAFE_FILENAME_PATTERN = r'^[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+$'
    SAFE_USERNAME_PATTERN = r'^[a-zA-Z0-9_-]{3,50}$'
    SAFE_USER_ID_PATTERN = r'^[a-zA-Z0-9_-]{1,50}$'
    SAFE_SESSION_ID_PATTERN = r'^[a-zA-Z0-9_-]{1,100}$'
    SAFE_ENDPOINT_PATTERN = r'^/[a-zA-Z0-9/_-]*$'
    
    # 금지된 문자열 (보안 위험)
    FORBIDDEN_STRINGS = {
        'script', 'javascript', 'vbscript', 'onload', 'onerror', 'onclick',
        'eval', 'exec', 'function', 'setTimeout', 'setInterval', 'alert',
        'confirm', 'prompt', 'document', 'window', 'location', 'history',
        'localStorage', 'sessionStorage', 'XMLHttpRequest', 'fetch',
        'import', 'require', 'module', 'process', 'global', '__proto__',
        'constructor', 'prototype', 'toString', 'valueOf', 'hasOwnProperty',
        'propertyIsEnumerable', 'isPrototypeOf', 'toLocaleString',
        'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter',
        'union', 'or', 'and', 'where', 'having', 'group', 'order', 'limit',
        '../', '..\\', '/etc/', '/proc/', '/dev/', '/sys/', 'C:\\', 'D:\\',
        'cmd.exe', 'powershell', 'bash', 'sh', 'cat', 'ls', 'rm', 'mkdir',
        'rmdir', 'mv', 'cp', 'chmod', 'chown', 'sudo', 'su', 'passwd'
    }
    
    # 한국어 특수 문자 허용 패턴
    KOREAN_TEXT_PATTERN = r'^[a-zA-Z0-9가-힣\s\.\,\!\?\-\(\)\[\]\{\}\:\"\'\_\+\=\*\&\%\$\#\@\~\`\|\\\<\>\/]*$'


class DatabaseConstants:
    """데이터베이스 관련 상수"""
    
    # Connection pool 설정
    POSTGRES_MIN_CONNECTIONS = 5
    POSTGRES_MAX_CONNECTIONS = 20
    POSTGRES_MAX_QUERIES = 50000
    POSTGRES_CONNECTION_TIMEOUT = 30.0
    
    # Neo4j 설정
    NEO4J_MAX_CONNECTION_POOL_SIZE = 50
    NEO4J_CONNECTION_ACQUISITION_TIMEOUT = 30.0
    NEO4J_MAX_TRANSACTION_RETRY_TIME = 15.0
    
    # Redis 설정
    REDIS_MAX_CONNECTIONS = 100
    REDIS_CONNECTION_TIMEOUT = 5.0
    REDIS_SOCKET_TIMEOUT = 5.0
    
    # 인덱스 이름
    NEO4J_INDEXES = {
        "session_idx": "FOR (n:Node) ON (n.session_id)",
        "global_idx": "FOR (n:Node) ON (n.global)",
        "timestamp_idx": "FOR (n:Node) ON (n.timestamp)",
        "user_idx": "FOR (u:User) ON (u.user_id)",
        "relationship_idx": "FOR (r:Relationship) ON (r.type)"
    }


class MonitoringConstants:
    """모니터링 관련 상수"""
    
    # 메트릭 수집 간격
    METRICS_COLLECTION_INTERVAL = 5  # 초
    PERFORMANCE_HISTORY_RETENTION_HOURS = 24
    
    # 알림 임계값
    MEMORY_USAGE_THRESHOLD = 80  # %
    CPU_USAGE_THRESHOLD = 80  # %
    RESPONSE_TIME_THRESHOLD = 2000  # ms
    ERROR_RATE_THRESHOLD = 5  # %
    
    # 배치 크기
    BATCH_SIZE_SMALL = 100
    BATCH_SIZE_MEDIUM = 500
    BATCH_SIZE_LARGE = 1000


class WebSocketConstants:
    """WebSocket 관련 상수"""
    
    # 연결 제한
    MAX_CONNECTIONS_PER_SESSION = 5
    MAX_TOTAL_CONNECTIONS = 1000
    
    # 메시지 크기 제한
    MAX_MESSAGE_SIZE = 64 * 1024  # 64KB
    
    # 하트비트 설정
    HEARTBEAT_INTERVAL = 30  # 초
    CONNECTION_TIMEOUT = 60  # 초
    
    # 채널 정리 간격
    CLEANUP_INTERVAL = 300  # 5분


class KoreanConstants:
    """한국어 처리 관련 상수"""
    
    # 뿌리산업 도메인
    ROOT_INDUSTRIES = [
        "주조", "금형", "소성가공", "용접", "표면처리", "열처리"
    ]
    
    # 기술 분야 키워드
    TECHNICAL_KEYWORDS = {
        "주조": ["주조", "주물", "용해", "주입", "응고", "합금"],
        "금형": ["금형", "프레스", "성형", "다이", "몰드", "사출"],
        "소성가공": ["소성가공", "단조", "압연", "인발", "압출", "성형"],
        "용접": ["용접", "아크", "레이저", "전자빔", "마찰", "브레이징"],
        "표면처리": ["표면처리", "도금", "코팅", "열처리", "연마", "청정"],
        "열처리": ["열처리", "담금질", "뜨임", "소준", "소둔", "침탄"]
    }
    
    # 품질 지표
    QUALITY_INDICATORS = [
        "치수정밀도", "표면거칠기", "기계적성질", "내구성", "신뢰성"
    ]