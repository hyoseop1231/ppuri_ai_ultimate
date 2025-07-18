"""
Security Middleware - PPuRI-AI Ultimate 보안 미들웨어
"""

import os
import time
import logging
from typing import Dict, Optional, List, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
import hashlib
import hmac

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Redis import with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..constants import SecurityConstants, APIErrors
from ..models.exceptions import (
    RateLimitException, 
    SecurityException,
    ValidationException
)
from ..models.responses import ErrorResponse

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """보안 헤더 미들웨어"""
    
    def __init__(self, app, config: Optional[Dict[str, str]] = None):
        super().__init__(app)
        self.config = config or {}
        self.default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' ws: wss:; "
                "frame-ancestors 'none';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            )
        }
    
    async def dispatch(self, request: Request, call_next):
        # 요청 처리
        response = await call_next(request)
        
        # 보안 헤더 추가
        for header, value in self.default_headers.items():
            response.headers[header] = self.config.get(header, value)
        
        # 서버 정보 숨김
        response.headers["Server"] = "PPuRI-AI"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate Limiting 미들웨어"""
    
    def __init__(
        self,
        app,
        redis_client: Optional[Any] = None,
        default_rate_limit: int = SecurityConstants.DEFAULT_RATE_LIMIT,
        api_rate_limit: int = SecurityConstants.API_RATE_LIMIT
    ):
        super().__init__(app)
        self.redis_client = redis_client if REDIS_AVAILABLE else None
        self.default_rate_limit = default_rate_limit
        self.api_rate_limit = api_rate_limit
        
        # In-memory fallback
        self.memory_store = defaultdict(lambda: deque(maxlen=1000))
        self.cleanup_interval = 60  # 1분
        self.last_cleanup = time.time()
        
        # 엔드포인트별 Rate Limit 설정
        self.endpoint_limits = {
            "/api/auth/login": 5,  # 로그인은 더 엄격하게
            "/api/chat/message": 60,  # 채팅은 조금 더 관대하게
            "/api/export": 10,  # 내보내기는 제한적으로
            "/api/performance": 30,  # 성능 모니터링은 적당히
        }
    
    async def dispatch(self, request: Request, call_next):
        # 정적 파일과 공개 경로는 Rate Limit 적용 안함
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # 클라이언트 식별
        client_id = self._get_client_id(request)
        
        # Rate Limit 확인
        if not await self._check_rate_limit(client_id, request.url.path):
            retry_after = self._get_retry_after(client_id, request.url.path)
            
            # HTTPException으로 변환하여 FastAPI가 처리할 수 있도록 함
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": "요청 제한을 초과했습니다.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # 요청 처리
        response = await call_next(request)
        
        # Rate Limit 헤더 추가
        limit_info = await self._get_rate_limit_info(client_id, request.url.path)
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """Rate Limit 면제 경로 확인"""
        exempt_paths = [
            "/static",
            "/favicon.ico",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/health"
        ]
        
        return any(path.startswith(exempt_path) for exempt_path in exempt_paths)
    
    def _get_client_id(self, request: Request) -> str:
        """클라이언트 식별자 생성"""
        # X-Forwarded-For 헤더 확인 (프록시 환경 대응)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host
        
        # 사용자 ID가 있으면 우선 사용
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        return f"ip:{client_ip}"
    
    def _get_endpoint_limit(self, path: str) -> int:
        """엔드포인트별 Rate Limit 조회"""
        # 정확한 매칭 먼저 확인
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # 패턴 매칭
        for pattern, limit in self.endpoint_limits.items():
            if path.startswith(pattern):
                return limit
        
        # API 경로인지 확인
        if path.startswith("/api/"):
            return self.api_rate_limit
        
        return self.default_rate_limit
    
    async def _check_rate_limit(self, client_id: str, path: str) -> bool:
        """Rate Limit 확인"""
        limit = self._get_endpoint_limit(path)
        window_seconds = 60  # 1분 윈도우
        
        if self.redis_client:
            return await self._check_redis_rate_limit(client_id, path, limit, window_seconds)
        else:
            return await self._check_memory_rate_limit(client_id, path, limit, window_seconds)
    
    async def _check_redis_rate_limit(
        self, 
        client_id: str, 
        path: str, 
        limit: int, 
        window_seconds: int
    ) -> bool:
        """Redis 기반 Rate Limit 확인"""
        try:
            key = f"rate_limit:{client_id}:{path}"
            current_time = int(time.time())
            pipeline = self.redis_client.pipeline()
            
            # 현재 시간 기준으로 윈도우 시작 시간 계산
            window_start = current_time - window_seconds
            
            # 오래된 요청 제거
            pipeline.zremrangebyscore(key, 0, window_start)
            
            # 현재 요청 추가
            pipeline.zadd(key, {str(current_time): current_time})
            
            # 현재 요청 수 확인
            pipeline.zcard(key)
            
            # TTL 설정
            pipeline.expire(key, window_seconds)
            
            results = pipeline.execute()
            current_count = results[2]
            
            return current_count <= limit
            
        except Exception as e:
            logger.error(f"Redis Rate Limit 확인 실패: {e}")
            # Redis 실패 시 memory fallback
            return await self._check_memory_rate_limit(client_id, path, limit, window_seconds)
    
    async def _check_memory_rate_limit(
        self, 
        client_id: str, 
        path: str, 
        limit: int, 
        window_seconds: int
    ) -> bool:
        """메모리 기반 Rate Limit 확인"""
        key = f"{client_id}:{path}"
        current_time = time.time()
        
        # 정기적으로 오래된 데이터 정리
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_memory_store(current_time, window_seconds)
            self.last_cleanup = current_time
        
        # 현재 윈도우 내의 요청들 확인
        request_times = self.memory_store[key]
        
        # 윈도우 밖의 요청들 제거
        window_start = current_time - window_seconds
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # 현재 요청 추가
        request_times.append(current_time)
        
        return len(request_times) <= limit
    
    def _cleanup_memory_store(self, current_time: float, window_seconds: int):
        """메모리 스토어 정리"""
        window_start = current_time - window_seconds
        
        for key in list(self.memory_store.keys()):
            request_times = self.memory_store[key]
            
            # 윈도우 밖의 요청들 제거
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # 빈 큐 제거
            if not request_times:
                del self.memory_store[key]
    
    async def _get_rate_limit_info(self, client_id: str, path: str) -> Dict[str, int]:
        """Rate Limit 정보 조회"""
        limit = self._get_endpoint_limit(path)
        window_seconds = 60
        
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}:{path}"
                current_time = int(time.time())
                window_start = current_time - window_seconds
                
                # 현재 요청 수 확인
                current_count = self.redis_client.zcount(key, window_start, current_time)
                
                return {
                    "limit": limit,
                    "remaining": max(0, limit - current_count),
                    "reset": current_time + window_seconds
                }
                
            except Exception as e:
                logger.error(f"Rate Limit 정보 조회 실패: {e}")
        
        # Fallback to memory store
        key = f"{client_id}:{path}"
        request_times = self.memory_store[key]
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # 윈도우 내 요청 수 계산
        current_count = sum(1 for t in request_times if t >= window_start)
        
        return {
            "limit": limit,
            "remaining": max(0, limit - current_count),
            "reset": int(current_time + window_seconds)
        }
    
    def _get_retry_after(self, client_id: str, path: str) -> int:
        """다음 요청 가능 시간 계산"""
        return 60  # 1분 후 재시도 권장


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF 보호 미들웨어"""
    
    def __init__(self, app, secret_key: str = None):
        super().__init__(app)
        self.secret_key = secret_key or os.getenv("CSRF_SECRET_KEY", "csrf-secret-key")
        self.safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}
        
    async def dispatch(self, request: Request, call_next):
        # 안전한 메서드는 CSRF 검증 안함
        if request.method in self.safe_methods:
            return await call_next(request)
        
        # WebSocket은 별도 처리
        if request.url.path.startswith("/ws/"):
            return await call_next(request)
        
        # API 엔드포인트만 CSRF 보호 적용
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        
        # CSRF 토큰 확인
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            raise SecurityException("CSRF 토큰이 필요합니다.")
        
        # 토큰 검증
        if not self._verify_csrf_token(csrf_token):
            raise SecurityException("유효하지 않은 CSRF 토큰입니다.")
        
        return await call_next(request)
    
    def _verify_csrf_token(self, token: str) -> bool:
        """CSRF 토큰 검증"""
        try:
            # 간단한 HMAC 기반 토큰 검증
            expected_token = hmac.new(
                self.secret_key.encode(),
                "csrf-token".encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(token, expected_token)
            
        except Exception as e:
            logger.error(f"CSRF 토큰 검증 실패: {e}")
            return False
    
    def generate_csrf_token(self) -> str:
        """CSRF 토큰 생성"""
        return hmac.new(
            self.secret_key.encode(),
            "csrf-token".encode(),
            hashlib.sha256
        ).hexdigest()


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """요청 검증 미들웨어"""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
        
    async def dispatch(self, request: Request, call_next):
        # 요청 크기 확인
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    raise ValidationException(
                        f"요청 크기가 제한을 초과합니다: {size} bytes"
                    )
            except ValueError:
                raise ValidationException("유효하지 않은 Content-Length 헤더")
        
        # Content-Type 확인 (POST/PUT 요청의 경우)
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            # 허용된 Content-Type인지 확인
            allowed_types = [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data",
                "text/plain"
            ]
            
            if not any(content_type.startswith(allowed) for allowed in allowed_types):
                raise ValidationException(f"허용되지 않은 Content-Type: {content_type}")
        
        # User-Agent 확인 (봇 차단)
        user_agent = request.headers.get("user-agent", "")
        if self._is_suspicious_user_agent(user_agent):
            raise SecurityException("허용되지 않은 User-Agent")
        
        return await call_next(request)
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """의심스러운 User-Agent 확인"""
        suspicious_patterns = [
            "bot", "crawler", "spider", "scraper",
            "wget", "curl", "python-requests",
            "scanner", "exploit", "attack"
        ]
        
        user_agent_lower = user_agent.lower()
        
        # 빈 User-Agent 차단
        if not user_agent_lower:
            return True
        
        # 의심스러운 패턴 확인
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower:
                return True
        
        return False


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP 화이트리스트 미들웨어"""
    
    def __init__(self, app, whitelist: List[str] = None):
        super().__init__(app)
        self.whitelist = set(whitelist or [])
        
        # 기본 허용 IP (로컬 개발용)
        self.whitelist.update([
            "127.0.0.1",
            "::1",
            "localhost"
        ])
    
    async def dispatch(self, request: Request, call_next):
        # 화이트리스트가 비어있으면 모든 IP 허용
        if not self.whitelist:
            return await call_next(request)
        
        # 클라이언트 IP 확인
        client_ip = self._get_client_ip(request)
        
        # 관리자 엔드포인트는 더 엄격하게 확인
        if request.url.path.startswith("/api/admin/"):
            admin_whitelist = {"127.0.0.1", "::1"}
            if client_ip not in admin_whitelist:
                raise SecurityException("관리자 기능에 접근할 수 없습니다.")
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 주소 추출"""
        # X-Forwarded-For 헤더 확인 (프록시 환경)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 직접 연결
        return request.client.host