"""
Authentication Middleware - PPuRI-AI Ultimate JWT 인증 미들웨어
"""

import os
import jwt
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import wraps
import logging

from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Redis import with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..constants import SecurityConstants, APIErrors
from ..models.exceptions import (
    AuthenticationException,
    AuthorizationException,
    SecurityException
)
from ..models.responses import ErrorResponse

logger = logging.getLogger(__name__)


class JWTManager:
    """JWT 토큰 관리 클래스"""
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.secret_key = os.getenv("JWT_SECRET_KEY", SecurityConstants.JWT_SECRET_KEY)
        self.algorithm = SecurityConstants.JWT_ALGORITHM
        self.expiration_minutes = SecurityConstants.JWT_EXPIRATION_MINUTES
        self.refresh_expiration_days = SecurityConstants.JWT_REFRESH_EXPIRATION_DAYS
        self.redis_client = redis_client if REDIS_AVAILABLE else None
        
        if not self.secret_key or self.secret_key == "your-secret-key-here":
            logger.warning("JWT_SECRET_KEY가 설정되지 않았습니다. 기본값을 사용합니다.")
    
    def create_access_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """액세스 토큰 생성"""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.expiration_minutes)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": now,
            "type": "access"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Redis에 토큰 저장 (블랙리스트 관리용)
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"token:{user_id}:{token[-10:]}",
                    self.expiration_minutes * 60,
                    "active"
                )
            except Exception as e:
                logger.warning(f"Redis 토큰 저장 실패: {e}")
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """리프레시 토큰 생성"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_expiration_days)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": now,
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Redis에 리프레시 토큰 저장
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"refresh:{user_id}",
                    self.refresh_expiration_days * 24 * 60 * 60,
                    token
                )
            except Exception as e:
                logger.warning(f"Redis 리프레시 토큰 저장 실패: {e}")
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # 토큰 타입 확인
            if payload.get("type") != "access":
                raise AuthenticationException("잘못된 토큰 타입입니다.")
            
            # 만료 시간 확인
            if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
                raise AuthenticationException("토큰이 만료되었습니다.")
            
            # 블랙리스트 확인
            if self.redis_client:
                try:
                    token_key = f"token:{payload['user_id']}:{token[-10:]}"
                    if not self.redis_client.get(token_key):
                        raise AuthenticationException("토큰이 무효화되었습니다.")
                except Exception as e:
                    logger.warning(f"Redis 토큰 확인 실패: {e}")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("토큰이 만료되었습니다.")
        except jwt.InvalidTokenError:
            raise AuthenticationException("유효하지 않은 토큰입니다.")
        except Exception as e:
            logger.error(f"토큰 검증 오류: {e}")
            raise AuthenticationException("토큰 검증 실패")
    
    def verify_refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """리프레시 토큰 검증"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            # 토큰 타입 확인
            if payload.get("type") != "refresh":
                raise AuthenticationException("잘못된 리프레시 토큰 타입입니다.")
            
            # Redis에서 리프레시 토큰 확인
            if self.redis_client:
                try:
                    stored_token = self.redis_client.get(f"refresh:{payload['user_id']}")
                    if not stored_token or stored_token.decode() != refresh_token:
                        raise AuthenticationException("리프레시 토큰이 무효화되었습니다.")
                except Exception as e:
                    logger.warning(f"Redis 리프레시 토큰 확인 실패: {e}")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("리프레시 토큰이 만료되었습니다.")
        except jwt.InvalidTokenError:
            raise AuthenticationException("유효하지 않은 리프레시 토큰입니다.")
        except Exception as e:
            logger.error(f"리프레시 토큰 검증 오류: {e}")
            raise AuthenticationException("리프레시 토큰 검증 실패")
    
    def revoke_token(self, user_id: str, token: str) -> bool:
        """토큰 무효화"""
        if self.redis_client:
            try:
                token_key = f"token:{user_id}:{token[-10:]}"
                self.redis_client.delete(token_key)
                return True
            except Exception as e:
                logger.error(f"토큰 무효화 실패: {e}")
                return False
        
        return True
    
    def revoke_refresh_token(self, user_id: str) -> bool:
        """리프레시 토큰 무효화"""
        if self.redis_client:
            try:
                self.redis_client.delete(f"refresh:{user_id}")
                return True
            except Exception as e:
                logger.error(f"리프레시 토큰 무효화 실패: {e}")
                return False
        
        return True


class AuthenticationMiddleware:
    """인증 미들웨어"""
    
    def __init__(self, jwt_manager: JWTManager):
        self.jwt_manager = jwt_manager
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request, call_next):
        """미들웨어 실행"""
        # 인증이 필요없는 경로 확인
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Authorization 헤더 확인
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise AuthenticationException("인증 토큰이 필요합니다.")
        
        try:
            # Bearer 토큰 추출
            if not authorization.startswith("Bearer "):
                raise AuthenticationException("Bearer 토큰이 필요합니다.")
            
            token = authorization.split(" ")[1]
            
            # 토큰 검증
            payload = self.jwt_manager.verify_token(token)
            
            # 요청에 사용자 정보 추가
            request.state.user_id = payload["user_id"]
            request.state.token_payload = payload
            
            return await call_next(request)
            
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"인증 미들웨어 오류: {e}")
            raise AuthenticationException("인증 처리 실패")
    
    def _is_public_path(self, path: str) -> bool:
        """공개 경로 확인"""
        public_paths = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/health",
            "/api/auth/login",
            "/api/auth/refresh",
            "/static",
            "/favicon.ico"
        ]
        
        return any(path.startswith(public_path) for public_path in public_paths)


class SessionManager:
    """세션 관리 클래스"""
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client if REDIS_AVAILABLE else None
        self.session_timeout = SecurityConstants.SESSION_TIMEOUT_MINUTES * 60
        self.cleanup_interval = SecurityConstants.SESSION_CLEANUP_INTERVAL_MINUTES * 60
        
        # Redis 없을 때 사용할 인메모리 저장소
        self.memory_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def create_session(self, user_id: str, session_data: Dict[str, Any] = None) -> str:
        """세션 생성"""
        import uuid
        
        session_id = str(uuid.uuid4())
        
        session_info = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "is_active": True,
            **(session_data or {})
        }
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_timeout,
                    str(session_info)
                )
            except Exception as e:
                logger.error(f"Redis 세션 생성 실패: {e}")
                # Redis 실패 시 메모리 저장소로 fallback
                self.memory_sessions[session_id] = session_info
        else:
            # Redis 없을 때는 메모리 저장소 사용
            self.memory_sessions[session_id] = session_info
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 조회"""
        if self.redis_client:
            try:
                session_data = self.redis_client.get(f"session:{session_id}")
                if session_data:
                    import ast
                    return ast.literal_eval(session_data.decode())
            except Exception as e:
                logger.error(f"Redis 세션 조회 실패: {e}")
        
        # Redis 실패 시 또는 Redis 없을 때 메모리 저장소 확인
        return self.memory_sessions.get(session_id)
    
    async def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """세션 업데이트"""
        # 기존 세션 조회
        existing_session = await self.get_session(session_id)
        if not existing_session:
            return False
        
        # 세션 데이터 업데이트
        existing_session.update(session_data)
        existing_session["last_activity"] = datetime.utcnow().isoformat()
        
        if self.redis_client:
            try:
                # Redis에 저장
                self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_timeout,
                    str(existing_session)
                )
                return True
            except Exception as e:
                logger.error(f"Redis 세션 업데이트 실패: {e}")
                # Redis 실패 시 메모리 저장소로 fallback
                self.memory_sessions[session_id] = existing_session
                return True
        else:
            # Redis 없을 때는 메모리 저장소 사용
            self.memory_sessions[session_id] = existing_session
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        success = True
        
        if self.redis_client:
            try:
                self.redis_client.delete(f"session:{session_id}")
            except Exception as e:
                logger.error(f"Redis 세션 삭제 실패: {e}")
                success = False
        
        # 메모리 저장소에서도 삭제
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]
        
        return success
    
    async def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        if self.redis_client:
            try:
                # 만료된 세션은 Redis의 TTL로 자동 삭제됨
                # 추가적인 정리 로직이 필요하다면 여기에 구현
                pass
            except Exception as e:
                logger.error(f"세션 정리 실패: {e}")


# 인증 의존성 함수들
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    jwt_manager: JWTManager = Depends()
) -> Dict[str, Any]:
    """현재 사용자 정보 조회"""
    try:
        payload = jwt_manager.verify_token(credentials.credentials)
        return payload
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"사용자 정보 조회 실패: {e}")
        raise AuthenticationException("사용자 정보 조회 실패")


async def get_current_user_id(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> str:
    """현재 사용자 ID 조회"""
    return current_user["user_id"]


def require_auth(func):
    """인증 필수 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # FastAPI의 Depends를 사용하여 인증 처리
        return await func(*args, **kwargs)
    return wrapper


def require_role(required_role: str):
    """역할 기반 인증 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 현재 사용자의 역할 확인
            current_user = kwargs.get("current_user")
            if not current_user:
                raise AuthenticationException("인증이 필요합니다.")
            
            user_role = current_user.get("role")
            if user_role != required_role:
                raise AuthorizationException("권한이 부족합니다.")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator