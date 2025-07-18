"""
Auth Routes - PPuRI-AI Ultimate 인증 라우터
"""

import logging
from typing import Dict, Any
from datetime import datetime
import os

# bcrypt import with fallback
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models.requests import (
    AuthLoginRequest,
    AuthTokenRequest
)
from ..models.responses import (
    SuccessResponse,
    ErrorResponse
)
from ..models.exceptions import (
    AuthenticationException,
    ValidationException,
    SecurityException
)
from ..middleware.auth import (
    JWTManager,
    SessionManager,
    get_current_user_id
)
from ..constants import HTTPStatus, SecurityConstants

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()


@router.post("/login", response_model=SuccessResponse)
async def login(
    request: AuthLoginRequest,
    jwt_manager: JWTManager = Depends(),
    session_manager: SessionManager = Depends()
):
    """사용자 로그인"""
    try:
        # 사용자 인증 (실제 구현에서는 데이터베이스 연동 필요)
        user_info = await authenticate_user(request.username, request.password)
        
        if not user_info:
            raise AuthenticationException("잘못된 사용자명 또는 비밀번호입니다.")
        
        # JWT 토큰 생성
        access_token = jwt_manager.create_access_token(
            user_id=user_info["user_id"],
            additional_claims={
                "role": user_info.get("role", "user"),
                "permissions": user_info.get("permissions", [])
            }
        )
        
        refresh_token = jwt_manager.create_refresh_token(user_info["user_id"])
        
        # 세션 생성
        session_id = await session_manager.create_session(
            user_id=user_info["user_id"],
            session_data={
                "login_time": datetime.utcnow().isoformat(),
                "ip_address": "127.0.0.1",  # 실제 구현에서는 request.client.host 사용
                "user_agent": "PPuRI-AI Client"  # 실제 구현에서는 request.headers.get("user-agent") 사용
            }
        )
        
        response_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": SecurityConstants.JWT_EXPIRATION_MINUTES * 60,
            "session_id": session_id,
            "user_info": {
                "user_id": user_info["user_id"],
                "username": user_info["username"],
                "role": user_info.get("role", "user"),
                "permissions": user_info.get("permissions", [])
            }
        }
        
        logger.info(f"사용자 로그인 성공: {user_info['user_id']}")
        
        return SuccessResponse(
            data=response_data,
            request_id=request.request_id
        )
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"로그인 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e, request_id=request.request_id).dict()
        )


@router.post("/refresh", response_model=SuccessResponse)
async def refresh_token(
    request: AuthTokenRequest,
    jwt_manager: JWTManager = Depends()
):
    """토큰 갱신"""
    try:
        # 리프레시 토큰 검증
        payload = jwt_manager.verify_refresh_token(request.refresh_token)
        
        user_id = payload["user_id"]
        
        # 사용자 정보 조회 (실제 구현에서는 데이터베이스 연동 필요)
        user_info = await get_user_info(user_id)
        
        if not user_info:
            raise AuthenticationException("사용자를 찾을 수 없습니다.")
        
        # 새 액세스 토큰 생성
        new_access_token = jwt_manager.create_access_token(
            user_id=user_id,
            additional_claims={
                "role": user_info.get("role", "user"),
                "permissions": user_info.get("permissions", [])
            }
        )
        
        # 새 리프레시 토큰 생성 (선택사항)
        new_refresh_token = jwt_manager.create_refresh_token(user_id)
        
        response_data = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": SecurityConstants.JWT_EXPIRATION_MINUTES * 60
        }
        
        logger.info(f"토큰 갱신 성공: {user_id}")
        
        return SuccessResponse(
            data=response_data,
            request_id=request.request_id
        )
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"토큰 갱신 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e, request_id=request.request_id).dict()
        )


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends(),
    session_manager: SessionManager = Depends(),
    current_user_id: str = Depends(get_current_user_id)
):
    """사용자 로그아웃"""
    try:
        # 현재 토큰 무효화
        jwt_manager.revoke_token(current_user_id, credentials.credentials)
        
        # 리프레시 토큰 무효화
        jwt_manager.revoke_refresh_token(current_user_id)
        
        # 모든 세션 무효화 (선택사항)
        # 실제 구현에서는 사용자의 모든 세션을 찾아서 무효화
        
        logger.info(f"사용자 로그아웃: {current_user_id}")
        
        return SuccessResponse(
            data={"message": "성공적으로 로그아웃되었습니다."}
        )
        
    except Exception as e:
        logger.error(f"로그아웃 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.get("/me", response_model=SuccessResponse)
async def get_current_user_info(
    current_user_id: str = Depends(get_current_user_id)
):
    """현재 사용자 정보 조회"""
    try:
        # 사용자 정보 조회 (실제 구현에서는 데이터베이스 연동 필요)
        user_info = await get_user_info(current_user_id)
        
        if not user_info:
            raise AuthenticationException("사용자를 찾을 수 없습니다.")
        
        # 민감한 정보 제거
        safe_user_info = {
            "user_id": user_info["user_id"],
            "username": user_info["username"],
            "role": user_info.get("role", "user"),
            "permissions": user_info.get("permissions", []),
            "created_at": user_info.get("created_at"),
            "last_login": user_info.get("last_login"),
            "profile": user_info.get("profile", {})
        }
        
        return SuccessResponse(data=safe_user_info)
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"사용자 정보 조회 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.post("/verify", response_model=SuccessResponse)
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTManager = Depends()
):
    """토큰 검증"""
    try:
        # 토큰 검증
        payload = jwt_manager.verify_token(credentials.credentials)
        
        response_data = {
            "valid": True,
            "user_id": payload["user_id"],
            "expires_at": payload["exp"],
            "issued_at": payload["iat"],
            "token_type": payload.get("type", "access")
        }
        
        return SuccessResponse(data=response_data)
        
    except AuthenticationException as e:
        return SuccessResponse(
            data={
                "valid": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"토큰 검증 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    old_password: str,
    new_password: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """비밀번호 변경"""
    try:
        # 입력 검증
        if len(new_password) < 8:
            raise ValidationException("새 비밀번호는 8자 이상이어야 합니다.")
        
        # 현재 비밀번호 확인
        user_info = await get_user_info(current_user_id)
        if not user_info:
            raise AuthenticationException("사용자를 찾을 수 없습니다.")
        
        # 기존 비밀번호 검증
        if not verify_password(old_password, user_info["password_hash"]):
            raise AuthenticationException("기존 비밀번호가 일치하지 않습니다.")
        
        # 새 비밀번호 해시
        new_password_hash = hash_password(new_password)
        
        # 비밀번호 업데이트 (실제 구현에서는 데이터베이스 연동 필요)
        success = await update_user_password(current_user_id, new_password_hash)
        
        if not success:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="비밀번호 변경에 실패했습니다."
            )
        
        logger.info(f"비밀번호 변경 성공: {current_user_id}")
        
        return SuccessResponse(
            data={"message": "비밀번호가 성공적으로 변경되었습니다."}
        )
        
    except (ValidationException, AuthenticationException):
        raise
    except Exception as e:
        logger.error(f"비밀번호 변경 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


# 헬퍼 함수들 (실제 구현에서는 데이터베이스 연동 필요)
async def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """사용자 인증"""
    # 실제 구현에서는 데이터베이스에서 사용자 정보 조회
    # 여기서는 예시 데이터 사용
    mock_users = {
        "admin": {
            "user_id": "admin_001",
            "username": "admin",
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "permissions": ["read", "write", "admin"]
        },
        "user": {
            "user_id": "user_001",
            "username": "user",
            "password_hash": hash_password("user123"),
            "role": "user",
            "permissions": ["read", "write"]
        }
    }
    
    user_info = mock_users.get(username)
    if not user_info:
        return None
    
    # 비밀번호 검증
    if not verify_password(password, user_info["password_hash"]):
        return None
    
    return user_info


async def get_user_info(user_id: str) -> Dict[str, Any]:
    """사용자 정보 조회"""
    # 실제 구현에서는 데이터베이스에서 사용자 정보 조회
    # 여기서는 예시 데이터 사용
    mock_users = {
        "admin_001": {
            "user_id": "admin_001",
            "username": "admin",
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "permissions": ["read", "write", "admin"],
            "created_at": "2024-01-01T00:00:00",
            "last_login": datetime.utcnow().isoformat(),
            "profile": {
                "name": "관리자",
                "email": "admin@ppuri.ai"
            }
        },
        "user_001": {
            "user_id": "user_001",
            "username": "user",
            "password_hash": hash_password("user123"),
            "role": "user",
            "permissions": ["read", "write"],
            "created_at": "2024-01-01T00:00:00",
            "last_login": datetime.utcnow().isoformat(),
            "profile": {
                "name": "사용자",
                "email": "user@ppuri.ai"
            }
        }
    }
    
    return mock_users.get(user_id)


async def update_user_password(user_id: str, password_hash: str) -> bool:
    """사용자 비밀번호 업데이트"""
    # 실제 구현에서는 데이터베이스에 비밀번호 업데이트
    # 여기서는 성공으로 시뮬레이션
    return True


def hash_password(password: str) -> str:
    """비밀번호 해시"""
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    else:
        # 임시 해시 (프로덕션에서는 bcrypt 사용 필수)
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    try:
        if BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        else:
            # 임시 검증 (프로덕션에서는 bcrypt 사용 필수)
            import hashlib
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == hashed
    except Exception:
        return False