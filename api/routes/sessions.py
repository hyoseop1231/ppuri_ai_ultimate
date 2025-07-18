"""
Session Routes - PPuRI-AI Ultimate 세션 관리 라우터
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from ..models.requests import CreateSessionRequest
from ..models.responses import (
    SuccessResponse, 
    ErrorResponse, 
    SessionResponse,
    HealthResponse
)
from ..models.exceptions import (
    SessionException, 
    ValidationException,
    ResourceNotFoundException
)
from ..middleware.auth import (
    JWTManager,
    SessionManager,
    get_current_user_id
)
from ..constants import HTTPStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(),
    jwt_manager: JWTManager = Depends()
):
    """새로운 세션 생성"""
    try:
        # 사용자 ID 생성 (게스트 사용자의 경우)
        user_id = request.user_id or f"guest_{uuid.uuid4().hex[:8]}"
        
        # 세션 생성
        session_id = await session_manager.create_session(
            user_id=user_id,
            session_data=request.metadata or {}
        )
        
        # JWT 토큰 생성 (선택사항)
        tokens = None
        if request.user_id:  # 인증된 사용자만 토큰 생성
            access_token = jwt_manager.create_access_token(user_id)
            refresh_token = jwt_manager.create_refresh_token(user_id)
            tokens = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
        
        # 세션 정보 조회
        session_info = await session_manager.get_session(session_id)
        
        response_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": session_info.get("created_at"),
            "expires_at": None,  # 세션 만료 시간 계산 필요
            "is_active": True,
            "metadata": request.metadata or {}
        }
        
        if tokens:
            response_data["tokens"] = tokens
        
        logger.info(f"새 세션 생성: {session_id} (사용자: {user_id})")
        
        return SessionResponse(
            data=response_data,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"세션 생성 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e, request_id=request.request_id).dict()
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_info(
    session_id: str,
    session_manager: SessionManager = Depends(),
    current_user_id: str = Depends(get_current_user_id)
):
    """세션 정보 조회"""
    try:
        # 세션 정보 조회
        session_info = await session_manager.get_session(session_id)
        
        if not session_info:
            raise ResourceNotFoundException("세션", session_id)
        
        # 권한 확인 (본인의 세션만 조회 가능)
        if session_info.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN.value,
                detail="접근 권한이 없습니다."
            )
        
        response_data = {
            "session_id": session_id,
            "user_id": session_info.get("user_id"),
            "created_at": session_info.get("created_at"),
            "last_activity": session_info.get("last_activity"),
            "is_active": session_info.get("is_active", True),
            "metadata": {
                k: v for k, v in session_info.items() 
                if k not in ["user_id", "created_at", "last_activity", "is_active"]
            }
        }
        
        logger.info(f"세션 정보 조회: {session_id}")
        
        return SessionResponse(data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 정보 조회 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    update_data: Dict[str, Any],
    session_manager: SessionManager = Depends(),
    current_user_id: str = Depends(get_current_user_id)
):
    """세션 정보 업데이트"""
    try:
        # 세션 존재 확인
        session_info = await session_manager.get_session(session_id)
        
        if not session_info:
            raise ResourceNotFoundException("세션", session_id)
        
        # 권한 확인
        if session_info.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN.value,
                detail="접근 권한이 없습니다."
            )
        
        # 보호된 필드 제거
        protected_fields = ["user_id", "created_at", "session_id"]
        filtered_data = {
            k: v for k, v in update_data.items() 
            if k not in protected_fields
        }
        
        # 세션 업데이트
        success = await session_manager.update_session(session_id, filtered_data)
        
        if not success:
            raise SessionException("세션 업데이트 실패", session_id)
        
        # 업데이트된 세션 정보 조회
        updated_session = await session_manager.get_session(session_id)
        
        response_data = {
            "session_id": session_id,
            "user_id": updated_session.get("user_id"),
            "created_at": updated_session.get("created_at"),
            "last_activity": updated_session.get("last_activity"),
            "is_active": updated_session.get("is_active", True),
            "metadata": {
                k: v for k, v in updated_session.items() 
                if k not in ["user_id", "created_at", "last_activity", "is_active"]
            }
        }
        
        logger.info(f"세션 업데이트: {session_id}")
        
        return SessionResponse(data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 업데이트 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.delete("/{session_id}", response_model=SuccessResponse)
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(),
    jwt_manager: JWTManager = Depends(),
    current_user_id: str = Depends(get_current_user_id)
):
    """세션 삭제"""
    try:
        # 세션 존재 확인
        session_info = await session_manager.get_session(session_id)
        
        if not session_info:
            raise ResourceNotFoundException("세션", session_id)
        
        # 권한 확인
        if session_info.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN.value,
                detail="접근 권한이 없습니다."
            )
        
        # 세션 삭제
        success = await session_manager.delete_session(session_id)
        
        if not success:
            raise SessionException("세션 삭제 실패", session_id)
        
        # 관련 토큰 무효화
        jwt_manager.revoke_refresh_token(current_user_id)
        
        logger.info(f"세션 삭제: {session_id}")
        
        return SuccessResponse(
            data={"message": "세션이 성공적으로 삭제되었습니다."}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 삭제 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.post("/{session_id}/refresh", response_model=SessionResponse)
async def refresh_session(
    session_id: str,
    session_manager: SessionManager = Depends(),
    current_user_id: str = Depends(get_current_user_id)
):
    """세션 갱신 (활성화)"""
    try:
        # 세션 존재 확인
        session_info = await session_manager.get_session(session_id)
        
        if not session_info:
            raise ResourceNotFoundException("세션", session_id)
        
        # 권한 확인
        if session_info.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN.value,
                detail="접근 권한이 없습니다."
            )
        
        # 세션 갱신
        refresh_data = {
            "last_activity": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        success = await session_manager.update_session(session_id, refresh_data)
        
        if not success:
            raise SessionException("세션 갱신 실패", session_id)
        
        # 갱신된 세션 정보 조회
        updated_session = await session_manager.get_session(session_id)
        
        response_data = {
            "session_id": session_id,
            "user_id": updated_session.get("user_id"),
            "created_at": updated_session.get("created_at"),
            "last_activity": updated_session.get("last_activity"),
            "is_active": updated_session.get("is_active", True),
            "metadata": {
                k: v for k, v in updated_session.items() 
                if k not in ["user_id", "created_at", "last_activity", "is_active"]
            }
        }
        
        logger.info(f"세션 갱신: {session_id}")
        
        return SessionResponse(data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 갱신 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.get("/{session_id}/health", response_model=HealthResponse)
async def check_session_health(
    session_id: str,
    session_manager: SessionManager = Depends()
):
    """세션 상태 확인"""
    try:
        # 세션 존재 확인
        session_info = await session_manager.get_session(session_id)
        
        if not session_info:
            raise ResourceNotFoundException("세션", session_id)
        
        # 세션 상태 분석
        last_activity = session_info.get("last_activity")
        is_active = session_info.get("is_active", True)
        
        # 마지막 활동 시간으로부터 경과 시간 계산
        if last_activity:
            try:
                last_activity_dt = datetime.fromisoformat(last_activity)
                elapsed_minutes = (datetime.utcnow() - last_activity_dt).total_seconds() / 60
            except:
                elapsed_minutes = 0
        else:
            elapsed_minutes = 0
        
        # 세션 상태 판단
        if not is_active:
            status = "inactive"
        elif elapsed_minutes > 30:  # 30분 이상 비활성
            status = "idle"
        elif elapsed_minutes > 5:  # 5분 이상 비활성
            status = "slow"
        else:
            status = "active"
        
        response_data = {
            "session_id": session_id,
            "status": status,
            "is_active": is_active,
            "last_activity": last_activity,
            "elapsed_minutes": elapsed_minutes,
            "health_check_time": datetime.utcnow().isoformat()
        }
        
        return HealthResponse(data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 상태 확인 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )