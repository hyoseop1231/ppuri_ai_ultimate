"""
Chat Routes - PPuRI-AI Ultimate 채팅 라우터
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse, StreamingResponse

from ..models.requests import (
    ChatMessageRequest,
    SearchRequest,
    ExportRequest
)
from ..models.responses import (
    SuccessResponse,
    ErrorResponse,
    ChatResponse,
    PaginatedResponse,
    ExportResponse
)
from ..models.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    ExternalServiceException
)
from ..middleware.auth import get_current_user_id
from ..constants import HTTPStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{session_id}/message", response_model=ChatResponse)
async def send_message(
    session_id: str,
    request: ChatMessageRequest,
    current_user_id: str = Depends(get_current_user_id),
    # ui_orchestrator dependency 필요 - 실제 구현에서 추가
):
    """채팅 메시지 전송"""
    try:
        # 세션 유효성 확인
        # session_manager를 통해 세션 확인 로직 필요
        
        # 입력 검증
        if not request.message.strip():
            raise ValidationException("메시지는 비어있을 수 없습니다.")
        
        # 메시지 처리 (실제 구현에서는 ui_orchestrator 사용)
        # 여기서는 예시 응답 구조만 제공
        
        # 메시지 ID 생성
        import uuid
        message_id = str(uuid.uuid4())
        
        # AI 응답 생성 로직 (실제 구현 필요)
        ai_response = await process_chat_message(
            session_id=session_id,
            message=request.message,
            message_type=request.message_type,
            attachments=request.attachments,
            context=request.context,
            user_id=current_user_id
        )
        
        response_data = {
            "message_id": message_id,
            "session_id": session_id,
            "user_message": request.message,
            "ai_response": ai_response.get("response", ""),
            "think_blocks": ai_response.get("think_blocks", []),
            "sources": ai_response.get("sources", []),
            "suggestions": ai_response.get("suggestions", []),
            "metadata": {
                "processing_time": ai_response.get("processing_time", 0),
                "model_used": ai_response.get("model_used", "unknown"),
                "confidence": ai_response.get("confidence", 0.0)
            }
        }
        
        logger.info(f"메시지 처리 완료: {session_id} - {message_id}")
        
        return ChatResponse(
            data=response_data,
            request_id=request.request_id
        )
        
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"메시지 전송 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e, request_id=request.request_id).dict()
        )


@router.get("/{session_id}/history", response_model=PaginatedResponse)
async def get_message_history(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user_id: str = Depends(get_current_user_id)
):
    """메시지 히스토리 조회"""
    try:
        # 세션 권한 확인
        # session_manager를 통해 권한 확인 로직 필요
        
        # 메시지 히스토리 조회 (실제 구현 필요)
        history_data = await get_session_message_history(
            session_id=session_id,
            limit=limit,
            offset=offset,
            user_id=current_user_id
        )
        
        # 페이지네이션 정보 계산
        total_count = history_data.get("total_count", 0)
        total_pages = (total_count + limit - 1) // limit
        current_page = (offset // limit) + 1
        
        pagination_info = {
            "current_page": current_page,
            "page_size": limit,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": offset + limit < total_count,
            "has_previous": offset > 0
        }
        
        logger.info(f"메시지 히스토리 조회: {session_id} (페이지: {current_page})")
        
        return PaginatedResponse(
            data=history_data.get("messages", []),
            pagination=pagination_info
        )
        
    except Exception as e:
        logger.error(f"메시지 히스토리 조회 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.get("/{session_id}/suggestions", response_model=SuccessResponse)
async def get_input_suggestions(
    session_id: str,
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    current_user_id: str = Depends(get_current_user_id)
):
    """입력 제안 조회"""
    try:
        # 세션 권한 확인
        # session_manager를 통해 권한 확인 로직 필요
        
        # 입력 제안 생성 (실제 구현 필요)
        suggestions = await generate_input_suggestions(
            session_id=session_id,
            query=query,
            limit=limit,
            user_id=current_user_id
        )
        
        response_data = {
            "query": query,
            "suggestions": suggestions,
            "suggestion_count": len(suggestions),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"입력 제안 생성: {session_id} - {query}")
        
        return SuccessResponse(data=response_data)
        
    except Exception as e:
        logger.error(f"입력 제안 생성 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.post("/{session_id}/export", response_model=ExportResponse)
async def export_conversation(
    session_id: str,
    request: ExportRequest,
    current_user_id: str = Depends(get_current_user_id)
):
    """대화 내용 내보내기"""
    try:
        # 세션 권한 확인
        # session_manager를 통해 권한 확인 로직 필요
        
        # 대화 내용 내보내기 (실제 구현 필요)
        export_data = await export_conversation_data(
            session_id=session_id,
            format=request.format,
            time_range=request.time_range,
            filters=request.filters,
            include_metadata=request.include_metadata,
            user_id=current_user_id
        )
        
        # 내보내기 ID 생성
        import uuid
        export_id = str(uuid.uuid4())
        
        response_data = {
            "export_id": export_id,
            "session_id": session_id,
            "format": request.format.value,
            "size": export_data.get("size", 0),
            "message_count": export_data.get("message_count", 0),
            "download_url": f"/api/exports/{export_id}/download",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": None,  # 만료 시간 계산 필요
            "metadata": export_data.get("metadata", {})
        }
        
        logger.info(f"대화 내용 내보내기: {session_id} - {export_id}")
        
        return ExportResponse(
            data=response_data,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"대화 내용 내보내기 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e, request_id=request.request_id).dict()
        )


@router.get("/{session_id}/stream")
async def stream_chat_response(
    session_id: str,
    message: str = Query(..., min_length=1),
    current_user_id: str = Depends(get_current_user_id)
):
    """스트리밍 채팅 응답"""
    try:
        # 세션 권한 확인
        # session_manager를 통해 권한 확인 로직 필요
        
        async def generate_stream():
            """스트리밍 응답 생성"""
            try:
                # 스트리밍 응답 생성 로직 (실제 구현 필요)
                async for chunk in stream_chat_response_generator(
                    session_id=session_id,
                    message=message,
                    user_id=current_user_id
                ):
                    yield f"data: {chunk}\n\n"
                
                # 스트리밍 종료 신호
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"스트리밍 응답 생성 실패: {e}")
                error_response = ErrorResponse.from_exception(e)
                yield f"data: {error_response.json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        logger.error(f"스트리밍 채팅 응답 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


@router.delete("/{session_id}/messages/{message_id}", response_model=SuccessResponse)
async def delete_message(
    session_id: str,
    message_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """메시지 삭제"""
    try:
        # 세션 권한 확인
        # session_manager를 통해 권한 확인 로직 필요
        
        # 메시지 삭제 (실제 구현 필요)
        success = await delete_chat_message(
            session_id=session_id,
            message_id=message_id,
            user_id=current_user_id
        )
        
        if not success:
            raise ResourceNotFoundException("메시지", message_id)
        
        logger.info(f"메시지 삭제: {session_id} - {message_id}")
        
        return SuccessResponse(
            data={"message": "메시지가 성공적으로 삭제되었습니다."}
        )
        
    except Exception as e:
        logger.error(f"메시지 삭제 실패: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=ErrorResponse.from_exception(e).dict()
        )


# ChatService 통합 헬퍼 함수들
async def process_chat_message(
    session_id: str,
    message: str,
    message_type: str,
    attachments: Optional[List[str]],
    context: Optional[Dict[str, Any]],
    user_id: str
) -> Dict[str, Any]:
    """채팅 메시지 처리 - ChatService 통합"""
    try:
        from core.services import get_chat_service, SearchMode

        chat_service = await get_chat_service()

        # 컨텍스트에서 검색 모드 추출
        search_mode = SearchMode.WEB_ENABLED
        industry_filter = None

        if context:
            mode_str = context.get("search_mode", "web_enabled")
            if mode_str == "documents_only":
                search_mode = SearchMode.DOCUMENTS_ONLY
            elif mode_str == "full_search":
                search_mode = SearchMode.FULL_SEARCH

            industry_filter = context.get("industry")

        # 채팅 서비스로 응답 생성
        response = await chat_service.chat(
            query=message,
            search_mode=search_mode,
            industry_filter=industry_filter,
            session_id=session_id
        )

        # 인용 정보를 소스로 변환
        sources = []
        for citation in response.citations:
            sources.append({
                "id": citation.id,
                "title": citation.title,
                "snippet": citation.content_snippet,
                "type": citation.source_type,
                "url": citation.url,
                "page": citation.page_number,
                "confidence": citation.confidence
            })

        # 추론 상세 정보
        think_blocks = []
        if response.reasoning_details:
            for detail in response.reasoning_details:
                think_blocks.append({
                    "type": detail.get("type", "thinking"),
                    "content": detail.get("content", "")
                })

        return {
            "response": response.answer,
            "think_blocks": think_blocks,
            "sources": sources,
            "suggestions": await generate_input_suggestions(session_id, message, 3, user_id),
            "processing_time": (response.search_time_ms + response.generation_time_ms) / 1000,
            "model_used": response.model_used,
            "confidence": 0.95,
            "citations_footnote": chat_service.get_citations_footnotes(response.citations)
        }

    except Exception as e:
        logger.error(f"ChatService processing failed: {e}")
        # 폴백 응답
        return {
            "response": f"죄송합니다, 일시적인 오류가 발생했습니다: {str(e)}",
            "think_blocks": [],
            "sources": [],
            "suggestions": [],
            "processing_time": 0,
            "model_used": "fallback",
            "confidence": 0.0
        }


async def get_session_message_history(
    session_id: str,
    limit: int,
    offset: int,
    user_id: str
) -> Dict[str, Any]:
    """세션 메시지 히스토리 조회"""
    # 실제 데이터베이스 연동 로직 구현 필요
    await asyncio.sleep(0.1)  # 시뮬레이션
    
    return {
        "messages": [
            {
                "message_id": f"msg_{i}",
                "user_message": f"사용자 메시지 {i}",
                "ai_response": f"AI 응답 {i}",
                "timestamp": datetime.utcnow().isoformat()
            }
            for i in range(limit)
        ],
        "total_count": 100
    }


async def generate_input_suggestions(
    session_id: str,
    query: str,
    limit: int,
    user_id: str
) -> List[str]:
    """입력 제안 생성"""
    # 실제 AI 기반 제안 생성 로직 구현 필요
    await asyncio.sleep(0.1)  # 시뮬레이션
    
    return [
        f"{query} 관련 질문 1",
        f"{query} 관련 질문 2",
        f"{query} 관련 질문 3"
    ][:limit]


async def export_conversation_data(
    session_id: str,
    format: str,
    time_range: Optional[str],
    filters: Optional[Dict[str, Any]],
    include_metadata: bool,
    user_id: str
) -> Dict[str, Any]:
    """대화 내용 내보내기"""
    # 실제 내보내기 로직 구현 필요
    await asyncio.sleep(0.1)  # 시뮬레이션
    
    return {
        "size": 1024,
        "message_count": 50,
        "metadata": {
            "export_format": format,
            "time_range": time_range,
            "include_metadata": include_metadata
        }
    }


async def stream_chat_response_generator(
    session_id: str,
    message: str,
    user_id: str
):
    """스트리밍 응답 생성기 - ChatService 통합"""
    import json

    try:
        from core.services import get_chat_service, SearchMode

        chat_service = await get_chat_service()

        async for chunk in chat_service.chat_stream(
            query=message,
            search_mode=SearchMode.WEB_ENABLED
        ):
            # 인용 메타데이터 처리
            if chunk.startswith("[CITATIONS_START]"):
                citations_json = chunk.replace("[CITATIONS_START]", "").replace("[CITATIONS_END]", "")
                yield json.dumps({
                    "type": "citations",
                    "data": json.loads(citations_json)
                })
            else:
                yield json.dumps({
                    "type": "text",
                    "chunk": chunk
                })

    except Exception as e:
        logger.error(f"Streaming response failed: {e}")
        yield json.dumps({
            "type": "error",
            "message": str(e)
        })


async def delete_chat_message(
    session_id: str,
    message_id: str,
    user_id: str
) -> bool:
    """채팅 메시지 삭제"""
    # 실제 데이터베이스 삭제 로직 구현 필요
    await asyncio.sleep(0.1)  # 시뮬레이션
    return True