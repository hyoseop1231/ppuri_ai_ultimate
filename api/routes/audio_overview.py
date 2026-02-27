"""
Audio Overview API Routes - PPuRI-AI Ultimate
NotebookLM 스타일 팟캐스트 생성 API
"""

import logging
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..middleware.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio-overview"])


# Request/Response Models
class AudioGenerationRequest(BaseModel):
    """Audio Overview 생성 요청"""
    title: str = Field(..., min_length=1, max_length=200)
    document_ids: List[str] = Field(..., min_items=1, max_items=10)
    style: str = Field(default="conversational", pattern="^(conversational|educational|technical)$")
    duration_minutes: int = Field(default=10, ge=3, le=30)
    tts_provider: str = Field(default="edge_tts", pattern="^(edge_tts|melo_tts|openai_tts)$")
    industry: Optional[str] = None


class AudioOverviewResponse(BaseModel):
    """Audio Overview 응답"""
    id: str
    title: str
    status: str
    audio_url: Optional[str] = None
    transcript_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None


class AudioStatusResponse(BaseModel):
    """Audio Overview 상태 응답"""
    id: str
    status: str  # pending, generating, completed, failed
    progress: float  # 0.0 ~ 1.0
    message: Optional[str] = None


# 진행 중인 생성 작업 추적
_generation_tasks = {}


@router.post("/generate", response_model=AudioOverviewResponse)
async def generate_audio_overview(
    request: AudioGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Audio Overview 생성 시작

    문서들을 분석하여 팟캐스트 스타일의 오디오 요약을 생성합니다.
    생성은 백그라운드에서 진행되며, status API로 진행 상황을 확인할 수 있습니다.
    """
    try:
        # 작업 ID 생성
        task_id = str(uuid.uuid4())

        # 작업 상태 초기화
        _generation_tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "생성 대기 중..."
        }

        # 백그라운드에서 생성 시작
        background_tasks.add_task(
            _generate_audio_background,
            task_id=task_id,
            title=request.title,
            document_ids=request.document_ids,
            style=request.style,
            duration_minutes=request.duration_minutes,
            tts_provider=request.tts_provider,
            industry=request.industry,
            user_id=current_user_id
        )

        return AudioOverviewResponse(
            id=task_id,
            title=request.title,
            status="pending",
            created_at=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Audio generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}/status", response_model=AudioStatusResponse)
async def get_generation_status(
    task_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """Audio Overview 생성 상태 확인"""
    if task_id not in _generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _generation_tasks[task_id]
    return AudioStatusResponse(
        id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task.get("message")
    )


@router.get("/{task_id}", response_model=AudioOverviewResponse)
async def get_audio_overview(
    task_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """Audio Overview 정보 조회"""
    if task_id not in _generation_tasks:
        raise HTTPException(status_code=404, detail="Audio overview not found")

    task = _generation_tasks[task_id]

    return AudioOverviewResponse(
        id=task_id,
        title=task.get("title", ""),
        status=task["status"],
        audio_url=f"/api/audio/{task_id}/download" if task["status"] == "completed" else None,
        transcript_url=f"/api/audio/{task_id}/transcript" if task["status"] == "completed" else None,
        duration_seconds=task.get("duration_seconds"),
        created_at=task.get("created_at", datetime.utcnow().isoformat()),
        completed_at=task.get("completed_at")
    )


@router.get("/{task_id}/download")
async def download_audio(
    task_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """Audio 파일 다운로드"""
    if task_id not in _generation_tasks:
        raise HTTPException(status_code=404, detail="Audio not found")

    task = _generation_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Audio generation not completed")

    audio_path = task.get("audio_path")
    if not audio_path:
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=f"{task.get('title', 'audio')}.mp3"
    )


@router.get("/{task_id}/transcript")
async def get_transcript(
    task_id: str,
    format: str = Query(default="text", pattern="^(text|srt|vtt)$"),
    current_user_id: str = Depends(get_current_user_id)
):
    """대본/자막 조회"""
    if task_id not in _generation_tasks:
        raise HTTPException(status_code=404, detail="Audio not found")

    task = _generation_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Audio generation not completed")

    transcript = task.get("transcript_text", "")

    if format == "srt":
        content = task.get("transcript_srt", transcript)
        media_type = "application/x-subrip"
        filename = f"{task.get('title', 'transcript')}.srt"
    elif format == "vtt":
        content = task.get("transcript_vtt", transcript)
        media_type = "text/vtt"
        filename = f"{task.get('title', 'transcript')}.vtt"
    else:
        content = transcript
        media_type = "text/plain"
        filename = f"{task.get('title', 'transcript')}.txt"

    return StreamingResponse(
        iter([content.encode()]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/list", response_model=List[AudioOverviewResponse])
async def list_audio_overviews(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user_id: str = Depends(get_current_user_id)
):
    """사용자의 Audio Overview 목록 조회"""
    # TODO: 데이터베이스에서 조회
    overviews = []
    for task_id, task in list(_generation_tasks.items())[offset:offset + limit]:
        overviews.append(AudioOverviewResponse(
            id=task_id,
            title=task.get("title", ""),
            status=task["status"],
            audio_url=f"/api/audio/{task_id}/download" if task["status"] == "completed" else None,
            duration_seconds=task.get("duration_seconds"),
            created_at=task.get("created_at", datetime.utcnow().isoformat())
        ))
    return overviews


@router.delete("/{task_id}")
async def delete_audio_overview(
    task_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """Audio Overview 삭제"""
    if task_id not in _generation_tasks:
        raise HTTPException(status_code=404, detail="Audio not found")

    # 파일 삭제
    task = _generation_tasks[task_id]
    audio_path = task.get("audio_path")
    if audio_path:
        import os
        try:
            os.remove(audio_path)
        except:
            pass

    del _generation_tasks[task_id]

    return {"message": "Audio overview deleted successfully"}


# 백그라운드 생성 함수
async def _generate_audio_background(
    task_id: str,
    title: str,
    document_ids: List[str],
    style: str,
    duration_minutes: int,
    tts_provider: str,
    industry: Optional[str],
    user_id: str
):
    """백그라운드에서 Audio Overview 생성"""
    try:
        _generation_tasks[task_id]["status"] = "generating"
        _generation_tasks[task_id]["progress"] = 0.1
        _generation_tasks[task_id]["message"] = "문서 로딩 중..."
        _generation_tasks[task_id]["title"] = title
        _generation_tasks[task_id]["created_at"] = datetime.utcnow().isoformat()

        # 의존성 로드
        from core.llm import get_openrouter_client
        from core.audio import AudioOverviewEngine, TTSProvider

        llm = await get_openrouter_client()

        # TTS 제공자 선택
        tts_enum = {
            "edge_tts": TTSProvider.EDGE_TTS,
            "melo_tts": TTSProvider.MELO_TTS,
            "openai_tts": TTSProvider.OPENAI_TTS
        }.get(tts_provider, TTSProvider.EDGE_TTS)

        # 엔진 초기화
        engine = AudioOverviewEngine(
            llm_client=llm,
            tts_provider=tts_enum
        )
        await engine.initialize()

        _generation_tasks[task_id]["progress"] = 0.2
        _generation_tasks[task_id]["message"] = "문서 분석 중..."

        # 문서 로드 (TODO: 실제 문서 로드 구현)
        documents = []
        for doc_id in document_ids:
            # 임시 더미 데이터
            documents.append({
                "id": doc_id,
                "title": f"Document {doc_id}",
                "content": "문서 내용..."  # TODO: 실제 문서 내용 로드
            })

        _generation_tasks[task_id]["progress"] = 0.4
        _generation_tasks[task_id]["message"] = "대화 스크립트 생성 중..."

        # Audio Overview 생성
        overview = await engine.generate_overview(
            documents=documents,
            title=title,
            style=style,
            duration_target_minutes=duration_minutes
        )

        _generation_tasks[task_id]["progress"] = 0.9
        _generation_tasks[task_id]["message"] = "최종 처리 중..."

        # 자막 생성
        transcript_text = await engine.generate_transcript(overview, format="text")
        transcript_srt = await engine.generate_transcript(overview, format="srt")
        transcript_vtt = await engine.generate_transcript(overview, format="vtt")

        # 결과 저장
        _generation_tasks[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "생성 완료",
            "audio_path": overview.audio_path,
            "duration_seconds": overview.duration_seconds,
            "transcript_text": transcript_text,
            "transcript_srt": transcript_srt,
            "transcript_vtt": transcript_vtt,
            "completed_at": datetime.utcnow().isoformat()
        })

        logger.info(f"Audio overview generated: {task_id}")

    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        _generation_tasks[task_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"생성 실패: {str(e)}"
        })
