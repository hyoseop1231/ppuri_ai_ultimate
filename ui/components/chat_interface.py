"""
Chat Interface - 실시간 스트리밍 대화 인터페이스

사용자와 AI 간의 자연스럽고 실시간적인 대화를 지원하는 
차세대 채팅 인터페이스 컴포넌트.

Features:
- 실시간 스트리밍 응답
- THINK 블록 통합 표시
- 한국어 최적화 입력/출력
- 첨부 파일 지원
- 메시지 히스토리 관리
- 자동 완성 및 제안
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """채팅 메시지"""
    id: str
    content: str
    role: str  # user, assistant, system
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    think_blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatSession:
    """채팅 세션"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_message: Optional[datetime] = None
    message_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputSuggestion:
    """입력 제안"""
    text: str
    type: str  # completion, template, command
    confidence: float
    category: Optional[str] = None


class ChatInterface:
    """
    실시간 스트리밍 채팅 인터페이스
    
    사용자와 AI 간의 자연스러운 대화를 지원하며,
    THINK 블록, 첨부파일, 실시간 스트리밍을 제공하는 컴포넌트.
    """
    
    def __init__(
        self,
        ui_orchestrator,
        korean_optimizer=None,
        max_message_history: int = 1000
    ):
        self.ui_orchestrator = ui_orchestrator
        self.korean_optimizer = korean_optimizer
        self.max_message_history = max_message_history
        
        # 메시지 관리
        self.message_history: Dict[str, List[ChatMessage]] = {}
        self.active_sessions: Dict[str, ChatSession] = {}
        
        # 입력 지원
        self.input_suggestions: Dict[str, List[InputSuggestion]] = {}
        self.command_templates = self._initialize_command_templates()
        
        # 설정
        self.config = {
            "stream_chunk_size": 50,
            "typing_indicator_delay": 1.0,
            "auto_save_interval": 30,
            "max_attachment_size": 10 * 1024 * 1024,  # 10MB
            "supported_file_types": [".txt", ".pdf", ".doc", ".docx", ".md"]
        }
        
        # 실시간 상태
        self.typing_users: Dict[str, datetime] = {}
        self.active_streams: Dict[str, bool] = {}
        
        logger.info("Chat Interface 초기화 완료")
    
    def _initialize_command_templates(self) -> Dict[str, Dict[str, Any]]:
        """명령어 템플릿 초기화"""
        
        return {
            # 뿌리산업 특화 명령어
            "주조_분석": {
                "template": "{재료}를 사용한 주조 공정을 분석해주세요. 온도는 {온도}도이고 냉각 조건은 {냉각조건}입니다.",
                "parameters": ["재료", "온도", "냉각조건"],
                "category": "주조",
                "description": "주조 공정 분석 요청"
            },
            "금형_설계": {
                "template": "{부품명}을 위한 금형 설계를 도와주세요. 재료는 {재료}이고 생산량은 {생산량}개입니다.",
                "parameters": ["부품명", "재료", "생산량"],
                "category": "금형",
                "description": "금형 설계 지원 요청"
            },
            "용접_품질": {
                "template": "{재료1}과 {재료2}의 {용접방법} 용접 품질을 평가해주세요.",
                "parameters": ["재료1", "재료2", "용접방법"],
                "category": "용접",
                "description": "용접 품질 평가 요청"
            },
            
            # 일반 명령어
            "문서_검색": {
                "template": "{검색어}에 대한 문서를 검색해주세요.",
                "parameters": ["검색어"],
                "category": "검색",
                "description": "문서 검색 요청"
            },
            "지식_그래프": {
                "template": "{개념}과 관련된 지식 그래프를 보여주세요.",
                "parameters": ["개념"],
                "category": "시각화",
                "description": "지식 그래프 시각화 요청"
            },
            "성능_분석": {
                "template": "현재 시스템 성능을 분석해주세요.",
                "parameters": [],
                "category": "시스템",
                "description": "시스템 성능 분석"
            }
        }
    
    async def create_chat_session(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """새 채팅 세션 생성"""
        
        session_id = str(uuid.uuid4())
        
        # UI 오케스트레이터에 세션 생성
        await self.ui_orchestrator.create_session(user_id, {
            "component": "chat_interface",
            **(initial_context or {})
        })
        
        # 채팅 세션 생성
        chat_session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            context=initial_context or {}
        )
        
        self.active_sessions[session_id] = chat_session
        self.message_history[session_id] = []
        
        # 환영 메시지 추가
        welcome_message = ChatMessage(
            id=str(uuid.uuid4()),
            content="안녕하세요! PPuRI-AI Ultimate입니다. 뿌리산업 관련 질문이나 작업을 도와드리겠습니다.",
            role="assistant",
            timestamp=datetime.now(),
            session_id=session_id,
            metadata={"type": "welcome", "auto_generated": True}
        )
        
        self.message_history[session_id].append(welcome_message)
        
        logger.info(f"채팅 세션 생성: {session_id}")
        return session_id
    
    async def send_message(
        self,
        session_id: str,
        content: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        메시지 전송 및 실시간 응답 스트리밍
        
        Args:
            session_id: 채팅 세션 ID
            content: 메시지 내용
            attachments: 첨부 파일들
            
        Yields:
            Dict[str, Any]: 실시간 응답 청크
        """
        
        if session_id not in self.active_sessions:
            yield {"error": "유효하지 않은 세션입니다."}
            return
        
        session = self.active_sessions[session_id]
        
        try:
            # 1. 사용자 메시지 처리
            user_message = await self._process_user_message(
                session_id, content, attachments
            )
            
            yield {
                "type": "user_message_received",
                "data": {
                    "message_id": user_message.id,
                    "content": content,
                    "timestamp": user_message.timestamp.isoformat()
                }
            }
            
            # 2. 입력 제안 업데이트 (백그라운드)
            asyncio.create_task(self._update_input_suggestions(session_id, content))
            
            # 3. 타이핑 인디케이터 시작
            yield {
                "type": "typing_start",
                "data": {"timestamp": datetime.now().isoformat()}
            }
            
            self.active_streams[session_id] = True
            
            # 4. UI 오케스트레이터를 통한 응답 스트리밍
            assistant_message_parts = []
            current_think_blocks = []
            
            async for response_chunk in self.ui_orchestrator.process_user_message(
                session_id, content, attachments
            ):
                
                if not self.active_streams.get(session_id, False):
                    break  # 스트림이 중단된 경우
                
                # 응답 타입별 처리
                if response_chunk["type"] == "think_update":
                    current_think_blocks.append(response_chunk["data"])
                    yield response_chunk
                
                elif response_chunk["type"] == "response_chunk":
                    assistant_message_parts.append(response_chunk["data"]["content"])
                    yield response_chunk
                
                elif response_chunk["type"] == "response_complete":
                    # 최종 어시스턴트 메시지 저장
                    final_content = "".join(assistant_message_parts)
                    
                    assistant_message = ChatMessage(
                        id=str(uuid.uuid4()),
                        content=final_content,
                        role="assistant",
                        timestamp=datetime.now(),
                        session_id=session_id,
                        think_blocks=current_think_blocks,
                        metadata={
                            "processing_time": response_chunk["data"]["processing_time"],
                            "response_length": len(final_content)
                        }
                    )
                    
                    await self._save_message(assistant_message)
                    
                    # 세션 통계 업데이트
                    session.last_message = datetime.now()
                    session.message_count += 2  # 사용자 + 어시스턴트
                    
                    yield response_chunk
                
                else:
                    yield response_chunk
            
            # 5. 타이핑 인디케이터 종료
            yield {
                "type": "typing_end",
                "data": {"timestamp": datetime.now().isoformat()}
            }
            
        except Exception as e:
            logger.error(f"메시지 처리 실패 ({session_id}): {e}")
            yield {
                "type": "error",
                "data": {"message": f"메시지 처리 중 오류가 발생했습니다: {str(e)}"}
            }
        finally:
            self.active_streams[session_id] = False
    
    async def _process_user_message(
        self,
        session_id: str,
        content: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> ChatMessage:
        """사용자 메시지 처리"""
        
        # 한국어 최적화
        processed_content = content
        if self.korean_optimizer:
            korean_result = await self.korean_optimizer.process_korean_text(content)
            processed_content = korean_result.normalized_text
        
        # 첨부 파일 처리
        processed_attachments = []
        if attachments:
            processed_attachments = await self._process_attachments(attachments)
        
        # 메시지 생성
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            content=processed_content,
            role="user",
            timestamp=datetime.now(),
            session_id=session_id,
            attachments=processed_attachments,
            metadata={
                "original_content": content if content != processed_content else None,
                "attachment_count": len(processed_attachments)
            }
        )
        
        # 저장
        await self._save_message(user_message)
        
        return user_message
    
    async def _process_attachments(
        self,
        attachments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """첨부 파일 처리"""
        
        processed = []
        
        for attachment in attachments:
            try:
                # 파일 크기 체크
                if attachment.get("size", 0) > self.config["max_attachment_size"]:
                    continue
                
                # 파일 타입 체크
                file_extension = attachment.get("name", "").lower()
                if not any(file_extension.endswith(ext) for ext in self.config["supported_file_types"]):
                    continue
                
                # 파일 내용 추출 (텍스트 파일인 경우)
                content_preview = ""
                if file_extension.endswith((".txt", ".md")):
                    content_preview = attachment.get("content", "")[:500]  # 처음 500자
                
                processed.append({
                    "id": str(uuid.uuid4()),
                    "name": attachment["name"],
                    "type": attachment.get("type", "unknown"),
                    "size": attachment.get("size", 0),
                    "content_preview": content_preview,
                    "processed_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"첨부 파일 처리 실패: {e}")
        
        return processed
    
    async def _save_message(self, message: ChatMessage):
        """메시지 저장"""
        
        if message.session_id not in self.message_history:
            self.message_history[message.session_id] = []
        
        self.message_history[message.session_id].append(message)
        
        # 메시지 히스토리 크기 제한
        if len(self.message_history[message.session_id]) > self.max_message_history:
            self.message_history[message.session_id] = \
                self.message_history[message.session_id][-self.max_message_history:]
    
    async def get_message_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        before_message_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """메시지 히스토리 조회"""
        
        if session_id not in self.message_history:
            return []
        
        messages = self.message_history[session_id]
        
        # before_message_id 이전 메시지들만 (페이지네이션)
        if before_message_id:
            before_index = None
            for i, msg in enumerate(messages):
                if msg.id == before_message_id:
                    before_index = i
                    break
            
            if before_index is not None:
                messages = messages[:before_index]
        
        # 제한 개수 적용
        if limit:
            messages = messages[-limit:]
        
        # 딕셔너리로 변환
        return [
            {
                "id": msg.id,
                "content": msg.content,
                "role": msg.role,
                "timestamp": msg.timestamp.isoformat(),
                "think_blocks": msg.think_blocks,
                "attachments": msg.attachments,
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    async def get_input_suggestions(
        self,
        session_id: str,
        partial_input: str = "",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """입력 제안 조회"""
        
        suggestions = []
        
        # 1. 명령어 템플릿 기반 제안
        for cmd_name, template_info in self.command_templates.items():
            if partial_input.lower() in template_info["description"].lower():
                suggestions.append({
                    "type": "template",
                    "text": template_info["template"],
                    "description": template_info["description"],
                    "category": template_info["category"],
                    "confidence": 0.9
                })
        
        # 2. 세션별 맞춤 제안
        if session_id in self.input_suggestions:
            session_suggestions = self.input_suggestions[session_id]
            
            for suggestion in session_suggestions:
                if partial_input.lower() in suggestion.text.lower():
                    suggestions.append({
                        "type": suggestion.type,
                        "text": suggestion.text,
                        "category": suggestion.category,
                        "confidence": suggestion.confidence
                    })
        
        # 3. 한국어 자동 완성 (Korean Optimizer 활용)
        if self.korean_optimizer and partial_input:
            korean_suggestions = await self._get_korean_completions(partial_input)
            suggestions.extend(korean_suggestions)
        
        # 신뢰도 순 정렬 및 제한
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:limit]
    
    async def _get_korean_completions(self, partial_input: str) -> List[Dict[str, Any]]:
        """한국어 자동 완성 제안"""
        
        try:
            # 뿌리산업 용어 기반 완성
            completions = []
            
            if hasattr(self.korean_optimizer, 'industry_terms'):
                for term_key, term_info in self.korean_optimizer.industry_terms.items():
                    if term_key.startswith(partial_input) and len(partial_input) >= 2:
                        completions.append({
                            "type": "completion",
                            "text": term_key,
                            "category": term_info.category,
                            "confidence": 0.8
                        })
            
            return completions[:3]  # 최대 3개
            
        except Exception as e:
            logger.error(f"한국어 자동 완성 실패: {e}")
            return []
    
    async def _update_input_suggestions(self, session_id: str, user_input: str):
        """입력 제안 업데이트 (백그라운드)"""
        
        try:
            if session_id not in self.input_suggestions:
                self.input_suggestions[session_id] = []
            
            # 자주 사용되는 패턴 학습
            words = user_input.split()
            if len(words) >= 3:
                # 3단어 이상 구문을 제안으로 추가
                suggestion = InputSuggestion(
                    text=user_input[:50],  # 처음 50자만
                    type="learned",
                    confidence=0.6,
                    category="사용자_패턴"
                )
                
                self.input_suggestions[session_id].append(suggestion)
                
                # 제한 개수 유지
                if len(self.input_suggestions[session_id]) > 50:
                    self.input_suggestions[session_id] = \
                        self.input_suggestions[session_id][-50:]
        
        except Exception as e:
            logger.error(f"입력 제안 업데이트 실패: {e}")
    
    def stop_message_stream(self, session_id: str):
        """메시지 스트림 중단"""
        
        self.active_streams[session_id] = False
        logger.debug(f"메시지 스트림 중단: {session_id}")
    
    def set_typing_status(self, session_id: str, is_typing: bool):
        """타이핑 상태 설정"""
        
        if is_typing:
            self.typing_users[session_id] = datetime.now()
        else:
            self.typing_users.pop(session_id, None)
    
    def get_typing_users(self, session_id: str) -> List[str]:
        """타이핑 중인 사용자 목록"""
        
        current_time = datetime.now()
        active_typing = []
        
        for user_session_id, last_typing in list(self.typing_users.items()):
            if (current_time - last_typing).total_seconds() < 5:  # 5초 내
                if user_session_id != session_id:  # 자신 제외
                    active_typing.append(user_session_id)
            else:
                del self.typing_users[user_session_id]
        
        return active_typing
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """세션 통계 조회"""
        
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        messages = self.message_history.get(session_id, [])
        
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_message": session.last_message.isoformat() if session.last_message else None,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "avg_message_length": sum(len(m.content) for m in messages) / len(messages) if messages else 0,
            "think_blocks_used": sum(len(m.think_blocks) for m in assistant_messages),
            "attachments_sent": sum(len(m.attachments) for m in user_messages)
        }
    
    async def export_conversation(
        self,
        session_id: str,
        format: str = "json"
    ) -> Optional[str]:
        """대화 내용 내보내기"""
        
        if session_id not in self.message_history:
            return None
        
        messages = self.message_history[session_id]
        session = self.active_sessions.get(session_id)
        
        if format == "json":
            export_data = {
                "session_info": {
                    "session_id": session_id,
                    "created_at": session.created_at.isoformat() if session else None,
                    "message_count": len(messages)
                },
                "messages": [
                    {
                        "id": msg.id,
                        "content": msg.content,
                        "role": msg.role,
                        "timestamp": msg.timestamp.isoformat(),
                        "think_blocks": msg.think_blocks,
                        "metadata": msg.metadata
                    }
                    for msg in messages
                ]
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        
        elif format == "markdown":
            lines = [f"# 대화 기록 - {session_id}\n"]
            
            if session:
                lines.append(f"**생성일**: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            lines.append(f"**총 메시지 수**: {len(messages)}\n\n---\n")
            
            for msg in messages:
                role_name = {"user": "사용자", "assistant": "AI", "system": "시스템"}.get(msg.role, msg.role)
                lines.append(f"## {role_name} ({msg.timestamp.strftime('%H:%M:%S')})\n")
                lines.append(f"{msg.content}\n")
                
                if msg.think_blocks:
                    lines.append("### 사고 과정\n")
                    for think in msg.think_blocks:
                        lines.append(f"- **{think.get('level', 'THINK')}**: {think.get('content', '')}\n")
                
                lines.append("\n---\n")
            
            return "".join(lines)
        
        return None
    
    async def cleanup_session(self, session_id: str):
        """세션 정리"""
        
        try:
            # 스트림 중단
            self.stop_message_stream(session_id)
            
            # 세션 데이터 정리
            self.active_sessions.pop(session_id, None)
            self.input_suggestions.pop(session_id, None)
            self.typing_users.pop(session_id, None)
            
            # 메시지 히스토리는 유지 (최근 100개만)
            if session_id in self.message_history:
                recent_messages = self.message_history[session_id][-100:]
                self.message_history[session_id] = recent_messages
            
            logger.info(f"채팅 세션 정리 완료: {session_id}")
            
        except Exception as e:
            logger.error(f"세션 정리 실패 ({session_id}): {e}")
    
    async def cleanup(self):
        """Chat Interface 정리"""
        
        # 모든 세션 정리
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)
        
        logger.info("Chat Interface 정리 완료")