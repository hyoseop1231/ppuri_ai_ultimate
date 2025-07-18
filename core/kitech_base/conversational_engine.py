"""
Conversational Engine - 대화형 AI 엔진 

KITECH 검증 패턴과 AdalFlow 자동 최적화를 통합한
뿌리산업 전용 대화형 AI 챗봇 엔진.

Features:
- Ollama 기반 대화 생성
- AdalFlow 자동 프롬프트 최적화  
- THINK 블록 실시간 표시
- 한국어 특화 처리
- 대화 내역 DB 저장 및 재활용
- MCP 도구 통합
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import aiohttp

from ..adalflow_engine import AutoPromptOptimizer, PerformanceTracker
from .config_manager import KitechConfigManager
from .korean_optimizer import KoreanLanguageOptimizer
from .think_ui import ThinkBlockManager, ThinkLevel

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """대화 메시지"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    think_blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class ConversationSession:
    """대화 세션"""
    session_id: str
    user_id: Optional[str] = None
    messages: List[ConversationMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    industry_domain: Optional[str] = None
    optimization_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResult:
    """대화 결과"""
    response: str
    session_id: str
    processing_time: float
    optimization_applied: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    think_blocks: List[Dict[str, Any]] = field(default_factory=list)
    industry_terms_detected: List[str] = field(default_factory=list)


class ConversationalEngine:
    """
    PPuRI-AI Ultimate 대화형 엔진
    
    KITECH 검증된 안정성과 AdalFlow 혁신적 최적화를 결합한
    차세대 뿌리산업 전용 대화형 AI 시스템.
    """
    
    def __init__(
        self,
        config_manager: KitechConfigManager,
        korean_optimizer: KoreanLanguageOptimizer,
        think_manager: ThinkBlockManager,
        auto_optimizer: Optional[AutoPromptOptimizer] = None,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        self.config_manager = config_manager
        self.korean_optimizer = korean_optimizer
        self.think_manager = think_manager
        self.auto_optimizer = auto_optimizer
        self.performance_tracker = performance_tracker
        
        # Ollama 설정
        self.ollama_config = config_manager.get_ollama_config()
        self.ollama_session: Optional[aiohttp.ClientSession] = None
        
        # 대화 세션 관리
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.session_timeout = timedelta(hours=2)  # 2시간 후 세션 만료
        
        # 성능 최적화 설정
        self.enable_auto_optimization = True
        self.optimization_threshold = 3  # 3번 대화 후 최적화
        
        # 뿌리산업 시스템 프롬프트
        self.system_prompt = self._create_system_prompt()
        
        # 대화 통계
        self.conversation_stats = {
            "total_conversations": 0,
            "total_sessions": 0,
            "avg_response_time": 0.0,
            "optimization_success_rate": 0.0
        }
        
        logger.info("Conversational Engine 초기화 완료")
    
    async def initialize(self):
        """엔진 초기화"""
        
        # Ollama 연결 초기화
        self.ollama_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.ollama_config["timeout"])
        )
        
        # 연결 테스트
        await self._test_ollama_connection()
        
        logger.info("Conversational Engine 초기화 및 연결 테스트 완료")
    
    async def cleanup(self):
        """리소스 정리"""
        if self.ollama_session:
            await self.ollama_session.close()
        
        # 세션 정리
        self.active_sessions.clear()
        
        logger.info("Conversational Engine 정리 완료")
    
    def _create_system_prompt(self) -> str:
        """뿌리산업 특화 시스템 프롬프트 생성"""
        
        return """당신은 뿌리산업 전문 AI 어시스턴트 '뿌리아이(PPuRI-AI)'입니다.

**전문 분야**: 주조, 금형, 소성가공, 용접, 표면처리, 열처리
**역할**: 뿌리산업 기술 상담, 문제 해결, 지식 전달

**응답 원칙**:
1. 🎯 **정확성**: 기술적으로 정확하고 신뢰할 수 있는 정보 제공
2. 🇰🇷 **한국어 최적화**: 자연스럽고 이해하기 쉬운 한국어 사용
3. 🏭 **실무 중심**: 현장에서 바로 적용 가능한 구체적 솔루션
4. 📚 **교육적**: 원리와 배경을 함께 설명하여 학습 효과 극대화
5. 🔧 **문제 해결**: 단계적이고 체계적인 문제 해결 접근

**특별 지시**:
- 전문 용어는 한국어 우선, 필요시 영어/한자 병기
- 수치와 기준은 국제 표준 및 국내 기준 모두 제시
- 안전 관련 내용은 반드시 강조
- 불확실한 내용은 명확히 표시하고 추가 확인 권장

지금부터 뿌리산업 전문가로서 최고 품질의 기술 상담을 제공해주세요."""
    
    async def _test_ollama_connection(self) -> bool:
        """Ollama 연결 테스트"""
        
        try:
            health_url = self.ollama_config["api_url"].replace("/api/generate", "/api/tags")
            
            async with self.ollama_session.get(health_url) as response:
                if response.status == 200:
                    logger.info("✅ Ollama 연결 테스트 성공")
                    return True
                else:
                    logger.warning(f"⚠️ Ollama 응답 상태 이상: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Ollama 연결 테스트 실패: {e}")
            return False
    
    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """새로운 대화 세션 시작"""
        
        session_id = str(uuid.uuid4())
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            context=initial_context or {}
        )
        
        # 시스템 메시지 추가
        session.messages.append(ConversationMessage(
            role="system",
            content=self.system_prompt
        ))
        
        self.active_sessions[session_id] = session
        self.conversation_stats["total_sessions"] += 1
        
        # THINK 세션 시작
        await self.think_manager.start_think_session(session_id, initial_context)
        
        logger.info(f"새로운 대화 세션 시작: {session_id}")
        return session_id
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        stream: bool = True
    ) -> AsyncGenerator[ConversationResult, None]:
        """
        사용자와 대화 진행
        
        Args:
            session_id: 대화 세션 ID
            user_message: 사용자 메시지
            stream: 스트리밍 응답 여부
            
        Yields:
            ConversationResult: 대화 결과 (스트리밍 시 부분 결과)
        """
        
        start_time = time.time()
        
        try:
            # 1. 세션 유효성 확인
            if session_id not in self.active_sessions:
                yield ConversationResult(
                    response="❌ 유효하지 않은 세션입니다. 새로운 대화를 시작해주세요.",
                    session_id=session_id,
                    processing_time=0.0
                )
                return
            
            session = self.active_sessions[session_id]
            session.last_activity = datetime.now()
            
            # 2. 사용자 메시지 추가
            user_msg = ConversationMessage(
                role="user",
                content=user_message
            )
            session.messages.append(user_msg)
            
            # 3. 한국어 처리 및 분석
            korean_result = await self.korean_optimizer.process_korean_text(user_message)
            
            # 뿌리산업 도메인 감지
            if korean_result.industry_terms:
                # 가장 많이 언급된 카테고리 추정
                categories = [
                    self.korean_optimizer.industry_terms[term].category 
                    for term in korean_result.industry_terms 
                    if term in self.korean_optimizer.industry_terms
                ]
                if categories:
                    from collections import Counter
                    session.industry_domain = Counter(categories).most_common(1)[0][0]
            
            # 4. 프롬프트 최적화 (필요시)
            optimized_prompt = await self._get_optimized_prompt(session, user_message)
            optimization_applied = optimized_prompt != user_message
            
            # 5. THINK 블록 생성 시작
            think_blocks = []
            if self.think_manager.enabled:
                async for think_block in self.think_manager.generate_progressive_think(
                    session_id, user_message, session.industry_domain
                ):
                    think_blocks.append({
                        "level": think_block.level.value,
                        "content": think_block.content,
                        "timestamp": think_block.timestamp.isoformat()
                    })
                    
                    if stream:
                        # THINK 블록을 실시간으로 전송
                        yield ConversationResult(
                            response=self.think_manager.format_think_block_for_display(think_block),
                            session_id=session_id,
                            processing_time=time.time() - start_time,
                            think_blocks=[think_blocks[-1]]
                        )
            
            # 6. Ollama로 응답 생성
            if stream:
                async for partial_result in self._generate_streaming_response(
                    session, optimized_prompt, start_time, optimization_applied, 
                    korean_result.industry_terms, think_blocks
                ):
                    yield partial_result
            else:
                final_result = await self._generate_single_response(
                    session, optimized_prompt, start_time, optimization_applied,
                    korean_result.industry_terms, think_blocks
                )
                yield final_result
                
        except Exception as e:
            logger.error(f"대화 처리 실패: {e}")
            yield ConversationResult(
                response=f"❌ 대화 처리 중 오류가 발생했습니다: {str(e)}",
                session_id=session_id,
                processing_time=time.time() - start_time
            )
    
    async def _get_optimized_prompt(
        self, 
        session: ConversationSession, 
        user_message: str
    ) -> str:
        """프롬프트 최적화"""
        
        if not self.auto_optimizer or not self.enable_auto_optimization:
            return user_message
        
        # 일정 대화 수 이후부터 최적화 적용
        if len(session.messages) < self.optimization_threshold:
            return user_message
        
        try:
            # 최적화 컨텍스트 구성
            context = {
                "user_profile": session.context.get("user_profile", {}),
                "conversation_history": [
                    {"role": msg.role, "content": msg.content[:200]}  # 처음 200자만
                    for msg in session.messages[-5:]  # 최근 5개 메시지만
                ],
                "industry_domain": session.industry_domain,
                "success_patterns": session.optimization_state.get("success_patterns", [])
            }
            
            # AdalFlow 최적화 실행
            optimization_result = await self.auto_optimizer.optimize_prompt(
                prompt=user_message,
                context=context,
                target_domain=session.industry_domain or "뿌리산업"
            )
            
            # 최적화 결과 저장
            session.optimization_state["last_optimization"] = {
                "original": user_message,
                "optimized": optimization_result.optimized_prompt,
                "performance_gain": optimization_result.performance_gain,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"프롬프트 최적화 적용: +{optimization_result.performance_gain:.1f}%")
            return optimization_result.optimized_prompt
            
        except Exception as e:
            logger.error(f"프롬프트 최적화 실패: {e}")
            return user_message
    
    async def _generate_streaming_response(
        self,
        session: ConversationSession,
        prompt: str,
        start_time: float,
        optimization_applied: bool,
        industry_terms: List[str],
        think_blocks: List[Dict[str, Any]]
    ) -> AsyncGenerator[ConversationResult, None]:
        """스트리밍 응답 생성"""
        
        try:
            # 대화 히스토리 구성
            messages = self._build_conversation_history(session, prompt)
            
            # Ollama 스트리밍 요청
            request_data = {
                "model": self.ollama_config["model"],
                "messages": messages,
                "stream": True,
                "options": self.ollama_config["options"]
            }
            
            full_response = ""
            
            async with self.ollama_session.post(
                self.ollama_config["api_url"].replace("/api/generate", "/api/chat"),
                json=request_data
            ) as response:
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode().strip())
                            
                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                full_response += chunk
                                
                                yield ConversationResult(
                                    response=chunk,
                                    session_id=session.session_id,
                                    processing_time=time.time() - start_time,
                                    optimization_applied=optimization_applied,
                                    industry_terms_detected=industry_terms,
                                    think_blocks=think_blocks
                                )
                            
                            # 스트림 완료 확인
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            # 응답 완료 후 처리
            await self._post_process_response(
                session, full_response, start_time, optimization_applied, industry_terms
            )
            
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 실패: {e}")
            yield ConversationResult(
                response=f"❌ 응답 생성 실패: {str(e)}",
                session_id=session.session_id,
                processing_time=time.time() - start_time
            )
    
    async def _generate_single_response(
        self,
        session: ConversationSession,
        prompt: str,
        start_time: float,
        optimization_applied: bool,
        industry_terms: List[str],
        think_blocks: List[Dict[str, Any]]
    ) -> ConversationResult:
        """단일 응답 생성"""
        
        try:
            # 대화 히스토리 구성
            messages = self._build_conversation_history(session, prompt)
            
            # Ollama 요청
            request_data = {
                "model": self.ollama_config["model"],
                "messages": messages,
                "stream": False,
                "options": self.ollama_config["options"]
            }
            
            async with self.ollama_session.post(
                self.ollama_config["api_url"].replace("/api/generate", "/api/chat"),
                json=request_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    ai_response = result["message"]["content"]
                    
                    # 응답 후처리
                    await self._post_process_response(
                        session, ai_response, start_time, optimization_applied, industry_terms
                    )
                    
                    return ConversationResult(
                        response=ai_response,
                        session_id=session.session_id,
                        processing_time=time.time() - start_time,
                        optimization_applied=optimization_applied,
                        industry_terms_detected=industry_terms,
                        think_blocks=think_blocks
                    )
                else:
                    raise Exception(f"Ollama API 오류: {response.status}")
            
        except Exception as e:
            logger.error(f"단일 응답 생성 실패: {e}")
            return ConversationResult(
                response=f"❌ 응답 생성 실패: {str(e)}",
                session_id=session.session_id,
                processing_time=time.time() - start_time
            )
    
    def _build_conversation_history(
        self, 
        session: ConversationSession, 
        current_prompt: str
    ) -> List[Dict[str, str]]:
        """대화 히스토리 구성"""
        
        messages = []
        
        # 시스템 메시지
        if session.messages and session.messages[0].role == "system":
            messages.append({
                "role": "system",
                "content": session.messages[0].content
            })
        
        # 최근 대화 (메모리 절약을 위해 제한)
        recent_messages = session.messages[-10:]  # 최근 10개 메시지
        
        for msg in recent_messages:
            if msg.role != "system":  # 시스템 메시지는 이미 추가됨
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # 현재 사용자 메시지 (아직 세션에 추가되지 않은 경우)
        if not messages or messages[-1]["content"] != current_prompt:
            messages.append({
                "role": "user", 
                "content": current_prompt
            })
        
        return messages
    
    async def _post_process_response(
        self,
        session: ConversationSession,
        ai_response: str,
        start_time: float,
        optimization_applied: bool,
        industry_terms: List[str]
    ):
        """응답 후처리"""
        
        # AI 응답을 세션에 추가
        ai_msg = ConversationMessage(
            role="assistant",
            content=ai_response,
            metadata={
                "optimization_applied": optimization_applied,
                "industry_terms": industry_terms,
                "processing_time": time.time() - start_time
            }
        )
        session.messages.append(ai_msg)
        
        # 성능 추적 (가능한 경우)
        if self.performance_tracker:
            await self.performance_tracker.track_interaction(
                session_id=session.session_id,
                prompt=session.messages[-2].content if len(session.messages) >= 2 else "",
                response=ai_response,
                user_id=session.user_id,
                context={
                    "optimization_applied": optimization_applied,
                    "industry_terms": industry_terms,
                    "response_time": time.time() - start_time
                }
            )
        
        # 통계 업데이트
        self._update_conversation_stats(time.time() - start_time, optimization_applied)
        
        # 세션 자동 정리 (너무 길어진 경우)
        if len(session.messages) > 50:  # 메시지가 50개 이상이면
            # 시스템 메시지 + 최근 30개 메시지만 유지
            system_msg = session.messages[0] if session.messages[0].role == "system" else None
            recent_msgs = session.messages[-30:]
            
            session.messages = ([system_msg] if system_msg else []) + recent_msgs
            logger.info(f"세션 {session.session_id} 메시지 정리 완료")
    
    def _update_conversation_stats(self, processing_time: float, optimization_applied: bool):
        """대화 통계 업데이트"""
        
        self.conversation_stats["total_conversations"] += 1
        
        # 평균 응답 시간 업데이트
        total = self.conversation_stats["total_conversations"]
        current_avg = self.conversation_stats["avg_response_time"]
        
        self.conversation_stats["avg_response_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # 최적화 성공률 업데이트
        if optimization_applied:
            current_rate = self.conversation_stats["optimization_success_rate"]
            self.conversation_stats["optimization_success_rate"] = (
                (current_rate * (total - 1) + 1.0) / total
            )
        else:
            current_rate = self.conversation_stats["optimization_success_rate"]
            self.conversation_stats["optimization_success_rate"] = (
                current_rate * (total - 1) / total
            )
    
    async def end_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """대화 세션 종료"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # THINK 세션 종료
        think_stats = self.think_manager.end_think_session(session_id)
        
        # 세션 통계 생성
        session_stats = {
            "session_id": session_id,
            "user_id": session.user_id,
            "duration": (datetime.now() - session.start_time).total_seconds(),
            "total_messages": len(session.messages),
            "industry_domain": session.industry_domain,
            "optimization_applied_count": len([
                msg for msg in session.messages 
                if msg.metadata.get("optimization_applied", False)
            ]),
            "think_stats": think_stats,
            "end_time": datetime.now().isoformat()
        }
        
        # 세션 삭제
        del self.active_sessions[session_id]
        
        logger.info(f"대화 세션 종료: {session_id} - {session_stats['total_messages']}개 메시지")
        return session_stats
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "start_time": session.start_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.messages),
            "industry_domain": session.industry_domain,
            "context": session.context,
            "optimization_state": session.optimization_state
        }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """대화 통계 조회"""
        
        active_sessions_count = len(self.active_sessions)
        
        return {
            **self.conversation_stats,
            "active_sessions": active_sessions_count,
            "engine_config": {
                "auto_optimization_enabled": self.enable_auto_optimization,
                "optimization_threshold": self.optimization_threshold,
                "session_timeout_hours": self.session_timeout.total_seconds() / 3600
            },
            "ollama_config": {
                "model": self.ollama_config["model"],
                "api_url": self.ollama_config["api_url"]
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.end_conversation(session_id)
            
        if expired_sessions:
            logger.info(f"만료된 세션 {len(expired_sessions)}개 정리 완료")
    
    async def export_conversation_history(
        self, 
        session_id: str, 
        format: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """대화 히스토리 내보내기"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        if format == "json":
            return {
                "session_info": {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "start_time": session.start_time.isoformat(),
                    "industry_domain": session.industry_domain
                },
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata,
                        "think_blocks": msg.think_blocks
                    }
                    for msg in session.messages
                ],
                "context": session.context,
                "optimization_state": session.optimization_state,
                "export_timestamp": datetime.now().isoformat()
            }
        
        return None