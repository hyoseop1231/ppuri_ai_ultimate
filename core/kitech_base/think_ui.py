"""
THINK Block UI Manager - 구조화된 사고 표시 시스템

KITECH RAG 챗봇에서 검증된 THINK 블록 UI 패턴을 구현하여
AI의 사고 과정을 단계별로 시각화하는 사용자 인터페이스 시스템.

Features:
- 단계별 사고 과정 표시
- 실시간 사고 진행 상황
- 한국어 최적화된 UI
- 뿌리산업 전문 용어 강조
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ThinkLevel(Enum):
    """사고 단계 레벨"""
    THINK = "think"           # 🧠 기본 분석
    MEGATHINK = "megathink"   # 🚀 복합 관계 분석  
    ULTRATHINK = "ultrathink" # ⚡ 최종 통합 결론


@dataclass
class ThinkBlock:
    """THINK 블록 데이터 구조"""
    level: ThinkLevel
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    confidence: float = 1.0
    industry_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkSession:
    """THINK 세션"""
    session_id: str
    blocks: List[ThinkBlock] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    total_processing_time: float = 0.0
    user_context: Dict[str, Any] = field(default_factory=dict)


class ThinkBlockManager:
    """
    THINK 블록 UI 관리자
    
    AI의 사고 과정을 구조화하여 표시하고
    사용자에게 투명한 추론 과정을 제공.
    """
    
    def __init__(self, config_manager, korean_optimizer=None):
        self.config_manager = config_manager
        self.korean_optimizer = korean_optimizer
        
        # THINK 블록 설정
        self.think_config = config_manager.get_config()
        self.enabled = self.think_config.enable_think_blocks
        self.styles = self.think_config.think_block_styles
        
        # 활성 세션
        self.active_sessions: Dict[str, ThinkSession] = {}
        
        # 뿌리산업 특화 사고 템플릿
        self.industry_think_templates = self._create_industry_templates()
        
        # 성능 통계
        self.usage_stats = {
            "total_sessions": 0,
            "total_blocks": 0,
            "avg_processing_time": 0.0,
            "most_used_level": ThinkLevel.THINK.value
        }
        
        logger.info(f"THINK Block Manager 초기화 - 활성화: {self.enabled}")
    
    def _create_industry_templates(self) -> Dict[str, Dict[str, str]]:
        """뿌리산업 특화 사고 템플릿 생성"""
        
        return {
            "주조": {
                "think": "용탕의 특성과 주형 조건을 분석하여",
                "megathink": "응고 과정과 결함 발생 가능성을 종합적으로 검토하여", 
                "ultrathink": "최적의 주조 공정 조건을 결정하면"
            },
            "금형": {
                "think": "제품 형상과 재료 특성을 고려하여",
                "megathink": "금형 구조와 성형 조건의 상관관계를 분석하여",
                "ultrathink": "최적의 금형 설계 방안을 제시하면"
            },
            "소성가공": {
                "think": "재료의 소성 특성과 가공 조건을 검토하여",
                "megathink": "변형률과 가공력의 관계를 종합 분석하여",
                "ultrathink": "효율적인 소성가공 공정을 도출하면"
            },
            "용접": {
                "think": "모재와 용접재료의 특성을 파악하여",
                "megathink": "입열량과 용접부 품질의 연관성을 분석하여",
                "ultrathink": "최적의 용접 조건을 결정하면"
            },
            "표면처리": {
                "think": "기재 특성과 요구 성능을 검토하여",
                "megathink": "전처리와 후처리 공정의 영향을 분석하여",
                "ultrathink": "최적의 표면처리 방법을 선택하면"
            },
            "열처리": {
                "think": "강종과 요구 특성을 고려하여",
                "megathink": "온도-시간-조직 변화의 관계를 분석하여",
                "ultrathink": "적절한 열처리 조건을 설정하면"
            }
        }
    
    async def start_think_session(
        self, 
        session_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ThinkSession:
        """THINK 세션 시작"""
        
        if not self.enabled:
            logger.debug("THINK 블록이 비활성화됨")
            return None
        
        session = ThinkSession(
            session_id=session_id,
            user_context=user_context or {}
        )
        
        self.active_sessions[session_id] = session
        self.usage_stats["total_sessions"] += 1
        
        logger.debug(f"THINK 세션 시작: {session_id}")
        return session
    
    async def add_think_block(
        self,
        session_id: str,
        level: ThinkLevel,
        content: str,
        processing_time: Optional[float] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThinkBlock:
        """THINK 블록 추가"""
        
        if not self.enabled or session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # 한국어 최적화 적용
        if self.korean_optimizer:
            content = await self._optimize_think_content(content, level)
        
        # 뿌리산업 용어 추출
        industry_terms = []
        if self.korean_optimizer:
            result = await self.korean_optimizer.process_korean_text(content)
            industry_terms = result.industry_terms
        
        # THINK 블록 생성
        think_block = ThinkBlock(
            level=level,
            content=content,
            processing_time=processing_time or 0.0,
            confidence=confidence,
            industry_terms=industry_terms,
            metadata=metadata or {}
        )
        
        session.blocks.append(think_block)
        session.total_processing_time += think_block.processing_time
        
        # 통계 업데이트
        self.usage_stats["total_blocks"] += 1
        self._update_usage_stats(think_block)
        
        logger.debug(f"THINK 블록 추가: {level.value} - {session_id}")
        return think_block
    
    async def _optimize_think_content(self, content: str, level: ThinkLevel) -> str:
        """THINK 내용 한국어 최적화"""
        
        # 기본 한국어 정규화
        optimized = self.korean_optimizer.normalize_korean_text(content)
        
        # 레벨별 표현 개선
        if level == ThinkLevel.THINK:
            # 기본 분석 단계 - 명확하고 직접적
            optimized = re.sub(r'^', '분석: ', optimized)
            optimized = re.sub(r'생각해보면', '검토하면', optimized)
            
        elif level == ThinkLevel.MEGATHINK:
            # 복합 분석 단계 - 종합적이고 체계적
            optimized = re.sub(r'^', '종합분석: ', optimized)
            optimized = re.sub(r'고려하면', '종합적으로 검토하면', optimized)
            
        elif level == ThinkLevel.ULTRATHINK:
            # 최종 결론 단계 - 결정적이고 명확
            optimized = re.sub(r'^', '결론: ', optimized)
            optimized = re.sub(r'결론적으로', '최종적으로', optimized)
        
        return optimized.strip()
    
    async def generate_progressive_think(
        self,
        session_id: str,
        topic: str,
        industry_domain: Optional[str] = None
    ) -> AsyncGenerator[ThinkBlock, None]:
        """점진적 THINK 블록 생성"""
        
        if not self.enabled:
            return
        
        # 도메인별 템플릿 선택
        template = self.industry_think_templates.get(
            industry_domain, 
            self.industry_think_templates["주조"]  # 기본값
        )
        
        # THINK 단계
        start_time = time.time()
        
        think_content = f"{template['think']} {topic}의 기본 조건을 검토해보겠습니다."
        think_block = await self.add_think_block(
            session_id, 
            ThinkLevel.THINK, 
            think_content,
            processing_time=time.time() - start_time
        )
        
        if think_block:
            yield think_block
        
        # 잠시 대기 (실제 처리 시뮬레이션)
        await asyncio.sleep(0.5)
        
        # MEGATHINK 단계  
        start_time = time.time()
        
        megathink_content = f"{template['megathink']} 다양한 요인들을 종합적으로 분석해보겠습니다."
        megathink_block = await self.add_think_block(
            session_id,
            ThinkLevel.MEGATHINK,
            megathink_content,
            processing_time=time.time() - start_time
        )
        
        if megathink_block:
            yield megathink_block
        
        await asyncio.sleep(0.3)
        
        # ULTRATHINK 단계
        start_time = time.time()
        
        ultrathink_content = f"{template['ultrathink']} 최적의 해결방안을 제시하겠습니다."
        ultrathink_block = await self.add_think_block(
            session_id,
            ThinkLevel.ULTRATHINK, 
            ultrathink_content,
            processing_time=time.time() - start_time
        )
        
        if ultrathink_block:
            yield ultrathink_block
    
    def format_think_block_for_display(self, think_block: ThinkBlock) -> str:
        """THINK 블록을 표시용으로 포맷팅"""
        
        if not think_block:
            return ""
        
        # 레벨별 스타일 적용
        level_style = self.styles.get(think_block.level.value, f"**{think_block.level.value.upper()}**")
        
        # 기본 포맷
        formatted = f"{level_style}: {think_block.content}"
        
        # 뿌리산업 용어 강조
        if think_block.industry_terms:
            for term in think_block.industry_terms:
                formatted = formatted.replace(term, f"**{term}**")
        
        # 신뢰도 표시 (낮은 경우만)
        if think_block.confidence < 0.8:
            formatted += f" (신뢰도: {think_block.confidence:.1f})"
        
        # 처리 시간 표시 (디버그 모드에서만)
        if self.config_manager.get_value("debug", False) and think_block.processing_time > 0:
            formatted += f" [{think_block.processing_time:.2f}s]"
        
        return formatted
    
    def format_session_summary(self, session_id: str) -> str:
        """세션 요약 포맷팅"""
        
        if session_id not in self.active_sessions:
            return "세션을 찾을 수 없습니다."
        
        session = self.active_sessions[session_id]
        
        if not session.blocks:
            return "사고 과정이 기록되지 않았습니다."
        
        # 전체 사고 과정 정리
        summary_lines = ["## 🧠 사고 과정 요약\n"]
        
        for i, block in enumerate(session.blocks, 1):
            formatted_block = self.format_think_block_for_display(block)
            summary_lines.append(f"{i}. {formatted_block}")
        
        # 통계 정보 추가
        summary_lines.append(f"\n📊 **처리 통계**:")
        summary_lines.append(f"- 총 사고 단계: {len(session.blocks)}개")
        summary_lines.append(f"- 총 처리 시간: {session.total_processing_time:.2f}초")
        
        # 발견된 산업 용어 정리
        all_terms = set()
        for block in session.blocks:
            all_terms.update(block.industry_terms)
        
        if all_terms:
            summary_lines.append(f"- 감지된 뿌리산업 용어: {', '.join(sorted(all_terms))}")
        
        return "\n".join(summary_lines)
    
    async def stream_think_process(
        self,
        session_id: str,
        thinking_function: callable,
        *args,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """사고 과정 실시간 스트리밍"""
        
        if not self.enabled:
            # THINK 블록이 비활성화된 경우 직접 실행
            result = await thinking_function(*args, **kwargs)
            yield str(result)
            return
        
        # 사고 과정 시작 알림
        yield "🧠 **사고 시작**...\n\n"
        
        try:
            # 실제 사고 함수 실행하면서 중간 과정 표시
            start_time = time.time()
            
            # THINK 단계
            await self.add_think_block(
                session_id,
                ThinkLevel.THINK,
                "문제 상황을 분석하고 접근 방법을 검토 중입니다.",
                processing_time=0.5
            )
            yield f"{self.styles['think']}: 문제 분석 중...\n\n"
            
            await asyncio.sleep(0.3)
            
            # MEGATHINK 단계 
            await self.add_think_block(
                session_id,
                ThinkLevel.MEGATHINK,
                "다양한 관점에서 종합적으로 검토하고 있습니다.",
                processing_time=1.0
            )
            yield f"{self.styles['megathink']}: 종합 분석 중...\n\n"
            
            await asyncio.sleep(0.5)
            
            # 실제 함수 실행
            result = await thinking_function(*args, **kwargs)
            
            # ULTRATHINK 단계
            total_time = time.time() - start_time
            await self.add_think_block(
                session_id,
                ThinkLevel.ULTRATHINK,
                "최적의 답변을 도출했습니다.",
                processing_time=total_time
            )
            yield f"{self.styles['ultrathink']}: 결론 도출 완료!\n\n"
            
            # 최종 결과
            yield "---\n\n"
            yield str(result)
            
        except Exception as e:
            yield f"❌ **사고 과정 오류**: {e}\n\n"
            logger.error(f"사고 과정 스트리밍 실패: {e}")
    
    def end_think_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """THINK 세션 종료 및 통계 반환"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # 세션 통계 생성
        session_stats = {
            "session_id": session_id,
            "total_blocks": len(session.blocks),
            "total_time": session.total_processing_time,
            "duration": (datetime.now() - session.start_time).total_seconds(),
            "blocks_by_level": {
                level.value: len([b for b in session.blocks if b.level == level])
                for level in ThinkLevel
            },
            "industry_terms_found": len(set(
                term for block in session.blocks for term in block.industry_terms
            )),
            "avg_confidence": sum(b.confidence for b in session.blocks) / len(session.blocks) if session.blocks else 0.0
        }
        
        # 세션 삭제
        del self.active_sessions[session_id]
        
        logger.info(f"THINK 세션 종료: {session_id} - {session_stats['total_blocks']}개 블록")
        return session_stats
    
    def _update_usage_stats(self, think_block: ThinkBlock):
        """사용 통계 업데이트"""
        
        # 평균 처리 시간 업데이트
        total_blocks = self.usage_stats["total_blocks"]
        current_avg = self.usage_stats["avg_processing_time"]
        
        self.usage_stats["avg_processing_time"] = (
            (current_avg * (total_blocks - 1) + think_block.processing_time) / total_blocks
        )
        
        # 가장 많이 사용된 레벨 업데이트
        level_counts = {}
        for session in self.active_sessions.values():
            for block in session.blocks:
                level = block.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            self.usage_stats["most_used_level"] = max(level_counts, key=level_counts.get)
    
    def get_think_analytics(self) -> Dict[str, Any]:
        """THINK 블록 사용 분석"""
        
        active_session_count = len(self.active_sessions)
        
        # 활성 세션들의 통계
        active_stats = {
            "total_active_blocks": sum(len(s.blocks) for s in self.active_sessions.values()),
            "avg_session_duration": 0.0,
            "most_common_industry_terms": []
        }
        
        if self.active_sessions:
            durations = [
                (datetime.now() - session.start_time).total_seconds()
                for session in self.active_sessions.values()
            ]
            active_stats["avg_session_duration"] = sum(durations) / len(durations)
            
            # 공통 산업 용어 분석
            all_terms = []
            for session in self.active_sessions.values():
                for block in session.blocks:
                    all_terms.extend(block.industry_terms)
            
            if all_terms:
                from collections import Counter
                term_counts = Counter(all_terms)
                active_stats["most_common_industry_terms"] = term_counts.most_common(5)
        
        return {
            "enabled": self.enabled,
            "usage_stats": self.usage_stats,
            "active_sessions": active_session_count,
            "active_session_stats": active_stats,
            "supported_levels": [level.value for level in ThinkLevel],
            "industry_templates": list(self.industry_think_templates.keys()),
            "last_updated": datetime.now().isoformat()
        }
    
    def configure_think_styles(self, custom_styles: Dict[str, str]):
        """THINK 블록 스타일 커스터마이징"""
        
        for level, style in custom_styles.items():
            if level in self.styles:
                self.styles[level] = style
                logger.info(f"THINK 스타일 업데이트: {level} -> {style}")
    
    def export_session_log(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 로그 내보내기"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "start_time": session.start_time.isoformat(),
            "user_context": session.user_context,
            "blocks": [
                {
                    "level": block.level.value,
                    "content": block.content,
                    "timestamp": block.timestamp.isoformat(),
                    "processing_time": block.processing_time,
                    "confidence": block.confidence,
                    "industry_terms": block.industry_terms,
                    "metadata": block.metadata
                }
                for block in session.blocks
            ],
            "total_processing_time": session.total_processing_time,
            "export_timestamp": datetime.now().isoformat()
        }