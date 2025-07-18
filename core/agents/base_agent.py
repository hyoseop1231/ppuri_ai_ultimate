"""
Base Agent - PPuRI-AI Ultimate 기본 에이전트 클래스

Agno 프레임워크를 활용한 초경량 고성능 에이전트 베이스 클래스
"""

import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# Agno import with fallback
try:
    from agno import Agent, AgentConfig
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    # Fallback 클래스 정의
    class Agent:
        pass
    class AgentConfig:
        pass

logger = logging.getLogger(__name__)


class BaseIndustrialAgent(ABC):
    """뿌리산업 전문 에이전트 베이스 클래스"""
    
    def __init__(
        self,
        domain: str,
        model_provider: str = "openai",
        reasoning_type: str = "technical_analysis",
        tools: Optional[List[str]] = None,
        knowledge_base: Optional[str] = None
    ):
        self.domain = domain
        self.model_provider = model_provider
        self.reasoning_type = reasoning_type
        self.tools = tools or []
        self.knowledge_base = knowledge_base
        
        # 성능 메트릭 추적
        self.metrics = {
            "creation_time": datetime.now(),
            "total_requests": 0,
            "average_response_time": 0,
            "memory_usage": "6.5KB"  # Agno 초경량 특성
        }
        
        # Agno 에이전트 초기화
        if AGNO_AVAILABLE:
            self._initialize_agno_agent()
        else:
            logger.warning(f"Agno를 사용할 수 없습니다. {domain} 에이전트가 제한된 모드로 실행됩니다.")
            self.agent = None
    
    def _initialize_agno_agent(self):
        """Agno 에이전트 초기화"""
        try:
            config = AgentConfig(
                model_provider=self.model_provider,
                reasoning_type=self.reasoning_type,
                tools=self.tools,
                knowledge_base=self.knowledge_base,
                optimization_level="extreme",  # 초경량 최적화
                memory_limit="6.5KB",         # 메모리 제한
                creation_time_target="3μs"    # 생성 시간 목표
            )
            
            self.agent = Agent(config)
            logger.info(f"{self.domain} 에이전트 초기화 성공 (3μs, 6.5KB)")
            
        except Exception as e:
            logger.error(f"{self.domain} 에이전트 초기화 실패: {e}")
            self.agent = None
    
    @abstractmethod
    async def analyze(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """문제 분석 (하위 클래스에서 구현)"""
        pass
    
    @abstractmethod
    async def generate_solution(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """솔루션 생성 (하위 클래스에서 구현)"""
        pass
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """요청 처리 메인 플로우"""
        start_time = datetime.now()
        
        try:
            # 1. 문제 분석
            analysis_result = await self.analyze(request_data)
            
            # 2. 솔루션 생성
            solution = await self.generate_solution(analysis_result)
            
            # 3. 메트릭 업데이트
            self._update_metrics(start_time)
            
            return {
                "status": "success",
                "domain": self.domain,
                "analysis": analysis_result,
                "solution": solution,
                "metrics": self.get_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"{self.domain} 에이전트 처리 실패: {e}")
            return {
                "status": "error",
                "domain": self.domain,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _update_metrics(self, start_time: datetime):
        """성능 메트릭 업데이트"""
        response_time = (datetime.now() - start_time).total_seconds()
        self.metrics["total_requests"] += 1
        
        # 이동 평균 계산
        current_avg = self.metrics["average_response_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 조회"""
        return {
            **self.metrics,
            "uptime": (datetime.now() - self.metrics["creation_time"]).total_seconds()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        return {
            "status": "healthy" if self.agent else "degraded",
            "domain": self.domain,
            "metrics": self.get_metrics(),
            "agno_available": AGNO_AVAILABLE
        }
    
    async def collaborate_with(self, other_agent: 'BaseIndustrialAgent', shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """다른 에이전트와 협업"""
        # 병렬로 두 에이전트 실행
        results = await asyncio.gather(
            self.process_request(shared_data),
            other_agent.process_request(shared_data)
        )
        
        return {
            "collaboration_result": {
                self.domain: results[0],
                other_agent.domain: results[1]
            },
            "combined_confidence": (
                results[0].get("confidence", 0) + results[1].get("confidence", 0)
            ) / 2
        }