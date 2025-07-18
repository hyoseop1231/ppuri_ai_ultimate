"""
Base Workflow - PPuRI-AI Ultimate 기본 워크플로우 클래스

LlamaIndex Workflows를 활용한 이벤트 드리븐 워크플로우 베이스
"""

import asyncio
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from dataclasses import dataclass
import json

# LlamaIndex Workflows import with fallback
try:
    from llama_index.core.workflow import (
        Workflow,
        step,
        Context,
        Event,
        StartEvent,
        StopEvent
    )
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Fallback 클래스 정의
    class Workflow:
        pass
    def step(func):
        return func
    class Context:
        pass
    class Event:
        pass
    class StartEvent(Event):
        pass
    class StopEvent(Event):
        pass

logger = logging.getLogger(__name__)


# 커스텀 이벤트 정의
@dataclass
class ProblemSubmissionEvent(Event):
    """문제 제출 이벤트"""
    problem_data: Dict[str, Any]
    priority: str = "normal"
    request_id: str = None


@dataclass
class AnalysisCompleteEvent(Event):
    """분석 완료 이벤트"""
    analysis_result: Dict[str, Any]
    domain_agents_used: List[str]
    confidence_score: float


@dataclass
class SolutionGeneratedEvent(Event):
    """솔루션 생성 이벤트"""
    solutions: List[Dict[str, Any]]
    implementation_plan: Dict[str, Any]
    estimated_impact: float


class WorkflowState:
    """워크플로우 상태 관리"""
    def __init__(self):
        self.workflow_id = None
        self.start_time = None
        self.current_step = None
        self.completed_steps = []
        self.errors = []
        self.results = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "errors": self.errors,
            "results": self.results
        }


class BaseIndustrialWorkflow(ABC):
    """뿌리산업 워크플로우 베이스 클래스"""
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.state = WorkflowState()
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0,
            "last_execution": None
        }
        
        if LLAMAINDEX_AVAILABLE:
            self._workflow = self._create_workflow()
        else:
            logger.warning(f"LlamaIndex Workflows를 사용할 수 없습니다. {workflow_name}이 제한된 모드로 실행됩니다.")
            self._workflow = None
    
    @abstractmethod
    def _create_workflow(self) -> Workflow:
        """워크플로우 생성 (하위 클래스에서 구현)"""
        pass
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 실행"""
        start_time = datetime.now()
        
        # 상태 초기화
        self.state = WorkflowState()
        self.state.workflow_id = f"{self.workflow_name}_{start_time.timestamp()}"
        self.state.start_time = start_time
        
        try:
            if self._workflow:
                # LlamaIndex Workflow 실행
                result = await self._execute_llamaindex_workflow(input_data)
            else:
                # Fallback 실행
                result = await self._execute_fallback_workflow(input_data)
            
            # 메트릭 업데이트
            self._update_metrics(start_time, success=True)
            
            return {
                "status": "success",
                "workflow_id": self.state.workflow_id,
                "result": result,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "state": self.state.to_dict()
            }
            
        except Exception as e:
            logger.error(f"{self.workflow_name} 실행 실패: {e}")
            self.state.errors.append(str(e))
            self._update_metrics(start_time, success=False)
            
            return {
                "status": "error",
                "workflow_id": self.state.workflow_id,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "state": self.state.to_dict()
            }
    
    async def _execute_llamaindex_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LlamaIndex Workflow 실행"""
        # StartEvent 생성
        start_event = StartEvent(input_data=input_data)
        
        # 워크플로우 실행
        result = await self._workflow.run(start_event)
        
        return result
    
    @abstractmethod
    async def _execute_fallback_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback 워크플로우 실행 (하위 클래스에서 구현)"""
        pass
    
    def _update_metrics(self, start_time: datetime, success: bool):
        """메트릭 업데이트"""
        execution_time = (datetime.now() - start_time).total_seconds()
        self.metrics["total_executions"] += 1
        
        if success:
            self.metrics["successful_executions"] += 1
        
        # 평균 실행 시간 업데이트
        current_avg = self.metrics["average_execution_time"]
        total = self.metrics["total_executions"]
        self.metrics["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
        
        self.metrics["last_execution"] = datetime.now().isoformat()
    
    def get_metrics(self) -> Dict[str, Any]:
        """워크플로우 메트릭 조회"""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_executions"] / self.metrics["total_executions"]
                if self.metrics["total_executions"] > 0 else 0
            )
        }
    
    async def save_state(self, storage_path: str):
        """워크플로우 상태 저장"""
        state_data = {
            "workflow_name": self.workflow_name,
            "state": self.state.to_dict(),
            "metrics": self.metrics,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(storage_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    async def load_state(self, storage_path: str):
        """워크플로우 상태 로드"""
        try:
            with open(storage_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 상태 복원
            self.state = WorkflowState()
            if state_data["state"]["workflow_id"]:
                self.state.workflow_id = state_data["state"]["workflow_id"]
            if state_data["state"]["start_time"]:
                self.state.start_time = datetime.fromisoformat(state_data["state"]["start_time"])
            self.state.current_step = state_data["state"]["current_step"]
            self.state.completed_steps = state_data["state"]["completed_steps"]
            self.state.errors = state_data["state"]["errors"]
            self.state.results = state_data["state"]["results"]
            
            # 메트릭 복원
            self.metrics = state_data["metrics"]
            
            logger.info(f"워크플로우 상태 로드 완료: {self.workflow_name}")
            
        except Exception as e:
            logger.error(f"워크플로우 상태 로드 실패: {e}")
            raise