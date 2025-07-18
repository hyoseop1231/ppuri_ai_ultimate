"""
MCP Ecosystem - 자동 진화 MCP 도구 생태계

Model Context Protocol 기반 도구들이 자율적으로 학습하고 진화하며
성능을 최적화하는 동적 생태계 시스템.

Features:
- 자동 도구 발견 및 등록
- 성능 기반 도구 진화
- 동적 도구 조합 및 체인
- 실시간 성능 모니터링
- 자율 학습 및 최적화
- 뿌리산업 특화 도구 생성
"""

from .tool_registry import ToolRegistry
from .evolution_manager import EvolutionManager
from .mcp_integrator import MCPIntegrator
from .tool_composer import ToolComposer
from .performance_monitor import ToolPerformanceMonitor
from .ecosystem_orchestrator import EcosystemOrchestrator

__all__ = [
    "ToolRegistry",
    "EvolutionManager",
    "MCPIntegrator",
    "ToolComposer", 
    "ToolPerformanceMonitor",
    "EcosystemOrchestrator"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__self_evolving__ = True