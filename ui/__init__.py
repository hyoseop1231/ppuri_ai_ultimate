"""
차세대 UI/UX 시스템 - PPuRI-AI Ultimate

모든 AI 시스템 컴포넌트를 통합하여 실시간 인터랙티브 
사용자 인터페이스를 제공하는 통합 UI 패키지.

Features:
- 실시간 스트리밍 대화 인터페이스
- THINK 블록 시각화
- 지식 그래프 탐색기  
- 성능 모니터링 대시보드
- MCP 도구 생태계 시각화
- 한국어 최적화 UI/UX
- WebSocket 기반 실시간 통신
- 반응형 웹 인터페이스
"""

from .ui_orchestrator import UIOrchestrator
from .components.chat_interface import ChatInterface
from .components.think_visualizer import ThinkVisualizer
from .components.knowledge_explorer import KnowledgeExplorer
from .components.performance_dashboard import PerformanceDashboard
from .components.mcp_monitor import MCPMonitor
from .web.web_server import WebServer

__all__ = [
    "UIOrchestrator",
    "ChatInterface", 
    "ThinkVisualizer",
    "KnowledgeExplorer",
    "PerformanceDashboard",
    "MCPMonitor",
    "WebServer"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__next_generation_ui__ = True