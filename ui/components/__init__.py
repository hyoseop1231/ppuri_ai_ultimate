"""
UI Components - PPuRI-AI Ultimate UI 컴포넌트 패키지

차세대 사용자 인터페이스를 구성하는 모든 컴포넌트들을 통합한 패키지.

Components:
- ChatInterface: 실시간 스트리밍 대화 인터페이스
- ThinkVisualizer: THINK 블록 실시간 시각화
- KnowledgeExplorer: 지식 그래프 인터랙티브 탐색기  
- PerformanceDashboard: 실시간 성능 모니터링 대시보드
- MCPMonitor: MCP 도구 생태계 모니터링

Features:
- 실시간 WebSocket 통신
- 한국어 최적화 지원
- 뿌리산업 특화 기능
- 반응형 UI/UX
- 성능 최적화
"""

from .chat_interface import ChatInterface
from .think_visualizer import ThinkVisualizer
from .knowledge_explorer import KnowledgeExplorer
from .performance_dashboard import PerformanceDashboard
from .mcp_monitor import MCPMonitor

__all__ = [
    "ChatInterface",
    "ThinkVisualizer", 
    "KnowledgeExplorer",
    "PerformanceDashboard",
    "MCPMonitor"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__description__ = "차세대 UI 컴포넌트 시스템"