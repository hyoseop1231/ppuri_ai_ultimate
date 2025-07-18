"""
Web Package - PPuRI-AI Ultimate 웹 서버 패키지

FastAPI 기반 웹 서버와 정적 파일 관리를 담당하는 패키지.

Features:
- FastAPI 기반 REST API
- WebSocket 실시간 통신
- 정적 파일 서빙
- CORS 지원
- 세션 관리
- 실시간 모니터링
"""

from .web_server import WebServer

__all__ = [
    "WebServer"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"
__description__ = "FastAPI 기반 웹 서버"