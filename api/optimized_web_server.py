"""
Optimized Web Server - PPuRI-AI Ultimate 최적화된 웹 서버
"""

import asyncio
import gc
import logging
import os
import weakref
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
import time
import psutil

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

from .middleware.auth import AuthenticationMiddleware, JWTManager, SessionManager
from .middleware.security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    CSRFProtectionMiddleware,
    RequestValidationMiddleware
)
from .routes.auth import router as auth_router
from .routes.sessions import router as sessions_router
from .routes.chat import router as chat_router
from .database.connection_pool import connection_pool_manager
from .constants import SecurityConstants

logger = logging.getLogger(__name__)


class OptimizedWebServer:
    """최적화된 웹 서버 클래스"""
    
    def __init__(
        self,
        ui_orchestrator,
        chat_interface,
        think_visualizer,
        knowledge_explorer,
        performance_dashboard,
        mcp_monitor,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        self.ui_orchestrator = ui_orchestrator
        self.chat_interface = chat_interface
        self.think_visualizer = think_visualizer
        self.knowledge_explorer = knowledge_explorer
        self.performance_dashboard = performance_dashboard
        self.mcp_monitor = mcp_monitor
        
        self.host = host
        self.port = port
        
        # 메모리 관리
        self.background_tasks: Set[asyncio.Task] = set()
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.session_websockets: Dict[str, List[str]] = {}
        
        # 캐시 관리 (크기 제한)
        self.message_cache = deque(maxlen=1000)
        self.think_cache = deque(maxlen=100)
        self.performance_cache = deque(maxlen=500)
        
        # 정리 작업 관리
        self.cleanup_interval = 300  # 5분
        self.last_cleanup = time.time()
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # 성능 모니터링
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # FastAPI 앱 생성
        self.app = self._create_app()
        
        logger.info("최적화된 웹 서버 초기화 완료")
    
    def _create_app(self) -> FastAPI:
        """FastAPI 앱 생성 및 설정"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """앱 생명주기 관리"""
            # 시작 시 실행
            await self._startup()
            try:
                yield
            finally:
                # 종료 시 실행
                await self._shutdown()
        
        app = FastAPI(
            title="PPuRI-AI Ultimate",
            description="뿌리산업 특화 AI 시스템 - 최적화된 버전",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=lifespan
        )
        
        # 보안 미들웨어 설정
        self._setup_security_middleware(app)
        
        # 라우터 등록
        self._setup_routes(app)
        
        # WebSocket 라우터 등록
        self._setup_websocket_routes(app)
        
        # 정적 파일 서빙
        self._setup_static_files(app)
        
        return app
    
    def _setup_security_middleware(self, app: FastAPI):
        """보안 미들웨어 설정"""
        # CORS 설정 (보안 강화)
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # 보안 헤더 미들웨어
        app.add_middleware(SecurityHeadersMiddleware)
        
        # Rate Limiting 미들웨어
        app.add_middleware(RateLimitMiddleware)
        
        # CSRF 보호 미들웨어
        app.add_middleware(CSRFProtectionMiddleware)
        
        # 요청 검증 미들웨어
        app.add_middleware(RequestValidationMiddleware)
        
        # 인증 미들웨어 (JWT 관리자 필요)
        jwt_manager = JWTManager()
        app.add_middleware(AuthenticationMiddleware, jwt_manager=jwt_manager)
    
    def _setup_routes(self, app: FastAPI):
        """라우터 설정"""
        # 인증 라우터
        app.include_router(auth_router)
        
        # 세션 라우터
        app.include_router(sessions_router)
        
        # 채팅 라우터
        app.include_router(chat_router)
        
        # 헬스 체크 엔드포인트
        @app.get("/api/health")
        async def health_check():
            return await self._health_check()
        
        # 루트 엔드포인트
        @app.get("/")
        async def root():
            return {"message": "PPuRI-AI Ultimate - 최적화된 버전"}
    
    def _setup_websocket_routes(self, app: FastAPI):
        """WebSocket 라우터 설정"""
        
        @app.websocket("/ws/chat/{session_id}")
        async def websocket_chat(websocket: WebSocket, session_id: str):
            await self._handle_websocket_connection(websocket, session_id, "chat")
        
        @app.websocket("/ws/performance")
        async def websocket_performance(websocket: WebSocket):
            await self._handle_websocket_connection(websocket, "performance", "performance")
        
        @app.websocket("/ws/mcp")
        async def websocket_mcp(websocket: WebSocket):
            await self._handle_websocket_connection(websocket, "mcp", "mcp")
    
    def _setup_static_files(self, app: FastAPI):
        """정적 파일 서빙 설정"""
        static_dir = Path(__file__).parent.parent / "ui" / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    async def _startup(self):
        """서버 시작 시 실행"""
        logger.info("서버 시작 초기화 중...")
        
        # 데이터베이스 연결 풀 초기화
        await connection_pool_manager.initialize()
        
        # 정리 작업 스케줄링
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info("서버 시작 초기화 완료")
    
    async def _shutdown(self):
        """서버 종료 시 실행"""
        logger.info("서버 종료 중...")
        
        # 정리 작업 중지
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 백그라운드 태스크 정리
        await self._cleanup_background_tasks()
        
        # WebSocket 연결 정리
        await self._cleanup_websocket_connections()
        
        # 데이터베이스 연결 풀 정리
        await connection_pool_manager.close_all()
        
        logger.info("서버 종료 완료")
    
    async def _handle_websocket_connection(
        self, 
        websocket: WebSocket, 
        connection_id: str, 
        connection_type: str
    ):
        """WebSocket 연결 처리"""
        await websocket.accept()
        
        # 약한 참조 사용하여 메모리 누수 방지
        connection_ref = weakref.ref(websocket)
        
        self.websocket_connections[connection_id] = websocket
        
        try:
            # 초기 데이터 전송
            await self._send_initial_websocket_data(websocket, connection_type)
            
            # 메시지 처리 루프
            while True:
                try:
                    # 타임아웃 설정으로 무한 대기 방지
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=60.0
                    )
                    
                    # 메시지 처리
                    await self._process_websocket_message(
                        websocket, connection_id, connection_type, message
                    )
                    
                except asyncio.TimeoutError:
                    # 하트비트 전송
                    await websocket.send_text('{"type": "heartbeat"}')
                    
                except WebSocketDisconnect:
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket 연결 오류: {e}")
        finally:
            # 연결 정리
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
            
            # 약한 참조 확인
            if connection_ref() is not None:
                try:
                    await websocket.close()
                except:
                    pass
    
    async def _send_initial_websocket_data(self, websocket: WebSocket, connection_type: str):
        """초기 WebSocket 데이터 전송"""
        try:
            if connection_type == "chat":
                await websocket.send_text('{"type": "welcome", "message": "채팅 준비 완료"}')
            elif connection_type == "performance":
                # 성능 데이터 전송
                performance_data = await self._get_performance_data()
                await websocket.send_text(f'{{"type": "performance", "data": {performance_data}}}')
            elif connection_type == "mcp":
                # MCP 데이터 전송
                mcp_data = await self._get_mcp_data()
                await websocket.send_text(f'{{"type": "mcp", "data": {mcp_data}}}')
        except Exception as e:
            logger.error(f"초기 WebSocket 데이터 전송 실패: {e}")
    
    async def _process_websocket_message(
        self, 
        websocket: WebSocket, 
        connection_id: str, 
        connection_type: str, 
        message: str
    ):
        """WebSocket 메시지 처리"""
        try:
            import json
            data = json.loads(message)
            
            if connection_type == "chat":
                await self._handle_chat_message(websocket, connection_id, data)
            elif connection_type == "performance":
                await self._handle_performance_message(websocket, data)
            elif connection_type == "mcp":
                await self._handle_mcp_message(websocket, data)
                
        except json.JSONDecodeError:
            await websocket.send_text('{"type": "error", "message": "잘못된 JSON 형식"}')
        except Exception as e:
            logger.error(f"WebSocket 메시지 처리 실패: {e}")
            await websocket.send_text(f'{{"type": "error", "message": "{str(e)}"}}')
    
    async def _handle_chat_message(self, websocket: WebSocket, session_id: str, data: Dict[str, Any]):
        """채팅 메시지 처리"""
        # 실제 채팅 로직 구현 필요
        message = data.get("message", "")
        
        # 캐시에 메시지 추가
        self.message_cache.append({
            "session_id": session_id,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 응답 전송
        await websocket.send_text(f'{{"type": "response", "message": "메시지 처리됨: {message}"}}')
    
    async def _handle_performance_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """성능 메시지 처리"""
        # 성능 데이터 업데이트
        performance_data = await self._get_performance_data()
        await websocket.send_text(f'{{"type": "performance_update", "data": {performance_data}}}')
    
    async def _handle_mcp_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """MCP 메시지 처리"""
        # MCP 데이터 업데이트
        mcp_data = await self._get_mcp_data()
        await websocket.send_text(f'{{"type": "mcp_update", "data": {mcp_data}}}')
    
    async def _periodic_cleanup(self):
        """정기적인 정리 작업"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # 백그라운드 태스크 정리
                await self._cleanup_background_tasks()
                
                # 메모리 정리
                await self._cleanup_memory()
                
                # 가비지 컬렉션
                gc.collect()
                
                logger.info("정기 정리 작업 완료")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"정기 정리 작업 실패: {e}")
    
    async def _cleanup_background_tasks(self):
        """백그라운드 태스크 정리"""
        # 완료된 태스크 제거
        completed_tasks = {task for task in self.background_tasks if task.done()}
        
        for task in completed_tasks:
            # 예외 확인 및 로깅
            if task.exception():
                logger.error(f"백그라운드 태스크 예외: {task.exception()}")
            
            self.background_tasks.discard(task)
        
        # 취소된 태스크 정리
        cancelled_tasks = {task for task in self.background_tasks if task.cancelled()}
        
        for task in cancelled_tasks:
            self.background_tasks.discard(task)
        
        logger.info(f"백그라운드 태스크 정리 완료: {len(completed_tasks)} 완료, {len(cancelled_tasks)} 취소")
    
    async def _cleanup_memory(self):
        """메모리 정리"""
        # 캐시 크기 확인 및 정리
        if len(self.message_cache) > 800:
            # 20% 제거
            for _ in range(200):
                if self.message_cache:
                    self.message_cache.popleft()
        
        # 시스템 메모리 사용량 확인
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            logger.warning(f"높은 메모리 사용률: {memory_usage}%")
            
            # 캐시 크기 축소
            while len(self.message_cache) > 500:
                self.message_cache.popleft()
            
            while len(self.think_cache) > 50:
                self.think_cache.popleft()
            
            while len(self.performance_cache) > 250:
                self.performance_cache.popleft()
            
            # 강제 가비지 컬렉션
            gc.collect()
    
    async def _cleanup_websocket_connections(self):
        """WebSocket 연결 정리"""
        closed_connections = []
        
        for connection_id, websocket in self.websocket_connections.items():
            try:
                # 연결 상태 확인
                await websocket.send_text('{"type": "ping"}')
            except:
                # 연결이 끊어진 경우
                closed_connections.append(connection_id)
        
        # 끊어진 연결 제거
        for connection_id in closed_connections:
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
        
        logger.info(f"WebSocket 연결 정리: {len(closed_connections)}개 연결 정리")
    
    async def _health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        # 시스템 상태 확인
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 데이터베이스 상태 확인
        db_status = await connection_pool_manager.health_check()
        
        # 서버 상태
        uptime = time.time() - self.start_time
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime,
            "system": {
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage
            },
            "databases": db_status,
            "websocket_connections": len(self.websocket_connections),
            "background_tasks": len(self.background_tasks),
            "cache_sizes": {
                "messages": len(self.message_cache),
                "thinks": len(self.think_cache),
                "performance": len(self.performance_cache)
            }
        }
    
    async def _get_performance_data(self) -> str:
        """성능 데이터 조회"""
        # 실제 성능 데이터 조회 로직 구현 필요
        return '{"cpu": 45, "memory": 60, "response_time": 120}'
    
    async def _get_mcp_data(self) -> str:
        """MCP 데이터 조회"""
        # 실제 MCP 데이터 조회 로직 구현 필요
        return '{"active_tools": 48, "total_tools": 50, "performance": 95.5}'
    
    def create_background_task(self, coro) -> asyncio.Task:
        """백그라운드 태스크 생성"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        
        # 태스크 완료 시 자동 정리
        task.add_done_callback(self.background_tasks.discard)
        
        return task
    
    async def start_server(self):
        """서버 시작"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True,
            use_colors=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """서버 실행"""
        asyncio.run(self.start_server())