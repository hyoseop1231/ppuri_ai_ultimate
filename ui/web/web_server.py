"""
Web Server - PPuRI-AI Ultimate 웹 서버

FastAPI 기반 웹 서버로 모든 UI 컴포넌트를 통합하여
실시간 웹 인터페이스를 제공하는 서버.

Features:
- FastAPI 기반 REST API
- WebSocket 실시간 통신
- 정적 파일 서빙
- CORS 지원
- 세션 관리
- 한국어 최적화 지원
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pathlib import Path

logger = logging.getLogger(__name__)


class WebServer:
    """
    PPuRI-AI Ultimate 웹 서버
    
    모든 UI 컴포넌트를 통합하여 웹 인터페이스를 제공하고
    실시간 통신을 지원하는 FastAPI 기반 서버.
    """
    
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
        
        # FastAPI 앱 생성
        self.app = FastAPI(
            title="PPuRI-AI Ultimate",
            description="뿌리산업 특화 AI 시스템",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # WebSocket 연결 관리
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.session_websockets: Dict[str, List[str]] = {}
        
        # 보안 설정
        self.security = HTTPBearer(auto_error=False)
        
        # 라우트 설정
        self._setup_middleware()
        self._setup_routes()
        self._setup_websocket_routes()
        self._setup_static_files()
        
        logger.info("Web Server 초기화 완료")
    
    def _setup_middleware(self):
        """미들웨어 설정"""
        
        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 개발용, 프로덕션에서는 특정 도메인만
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """REST API 라우트 설정"""
        
        # === 기본 라우트 ===
        @self.app.get("/")
        async def root():
            return {"message": "PPuRI-AI Ultimate", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # === 세션 관리 ===
        @self.app.post("/api/sessions")
        async def create_session(user_id: Optional[str] = None):
            try:
                session_id = await self.ui_orchestrator.create_session(user_id)
                return {"session_id": session_id, "created_at": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"세션 생성 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/sessions/{session_id}")
        async def get_session_info(session_id: str):
            try:
                session_info = self.ui_orchestrator.get_session_info(session_id)
                if not session_info:
                    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
                return session_info
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"세션 정보 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === 채팅 인터페이스 API ===
        @self.app.post("/api/chat/{session_id}/messages")
        async def send_message(session_id: str, message: Dict[str, Any]):
            try:
                content = message.get("content", "")
                attachments = message.get("attachments", [])
                
                # 비동기 생성기를 리스트로 변환 (실제로는 WebSocket 사용 권장)
                responses = []
                async for response_chunk in self.chat_interface.send_message(
                    session_id, content, attachments
                ):
                    responses.append(response_chunk)
                
                return {"responses": responses}
            except Exception as e:
                logger.error(f"메시지 전송 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/chat/{session_id}/history")
        async def get_message_history(
            session_id: str,
            limit: Optional[int] = None,
            before_message_id: Optional[str] = None
        ):
            try:
                history = await self.chat_interface.get_message_history(
                    session_id, limit, before_message_id
                )
                return {"messages": history}
            except Exception as e:
                logger.error(f"메시지 히스토리 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/chat/{session_id}/suggestions")
        async def get_input_suggestions(
            session_id: str,
            partial_input: str = "",
            limit: int = 5
        ):
            try:
                suggestions = await self.chat_interface.get_input_suggestions(
                    session_id, partial_input, limit
                )
                return {"suggestions": suggestions}
            except Exception as e:
                logger.error(f"입력 제안 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === 지식 그래프 API ===
        @self.app.get("/api/knowledge/{session_id}/graph")
        async def get_knowledge_graph(
            session_id: str,
            format: str = "hierarchical",
            filters: Optional[str] = None
        ):
            try:
                filter_dict = json.loads(filters) if filters else None
                graph_data = await self.knowledge_explorer.get_think_tree_data(
                    session_id, format
                )
                return graph_data
            except Exception as e:
                logger.error(f"지식 그래프 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/knowledge/{session_id}/search")
        async def search_knowledge_nodes(
            session_id: str,
            search_request: Dict[str, Any]
        ):
            try:
                query = search_request.get("query", "")
                search_type = search_request.get("type", "text")
                
                results = await self.knowledge_explorer.search_nodes(
                    session_id, query, search_type
                )
                
                return {
                    "results": [
                        {
                            "id": node.id,
                            "label": node.label,
                            "type": node.type,
                            "properties": node.properties
                        }
                        for node in results
                    ]
                }
            except Exception as e:
                logger.error(f"지식 노드 검색 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/knowledge/{session_id}/nodes/{node_id}/neighbors")
        async def get_node_neighbors(
            session_id: str,
            node_id: str,
            depth: int = 1
        ):
            try:
                neighbors = await self.knowledge_explorer.get_node_neighbors(
                    session_id, node_id, depth
                )
                return neighbors
            except Exception as e:
                logger.error(f"노드 이웃 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === 성능 대시보드 API ===
        @self.app.get("/api/performance/dashboard")
        async def get_performance_dashboard(time_range_minutes: int = 60):
            try:
                dashboard_data = await self.performance_dashboard.get_real_time_dashboard_data(
                    time_range_minutes
                )
                return dashboard_data
            except Exception as e:
                logger.error(f"성능 대시보드 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/performance/components/{component_name}")
        async def get_component_metrics(
            component_name: str,
            time_range_hours: int = 24
        ):
            try:
                metrics = await self.performance_dashboard.get_component_detailed_metrics(
                    component_name, time_range_hours
                )
                return metrics
            except Exception as e:
                logger.error(f"컴포넌트 메트릭 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/performance/alerts/{alert_id}/acknowledge")
        async def acknowledge_performance_alert(alert_id: str):
            try:
                success = await self.performance_dashboard.acknowledge_alert(alert_id)
                return {"success": success}
            except Exception as e:
                logger.error(f"경고 승인 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === MCP 모니터 API ===
        @self.app.get("/api/mcp/ecosystem")
        async def get_mcp_ecosystem():
            try:
                overview = await self.mcp_monitor.get_ecosystem_overview()
                return overview
            except Exception as e:
                logger.error(f"MCP 생태계 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/mcp/tools/{tool_name}")
        async def get_tool_details(tool_name: str):
            try:
                details = await self.mcp_monitor.get_tool_details(tool_name)
                if not details:
                    raise HTTPException(status_code=404, detail="도구를 찾을 수 없습니다")
                return details
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"도구 상세 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/mcp/network")
        async def get_tool_network():
            try:
                network_data = await self.mcp_monitor.get_network_visualization_data()
                return network_data
            except Exception as e:
                logger.error(f"도구 네트워크 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/mcp/tools/{tool_name}/evolve")
        async def trigger_tool_evolution(tool_name: str, evolution_request: Dict[str, Any]):
            try:
                strategy = evolution_request.get("strategy", "general_optimization")
                success = await self.mcp_monitor.trigger_manual_evolution(tool_name, strategy)
                return {"success": success}
            except Exception as e:
                logger.error(f"도구 진화 트리거 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === THINK 시각화 API ===
        @self.app.get("/api/think/{session_id}/tree")
        async def get_think_tree(session_id: str, format: str = "hierarchical"):
            try:
                tree_data = await self.think_visualizer.get_think_tree_data(session_id, format)
                return tree_data
            except Exception as e:
                logger.error(f"사고 트리 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/think/{session_id}/statistics")
        async def get_think_statistics(session_id: str):
            try:
                stats = await self.think_visualizer.get_session_statistics(session_id)
                return stats
            except Exception as e:
                logger.error(f"사고 통계 조회 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # === 내보내기 API ===
        @self.app.get("/api/export/conversation/{session_id}")
        async def export_conversation(session_id: str, format: str = "json"):
            try:
                exported_data = await self.chat_interface.export_conversation(session_id, format)
                if not exported_data:
                    raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
                
                content_type = "application/json" if format == "json" else "text/markdown"
                return JSONResponse(
                    content={"data": exported_data, "format": format},
                    media_type=content_type
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"대화 내보내기 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/export/performance")
        async def export_performance_report(
            format: str = "json",
            time_range_hours: int = 24
        ):
            try:
                report = await self.performance_dashboard.export_performance_report(
                    format, time_range_hours
                )
                if not report:
                    raise HTTPException(status_code=500, detail="리포트 생성 실패")
                
                content_type = "application/json" if format == "json" else "text/markdown"
                return JSONResponse(
                    content={"data": report, "format": format},
                    media_type=content_type
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"성능 리포트 내보내기 실패: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_websocket_routes(self):
        """WebSocket 라우트 설정"""
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_websocket_connection(websocket, session_id)
        
        @self.app.websocket("/ws/performance")
        async def performance_websocket(websocket: WebSocket):
            """성능 모니터링 전용 WebSocket"""
            await self._handle_performance_websocket(websocket)
        
        @self.app.websocket("/ws/mcp")
        async def mcp_websocket(websocket: WebSocket):
            """MCP 모니터링 전용 WebSocket"""
            await self._handle_mcp_websocket(websocket)
    
    async def _handle_websocket_connection(self, websocket: WebSocket, session_id: str):
        """WebSocket 연결 처리"""
        
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            # 연결 등록
            self.websocket_connections[connection_id] = websocket
            
            if session_id not in self.session_websockets:
                self.session_websockets[session_id] = []
            self.session_websockets[session_id].append(connection_id)
            
            # 각 컴포넌트에 WebSocket 연결 등록
            self.think_visualizer.add_websocket_connection(session_id, websocket)
            
            logger.info(f"WebSocket 연결: {session_id} ({connection_id})")
            
            # 초기 데이터 전송
            await self._send_initial_data(websocket, session_id)
            
            # 메시지 수신 루프
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    await self._handle_websocket_message(websocket, session_id, message)
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket 메시지 처리 오류: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket 연결 해제: {session_id} ({connection_id})")
        
        except Exception as e:
            logger.error(f"WebSocket 연결 오류: {e}")
        
        finally:
            # 연결 정리
            self.websocket_connections.pop(connection_id, None)
            
            if session_id in self.session_websockets:
                try:
                    self.session_websockets[session_id].remove(connection_id)
                    if not self.session_websockets[session_id]:
                        del self.session_websockets[session_id]
                except ValueError:
                    pass
            
            # 컴포넌트에서 연결 제거
            self.think_visualizer.remove_websocket_connection(session_id, websocket)
    
    async def _send_initial_data(self, websocket: WebSocket, session_id: str):
        """초기 데이터 전송"""
        
        try:
            # 세션 정보
            session_info = self.ui_orchestrator.get_session_info(session_id)
            if session_info:
                await websocket.send_text(json.dumps({
                    "type": "session_info",
                    "data": session_info
                }))
            
            # 최근 메시지 히스토리
            recent_messages = await self.chat_interface.get_message_history(session_id, 10)
            await websocket.send_text(json.dumps({
                "type": "message_history",
                "data": recent_messages
            }))
            
            # THINK 트리 데이터
            think_tree = await self.think_visualizer.get_think_tree_data(session_id)
            await websocket.send_text(json.dumps({
                "type": "think_tree",
                "data": think_tree
            }))
            
        except Exception as e:
            logger.error(f"초기 데이터 전송 실패: {e}")
    
    async def _handle_websocket_message(
        self,
        websocket: WebSocket,
        session_id: str,
        message: Dict[str, Any]
    ):
        """WebSocket 메시지 처리"""
        
        message_type = message.get("type")
        data = message.get("data", {})
        
        try:
            if message_type == "chat_message":
                # 채팅 메시지 처리
                content = data.get("content", "")
                attachments = data.get("attachments", [])
                
                # 실시간 응답 스트리밍
                async for response_chunk in self.chat_interface.send_message(
                    session_id, content, attachments
                ):
                    await websocket.send_text(json.dumps({
                        "type": "chat_response",
                        "data": response_chunk
                    }))
            
            elif message_type == "typing_start":
                # 타이핑 시작 알림
                self.chat_interface.set_typing_status(session_id, True)
                await self._broadcast_to_session(session_id, {
                    "type": "typing_status",
                    "data": {"session_id": session_id, "typing": True}
                }, exclude_websocket=websocket)
            
            elif message_type == "typing_end":
                # 타이핑 종료 알림
                self.chat_interface.set_typing_status(session_id, False)
                await self._broadcast_to_session(session_id, {
                    "type": "typing_status",
                    "data": {"session_id": session_id, "typing": False}
                }, exclude_websocket=websocket)
            
            elif message_type == "knowledge_filter_update":
                # 지식 그래프 필터 업데이트
                filter_updates = data.get("filters", {})
                await self.knowledge_explorer.update_filter(session_id, filter_updates)
                
                # 업데이트된 그래프 데이터 전송
                updated_graph = await self.knowledge_explorer.get_think_tree_data(session_id)
                await websocket.send_text(json.dumps({
                    "type": "knowledge_graph_updated",
                    "data": updated_graph
                }))
            
            elif message_type == "think_config_update":
                # THINK 시각화 설정 업데이트
                config_updates = data.get("config", {})
                await self.think_visualizer.update_visualization_config(
                    session_id, config_updates
                )
            
            elif message_type == "ping":
                # 하트비트 응답
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                logger.warning(f"알 수 없는 WebSocket 메시지 타입: {message_type}")
        
        except Exception as e:
            logger.error(f"WebSocket 메시지 처리 실패: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def _broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        exclude_websocket: Optional[WebSocket] = None
    ):
        """세션 내 모든 WebSocket에 브로드캐스트"""
        
        if session_id not in self.session_websockets:
            return
        
        message_text = json.dumps(message)
        
        for connection_id in self.session_websockets[session_id]:
            websocket = self.websocket_connections.get(connection_id)
            
            if websocket and websocket != exclude_websocket:
                try:
                    await websocket.send_text(message_text)
                except Exception as e:
                    logger.error(f"브로드캐스트 실패: {e}")
    
    def _setup_static_files(self):
        """정적 파일 설정"""
        
        # 정적 파일 경로 설정
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # 템플릿 파일 서빙 (개발용)
        @self.app.get("/ui", response_class=HTMLResponse)
        async def serve_ui():
            # 실제로는 React/Vue 등의 SPA 빌드 파일 서빙
            return """
            <!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PPuRI-AI Ultimate</title>
                <style>
                    body { font-family: 'Noto Sans KR', sans-serif; margin: 0; padding: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .component { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                    .status { color: #27AE60; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🏭 PPuRI-AI Ultimate</h1>
                        <p>뿌리산업 특화 AI 시스템</p>
                        <p class="status">시스템 정상 운영 중</p>
                    </div>
                    
                    <div class="component">
                        <h3>💬 채팅 인터페이스</h3>
                        <p>실시간 대화형 AI 상담</p>
                        <a href="/docs#/Chat%20Interface" target="_blank">API 문서 보기</a>
                    </div>
                    
                    <div class="component">
                        <h3>🧠 THINK 시각화</h3>
                        <p>AI 사고 과정 실시간 시각화</p>
                        <a href="/docs#/Think%20Visualizer" target="_blank">API 문서 보기</a>
                    </div>
                    
                    <div class="component">
                        <h3>🕸️ 지식 그래프 탐색기</h3>
                        <p>대화형 지식 그래프 탐색</p>
                        <a href="/docs#/Knowledge%20Explorer" target="_blank">API 문서 보기</a>
                    </div>
                    
                    <div class="component">
                        <h3>📊 성능 대시보드</h3>
                        <p>실시간 시스템 성능 모니터링</p>
                        <a href="/docs#/Performance%20Dashboard" target="_blank">API 문서 보기</a>
                    </div>
                    
                    <div class="component">
                        <h3>🔧 MCP 도구 모니터</h3>
                        <p>자동 진화 도구 생태계 모니터링</p>
                        <a href="/docs#/MCP%20Monitor" target="_blank">API 문서 보기</a>
                    </div>
                    
                    <div class="component">
                        <h3>🔗 연결 테스트</h3>
                        <p>WebSocket 연결: <span id="ws-status">연결 중...</span></p>
                        <button onclick="testWebSocket()">WebSocket 테스트</button>
                    </div>
                </div>
                
                <script>
                    // WebSocket 연결 테스트
                    function testWebSocket() {
                        const wsStatus = document.getElementById('ws-status');
                        
                        // 임시 세션 ID
                        const sessionId = 'test-' + Date.now();
                        const ws = new WebSocket(`ws://localhost:${window.location.port}/ws/${sessionId}`);
                        
                        ws.onopen = () => {
                            wsStatus.textContent = '✅ 연결됨';
                            wsStatus.style.color = '#27AE60';
                            
                            // Ping 테스트
                            ws.send(JSON.stringify({type: 'ping'}));
                        };
                        
                        ws.onmessage = (event) => {
                            const message = JSON.parse(event.data);
                            console.log('WebSocket 메시지:', message);
                        };
                        
                        ws.onerror = () => {
                            wsStatus.textContent = '❌ 연결 실패';
                            wsStatus.style.color = '#E74C3C';
                        };
                        
                        ws.onclose = () => {
                            wsStatus.textContent = '⚪ 연결 해제됨';
                            wsStatus.style.color = '#95A5A6';
                        };
                        
                        // 5초 후 연결 해제
                        setTimeout(() => ws.close(), 5000);
                    }
                    
                    // 페이지 로드 시 자동 테스트
                    window.onload = () => {
                        setTimeout(testWebSocket, 1000);
                    };
                </script>
            </body>
            </html>
            """
    
    async def start_server(self):
        """서버 시작"""
        
        # 각 컴포넌트 초기화
        await self.ui_orchestrator.initialize()
        
        if hasattr(self.performance_dashboard, 'start_monitoring'):
            await self.performance_dashboard.start_monitoring()
        
        if hasattr(self.mcp_monitor, 'start_monitoring'):
            await self.mcp_monitor.start_monitoring()
        
        logger.info(f"웹 서버 시작: http://{self.host}:{self.port}")
        
        # uvicorn 설정
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            ws_ping_interval=20,
            ws_ping_timeout=10
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """서버 실행 (동기)"""
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
    
    async def stop_server(self):
        """서버 중단"""
        
        # 모든 WebSocket 연결 정리
        for websocket in self.websocket_connections.values():
            try:
                await websocket.close()
            except:
                pass
        
        # 컴포넌트 정리
        if hasattr(self.performance_dashboard, 'stop_monitoring'):
            await self.performance_dashboard.stop_monitoring()
        
        if hasattr(self.mcp_monitor, 'stop_monitoring'):
            await self.mcp_monitor.stop_monitoring()
        
        # 각 컴포넌트 정리
        await self.ui_orchestrator.cleanup()
        await self.chat_interface.cleanup()
        await self.think_visualizer.cleanup()
        await self.knowledge_explorer.cleanup()
        await self.performance_dashboard.cleanup()
        await self.mcp_monitor.cleanup()
        
        logger.info("웹 서버 중단 완료")
    
    async def _handle_performance_websocket(self, websocket: WebSocket):
        """성능 모니터링 전용 WebSocket 핸들러"""
        
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            logger.info(f"성능 모니터링 WebSocket 연결: {connection_id}")
            
            # 초기 성능 데이터 전송
            await self._send_initial_performance_data(websocket)
            
            # 주기적 성능 업데이트 태스크
            update_task = asyncio.create_task(
                self._performance_update_loop(websocket, connection_id)
            )
            
            # 메시지 수신 루프
            try:
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        await self._handle_performance_websocket_message(websocket, message)
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"성능 WebSocket 메시지 처리 오류: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))
            
            except WebSocketDisconnect:
                logger.info(f"성능 모니터링 WebSocket 연결 해제: {connection_id}")
            
            except Exception as e:
                logger.error(f"성능 WebSocket 연결 오류: {e}")
            
            finally:
                # 업데이트 태스크 취소
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"성능 WebSocket 초기화 실패: {e}")
    
    async def _handle_mcp_websocket(self, websocket: WebSocket):
        """MCP 모니터링 전용 WebSocket 핸들러"""
        
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            logger.info(f"MCP 모니터링 WebSocket 연결: {connection_id}")
            
            # 초기 MCP 생태계 데이터 전송
            await self._send_initial_mcp_data(websocket)
            
            # 주기적 MCP 업데이트 태스크
            update_task = asyncio.create_task(
                self._mcp_update_loop(websocket, connection_id)
            )
            
            # 메시지 수신 루프
            try:
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        await self._handle_mcp_websocket_message(websocket, message)
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"MCP WebSocket 메시지 처리 오류: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e)
                        }))
            
            except WebSocketDisconnect:
                logger.info(f"MCP 모니터링 WebSocket 연결 해제: {connection_id}")
            
            except Exception as e:
                logger.error(f"MCP WebSocket 연결 오류: {e}")
            
            finally:
                # 업데이트 태스크 취소
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"MCP WebSocket 초기화 실패: {e}")
    
    async def _send_initial_performance_data(self, websocket: WebSocket):
        """초기 성능 데이터 전송"""
        
        try:
            # 실시간 대시보드 데이터
            dashboard_data = await self.performance_dashboard.get_real_time_dashboard_data(60)
            
            await websocket.send_text(json.dumps({
                "type": "performance_dashboard_init",
                "data": dashboard_data
            }))
            
            # 컴포넌트별 상세 메트릭
            components = ["conversational_engine", "rag_orchestrator", "graph_manager", "mcp_orchestrator"]
            
            for component in components:
                try:
                    metrics = await self.performance_dashboard.get_component_detailed_metrics(component, 1)
                    
                    await websocket.send_text(json.dumps({
                        "type": "component_metrics_init",
                        "data": {
                            "component": component,
                            "metrics": metrics
                        }
                    }))
                    
                except Exception as e:
                    logger.warning(f"컴포넌트 {component} 메트릭 로드 실패: {e}")
            
            # 시스템 헬스 상태
            health_data = await self.ui_orchestrator.get_performance_dashboard_data()
            
            await websocket.send_text(json.dumps({
                "type": "system_health_init",
                "data": health_data.get("system_health", {})
            }))
            
        except Exception as e:
            logger.error(f"초기 성능 데이터 전송 실패: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"초기 데이터 로드 실패: {str(e)}"
            }))
    
    async def _send_initial_mcp_data(self, websocket: WebSocket):
        """초기 MCP 데이터 전송"""
        
        try:
            # MCP 생태계 개요
            ecosystem_overview = await self.mcp_monitor.get_ecosystem_overview()
            
            await websocket.send_text(json.dumps({
                "type": "mcp_ecosystem_init",
                "data": ecosystem_overview
            }))
            
            # 도구 네트워크 시각화 데이터
            network_data = await self.mcp_monitor.get_network_visualization_data()
            
            await websocket.send_text(json.dumps({
                "type": "mcp_network_init",
                "data": network_data
            }))
            
            # 최근 도구 실행 로그
            execution_logs = await self.mcp_monitor.get_recent_execution_logs(50)
            
            await websocket.send_text(json.dumps({
                "type": "mcp_execution_logs_init",
                "data": execution_logs
            }))
            
            # 진화 히스토리
            evolution_history = await self.mcp_monitor.get_evolution_history(20)
            
            await websocket.send_text(json.dumps({
                "type": "mcp_evolution_history_init",
                "data": evolution_history
            }))
            
        except Exception as e:
            logger.error(f"초기 MCP 데이터 전송 실패: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"초기 MCP 데이터 로드 실패: {str(e)}"
            }))
    
    async def _performance_update_loop(self, websocket: WebSocket, connection_id: str):
        """성능 데이터 주기적 업데이트 루프"""
        
        while True:
            try:
                # 2초마다 업데이트
                await asyncio.sleep(2)
                
                # 실시간 메트릭 업데이트
                current_metrics = {
                    "active_sessions": len(self.ui_orchestrator.active_sessions),
                    "websocket_connections": len(self.websocket_connections),
                    "total_messages": sum(
                        len(stream) for stream in self.ui_orchestrator.message_streams.values()
                    ),
                    "think_blocks": sum(
                        len(stream) for stream in self.ui_orchestrator.think_streams.values()
                    ),
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_text(json.dumps({
                    "type": "performance_realtime_update",
                    "data": current_metrics
                }))
                
                # 5초마다 상세 시스템 상태 체크
                if int(datetime.now().timestamp()) % 5 == 0:
                    try:
                        system_health = await self.ui_orchestrator._check_system_health()
                        
                        await websocket.send_text(json.dumps({
                            "type": "system_health_update",
                            "data": system_health
                        }))
                        
                    except Exception as e:
                        logger.warning(f"시스템 헬스 체크 실패: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"성능 업데이트 루프 오류: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "update_error",
                        "message": str(e)
                    }))
                except:
                    break  # WebSocket 연결이 끊어진 경우
    
    async def _mcp_update_loop(self, websocket: WebSocket, connection_id: str):
        """MCP 데이터 주기적 업데이트 루프"""
        
        while True:
            try:
                # 5초마다 업데이트
                await asyncio.sleep(5)
                
                # MCP 생태계 상태 업데이트
                try:
                    ecosystem_status = await self.ui_orchestrator.get_mcp_ecosystem_status()
                    
                    await websocket.send_text(json.dumps({
                        "type": "mcp_ecosystem_update",
                        "data": ecosystem_status
                    }))
                    
                except Exception as e:
                    logger.warning(f"MCP 생태계 상태 업데이트 실패: {e}")
                
                # 최근 실행 로그 업데이트 (10초마다)
                if int(datetime.now().timestamp()) % 10 == 0:
                    try:
                        recent_executions = await self.mcp_monitor.get_recent_execution_logs(10)
                        
                        await websocket.send_text(json.dumps({
                            "type": "mcp_recent_executions",
                            "data": recent_executions
                        }))
                        
                    except Exception as e:
                        logger.warning(f"최근 MCP 실행 로그 업데이트 실패: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MCP 업데이트 루프 오류: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "update_error",
                        "message": str(e)
                    }))
                except:
                    break  # WebSocket 연결이 끊어진 경우
    
    async def _handle_performance_websocket_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ):
        """성능 WebSocket 메시지 처리"""
        
        message_type = message.get("type")
        data = message.get("data", {})
        
        try:
            if message_type == "request_component_details":
                # 특정 컴포넌트 상세 정보 요청
                component_name = data.get("component")
                time_range_hours = data.get("time_range_hours", 1)
                
                if component_name:
                    metrics = await self.performance_dashboard.get_component_detailed_metrics(
                        component_name, time_range_hours
                    )
                    
                    await websocket.send_text(json.dumps({
                        "type": "component_details_response",
                        "data": {
                            "component": component_name,
                            "metrics": metrics
                        }
                    }))
            
            elif message_type == "request_alert_history":
                # 경고 히스토리 요청
                limit = data.get("limit", 50)
                
                alert_history = await self.performance_dashboard.get_alert_history(limit)
                
                await websocket.send_text(json.dumps({
                    "type": "alert_history_response",
                    "data": alert_history
                }))
            
            elif message_type == "acknowledge_alert":
                # 경고 승인
                alert_id = data.get("alert_id")
                
                if alert_id:
                    success = await self.performance_dashboard.acknowledge_alert(alert_id)
                    
                    await websocket.send_text(json.dumps({
                        "type": "alert_acknowledged",
                        "data": {
                            "alert_id": alert_id,
                            "success": success
                        }
                    }))
            
            elif message_type == "export_performance_report":
                # 성능 리포트 내보내기 요청
                format_type = data.get("format", "json")
                time_range_hours = data.get("time_range_hours", 24)
                
                report = await self.performance_dashboard.export_performance_report(
                    format_type, time_range_hours
                )
                
                await websocket.send_text(json.dumps({
                    "type": "performance_report_ready",
                    "data": {
                        "format": format_type,
                        "report": report
                    }
                }))
            
            elif message_type == "ping":
                # 하트비트 응답
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                logger.warning(f"알 수 없는 성능 WebSocket 메시지 타입: {message_type}")
        
        except Exception as e:
            logger.error(f"성능 WebSocket 메시지 처리 실패: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def _handle_mcp_websocket_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ):
        """MCP WebSocket 메시지 처리"""
        
        message_type = message.get("type")
        data = message.get("data", {})
        
        try:
            if message_type == "request_tool_details":
                # 특정 도구 상세 정보 요청
                tool_name = data.get("tool_name")
                
                if tool_name:
                    details = await self.mcp_monitor.get_tool_details(tool_name)
                    
                    await websocket.send_text(json.dumps({
                        "type": "tool_details_response",
                        "data": {
                            "tool_name": tool_name,
                            "details": details
                        }
                    }))
            
            elif message_type == "trigger_tool_evolution":
                # 도구 진화 트리거
                tool_name = data.get("tool_name")
                evolution_strategy = data.get("strategy", "general_optimization")
                
                if tool_name:
                    success = await self.mcp_monitor.trigger_manual_evolution(
                        tool_name, evolution_strategy
                    )
                    
                    await websocket.send_text(json.dumps({
                        "type": "tool_evolution_triggered",
                        "data": {
                            "tool_name": tool_name,
                            "strategy": evolution_strategy,
                            "success": success
                        }
                    }))
            
            elif message_type == "request_execution_history":
                # 실행 히스토리 요청
                tool_name = data.get("tool_name")
                limit = data.get("limit", 100)
                
                history = await self.mcp_monitor.get_tool_execution_history(
                    tool_name, limit
                )
                
                await websocket.send_text(json.dumps({
                    "type": "execution_history_response",
                    "data": {
                        "tool_name": tool_name,
                        "history": history
                    }
                }))
            
            elif message_type == "request_network_update":
                # 네트워크 시각화 업데이트 요청
                updated_network = await self.mcp_monitor.get_network_visualization_data()
                
                await websocket.send_text(json.dumps({
                    "type": "network_update_response",
                    "data": updated_network
                }))
            
            elif message_type == "configure_tool_monitoring":
                # 도구 모니터링 설정
                tool_name = data.get("tool_name")
                monitoring_config = data.get("config", {})
                
                if tool_name:
                    success = await self.mcp_monitor.configure_tool_monitoring(
                        tool_name, monitoring_config
                    )
                    
                    await websocket.send_text(json.dumps({
                        "type": "tool_monitoring_configured",
                        "data": {
                            "tool_name": tool_name,
                            "success": success
                        }
                    }))
            
            elif message_type == "ping":
                # 하트비트 응답
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                logger.warning(f"알 수 없는 MCP WebSocket 메시지 타입: {message_type}")
        
        except Exception as e:
            logger.error(f"MCP WebSocket 메시지 처리 실패: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))