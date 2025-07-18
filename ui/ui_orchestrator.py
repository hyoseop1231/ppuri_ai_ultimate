"""
UI Orchestrator - 차세대 UI/UX 시스템 총괄 관리자

모든 시스템 컴포넌트를 통합하여 실시간 인터랙티브 
사용자 인터페이스를 제공하는 통합 조정 시스템.

Features:
- 실시간 THINK 블록 시각화
- 스트리밍 대화 인터페이스
- 지식 그래프 탐색기
- 성능 대시보드
- MCP 도구 생태계 모니터링
- 한국어 최적화 UI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class UISession:
    """UI 세션"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    ui_preferences: Dict[str, Any] = field(default_factory=dict)
    active_components: List[str] = field(default_factory=list)


@dataclass
class ThinkBlockState:
    """THINK 블록 상태"""
    session_id: str
    current_level: str  # think, megathink, ultrathink
    content: str
    progress: float  # 0.0 ~ 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMessage:
    """대화 메시지"""
    id: str
    session_id: str
    content: str
    role: str  # user, assistant, system
    timestamp: datetime
    think_blocks: List[ThinkBlockState] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    component: str
    metric_type: str
    value: float
    timestamp: datetime
    unit: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphVisualization:
    """지식 그래프 시각화 데이터"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: str = "force"
    filters: Dict[str, Any] = field(default_factory=dict)
    viewport: Dict[str, Any] = field(default_factory=dict)


class UIOrchestrator:
    """
    차세대 UI/UX 시스템 오케스트레이터
    
    모든 시스템 컴포넌트를 통합하여 실시간 반응형 
    사용자 인터페이스를 제공하는 중앙 조정 시스템.
    """
    
    def __init__(
        self,
        config_manager,
        conversational_engine,
        korean_optimizer,
        graph_manager,
        rag_orchestrator,
        mcp_orchestrator,
        think_block_manager
    ):
        self.config_manager = config_manager
        self.conversational_engine = conversational_engine
        self.korean_optimizer = korean_optimizer
        self.graph_manager = graph_manager
        self.rag_orchestrator = rag_orchestrator
        self.mcp_orchestrator = mcp_orchestrator
        self.think_block_manager = think_block_manager
        
        # UI 세션 관리
        self.active_sessions: Dict[str, UISession] = {}
        
        # 실시간 데이터 스트림
        self.message_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.think_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # WebSocket 연결 관리
        self.websocket_connections: Dict[str, List[Any]] = defaultdict(list)
        
        # 이벤트 리스너
        self.event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        
        # UI 컴포넌트 상태
        self.component_states = {
            "chat_interface": {"active": True, "mode": "streaming"},
            "think_visualizer": {"active": True, "show_all_levels": True},
            "knowledge_explorer": {"active": False, "layout": "force"},
            "performance_dashboard": {"active": False, "refresh_interval": 5},
            "mcp_monitor": {"active": False, "show_inactive": False}
        }
        
        # 실시간 업데이트 태스크
        self.update_tasks: List[asyncio.Task] = []
        
        logger.info("UI Orchestrator 초기화 완료")
    
    async def initialize(self):
        """UI 시스템 초기화"""
        
        logger.info("UI 시스템 초기화 중...")
        
        # 1. 이벤트 리스너 등록
        await self._register_event_listeners()
        
        # 2. 실시간 업데이트 태스크 시작
        await self._start_update_tasks()
        
        # 3. WebSocket 서버 준비
        await self._setup_websocket_server()
        
        # 4. 정적 파일 및 템플릿 준비
        await self._prepare_static_assets()
        
        logger.info("✅ UI 시스템 초기화 완료")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """새 UI 세션 생성"""
        
        session_id = str(uuid.uuid4())
        
        session = UISession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ui_preferences=preferences or self._get_default_preferences(),
            active_components=["chat_interface", "think_visualizer"]
        )
        
        self.active_sessions[session_id] = session
        
        # 대화 엔진에 세션 생성
        await self.conversational_engine.start_conversation(
            session_id=session_id,
            initial_context={"ui_session": True, "user_id": user_id}
        )
        
        logger.info(f"UI 세션 생성: {session_id}")
        return session_id
    
    async def process_user_message(
        self,
        session_id: str,
        content: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        사용자 메시지 처리 및 실시간 응답 스트리밍
        
        Args:
            session_id: UI 세션 ID
            content: 사용자 메시지 내용
            attachments: 첨부 파일들
            
        Yields:
            Dict[str, Any]: 실시간 응답 데이터
        """
        
        if session_id not in self.active_sessions:
            yield {"error": "유효하지 않은 세션"}
            return
        
        session = self.active_sessions[session_id]
        session.last_activity = datetime.now()
        
        try:
            # 1. 사용자 메시지 저장
            user_message = ConversationMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                content=content,
                role="user",
                timestamp=datetime.now(),
                attachments=attachments or []
            )
            
            self.message_streams[session_id].append(user_message)
            
            # 사용자 메시지 전송
            yield {
                "type": "user_message",
                "data": {
                    "id": user_message.id,
                    "content": content,
                    "timestamp": user_message.timestamp.isoformat()
                }
            }
            
            # 2. THINK 블록 활성화
            yield {
                "type": "think_start",
                "data": {"level": "think", "content": "사용자 요청 분석 중..."}
            }
            
            # 3. 대화 엔진으로 스트리밍 처리
            assistant_message_parts = []
            current_think_level = "think"
            
            async for result in self.conversational_engine.chat(
                session_id, content, stream=True
            ):
                
                # THINK 블록 처리
                if hasattr(result, 'think_block') and result.think_block:
                    think_state = ThinkBlockState(
                        session_id=session_id,
                        current_level=result.think_block.level,
                        content=result.think_block.content,
                        progress=result.think_block.progress,
                        timestamp=datetime.now()
                    )
                    
                    self.think_streams[session_id].append(think_state)
                    
                    yield {
                        "type": "think_update",
                        "data": {
                            "level": think_state.current_level,
                            "content": think_state.content,
                            "progress": think_state.progress,
                            "timestamp": think_state.timestamp.isoformat()
                        }
                    }
                    
                    current_think_level = result.think_block.level
                
                # 응답 스트리밍
                if hasattr(result, 'response') and result.response:
                    assistant_message_parts.append(result.response)
                    
                    yield {
                        "type": "response_chunk",
                        "data": {
                            "content": result.response,
                            "accumulated": "".join(assistant_message_parts)
                        }
                    }
            
            # 4. 최종 응답 메시지 저장
            final_response = "".join(assistant_message_parts)
            
            assistant_message = ConversationMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                content=final_response,
                role="assistant",
                timestamp=datetime.now(),
                think_blocks=list(self.think_streams[session_id])[-5:],  # 최근 5개
                metadata={
                    "processing_time": (datetime.now() - user_message.timestamp).total_seconds(),
                    "think_levels_used": [current_think_level]
                }
            )
            
            self.message_streams[session_id].append(assistant_message)
            
            # 5. 지식 그래프 업데이트 (백그라운드)
            asyncio.create_task(self._update_knowledge_graph(session_id, content, final_response))
            
            # 6. 최종 완료 신호
            yield {
                "type": "response_complete",
                "data": {
                    "message_id": assistant_message.id,
                    "total_length": len(final_response),
                    "processing_time": assistant_message.metadata["processing_time"]
                }
            }
            
        except Exception as e:
            logger.error(f"메시지 처리 실패 ({session_id}): {e}")
            yield {
                "type": "error",
                "data": {"message": f"처리 중 오류가 발생했습니다: {str(e)}"}
            }
    
    async def get_knowledge_graph_data(
        self,
        session_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> KnowledgeGraphVisualization:
        """지식 그래프 시각화 데이터 조회"""
        
        try:
            # 노드 조회
            nodes_query = """
            MATCH (n)
            WHERE n.session_id = $session_id OR n.global = true
            RETURN n
            LIMIT 100
            """
            
            nodes_result = await self.graph_manager.execute_query(
                nodes_query, {"session_id": session_id}
            )
            
            # 관계 조회
            edges_query = """
            MATCH (a)-[r]->(b)
            WHERE (a.session_id = $session_id OR a.global = true)
              AND (b.session_id = $session_id OR b.global = true)
            RETURN a, r, b
            LIMIT 200
            """
            
            edges_result = await self.graph_manager.execute_query(
                edges_query, {"session_id": session_id}
            )
            
            # 시각화 형태로 변환
            nodes = []
            for record in nodes_result:
                node = record['n']
                nodes.append({
                    "id": node.id,
                    "label": node.properties.get("name", ""),
                    "type": node.labels[0] if node.labels else "Unknown",
                    "properties": node.properties,
                    "size": min(100, max(20, len(node.properties.get("name", "")) * 3))
                })
            
            edges = []
            for record in edges_result:
                edge = record['r']
                edges.append({
                    "id": edge.id,
                    "source": record['a'].id,
                    "target": record['b'].id,
                    "type": edge.type,
                    "properties": edge.properties,
                    "weight": edge.properties.get("strength", 0.5)
                })
            
            return KnowledgeGraphVisualization(
                nodes=nodes,
                edges=edges,
                layout="force",
                filters=filters or {},
                viewport={"zoom": 1.0, "center": [0, 0]}
            )
            
        except Exception as e:
            logger.error(f"지식 그래프 데이터 조회 실패: {e}")
            return KnowledgeGraphVisualization(nodes=[], edges=[])
    
    async def get_performance_dashboard_data(
        self,
        time_range: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """성능 대시보드 데이터 조회"""
        
        cutoff_time = datetime.now() - time_range
        
        dashboard_data = {
            "overview": {
                "active_sessions": len(self.active_sessions),
                "total_messages": sum(len(stream) for stream in self.message_streams.values()),
                "avg_response_time": 0.0,
                "think_blocks_generated": sum(len(stream) for stream in self.think_streams.values())
            },
            "components": {},
            "realtime_metrics": [],
            "system_health": {}
        }
        
        try:
            # 컴포넌트별 성능
            dashboard_data["components"] = {
                "conversational_engine": await self._get_component_metrics("conversational_engine"),
                "rag_orchestrator": await self._get_component_metrics("rag_orchestrator"),
                "graph_manager": await self._get_component_metrics("graph_manager"),
                "mcp_orchestrator": await self._get_component_metrics("mcp_orchestrator"),
                "korean_optimizer": await self._get_component_metrics("korean_optimizer")
            }
            
            # 실시간 메트릭
            for component, metrics in self.performance_streams.items():
                recent_metrics = [
                    m for m in metrics 
                    if m.timestamp >= cutoff_time
                ]
                
                if recent_metrics:
                    dashboard_data["realtime_metrics"].append({
                        "component": component,
                        "data_points": [
                            {
                                "timestamp": m.timestamp.isoformat(),
                                "value": m.value,
                                "type": m.metric_type
                            }
                            for m in recent_metrics[-50:]  # 최근 50개
                        ]
                    })
            
            # 시스템 헬스 체크
            dashboard_data["system_health"] = await self._check_system_health()
            
        except Exception as e:
            logger.error(f"대시보드 데이터 조회 실패: {e}")
        
        return dashboard_data
    
    async def get_mcp_ecosystem_status(self) -> Dict[str, Any]:
        """MCP 생태계 상태 조회"""
        
        try:
            ecosystem_status = await self.mcp_orchestrator.get_ecosystem_status()
            
            # UI 친화적 형태로 변환
            ui_status = {
                "summary": {
                    "total_tools": ecosystem_status.get("total_tools", 0),
                    "active_tools": ecosystem_status.get("active_tools", 0),
                    "success_rate": ecosystem_status.get("avg_success_rate", 0.0),
                    "evolution_count": ecosystem_status.get("evolution_count", 0)
                },
                "tools_by_category": ecosystem_status.get("tools_by_category", {}),
                "top_performing_tools": ecosystem_status.get("top_performing_tools", []),
                "recent_evolutions": ecosystem_status.get("recent_evolutions", []),
                "tool_network": await self._build_tool_network_visualization()
            }
            
            return ui_status
            
        except Exception as e:
            logger.error(f"MCP 생태계 상태 조회 실패: {e}")
            return {"error": str(e)}
    
    async def update_component_state(
        self,
        component: str,
        state_updates: Dict[str, Any]
    ) -> bool:
        """UI 컴포넌트 상태 업데이트"""
        
        if component in self.component_states:
            self.component_states[component].update(state_updates)
            
            # 모든 세션에 상태 변경 알림
            await self._broadcast_component_update(component, state_updates)
            
            logger.debug(f"컴포넌트 상태 업데이트: {component}")
            return True
        
        return False
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """기본 UI 설정"""
        
        return {
            "theme": "dark",
            "language": "ko",
            "font_size": "medium",
            "animations": True,
            "auto_scroll": True,
            "think_block_style": "expandable",
            "graph_layout": "force",
            "dashboard_refresh": 5,
            "show_performance_overlay": False
        }
    
    async def _register_event_listeners(self):
        """이벤트 리스너 등록"""
        
        # 대화 엔진 이벤트
        self.event_listeners["message_processed"].append(self._on_message_processed)
        self.event_listeners["think_block_updated"].append(self._on_think_block_updated)
        
        # 그래프 이벤트
        self.event_listeners["node_created"].append(self._on_node_created)
        self.event_listeners["relationship_created"].append(self._on_relationship_created)
        
        # MCP 이벤트
        self.event_listeners["tool_executed"].append(self._on_tool_executed)
        self.event_listeners["tool_evolved"].append(self._on_tool_evolved)
        
        # 성능 이벤트
        self.event_listeners["performance_metric"].append(self._on_performance_metric)
    
    async def _start_update_tasks(self):
        """실시간 업데이트 태스크 시작"""
        
        # 성능 메트릭 수집
        self.update_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        # 세션 정리
        self.update_tasks.append(
            asyncio.create_task(self._session_cleanup_loop())
        )
        
        # WebSocket 상태 체크
        self.update_tasks.append(
            asyncio.create_task(self._websocket_heartbeat_loop())
        )
    
    async def _performance_monitoring_loop(self):
        """성능 모니터링 루프"""
        
        while True:
            try:
                # 각 컴포넌트 성능 수집
                await self._collect_performance_metrics()
                await asyncio.sleep(1)  # 1초마다
                
            except Exception as e:
                logger.error(f"성능 모니터링 오류: {e}")
                await asyncio.sleep(5)
    
    async def _session_cleanup_loop(self):
        """세션 정리 루프"""
        
        while True:
            try:
                # 비활성 세션 정리 (1시간 이상 비활성)
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                inactive_sessions = [
                    session_id for session_id, session in self.active_sessions.items()
                    if session.last_activity < cutoff_time
                ]
                
                for session_id in inactive_sessions:
                    await self._cleanup_session(session_id)
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                logger.error(f"세션 정리 오류: {e}")
                await asyncio.sleep(300)
    
    async def _websocket_heartbeat_loop(self):
        """WebSocket 하트비트 루프"""
        
        while True:
            try:
                # 연결된 모든 WebSocket에 하트비트 전송
                for session_id, connections in self.websocket_connections.items():
                    active_connections = []
                    
                    for conn in connections:
                        try:
                            await conn.send(json.dumps({
                                "type": "heartbeat",
                                "timestamp": datetime.now().isoformat()
                            }))
                            active_connections.append(conn)
                        except:
                            # 연결 끊어진 경우 제거
                            pass
                    
                    self.websocket_connections[session_id] = active_connections
                
                await asyncio.sleep(30)  # 30초마다
                
            except Exception as e:
                logger.error(f"WebSocket 하트비트 오류: {e}")
                await asyncio.sleep(30)
    
    async def _collect_performance_metrics(self):
        """성능 메트릭 수집"""
        
        timestamp = datetime.now()
        
        # UI 세션 수
        self.performance_streams["ui_orchestrator"].append(
            PerformanceMetrics(
                component="ui_orchestrator",
                metric_type="active_sessions",
                value=len(self.active_sessions),
                timestamp=timestamp,
                unit="count"
            )
        )
        
        # 메시지 스트림 크기
        total_messages = sum(len(stream) for stream in self.message_streams.values())
        self.performance_streams["ui_orchestrator"].append(
            PerformanceMetrics(
                component="ui_orchestrator",
                metric_type="total_messages",
                value=total_messages,
                timestamp=timestamp,
                unit="count"
            )
        )
        
        # WebSocket 연결 수
        total_connections = sum(len(conns) for conns in self.websocket_connections.values())
        self.performance_streams["ui_orchestrator"].append(
            PerformanceMetrics(
                component="ui_orchestrator",
                metric_type="websocket_connections",
                value=total_connections,
                timestamp=timestamp,
                unit="count"
            )
        )
    
    async def _update_knowledge_graph(self, session_id: str, user_message: str, assistant_response: str):
        """지식 그래프 업데이트 (백그라운드)"""
        
        try:
            # 지식 추출
            from ..knowledge_graph.knowledge_extractor import KnowledgeExtractor
            
            extractor = KnowledgeExtractor(self.korean_optimizer)
            
            # 사용자 메시지에서 지식 추출
            user_knowledge = await extractor.extract_knowledge(
                user_message, {"session_id": session_id, "role": "user"}
            )
            
            # 어시스턴트 응답에서 지식 추출
            assistant_knowledge = await extractor.extract_knowledge(
                assistant_response, {"session_id": session_id, "role": "assistant"}
            )
            
            # 관계 구축
            from ..knowledge_graph.relationship_builder import RelationshipBuilder, ConversationContext
            
            builder = RelationshipBuilder(self.graph_manager, self.korean_optimizer)
            
            context = ConversationContext(
                session_id=session_id,
                user_id=self.active_sessions[session_id].user_id,
                message_sequence=len(self.message_streams[session_id]),
                timestamp=datetime.now(),
                domain=None  # 자동 감지됨
            )
            
            # 관계 후보 생성
            all_entities = user_knowledge.entities + assistant_knowledge.entities
            all_relations = user_knowledge.relations + assistant_knowledge.relations
            all_concepts = user_knowledge.concepts + assistant_knowledge.concepts
            
            relationship_candidates = await builder.build_relationships(
                all_entities, all_relations, all_concepts, context
            )
            
            # 관계 저장
            await builder.persist_relationships(relationship_candidates)
            
            logger.debug(f"지식 그래프 업데이트 완료: {len(relationship_candidates)}개 관계")
            
        except Exception as e:
            logger.error(f"지식 그래프 업데이트 실패: {e}")
    
    async def _get_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """컴포넌트별 성능 메트릭 조회"""
        
        if component_name == "conversational_engine":
            # 대화 엔진 통계 (더미 데이터)
            return {
                "avg_response_time": 1.2,
                "total_conversations": len(self.active_sessions),
                "success_rate": 0.95,
                "think_blocks_per_response": 2.3
            }
        elif component_name == "rag_orchestrator":
            return await self.rag_orchestrator.get_performance_statistics()
        elif component_name == "mcp_orchestrator":
            status = await self.mcp_orchestrator.get_ecosystem_status()
            return {
                "total_tools": status.get("total_tools", 0),
                "active_tools": status.get("active_tools", 0),
                "avg_success_rate": status.get("avg_success_rate", 0.0),
                "total_executions": status.get("total_executions", 0)
            }
        else:
            return {"status": "unknown"}
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """시스템 헬스 체크"""
        
        health = {
            "overall": "healthy",
            "components": {},
            "alerts": [],
            "uptime": (datetime.now() - datetime.now()).total_seconds()  # 실제로는 시작 시간 필요
        }
        
        # 각 컴포넌트 헬스 체크
        components = [
            ("conversational_engine", self.conversational_engine),
            ("graph_manager", self.graph_manager), 
            ("rag_orchestrator", self.rag_orchestrator),
            ("mcp_orchestrator", self.mcp_orchestrator)
        ]
        
        for name, component in components:
            try:
                # 간단한 헬스 체크 (hasattr으로 health_check 메소드 확인)
                if hasattr(component, 'health_check'):
                    health_status = await component.health_check()
                else:
                    health_status = "unknown"
                
                health["components"][name] = health_status
                
            except Exception as e:
                health["components"][name] = "unhealthy"
                health["alerts"].append(f"{name}: {str(e)}")
        
        # 전체 헬스 상태 결정
        unhealthy_count = sum(1 for status in health["components"].values() if status == "unhealthy")
        if unhealthy_count > 0:
            health["overall"] = "degraded" if unhealthy_count < len(components) // 2 else "unhealthy"
        
        return health
    
    async def _build_tool_network_visualization(self) -> Dict[str, Any]:
        """도구 네트워크 시각화 데이터 구축"""
        
        # MCP 도구들 간의 연결 관계 시각화
        try:
            ecosystem_status = await self.mcp_orchestrator.get_ecosystem_status()
            tools_by_category = ecosystem_status.get("tools_by_category", {})
            
            nodes = []
            edges = []
            node_id = 0
            
            # 카테고리별 노드 생성
            category_nodes = {}
            for category, tool_list in tools_by_category.items():
                category_node_id = f"cat_{node_id}"
                category_nodes[category] = category_node_id
                
                nodes.append({
                    "id": category_node_id,
                    "label": category,
                    "type": "category",
                    "size": len(tool_list) * 10 + 20,
                    "color": self._get_category_color(category)
                })
                node_id += 1
                
                # 도구 노드들
                for tool_name in tool_list[:5]:  # 최대 5개만
                    tool_node_id = f"tool_{node_id}"
                    nodes.append({
                        "id": tool_node_id,
                        "label": tool_name,
                        "type": "tool",
                        "size": 15,
                        "color": "#4A90E2"
                    })
                    
                    # 카테고리-도구 연결
                    edges.append({
                        "id": f"edge_{len(edges)}",
                        "source": category_node_id,
                        "target": tool_node_id,
                        "type": "contains"
                    })
                    node_id += 1
            
            return {"nodes": nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"도구 네트워크 시각화 구축 실패: {e}")
            return {"nodes": [], "edges": []}
    
    def _get_category_color(self, category: str) -> str:
        """카테고리별 색상 반환"""
        
        color_map = {
            "search": "#FF6B6B",
            "generation": "#4ECDC4", 
            "knowledge": "#45B7D1",
            "language": "#96CEB4",
            "optimization": "#FECA57",
            "industry": "#FF9FF3",
            "mcp_integration": "#54A0FF"
        }
        
        return color_map.get(category, "#95A5A6")
    
    async def _broadcast_component_update(self, component: str, updates: Dict[str, Any]):
        """컴포넌트 업데이트를 모든 연결된 클라이언트에 브로드캐스트"""
        
        message = {
            "type": "component_update",
            "data": {
                "component": component,
                "updates": updates,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # 모든 WebSocket 연결에 전송
        for session_id, connections in self.websocket_connections.items():
            for conn in connections:
                try:
                    await conn.send(json.dumps(message))
                except:
                    pass  # 연결 끊어진 경우 무시
    
    async def _cleanup_session(self, session_id: str):
        """세션 정리"""
        
        try:
            # 대화 엔진 세션 종료
            await self.conversational_engine.end_conversation(session_id)
            
            # UI 세션 제거
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # 스트림 정리 (일부만 유지)
            if session_id in self.message_streams:
                recent_messages = list(self.message_streams[session_id])[-50:]  # 최근 50개만
                self.message_streams[session_id] = deque(recent_messages, maxlen=1000)
            
            if session_id in self.think_streams:
                del self.think_streams[session_id]
            
            # WebSocket 연결 정리
            if session_id in self.websocket_connections:
                for conn in self.websocket_connections[session_id]:
                    try:
                        await conn.close()
                    except:
                        pass
                del self.websocket_connections[session_id]
            
            logger.info(f"세션 정리 완료: {session_id}")
            
        except Exception as e:
            logger.error(f"세션 정리 실패 ({session_id}): {e}")
    
    # 이벤트 핸들러들
    async def _on_message_processed(self, event_data: Dict[str, Any]):
        """메시지 처리 이벤트 핸들러"""
        
        try:
            session_id = event_data.get("session_id")
            message_id = event_data.get("message_id")
            processing_time = event_data.get("processing_time", 0.0)
            
            if not session_id:
                return
            
            # 성능 메트릭 수집
            self.performance_streams["conversational_engine"].append(
                PerformanceMetrics(
                    component="conversational_engine",
                    metric_type="message_processing_time",
                    value=processing_time,
                    timestamp=datetime.now(),
                    unit="seconds",
                    context={"session_id": session_id, "message_id": message_id}
                )
            )
            
            # WebSocket으로 완료 알림 전송
            if session_id in self.websocket_connections:
                message = {
                    "type": "message_processed",
                    "data": {
                        "message_id": message_id,
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                await self._broadcast_to_session(session_id, message)
            
            logger.debug(f"메시지 처리 완료: {message_id} ({processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"메시지 처리 이벤트 핸들러 실패: {e}")
    
    async def _on_think_block_updated(self, event_data: Dict[str, Any]):
        """THINK 블록 업데이트 이벤트 핸들러"""
        
        try:
            session_id = event_data.get("session_id")
            think_level = event_data.get("level", "think")
            content = event_data.get("content", "")
            progress = event_data.get("progress", 0.0)
            
            if not session_id:
                return
            
            # THINK 블록 상태 업데이트
            think_state = ThinkBlockState(
                session_id=session_id,
                current_level=think_level,
                content=content,
                progress=progress,
                timestamp=datetime.now(),
                metadata=event_data.get("metadata", {})
            )
            
            self.think_streams[session_id].append(think_state)
            
            # WebSocket으로 실시간 업데이트 전송
            if session_id in self.websocket_connections:
                message = {
                    "type": "think_block_update",
                    "data": {
                        "level": think_level,
                        "content": content,
                        "progress": progress,
                        "timestamp": think_state.timestamp.isoformat(),
                        "metadata": think_state.metadata
                    }
                }
                
                await self._broadcast_to_session(session_id, message)
            
            # 성능 통계 업데이트
            self.performance_streams["think_block_manager"].append(
                PerformanceMetrics(
                    component="think_block_manager",
                    metric_type="think_block_updates",
                    value=1,
                    timestamp=datetime.now(),
                    unit="count",
                    context={"level": think_level, "session_id": session_id}
                )
            )
            
            logger.debug(f"THINK 블록 업데이트: {session_id} [{think_level}] {progress:.1%}")
            
        except Exception as e:
            logger.error(f"THINK 블록 업데이트 이벤트 핸들러 실패: {e}")
    
    async def _on_node_created(self, event_data: Dict[str, Any]):
        """노드 생성 이벤트 핸들러"""
        
        try:
            session_id = event_data.get("session_id")
            node_id = event_data.get("node_id")
            node_type = event_data.get("node_type", "Entity")
            properties = event_data.get("properties", {})
            
            if not session_id or not node_id:
                return
            
            # 지식 그래프 통계 업데이트
            self.performance_streams["graph_manager"].append(
                PerformanceMetrics(
                    component="graph_manager",
                    metric_type="nodes_created",
                    value=1,
                    timestamp=datetime.now(),
                    unit="count",
                    context={
                        "session_id": session_id,
                        "node_type": node_type,
                        "node_id": node_id
                    }
                )
            )
            
            # 세션의 지식 그래프가 활성화된 경우 실시간 업데이트
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if "knowledge_explorer" in session.active_components:
                    
                    # WebSocket으로 노드 생성 알림
                    if session_id in self.websocket_connections:
                        message = {
                            "type": "knowledge_node_created",
                            "data": {
                                "node_id": node_id,
                                "node_type": node_type,
                                "properties": properties,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        
                        await self._broadcast_to_session(session_id, message)
            
            logger.debug(f"지식 노드 생성: {session_id} - {node_type} ({node_id})")
            
        except Exception as e:
            logger.error(f"노드 생성 이벤트 핸들러 실패: {e}")
    
    async def _on_relationship_created(self, event_data: Dict[str, Any]):
        """관계 생성 이벤트 핸들러"""
        
        try:
            session_id = event_data.get("session_id")
            source_id = event_data.get("source_id")
            target_id = event_data.get("target_id")
            relationship_type = event_data.get("relationship_type")
            strength = event_data.get("strength", 0.5)
            
            if not all([session_id, source_id, target_id, relationship_type]):
                return
            
            # 관계 생성 통계
            self.performance_streams["graph_manager"].append(
                PerformanceMetrics(
                    component="graph_manager",
                    metric_type="relationships_created",
                    value=1,
                    timestamp=datetime.now(),
                    unit="count",
                    context={
                        "session_id": session_id,
                        "relationship_type": relationship_type,
                        "strength": strength
                    }
                )
            )
            
            # 지식 그래프 활성 세션에 업데이트 알림
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if "knowledge_explorer" in session.active_components:
                    
                    if session_id in self.websocket_connections:
                        message = {
                            "type": "knowledge_relationship_created",
                            "data": {
                                "source_id": source_id,
                                "target_id": target_id,
                                "relationship_type": relationship_type,
                                "strength": strength,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        
                        await self._broadcast_to_session(session_id, message)
            
            logger.debug(f"지식 관계 생성: {session_id} - {source_id} -[{relationship_type}]-> {target_id}")
            
        except Exception as e:
            logger.error(f"관계 생성 이벤트 핸들러 실패: {e}")
    
    async def _on_tool_executed(self, event_data: Dict[str, Any]):
        """도구 실행 이벤트 핸들러"""
        
        try:
            tool_name = event_data.get("tool_name")
            execution_time = event_data.get("execution_time", 0.0)
            success = event_data.get("success", False)
            session_id = event_data.get("session_id")
            result_summary = event_data.get("result_summary", "")
            
            if not tool_name:
                return
            
            # 도구 실행 통계
            self.performance_streams["mcp_orchestrator"].append(
                PerformanceMetrics(
                    component="mcp_orchestrator",
                    metric_type="tool_execution_time",
                    value=execution_time,
                    timestamp=datetime.now(),
                    unit="seconds",
                    context={
                        "tool_name": tool_name,
                        "success": success,
                        "session_id": session_id
                    }
                )
            )
            
            # 도구 성공률 통계
            self.performance_streams["mcp_orchestrator"].append(
                PerformanceMetrics(
                    component="mcp_orchestrator",
                    metric_type="tool_success_rate",
                    value=1.0 if success else 0.0,
                    timestamp=datetime.now(),
                    unit="ratio",
                    context={"tool_name": tool_name}
                )
            )
            
            # MCP 모니터 활성 세션에 알림
            for session_id_key, session in self.active_sessions.items():
                if "mcp_monitor" in session.active_components:
                    
                    if session_id_key in self.websocket_connections:
                        message = {
                            "type": "mcp_tool_executed",
                            "data": {
                                "tool_name": tool_name,
                                "execution_time": execution_time,
                                "success": success,
                                "result_summary": result_summary,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        
                        await self._broadcast_to_session(session_id_key, message)
            
            logger.debug(f"MCP 도구 실행: {tool_name} ({'성공' if success else '실패'}, {execution_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"도구 실행 이벤트 핸들러 실패: {e}")
    
    async def _on_tool_evolved(self, event_data: Dict[str, Any]):
        """도구 진화 이벤트 핸들러"""
        
        try:
            tool_name = event_data.get("tool_name")
            evolution_type = event_data.get("evolution_type", "optimization")
            improvement_metrics = event_data.get("improvement_metrics", {})
            new_capabilities = event_data.get("new_capabilities", [])
            
            if not tool_name:
                return
            
            # 도구 진화 통계
            self.performance_streams["mcp_orchestrator"].append(
                PerformanceMetrics(
                    component="mcp_orchestrator",
                    metric_type="tool_evolutions",
                    value=1,
                    timestamp=datetime.now(),
                    unit="count",
                    context={
                        "tool_name": tool_name,
                        "evolution_type": evolution_type,
                        "improvement_count": len(improvement_metrics)
                    }
                )
            )
            
            # 전체 시스템에 진화 알림 (중요 이벤트)
            evolution_message = {
                "type": "mcp_tool_evolved",
                "data": {
                    "tool_name": tool_name,
                    "evolution_type": evolution_type,
                    "improvement_metrics": improvement_metrics,
                    "new_capabilities": new_capabilities,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # 모든 활성 세션에 브로드캐스트
            for session_id, connections in self.websocket_connections.items():
                await self._broadcast_to_session(session_id, evolution_message)
            
            logger.info(f"🚀 MCP 도구 진화: {tool_name} [{evolution_type}] - {len(new_capabilities)}개 새 기능")
            
        except Exception as e:
            logger.error(f"도구 진화 이벤트 핸들러 실패: {e}")
    
    async def _on_performance_metric(self, event_data: Dict[str, Any]):
        """성능 메트릭 이벤트 핸들러"""
        
        try:
            component = event_data.get("component")
            metric_type = event_data.get("metric_type")
            value = event_data.get("value")
            timestamp = event_data.get("timestamp")
            
            if not all([component, metric_type, value is not None]):
                return
            
            # 타임스탬프 처리
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif timestamp is None:
                timestamp = datetime.now()
            
            # 성능 메트릭 저장
            metric = PerformanceMetrics(
                component=component,
                metric_type=metric_type,
                value=float(value),
                timestamp=timestamp,
                unit=event_data.get("unit", ""),
                context=event_data.get("context", {})
            )
            
            self.performance_streams[component].append(metric)
            
            # 임계값 체크 (예: 응답 시간이 5초 초과)
            alert_thresholds = {
                "response_time": 5.0,
                "error_rate": 0.1,
                "memory_usage": 0.8,
                "cpu_usage": 0.8
            }
            
            threshold = alert_thresholds.get(metric_type)
            if threshold and value > threshold:
                # 경고 알림
                alert_message = {
                    "type": "performance_alert",
                    "data": {
                        "component": component,
                        "metric_type": metric_type,
                        "current_value": value,
                        "threshold": threshold,
                        "severity": "warning" if value < threshold * 1.5 else "critical",
                        "timestamp": timestamp.isoformat()
                    }
                }
                
                # 성능 대시보드 활성 세션에 알림
                for session_id, session in self.active_sessions.items():
                    if "performance_dashboard" in session.active_components:
                        if session_id in self.websocket_connections:
                            await self._broadcast_to_session(session_id, alert_message)
                
                logger.warning(f"⚠️ 성능 경고: {component}.{metric_type} = {value} (임계값: {threshold})")
            
            # 실시간 성능 업데이트 (일부 메트릭만)
            if metric_type in ["response_time", "active_sessions", "tool_executions"]:
                update_message = {
                    "type": "performance_metric_update",
                    "data": {
                        "component": component,
                        "metric_type": metric_type,
                        "value": value,
                        "timestamp": timestamp.isoformat()
                    }
                }
                
                # 성능 대시보드 활성 세션에 업데이트
                for session_id, session in self.active_sessions.items():
                    if "performance_dashboard" in session.active_components:
                        if session_id in self.websocket_connections:
                            await self._broadcast_to_session(session_id, update_message)
            
        except Exception as e:
            logger.error(f"성능 메트릭 이벤트 핸들러 실패: {e}")
    
    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """세션별 WebSocket 브로드캐스트 (내부 메소드)"""
        
        if session_id not in self.websocket_connections:
            return
        
        message_text = json.dumps(message)
        
        for connection in self.websocket_connections[session_id]:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.debug(f"WebSocket 전송 실패: {e}")
    
    async def _setup_websocket_server(self):
        """WebSocket 서버 설정"""
        # WebSocket 서버 설정은 실제 웹 프레임워크와 함께 구현
        logger.debug("WebSocket 서버 설정 준비 완료")
    
    async def _prepare_static_assets(self):
        """정적 파일 및 템플릿 준비"""
        # 정적 파일 준비는 실제 웹 서버와 함께 구현
        logger.debug("정적 자원 준비 완료")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "active_components": session.active_components,
            "preferences": session.ui_preferences,
            "message_count": len(self.message_streams[session_id]),
            "think_blocks_count": len(self.think_streams[session_id])
        }
    
    async def cleanup(self):
        """UI 시스템 정리"""
        
        # 모든 업데이트 태스크 취소
        for task in self.update_tasks:
            task.cancel()
        
        # 모든 세션 정리
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self._cleanup_session(session_id)
        
        logger.info("UI Orchestrator 정리 완료")