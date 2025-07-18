"""
Think Visualizer - THINK 블록 실시간 시각화 컴포넌트

AI의 사고 과정을 3단계(THINK, MEGATHINK, ULTRATHINK)로 
실시간 시각화하여 투명하고 이해하기 쉬운 AI 상호작용을 제공.

Features:
- 3단계 사고 레벨 시각화
- 실시간 진행 상황 표시
- 사고 트리 구조 표현
- 한국어 최적화 표시
- 인터랙티브 확장/축소
- 성능 메트릭 통합
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ThinkLevel:
    """사고 레벨 정의"""
    name: str
    icon: str
    color: str
    description: str
    priority: int


@dataclass
class ThinkBlock:
    """THINK 블록"""
    id: str
    session_id: str
    level: str  # think, megathink, ultrathink
    content: str
    progress: float  # 0.0 ~ 1.0
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, completed, error


@dataclass
class ThinkTree:
    """사고 트리 구조"""
    session_id: str
    root_blocks: List[str]
    all_blocks: Dict[str, ThinkBlock] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class VisualizationConfig:
    """시각화 설정"""
    show_all_levels: bool = True
    animation_speed: float = 1.0
    auto_expand: bool = True
    max_display_blocks: int = 50
    color_scheme: str = "default"
    layout_style: str = "tree"  # tree, flow, timeline


class ThinkVisualizer:
    """
    THINK 블록 실시간 시각화기
    
    AI의 사고 과정을 직관적이고 아름다운 형태로 
    실시간 시각화하여 사용자 이해를 돕는 컴포넌트.
    """
    
    def __init__(
        self,
        ui_orchestrator,
        korean_optimizer=None,
        max_history_size: int = 1000
    ):
        self.ui_orchestrator = ui_orchestrator
        self.korean_optimizer = korean_optimizer
        self.max_history_size = max_history_size
        
        # 사고 레벨 정의
        self.think_levels = self._initialize_think_levels()
        
        # 세션별 사고 트리
        self.think_trees: Dict[str, ThinkTree] = {}
        
        # 실시간 스트림
        self.active_streams: Dict[str, deque] = {}
        
        # 시각화 설정
        self.visualization_configs: Dict[str, VisualizationConfig] = {}
        
        # 성능 통계
        self.performance_stats = {
            "total_think_blocks": 0,
            "avg_think_duration": 0.0,
            "level_distribution": {"think": 0, "megathink": 0, "ultrathink": 0},
            "avg_blocks_per_session": 0.0
        }
        
        # WebSocket 연결 (think block 업데이트용)
        self.websocket_connections: Dict[str, List[Any]] = {}
        
        logger.info("Think Visualizer 초기화 완료")
    
    def _initialize_think_levels(self) -> Dict[str, ThinkLevel]:
        """사고 레벨 초기화"""
        
        return {
            "think": ThinkLevel(
                name="THINK",
                icon="🧠",
                color="#4A90E2",
                description="기본 분석 및 이해",
                priority=1
            ),
            "megathink": ThinkLevel(
                name="MEGATHINK", 
                icon="🚀",
                color="#E74C3C",
                description="복합 관계 및 최적화 고려",
                priority=2
            ),
            "ultrathink": ThinkLevel(
                name="ULTRATHINK",
                icon="⚡",
                color="#9B59B6",
                description="최종 통합 결론 및 실행 계획",
                priority=3
            )
        }
    
    async def initialize_session(
        self,
        session_id: str,
        config: Optional[VisualizationConfig] = None
    ):
        """세션 초기화"""
        
        # 사고 트리 생성
        self.think_trees[session_id] = ThinkTree(
            session_id=session_id,
            root_blocks=[]
        )
        
        # 실시간 스트림 생성
        self.active_streams[session_id] = deque(maxlen=self.max_history_size)
        
        # 시각화 설정
        self.visualization_configs[session_id] = config or VisualizationConfig()
        
        logger.debug(f"Think Visualizer 세션 초기화: {session_id}")
    
    async def create_think_block(
        self,
        session_id: str,
        level: str,
        content: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """새 THINK 블록 생성"""
        
        if session_id not in self.think_trees:
            await self.initialize_session(session_id)
        
        think_tree = self.think_trees[session_id]
        block_id = str(uuid.uuid4())
        
        # 한국어 최적화
        optimized_content = content
        if self.korean_optimizer:
            korean_result = await self.korean_optimizer.process_korean_text(content)
            optimized_content = korean_result.normalized_text
        
        # THINK 블록 생성
        think_block = ThinkBlock(
            id=block_id,
            session_id=session_id,
            level=level,
            content=optimized_content,
            progress=0.0,
            start_time=datetime.now(),
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        # 트리에 추가
        think_tree.all_blocks[block_id] = think_block
        think_tree.last_updated = datetime.now()
        
        # 부모-자식 관계 설정
        if parent_id and parent_id in think_tree.all_blocks:
            think_tree.all_blocks[parent_id].children_ids.append(block_id)
        else:
            think_tree.root_blocks.append(block_id)
        
        # 실시간 스트림에 추가
        self.active_streams[session_id].append({
            "type": "block_created",
            "block_id": block_id,
            "data": self._serialize_think_block(think_block),
            "timestamp": datetime.now().isoformat()
        })
        
        # 통계 업데이트
        self._update_performance_stats("create", level)
        
        # WebSocket으로 브로드캐스트
        await self._broadcast_think_update(session_id, "block_created", think_block)
        
        logger.debug(f"THINK 블록 생성: {level} - {block_id}")
        return block_id
    
    async def update_think_block(
        self,
        session_id: str,
        block_id: str,
        content: Optional[str] = None,
        progress: Optional[float] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """THINK 블록 업데이트"""
        
        if session_id not in self.think_trees:
            return
        
        think_tree = self.think_trees[session_id]
        
        if block_id not in think_tree.all_blocks:
            return
        
        think_block = think_tree.all_blocks[block_id]
        
        # 업데이트 적용
        if content is not None:
            # 한국어 최적화
            if self.korean_optimizer:
                korean_result = await self.korean_optimizer.process_korean_text(content)
                think_block.content = korean_result.normalized_text
            else:
                think_block.content = content
        
        if progress is not None:
            think_block.progress = max(0.0, min(1.0, progress))
        
        if status is not None:
            think_block.status = status
            if status == "completed":
                think_block.end_time = datetime.now()
        
        if metadata is not None:
            think_block.metadata.update(metadata)
        
        think_tree.last_updated = datetime.now()
        
        # 실시간 스트림에 추가
        self.active_streams[session_id].append({
            "type": "block_updated",
            "block_id": block_id,
            "data": self._serialize_think_block(think_block),
            "timestamp": datetime.now().isoformat()
        })
        
        # WebSocket으로 브로드캐스트
        await self._broadcast_think_update(session_id, "block_updated", think_block)
        
        logger.debug(f"THINK 블록 업데이트: {block_id} - {status}")
    
    async def complete_think_block(
        self,
        session_id: str,
        block_id: str,
        final_content: Optional[str] = None
    ):
        """THINK 블록 완료"""
        
        await self.update_think_block(
            session_id,
            block_id,
            content=final_content,
            progress=1.0,
            status="completed"
        )
        
        # 완료 통계 업데이트
        if session_id in self.think_trees and block_id in self.think_trees[session_id].all_blocks:
            think_block = self.think_trees[session_id].all_blocks[block_id]
            duration = (think_block.end_time - think_block.start_time).total_seconds()
            self._update_performance_stats("complete", think_block.level, duration)
    
    async def get_think_tree_data(
        self,
        session_id: str,
        format: str = "hierarchical"
    ) -> Dict[str, Any]:
        """사고 트리 데이터 조회"""
        
        if session_id not in self.think_trees:
            return {"nodes": [], "edges": [], "metadata": {}}
        
        think_tree = self.think_trees[session_id]
        config = self.visualization_configs.get(session_id, VisualizationConfig())
        
        if format == "hierarchical":
            return self._build_hierarchical_data(think_tree, config)
        elif format == "timeline":
            return self._build_timeline_data(think_tree, config)
        elif format == "graph":
            return self._build_graph_data(think_tree, config)
        else:
            return self._build_hierarchical_data(think_tree, config)
    
    def _build_hierarchical_data(
        self,
        think_tree: ThinkTree,
        config: VisualizationConfig
    ) -> Dict[str, Any]:
        """계층 구조 데이터 구축"""
        
        nodes = []
        edges = []
        
        # 모든 블록을 노드로 변환
        for block_id, think_block in think_tree.all_blocks.items():
            level_info = self.think_levels.get(think_block.level, self.think_levels["think"])
            
            node = {
                "id": block_id,
                "label": f"{level_info.icon} {level_info.name}",
                "content": think_block.content[:100] + "..." if len(think_block.content) > 100 else think_block.content,
                "level": think_block.level,
                "progress": think_block.progress,
                "status": think_block.status,
                "color": level_info.color,
                "size": 20 + (level_info.priority * 10),
                "metadata": {
                    "start_time": think_block.start_time.isoformat(),
                    "end_time": think_block.end_time.isoformat() if think_block.end_time else None,
                    "duration": (think_block.end_time - think_block.start_time).total_seconds() if think_block.end_time else None,
                    **think_block.metadata
                }
            }
            
            nodes.append(node)
            
            # 부모-자식 관계를 엣지로 변환
            if think_block.parent_id:
                edges.append({
                    "id": f"{think_block.parent_id}_{block_id}",
                    "source": think_block.parent_id,
                    "target": block_id,
                    "type": "parent_child",
                    "arrow": True
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "tree",
            "metadata": {
                "session_id": think_tree.session_id,
                "total_blocks": len(think_tree.all_blocks),
                "root_blocks": len(think_tree.root_blocks),
                "created_at": think_tree.created_at.isoformat(),
                "last_updated": think_tree.last_updated.isoformat()
            }
        }
    
    def _build_timeline_data(
        self,
        think_tree: ThinkTree,
        config: VisualizationConfig
    ) -> Dict[str, Any]:
        """타임라인 데이터 구축"""
        
        timeline_items = []
        
        # 시간순으로 정렬
        sorted_blocks = sorted(
            think_tree.all_blocks.values(),
            key=lambda b: b.start_time
        )
        
        for think_block in sorted_blocks:
            level_info = self.think_levels.get(think_block.level, self.think_levels["think"])
            
            timeline_items.append({
                "id": think_block.id,
                "start": think_block.start_time.isoformat(),
                "end": think_block.end_time.isoformat() if think_block.end_time else None,
                "title": f"{level_info.icon} {level_info.name}",
                "content": think_block.content,
                "level": think_block.level,
                "status": think_block.status,
                "progress": think_block.progress,
                "color": level_info.color,
                "metadata": think_block.metadata
            })
        
        return {
            "timeline": timeline_items,
            "metadata": {
                "session_id": think_tree.session_id,
                "duration": (think_tree.last_updated - think_tree.created_at).total_seconds(),
                "total_blocks": len(think_tree.all_blocks)
            }
        }
    
    def _build_graph_data(
        self,
        think_tree: ThinkTree,
        config: VisualizationConfig
    ) -> Dict[str, Any]:
        """그래프 네트워크 데이터 구축"""
        
        nodes = []
        edges = []
        
        # 레벨별 클러스터링
        level_clusters = {"think": [], "megathink": [], "ultrathink": []}
        
        for block_id, think_block in think_tree.all_blocks.items():
            level_info = self.think_levels.get(think_block.level, self.think_levels["think"])
            
            node = {
                "id": block_id,
                "label": think_block.content[:50] + "..." if len(think_block.content) > 50 else think_block.content,
                "group": think_block.level,
                "size": 15 + (think_block.progress * 20),
                "color": level_info.color,
                "status": think_block.status,
                "metadata": {
                    "level": think_block.level,
                    "level_icon": level_info.icon,
                    "progress": think_block.progress,
                    "start_time": think_block.start_time.isoformat()
                }
            }
            
            nodes.append(node)
            level_clusters[think_block.level].append(block_id)
            
            # 엣지 생성 (부모-자식 + 시간적 순서)
            if think_block.parent_id:
                edges.append({
                    "id": f"parent_{think_block.parent_id}_{block_id}",
                    "source": think_block.parent_id,
                    "target": block_id,
                    "type": "hierarchy",
                    "weight": 3
                })
        
        # 시간적 순서 엣지 추가
        sorted_blocks = sorted(think_tree.all_blocks.values(), key=lambda b: b.start_time)
        for i in range(len(sorted_blocks) - 1):
            current_block = sorted_blocks[i]
            next_block = sorted_blocks[i + 1]
            
            edges.append({
                "id": f"temporal_{current_block.id}_{next_block.id}",
                "source": current_block.id,
                "target": next_block.id,
                "type": "temporal",
                "weight": 1,
                "style": "dashed"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": level_clusters,
            "layout": "force",
            "metadata": {
                "session_id": think_tree.session_id,
                "cluster_count": len([c for c in level_clusters.values() if c])
            }
        }
    
    def _serialize_think_block(self, think_block: ThinkBlock) -> Dict[str, Any]:
        """THINK 블록 직렬화"""
        
        level_info = self.think_levels.get(think_block.level, self.think_levels["think"])
        
        return {
            "id": think_block.id,
            "session_id": think_block.session_id,
            "level": think_block.level,
            "level_info": {
                "name": level_info.name,
                "icon": level_info.icon,
                "color": level_info.color,
                "description": level_info.description
            },
            "content": think_block.content,
            "progress": think_block.progress,
            "status": think_block.status,
            "start_time": think_block.start_time.isoformat(),
            "end_time": think_block.end_time.isoformat() if think_block.end_time else None,
            "parent_id": think_block.parent_id,
            "children_ids": think_block.children_ids,
            "metadata": think_block.metadata
        }
    
    async def _broadcast_think_update(
        self,
        session_id: str,
        update_type: str,
        think_block: ThinkBlock
    ):
        """THINK 업데이트 브로드캐스트"""
        
        message = {
            "type": "think_visualizer_update",
            "data": {
                "update_type": update_type,
                "session_id": session_id,
                "block": self._serialize_think_block(think_block),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # WebSocket 연결들에 전송
        if session_id in self.websocket_connections:
            for conn in self.websocket_connections[session_id]:
                try:
                    await conn.send(json.dumps(message))
                except:
                    pass  # 연결 끊어진 경우 무시
    
    def _update_performance_stats(
        self,
        operation: str,
        level: str,
        duration: Optional[float] = None
    ):
        """성능 통계 업데이트"""
        
        if operation == "create":
            self.performance_stats["total_think_blocks"] += 1
            self.performance_stats["level_distribution"][level] += 1
            
            # 세션당 평균 블록 수 업데이트
            total_sessions = len(self.think_trees)
            if total_sessions > 0:
                self.performance_stats["avg_blocks_per_session"] = \
                    self.performance_stats["total_think_blocks"] / total_sessions
        
        elif operation == "complete" and duration is not None:
            # 평균 지속 시간 업데이트
            current_avg = self.performance_stats["avg_think_duration"]
            total_completed = sum(
                tree.all_blocks.values() 
                for tree in self.think_trees.values()
                if any(block.status == "completed" for block in tree.all_blocks.values())
            )
            
            if len(list(total_completed)) > 0:
                self.performance_stats["avg_think_duration"] = \
                    (current_avg * (len(list(total_completed)) - 1) + duration) / len(list(total_completed))
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """세션 통계 조회"""
        
        if session_id not in self.think_trees:
            return {}
        
        think_tree = self.think_trees[session_id]
        blocks = list(think_tree.all_blocks.values())
        
        # 레벨별 통계
        level_stats = {"think": 0, "megathink": 0, "ultrathink": 0}
        completed_blocks = 0
        total_duration = 0.0
        
        for block in blocks:
            level_stats[block.level] += 1
            
            if block.status == "completed" and block.end_time:
                completed_blocks += 1
                total_duration += (block.end_time - block.start_time).total_seconds()
        
        return {
            "session_id": session_id,
            "total_blocks": len(blocks),
            "completed_blocks": completed_blocks,
            "level_distribution": level_stats,
            "avg_duration": total_duration / completed_blocks if completed_blocks > 0 else 0.0,
            "completion_rate": completed_blocks / len(blocks) if blocks else 0.0,
            "session_duration": (think_tree.last_updated - think_tree.created_at).total_seconds(),
            "blocks_per_minute": len(blocks) / ((think_tree.last_updated - think_tree.created_at).total_seconds() / 60) if (think_tree.last_updated - think_tree.created_at).total_seconds() > 0 else 0.0
        }
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """전역 통계 조회"""
        
        return {
            **self.performance_stats,
            "active_sessions": len(self.think_trees),
            "total_sessions": len(self.think_trees),  # 실제로는 전체 세션 수 추적 필요
            "avg_session_duration": sum(
                (tree.last_updated - tree.created_at).total_seconds()
                for tree in self.think_trees.values()
            ) / len(self.think_trees) if self.think_trees else 0.0,
            "think_levels_info": {
                level: {
                    "name": info.name,
                    "icon": info.icon,
                    "color": info.color,
                    "description": info.description
                }
                for level, info in self.think_levels.items()
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def update_visualization_config(
        self,
        session_id: str,
        config_updates: Dict[str, Any]
    ):
        """시각화 설정 업데이트"""
        
        if session_id not in self.visualization_configs:
            self.visualization_configs[session_id] = VisualizationConfig()
        
        config = self.visualization_configs[session_id]
        
        # 설정 업데이트
        for key, value in config_updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 클라이언트에 설정 변경 알림
        await self._broadcast_config_update(session_id, config_updates)
    
    async def _broadcast_config_update(
        self,
        session_id: str,
        config_updates: Dict[str, Any]
    ):
        """설정 변경 브로드캐스트"""
        
        message = {
            "type": "think_visualizer_config_update",
            "data": {
                "session_id": session_id,
                "config_updates": config_updates,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if session_id in self.websocket_connections:
            for conn in self.websocket_connections[session_id]:
                try:
                    await conn.send(json.dumps(message))
                except:
                    pass
    
    def add_websocket_connection(self, session_id: str, websocket):
        """WebSocket 연결 추가"""
        
        if session_id not in self.websocket_connections:
            self.websocket_connections[session_id] = []
        
        self.websocket_connections[session_id].append(websocket)
    
    def remove_websocket_connection(self, session_id: str, websocket):
        """WebSocket 연결 제거"""
        
        if session_id in self.websocket_connections:
            try:
                self.websocket_connections[session_id].remove(websocket)
            except ValueError:
                pass
    
    async def export_think_data(
        self,
        session_id: str,
        format: str = "json"
    ) -> Optional[str]:
        """사고 데이터 내보내기"""
        
        if session_id not in self.think_trees:
            return None
        
        think_tree = self.think_trees[session_id]
        
        if format == "json":
            export_data = {
                "session_info": {
                    "session_id": session_id,
                    "created_at": think_tree.created_at.isoformat(),
                    "last_updated": think_tree.last_updated.isoformat(),
                    "total_blocks": len(think_tree.all_blocks)
                },
                "think_blocks": [
                    self._serialize_think_block(block)
                    for block in think_tree.all_blocks.values()
                ],
                "statistics": await self.get_session_statistics(session_id)
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        
        elif format == "mermaid":
            # Mermaid 다이어그램 형태로 내보내기
            lines = ["graph TD"]
            
            for block_id, block in think_tree.all_blocks.items():
                level_info = self.think_levels[block.level]
                node_label = f"{level_info.icon} {block.content[:30]}..."
                lines.append(f"    {block_id}[\"{node_label}\"]")
                
                # 부모-자식 관계
                if block.parent_id:
                    lines.append(f"    {block.parent_id} --> {block_id}")
            
            return "\n".join(lines)
        
        return None
    
    async def cleanup_session(self, session_id: str):
        """세션 정리"""
        
        try:
            # WebSocket 연결 정리
            if session_id in self.websocket_connections:
                for conn in self.websocket_connections[session_id]:
                    try:
                        await conn.close()
                    except:
                        pass
                del self.websocket_connections[session_id]
            
            # 데이터 정리
            self.think_trees.pop(session_id, None)
            self.active_streams.pop(session_id, None)
            self.visualization_configs.pop(session_id, None)
            
            logger.debug(f"Think Visualizer 세션 정리: {session_id}")
            
        except Exception as e:
            logger.error(f"Think Visualizer 세션 정리 실패 ({session_id}): {e}")
    
    async def cleanup(self):
        """Think Visualizer 정리"""
        
        # 모든 세션 정리
        session_ids = list(self.think_trees.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)
        
        logger.info("Think Visualizer 정리 완료")