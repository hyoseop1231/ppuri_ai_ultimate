"""
Knowledge Explorer - 지식 그래프 인터랙티브 탐색 컴포넌트

Neo4j 기반 지식 그래프를 직관적이고 인터랙티브하게 
탐색할 수 있는 시각화 및 분석 도구.

Features:
- 실시간 그래프 시각화
- 노드/엣지 필터링 및 검색
- 관계 패턴 분석
- 클러스터링 및 레이아웃 최적화
- 드릴다운 탐색
- 뿌리산업 도메인 특화 뷰
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import uuid
import math

logger = logging.getLogger(__name__)


@dataclass
class GraphFilter:
    """그래프 필터"""
    node_types: List[str] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    date_range: Optional[Tuple[datetime, datetime]] = None
    min_strength: float = 0.0
    max_depth: int = 3
    keywords: List[str] = field(default_factory=list)


@dataclass
class GraphLayout:
    """그래프 레이아웃 설정"""
    algorithm: str = "force"  # force, hierarchical, circular, grid
    node_size_metric: str = "degree"  # degree, betweenness, pagerank
    edge_weight_metric: str = "strength"
    clustering_enabled: bool = True
    physics_enabled: bool = True
    stabilization_iterations: int = 100


@dataclass
class GraphNode:
    """그래프 노드"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any]
    size: float = 20.0
    color: str = "#4A90E2"
    position: Optional[Tuple[float, float]] = None
    cluster_id: Optional[str] = None


@dataclass
class GraphEdge:
    """그래프 엣지"""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    color: str = "#95A5A6"
    style: str = "solid"  # solid, dashed, dotted


@dataclass
class GraphCluster:
    """그래프 클러스터"""
    id: str
    nodes: List[str]
    label: str
    color: str
    size: int
    density: float
    centrality: float


class KnowledgeExplorer:
    """
    지식 그래프 인터랙티브 탐색기
    
    Neo4j 기반 지식 그래프를 시각적으로 탐색하고 
    분석할 수 있는 고급 인터페이스 컴포넌트.
    """
    
    def __init__(
        self,
        ui_orchestrator,
        graph_manager,
        korean_optimizer=None
    ):
        self.ui_orchestrator = ui_orchestrator
        self.graph_manager = graph_manager
        self.korean_optimizer = korean_optimizer
        
        # 세션별 상태
        self.session_graphs: Dict[str, Dict[str, Any]] = {}
        self.session_filters: Dict[str, GraphFilter] = {}
        self.session_layouts: Dict[str, GraphLayout] = {}
        
        # 노드/엣지 캐시
        self.node_cache: Dict[str, GraphNode] = {}
        self.edge_cache: Dict[str, GraphEdge] = {}
        
        # 색상 팔레트
        self.color_palettes = self._initialize_color_palettes()
        
        # 뿌리산업 도메인 설정
        self.industry_domains = self._initialize_industry_domains()
        
        # 성능 통계
        self.exploration_stats = {
            "total_queries": 0,
            "avg_query_time": 0.0,
            "popular_node_types": Counter(),
            "popular_edge_types": Counter(),
            "exploration_patterns": defaultdict(int)
        }
        
        logger.info("Knowledge Explorer 초기화 완료")
    
    def _initialize_color_palettes(self) -> Dict[str, Dict[str, str]]:
        """색상 팔레트 초기화"""
        
        return {
            "node_types": {
                "Entity": "#4A90E2",
                "Concept": "#50C8A3",
                "Material": "#F39C12",
                "Process": "#E74C3C",
                "Equipment": "#9B59B6",
                "Parameter": "#1ABC9C",
                "Quality": "#34495E",
                "Document": "#95A5A6"
            },
            "edge_types": {
                "USES_IN": "#E74C3C",
                "AFFECTS": "#F39C12", 
                "REQUIRES": "#9B59B6",
                "CAUSES": "#E67E22",
                "IMPROVES": "#27AE60",
                "MEASURES": "#3498DB",
                "RELATED_TO": "#95A5A6",
                "PART_OF": "#8E44AD",
                "SIMILAR_TO": "#16A085"
            },
            "domains": {
                "주조": "#E74C3C",
                "금형": "#F39C12",
                "소성가공": "#27AE60",
                "용접": "#3498DB",
                "표면처리": "#9B59B6",
                "열처리": "#E67E22"
            }
        }
    
    def _initialize_industry_domains(self) -> Dict[str, Dict[str, Any]]:
        """뿌리산업 도메인 설정"""
        
        return {
            "주조": {
                "keywords": ["주조", "캐스팅", "용탕", "응고", "주형"],
                "node_types": ["Material", "Process", "Equipment", "Parameter"],
                "edge_types": ["USES_IN", "AFFECTS", "REQUIRES"]
            },
            "금형": {
                "keywords": ["금형", "다이", "몰드", "성형", "프레스"],
                "node_types": ["Equipment", "Process", "Material", "Quality"],
                "edge_types": ["PART_OF", "USES_IN", "AFFECTS"]
            },
            "소성가공": {
                "keywords": ["소성가공", "단조", "압연", "인발", "전조"],
                "node_types": ["Process", "Material", "Parameter", "Equipment"],
                "edge_types": ["USES_IN", "AFFECTS", "REQUIRES", "IMPROVES"]
            },
            "용접": {
                "keywords": ["용접", "접합", "아크", "저항", "레이저"],
                "node_types": ["Process", "Material", "Equipment", "Quality"],
                "edge_types": ["USES_IN", "AFFECTS", "CAUSES", "MEASURES"]
            },
            "표면처리": {
                "keywords": ["표면처리", "코팅", "도금", "침탄", "질화"],
                "node_types": ["Process", "Material", "Parameter", "Quality"],
                "edge_types": ["IMPROVES", "AFFECTS", "REQUIRES"]
            },
            "열처리": {
                "keywords": ["열처리", "소입", "소성", "담금질", "풀림"],
                "node_types": ["Process", "Parameter", "Material", "Equipment"],
                "edge_types": ["AFFECTS", "REQUIRES", "IMPROVES", "CAUSES"]
            }
        }
    
    async def initialize_session(
        self,
        session_id: str,
        initial_filter: Optional[GraphFilter] = None,
        layout_config: Optional[GraphLayout] = None
    ):
        """세션 초기화"""
        
        # 필터 설정
        self.session_filters[session_id] = initial_filter or GraphFilter()
        
        # 레이아웃 설정
        self.session_layouts[session_id] = layout_config or GraphLayout()
        
        # 초기 그래프 데이터 로드
        await self.load_graph_data(session_id)
        
        logger.debug(f"Knowledge Explorer 세션 초기화: {session_id}")
    
    async def load_graph_data(
        self,
        session_id: str,
        focus_node_id: Optional[str] = None,
        max_nodes: int = 100
    ) -> Dict[str, Any]:
        """그래프 데이터 로드"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            graph_filter = self.session_filters.get(session_id, GraphFilter())
            
            # 노드 쿼리 구성
            node_query = self._build_node_query(graph_filter, focus_node_id, max_nodes)
            
            # 엣지 쿼리 구성
            edge_query = self._build_edge_query(graph_filter)
            
            # 병렬 실행
            nodes_result, edges_result = await asyncio.gather(
                self.graph_manager.execute_query(node_query["query"], node_query["params"]),
                self.graph_manager.execute_query(edge_query["query"], edge_query["params"])
            )
            
            # 데이터 변환
            graph_nodes = self._process_nodes(nodes_result)
            graph_edges = self._process_edges(edges_result)
            
            # 클러스터링
            clusters = await self._detect_clusters(graph_nodes, graph_edges)
            
            # 레이아웃 계산
            layout_data = await self._calculate_layout(
                graph_nodes, graph_edges, self.session_layouts[session_id]
            )
            
            # 통계 계산
            statistics = self._calculate_graph_statistics(graph_nodes, graph_edges)
            
            # 세션 그래프 업데이트
            self.session_graphs[session_id] = {
                "nodes": graph_nodes,
                "edges": graph_edges,
                "clusters": clusters,
                "layout": layout_data,
                "statistics": statistics,
                "last_updated": datetime.now().isoformat()
            }
            
            # 성능 통계 업데이트
            query_time = asyncio.get_event_loop().time() - start_time
            self._update_exploration_stats("load_graph", query_time, len(graph_nodes), len(graph_edges))
            
            return self.session_graphs[session_id]
            
        except Exception as e:
            logger.error(f"그래프 데이터 로드 실패 ({session_id}): {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def _build_node_query(
        self,
        graph_filter: GraphFilter,
        focus_node_id: Optional[str],
        max_nodes: int
    ) -> Dict[str, Any]:
        """노드 쿼리 구성"""
        
        conditions = []
        params = {"max_nodes": max_nodes}
        
        # 포커스 노드 기반 탐색
        if focus_node_id:
            base_query = """
            MATCH (focus {id: $focus_id})
            MATCH (n)-[*1..$max_depth]-(focus)
            """
            params["focus_id"] = focus_node_id
            params["max_depth"] = graph_filter.max_depth
        else:
            base_query = "MATCH (n)"
        
        # 노드 타입 필터
        if graph_filter.node_types:
            type_conditions = " OR ".join(f"n:`{node_type}`" for node_type in graph_filter.node_types)
            conditions.append(f"({type_conditions})")
        
        # 도메인 필터
        if graph_filter.domains:
            conditions.append("n.domain IN $domains")
            params["domains"] = graph_filter.domains
        
        # 키워드 필터
        if graph_filter.keywords:
            keyword_conditions = []
            for i, keyword in enumerate(graph_filter.keywords):
                key = f"keyword_{i}"
                keyword_conditions.append(f"n.name CONTAINS ${key}")
                params[key] = keyword
            conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # 날짜 범위 필터
        if graph_filter.date_range:
            conditions.append("datetime(n.created_at) >= datetime($start_date)")
            conditions.append("datetime(n.created_at) <= datetime($end_date)")
            params["start_date"] = graph_filter.date_range[0].isoformat()
            params["end_date"] = graph_filter.date_range[1].isoformat()
        
        # WHERE 절 구성
        where_clause = " AND ".join(conditions) if conditions else ""
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        # 최종 쿼리
        query = f"""
        {base_query}
        {where_clause}
        RETURN DISTINCT n
        LIMIT $max_nodes
        """
        
        return {"query": query, "params": params}
    
    def _build_edge_query(self, graph_filter: GraphFilter) -> Dict[str, Any]:
        """엣지 쿼리 구성"""
        
        conditions = []
        params = {}
        
        base_query = "MATCH (a)-[r]->(b)"
        
        # 엣지 타입 필터
        if graph_filter.edge_types:
            type_conditions = " OR ".join(f"type(r) = '{edge_type}'" for edge_type in graph_filter.edge_types)
            conditions.append(f"({type_conditions})")
        
        # 강도 필터
        if graph_filter.min_strength > 0:
            conditions.append("r.strength >= $min_strength")
            params["min_strength"] = graph_filter.min_strength
        
        # WHERE 절 구성
        where_clause = " AND ".join(conditions) if conditions else ""
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        # 최종 쿼리
        query = f"""
        {base_query}
        {where_clause}
        RETURN a, r, b
        LIMIT 500
        """
        
        return {"query": query, "params": params}
    
    def _process_nodes(self, nodes_result: List[Dict[str, Any]]) -> List[GraphNode]:
        """노드 데이터 처리"""
        
        processed_nodes = []
        
        for record in nodes_result:
            node_data = record['n']
            
            # 노드 타입 결정
            node_type = node_data.labels[0] if node_data.labels else "Unknown"
            
            # 색상 결정
            color = self.color_palettes["node_types"].get(node_type, "#95A5A6")
            
            # 도메인별 색상 오버라이드
            if node_data.properties.get("domain") in self.color_palettes["domains"]:
                color = self.color_palettes["domains"][node_data.properties["domain"]]
            
            # 크기 계산 (연결 수 기반)
            size = 20 + min(50, node_data.properties.get("degree", 0) * 2)
            
            graph_node = GraphNode(
                id=node_data.id,
                label=node_data.properties.get("name", f"Node_{node_data.id}"),
                type=node_type,
                properties=node_data.properties,
                size=size,
                color=color
            )
            
            processed_nodes.append(graph_node)
            self.node_cache[graph_node.id] = graph_node
        
        return processed_nodes
    
    def _process_edges(self, edges_result: List[Dict[str, Any]]) -> List[GraphEdge]:
        """엣지 데이터 처리"""
        
        processed_edges = []
        
        for record in edges_result:
            edge_data = record['r']
            source_id = record['a'].id
            target_id = record['b'].id
            
            # 엣지 타입
            edge_type = edge_data.type
            
            # 색상 결정
            color = self.color_palettes["edge_types"].get(edge_type, "#95A5A6")
            
            # 가중치 계산
            weight = edge_data.properties.get("strength", 1.0)
            
            # 스타일 결정
            style = "solid"
            if edge_type in ["RELATED_TO", "SIMILAR_TO"]:
                style = "dashed"
            elif weight < 0.5:
                style = "dotted"
            
            graph_edge = GraphEdge(
                id=edge_data.id,
                source=source_id,
                target=target_id,
                type=edge_type,
                properties=edge_data.properties,
                weight=weight,
                color=color,
                style=style
            )
            
            processed_edges.append(graph_edge)
            self.edge_cache[graph_edge.id] = graph_edge
        
        return processed_edges
    
    async def _detect_clusters(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> List[GraphCluster]:
        """클러스터 탐지"""
        
        # 간단한 연결 기반 클러스터링
        adjacency = defaultdict(set)
        
        for edge in edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        
        visited = set()
        clusters = []
        
        for node in nodes:
            if node.id not in visited:
                # DFS로 연결된 컴포넌트 찾기
                cluster_nodes = []
                stack = [node.id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster_nodes.append(current)
                        
                        # 인접 노드들 추가
                        for neighbor in adjacency.get(current, set()):
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if len(cluster_nodes) > 1:  # 단일 노드 클러스터 제외
                    cluster = GraphCluster(
                        id=f"cluster_{len(clusters)}",
                        nodes=cluster_nodes,
                        label=f"클러스터 {len(clusters) + 1}",
                        color=self._get_cluster_color(len(clusters)),
                        size=len(cluster_nodes),
                        density=self._calculate_cluster_density(cluster_nodes, edges),
                        centrality=0.0  # 추후 계산
                    )
                    clusters.append(cluster)
        
        return clusters
    
    def _get_cluster_color(self, cluster_index: int) -> str:
        """클러스터 색상 할당"""
        
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF"]
        return colors[cluster_index % len(colors)]
    
    def _calculate_cluster_density(
        self,
        cluster_nodes: List[str],
        all_edges: List[GraphEdge]
    ) -> float:
        """클러스터 밀도 계산"""
        
        cluster_node_set = set(cluster_nodes)
        internal_edges = 0
        
        for edge in all_edges:
            if edge.source in cluster_node_set and edge.target in cluster_node_set:
                internal_edges += 1
        
        # 최대 가능 엣지 수
        max_edges = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        
        return internal_edges / max_edges if max_edges > 0 else 0.0
    
    async def _calculate_layout(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        layout_config: GraphLayout
    ) -> Dict[str, Any]:
        """레이아웃 계산"""
        
        # 간단한 force-directed 레이아웃 시뮬레이션
        if layout_config.algorithm == "force":
            return self._calculate_force_layout(nodes, edges)
        elif layout_config.algorithm == "circular":
            return self._calculate_circular_layout(nodes)
        elif layout_config.algorithm == "hierarchical":
            return self._calculate_hierarchical_layout(nodes, edges)
        else:
            return self._calculate_grid_layout(nodes)
    
    def _calculate_force_layout(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> Dict[str, Any]:
        """Force-directed 레이아웃"""
        
        import random
        
        # 초기 위치 랜덤 설정
        positions = {}
        for node in nodes:
            positions[node.id] = (
                random.uniform(-100, 100),
                random.uniform(-100, 100)
            )
            node.position = positions[node.id]
        
        # 간단한 시뮬레이션 (실제로는 더 정교한 알고리즘 필요)
        iterations = 50
        for _ in range(iterations):
            forces = {node_id: [0.0, 0.0] for node_id in positions}
            
            # 척력 (모든 노드 쌍)
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    dx = positions[node2.id][0] - positions[node1.id][0]
                    dy = positions[node2.id][1] - positions[node1.id][1]
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    repulsion = 1000 / (distance * distance)
                    fx = repulsion * dx / distance
                    fy = repulsion * dy / distance
                    
                    forces[node1.id][0] -= fx
                    forces[node1.id][1] -= fy
                    forces[node2.id][0] += fx
                    forces[node2.id][1] += fy
            
            # 인력 (연결된 노드들)
            for edge in edges:
                if edge.source in positions and edge.target in positions:
                    dx = positions[edge.target][0] - positions[edge.source][0]
                    dy = positions[edge.target][1] - positions[edge.source][1]
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    attraction = distance * 0.1 * edge.weight
                    fx = attraction * dx / distance
                    fy = attraction * dy / distance
                    
                    forces[edge.source][0] += fx
                    forces[edge.source][1] += fy
                    forces[edge.target][0] -= fx
                    forces[edge.target][1] -= fy
            
            # 위치 업데이트
            for node_id in positions:
                positions[node_id] = (
                    positions[node_id][0] + forces[node_id][0] * 0.01,
                    positions[node_id][1] + forces[node_id][1] * 0.01
                )
        
        # 노드 객체에 위치 설정
        for node in nodes:
            node.position = positions[node.id]
        
        return {
            "algorithm": "force",
            "iterations": iterations,
            "node_positions": positions
        }
    
    def _calculate_circular_layout(self, nodes: List[GraphNode]) -> Dict[str, Any]:
        """원형 레이아웃"""
        
        import math
        
        radius = max(50, len(nodes) * 5)
        positions = {}
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[node.id] = (x, y)
            node.position = (x, y)
        
        return {
            "algorithm": "circular",
            "radius": radius,
            "node_positions": positions
        }
    
    def _calculate_hierarchical_layout(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> Dict[str, Any]:
        """계층적 레이아웃"""
        
        # 간단한 레벨 기반 배치
        levels = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 진입 차수 계산
        for edge in edges:
            in_degree[edge.target] += 1
        
        # 레벨 할당 (위상 정렬 기반)
        current_level = 0
        remaining_nodes = {node.id for node in nodes}
        
        while remaining_nodes:
            level_nodes = [node_id for node_id in remaining_nodes if in_degree[node_id] == 0]
            
            if not level_nodes:  # 순환 참조인 경우
                level_nodes = list(remaining_nodes)[:5]  # 임의로 5개 선택
            
            levels[current_level] = level_nodes
            
            # 선택된 노드들의 나가는 엣지 제거
            for node_id in level_nodes:
                remaining_nodes.remove(node_id)
                for edge in edges:
                    if edge.source == node_id and edge.target in remaining_nodes:
                        in_degree[edge.target] -= 1
            
            current_level += 1
        
        # 위치 계산
        positions = {}
        level_height = 100
        
        for level, level_nodes in levels.items():
            y = level * level_height
            node_width = 150
            total_width = len(level_nodes) * node_width
            start_x = -total_width / 2
            
            for i, node_id in enumerate(level_nodes):
                x = start_x + i * node_width
                positions[node_id] = (x, y)
        
        # 노드 객체에 위치 설정
        for node in nodes:
            if node.id in positions:
                node.position = positions[node.id]
        
        return {
            "algorithm": "hierarchical",
            "levels": dict(levels),
            "node_positions": positions
        }
    
    def _calculate_grid_layout(self, nodes: List[GraphNode]) -> Dict[str, Any]:
        """격자 레이아웃"""
        
        import math
        
        grid_size = math.ceil(math.sqrt(len(nodes)))
        positions = {}
        cell_size = 100
        
        for i, node in enumerate(nodes):
            row = i // grid_size
            col = i % grid_size
            x = col * cell_size - (grid_size * cell_size) / 2
            y = row * cell_size - (grid_size * cell_size) / 2
            positions[node.id] = (x, y)
            node.position = (x, y)
        
        return {
            "algorithm": "grid",
            "grid_size": grid_size,
            "cell_size": cell_size,
            "node_positions": positions
        }
    
    def _calculate_graph_statistics(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> Dict[str, Any]:
        """그래프 통계 계산"""
        
        # 기본 통계
        node_count = len(nodes)
        edge_count = len(edges)
        
        # 노드 타입 분포
        node_type_distribution = Counter(node.type for node in nodes)
        
        # 엣지 타입 분포
        edge_type_distribution = Counter(edge.type for edge in edges)
        
        # 연결도 통계
        degrees = defaultdict(int)
        for edge in edges:
            degrees[edge.source] += 1
            degrees[edge.target] += 1
        
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        max_degree = max(degrees.values()) if degrees else 0
        
        # 밀도 계산
        max_possible_edges = node_count * (node_count - 1) / 2
        density = edge_count / max_possible_edges if max_possible_edges > 0 else 0
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": density,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "node_type_distribution": dict(node_type_distribution),
            "edge_type_distribution": dict(edge_type_distribution),
            "connected_components": len(self._find_connected_components(nodes, edges))
        }
    
    def _find_connected_components(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge]
    ) -> List[List[str]]:
        """연결된 컴포넌트 찾기"""
        
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        
        visited = set()
        components = []
        
        for node in nodes:
            if node.id not in visited:
                component = []
                stack = [node.id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        for neighbor in adjacency.get(current, set()):
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                components.append(component)
        
        return components
    
    async def search_nodes(
        self,
        session_id: str,
        query: str,
        search_type: str = "text"  # text, semantic, property
    ) -> List[GraphNode]:
        """노드 검색"""
        
        if session_id not in self.session_graphs:
            return []
        
        nodes = self.session_graphs[session_id]["nodes"]
        results = []
        
        if search_type == "text":
            # 텍스트 기반 검색
            query_lower = query.lower()
            for node in nodes:
                if (query_lower in node.label.lower() or
                    any(query_lower in str(value).lower() 
                        for value in node.properties.values() if isinstance(value, str))):
                    results.append(node)
        
        elif search_type == "property":
            # 속성 기반 검색
            for node in nodes:
                for key, value in node.properties.items():
                    if query.lower() in f"{key}:{value}".lower():
                        results.append(node)
                        break
        
        elif search_type == "semantic" and self.korean_optimizer:
            # 의미적 검색 (한국어 최적화 활용)
            korean_result = await self.korean_optimizer.process_korean_text(query)
            normalized_query = korean_result.normalized_text
            
            for node in nodes:
                node_text = f"{node.label} {' '.join(map(str, node.properties.values()))}"
                node_result = await self.korean_optimizer.process_korean_text(node_text)
                
                # 산업 용어 매칭
                if set(korean_result.industry_terms) & set(node_result.industry_terms):
                    results.append(node)
        
        return results[:20]  # 최대 20개 결과
    
    async def get_node_neighbors(
        self,
        session_id: str,
        node_id: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """노드 이웃 조회"""
        
        if session_id not in self.session_graphs:
            return {"nodes": [], "edges": []}
        
        edges = self.session_graphs[session_id]["edges"]
        
        # 직접 연결된 노드들 찾기
        neighbor_ids = set()
        relevant_edges = []
        
        for edge in edges:
            if edge.source == node_id:
                neighbor_ids.add(edge.target)
                relevant_edges.append(edge)
            elif edge.target == node_id:
                neighbor_ids.add(edge.source)
                relevant_edges.append(edge)
        
        # depth > 1인 경우 재귀적으로 확장
        if depth > 1:
            current_neighbors = neighbor_ids.copy()
            for _ in range(depth - 1):
                new_neighbors = set()
                for neighbor_id in current_neighbors:
                    for edge in edges:
                        if edge.source == neighbor_id and edge.target not in neighbor_ids:
                            new_neighbors.add(edge.target)
                            relevant_edges.append(edge)
                        elif edge.target == neighbor_id and edge.source not in neighbor_ids:
                            new_neighbors.add(edge.source)
                            relevant_edges.append(edge)
                
                neighbor_ids.update(new_neighbors)
                current_neighbors = new_neighbors
        
        # 해당 노드들과 엣지들 반환
        nodes = self.session_graphs[session_id]["nodes"]
        neighbor_nodes = [node for node in nodes if node.id in neighbor_ids]
        
        return {
            "nodes": neighbor_nodes,
            "edges": relevant_edges,
            "center_node_id": node_id,
            "depth": depth
        }
    
    async def update_filter(
        self,
        session_id: str,
        filter_updates: Dict[str, Any]
    ):
        """필터 업데이트"""
        
        if session_id not in self.session_filters:
            self.session_filters[session_id] = GraphFilter()
        
        graph_filter = self.session_filters[session_id]
        
        # 필터 업데이트
        for key, value in filter_updates.items():
            if hasattr(graph_filter, key):
                setattr(graph_filter, key, value)
        
        # 그래프 데이터 재로드
        await self.load_graph_data(session_id)
    
    async def update_layout(
        self,
        session_id: str,
        layout_updates: Dict[str, Any]
    ):
        """레이아웃 업데이트"""
        
        if session_id not in self.session_layouts:
            self.session_layouts[session_id] = GraphLayout()
        
        layout_config = self.session_layouts[session_id]
        
        # 레이아웃 설정 업데이트
        for key, value in layout_updates.items():
            if hasattr(layout_config, key):
                setattr(layout_config, key, value)
        
        # 레이아웃 재계산
        if session_id in self.session_graphs:
            nodes = self.session_graphs[session_id]["nodes"]
            edges = self.session_graphs[session_id]["edges"]
            
            new_layout = await self._calculate_layout(nodes, edges, layout_config)
            self.session_graphs[session_id]["layout"] = new_layout
    
    def _update_exploration_stats(
        self,
        operation: str,
        query_time: float,
        node_count: int,
        edge_count: int
    ):
        """탐색 통계 업데이트"""
        
        self.exploration_stats["total_queries"] += 1
        
        # 평균 쿼리 시간 업데이트
        total_queries = self.exploration_stats["total_queries"]
        current_avg = self.exploration_stats["avg_query_time"]
        self.exploration_stats["avg_query_time"] = \
            (current_avg * (total_queries - 1) + query_time) / total_queries
        
        # 패턴 추적
        self.exploration_stats["exploration_patterns"][operation] += 1
    
    async def get_domain_overview(
        self,
        session_id: str,
        domain: str
    ) -> Dict[str, Any]:
        """도메인별 개요"""
        
        if domain not in self.industry_domains:
            return {"error": "지원하지 않는 도메인"}
        
        domain_config = self.industry_domains[domain]
        
        # 도메인 특화 필터 적용
        domain_filter = GraphFilter(
            domains=[domain],
            node_types=domain_config["node_types"],
            edge_types=domain_config["edge_types"]
        )
        
        # 임시로 필터 적용하여 데이터 로드
        original_filter = self.session_filters.get(session_id)
        self.session_filters[session_id] = domain_filter
        
        try:
            domain_data = await self.load_graph_data(session_id)
            
            # 도메인 특화 분석
            analysis = {
                "domain": domain,
                "summary": self._analyze_domain_patterns(domain_data, domain_config),
                "key_nodes": self._find_key_nodes(domain_data["nodes"], domain),
                "critical_relationships": self._find_critical_relationships(domain_data["edges"]),
                "recommendations": self._generate_domain_recommendations(domain_data, domain)
            }
            
            return analysis
            
        finally:
            # 원래 필터 복원
            if original_filter:
                self.session_filters[session_id] = original_filter
    
    def _analyze_domain_patterns(
        self,
        domain_data: Dict[str, Any],
        domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """도메인 패턴 분석"""
        
        nodes = domain_data["nodes"]
        edges = domain_data["edges"]
        
        return {
            "total_entities": len(nodes),
            "total_relationships": len(edges),
            "coverage": len(nodes) / 100,  # 가정된 전체 노드 수 대비
            "complexity": len(edges) / len(nodes) if nodes else 0,
            "dominant_node_types": Counter(node.type for node in nodes).most_common(3),
            "dominant_edge_types": Counter(edge.type for edge in edges).most_common(3)
        }
    
    def _find_key_nodes(
        self,
        nodes: List[GraphNode],
        domain: str
    ) -> List[Dict[str, Any]]:
        """핵심 노드 찾기"""
        
        # 크기(연결도) 기준으로 정렬
        sorted_nodes = sorted(nodes, key=lambda n: n.size, reverse=True)
        
        return [
            {
                "id": node.id,
                "label": node.label,
                "type": node.type,
                "size": node.size,
                "importance": node.size / max(n.size for n in nodes) if nodes else 0
            }
            for node in sorted_nodes[:5]  # 상위 5개
        ]
    
    def _find_critical_relationships(
        self,
        edges: List[GraphEdge]
    ) -> List[Dict[str, Any]]:
        """중요 관계 찾기"""
        
        # 가중치 기준으로 정렬
        sorted_edges = sorted(edges, key=lambda e: e.weight, reverse=True)
        
        return [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "weight": edge.weight,
                "importance": edge.weight
            }
            for edge in sorted_edges[:5]  # 상위 5개
        ]
    
    def _generate_domain_recommendations(
        self,
        domain_data: Dict[str, Any],
        domain: str
    ) -> List[str]:
        """도메인별 추천사항"""
        
        recommendations = []
        
        nodes = domain_data["nodes"]
        edges = domain_data["edges"]
        statistics = domain_data["statistics"]
        
        # 연결도가 낮은 경우
        if statistics["avg_degree"] < 2:
            recommendations.append(f"{domain} 영역의 개체들 간 연결성을 강화하는 것이 좋겠습니다.")
        
        # 특정 노드 타입이 부족한 경우
        node_types = Counter(node.type for node in nodes)
        if "Process" not in node_types or node_types["Process"] < 3:
            recommendations.append(f"{domain} 공정 정보를 더 추가하면 도움이 될 것 같습니다.")
        
        # 밀도가 낮은 경우
        if statistics["density"] < 0.1:
            recommendations.append("관련 개념들 간의 관계를 더 상세히 정의해보세요.")
        
        return recommendations
    
    async def export_graph_data(
        self,
        session_id: str,
        format: str = "json"
    ) -> Optional[str]:
        """그래프 데이터 내보내기"""
        
        if session_id not in self.session_graphs:
            return None
        
        graph_data = self.session_graphs[session_id]
        
        if format == "json":
            export_data = {
                "metadata": {
                    "session_id": session_id,
                    "exported_at": datetime.now().isoformat(),
                    "node_count": len(graph_data["nodes"]),
                    "edge_count": len(graph_data["edges"])
                },
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type,
                        "properties": node.properties,
                        "position": node.position
                    }
                    for node in graph_data["nodes"]
                ],
                "edges": [
                    {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "properties": edge.properties,
                        "weight": edge.weight
                    }
                    for edge in graph_data["edges"]
                ],
                "statistics": graph_data["statistics"]
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        
        elif format == "gexf":
            # GEXF 형식으로 내보내기 (Gephi 호환)
            lines = ['<?xml version="1.0" encoding="UTF-8"?>']
            lines.append('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">')
            lines.append('<graph mode="static" defaultedgetype="directed">')
            
            # 노드들
            lines.append('<nodes>')
            for node in graph_data["nodes"]:
                x, y = node.position or (0, 0)
                lines.append(f'<node id="{node.id}" label="{node.label}">')
                lines.append(f'<viz:position x="{x}" y="{y}"/>')
                lines.append(f'<viz:color r="128" g="128" b="128"/>')
                lines.append('</node>')
            lines.append('</nodes>')
            
            # 엣지들
            lines.append('<edges>')
            for edge in graph_data["edges"]:
                lines.append(f'<edge id="{edge.id}" source="{edge.source}" target="{edge.target}" weight="{edge.weight}"/>')
            lines.append('</edges>')
            
            lines.append('</graph>')
            lines.append('</gexf>')
            
            return '\n'.join(lines)
        
        return None
    
    async def cleanup_session(self, session_id: str):
        """세션 정리"""
        
        try:
            # 세션 데이터 정리
            self.session_graphs.pop(session_id, None)
            self.session_filters.pop(session_id, None)
            self.session_layouts.pop(session_id, None)
            
            # 캐시에서 해당 세션 노드/엣지 제거 (선택적)
            # 실제로는 더 정교한 캐시 관리 필요
            
            logger.debug(f"Knowledge Explorer 세션 정리: {session_id}")
            
        except Exception as e:
            logger.error(f"Knowledge Explorer 세션 정리 실패 ({session_id}): {e}")
    
    async def cleanup(self):
        """Knowledge Explorer 정리"""
        
        # 모든 세션 정리
        session_ids = list(self.session_graphs.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)
        
        # 캐시 정리
        self.node_cache.clear()
        self.edge_cache.clear()
        
        logger.info("Knowledge Explorer 정리 완료")