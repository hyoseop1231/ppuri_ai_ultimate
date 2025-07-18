"""
Graph Manager - Neo4j 기반 지식 그래프 관리자

RedPlanet Core에서 영감을 받은 포터블 메모리 그래프 시스템의 
핵심 관리자로 Neo4j 연결 및 그래프 작업을 담당.

Features:
- Neo4j 연결 및 세션 관리
- 그래프 스키마 초기화
- CRUD 작업 추상화
- 트랜잭션 관리
- 성능 최적화
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import uuid

try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j 드라이버가 설치되지 않음. pip install neo4j 실행 필요")

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """그래프 노드"""
    id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass  
class GraphRelationship:
    """그래프 관계"""
    id: str
    start_node_id: str
    end_node_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphQuery:
    """그래프 쿼리"""
    cypher: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30


class GraphManager:
    """
    Neo4j 기반 지식 그래프 관리자
    
    RedPlanet Core의 포터블 메모리 그래프 개념을 구현하여
    대화, 지식, 관계를 효율적으로 관리.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j", 
        password: str = "password",
        database: str = "ppuri_ai"
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j 드라이버가 필요합니다: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        # Neo4j 드라이버
        self.driver: Optional[Driver] = None
        self.connected = False
        
        # 성능 통계
        self.query_stats = {
            "total_queries": 0,
            "avg_query_time": 0.0,
            "failed_queries": 0
        }
        
        # 뿌리산업 그래프 스키마
        self.schema_initialized = False
        
        logger.info("Graph Manager 초기화 완료")
    
    async def connect(self) -> bool:
        """Neo4j 데이터베이스 연결"""
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,  # 1시간
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # 연결 테스트
            await self._verify_connection()
            
            # 스키마 초기화
            await self._initialize_schema()
            
            self.connected = True
            logger.info("✅ Neo4j 연결 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ Neo4j 연결 실패: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Neo4j 연결 종료"""
        
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j 연결 종료")
    
    async def _verify_connection(self):
        """연결 검증"""
        
        with self.driver.session(database=self.database) as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            
            if record["test"] != 1:
                raise Exception("연결 검증 실패")
    
    async def _initialize_schema(self):
        """뿌리산업 그래프 스키마 초기화"""
        
        if self.schema_initialized:
            return
        
        schema_queries = [
            # === 노드 제약 조건 ===
            
            # 사용자 노드
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            
            # 대화 세션 노드  
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
            
            # 메시지 노드
            "CREATE CONSTRAINT message_id IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
            
            # 지식 개념 노드
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            
            # 뿌리산업 도메인 노드
            "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
            
            # 기술 용어 노드
            "CREATE CONSTRAINT term_id IF NOT EXISTS FOR (t:Term) REQUIRE t.id IS UNIQUE",
            
            # === 인덱스 생성 ===
            
            # 시간 기반 검색
            "CREATE INDEX message_timestamp IF NOT EXISTS FOR (m:Message) ON (m.timestamp)",
            "CREATE INDEX session_timestamp IF NOT EXISTS FOR (s:Session) ON (s.created_at)",
            
            # 텍스트 검색
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX term_name IF NOT EXISTS FOR (t:Term) ON (t.name)",
            
            # 카테고리 검색
            "CREATE INDEX domain_category IF NOT EXISTS FOR (d:Domain) ON (d.category)",
            
            # === 기본 도메인 노드 생성 ===
            """
            MERGE (d1:Domain {name: "주조", category: "뿌리산업"})
            SET d1.description = "금속을 용융시켜 주형에 부어 제품을 만드는 기술"
            """,
            
            """
            MERGE (d2:Domain {name: "금형", category: "뿌리산업"})  
            SET d2.description = "제품 성형을 위한 금형 설계 및 제작 기술"
            """,
            
            """
            MERGE (d3:Domain {name: "소성가공", category: "뿌리산업"})
            SET d3.description = "금속의 소성 변형을 이용한 성형 가공 기술"
            """,
            
            """
            MERGE (d4:Domain {name: "용접", category: "뿌리산업"})
            SET d4.description = "금속 재료의 접합을 위한 용접 기술"
            """,
            
            """
            MERGE (d5:Domain {name: "표면처리", category: "뿌리산업"})
            SET d5.description = "제품 표면의 특성 개선을 위한 처리 기술"
            """,
            
            """
            MERGE (d6:Domain {name: "열처리", category: "뿌리산업"})
            SET d6.description = "금속의 조직과 성질 개선을 위한 열처리 기술"
            """
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for query in schema_queries:
                    session.run(query)
            
            self.schema_initialized = True
            logger.info("그래프 스키마 초기화 완료")
            
        except Exception as e:
            logger.error(f"스키마 초기화 실패: {e}")
            raise
    
    async def execute_query(
        self, 
        query: Union[str, GraphQuery],
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """그래프 쿼리 실행"""
        
        if not self.connected:
            raise Exception("Neo4j에 연결되지 않음")
        
        if isinstance(query, str):
            cypher = query
            params = parameters or {}
            timeout = 30
        else:
            cypher = query.cypher
            params = query.parameters
            timeout = query.timeout
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, params)
                records = [record.data() for record in result]
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_query_stats(execution_time, success=True)
            
            logger.debug(f"쿼리 실행 완료: {execution_time:.3f}초, {len(records)}개 결과")
            return records
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_query_stats(execution_time, success=False)
            
            logger.error(f"쿼리 실행 실패: {e}")
            raise
    
    async def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> GraphNode:
        """노드 생성"""
        
        if node_id is None:
            node_id = str(uuid.uuid4())
        
        # 기본 속성 추가
        properties.update({
            "id": node_id,
            "created_at": datetime.now().isoformat()
        })
        
        # 레이블 문자열 생성
        label_str = ":".join(labels)
        
        query = f"""
        CREATE (n:{label_str})
        SET n = $properties
        RETURN n
        """
        
        result = await self.execute_query(query, {"properties": properties})
        
        if result:
            return GraphNode(
                id=node_id,
                labels=labels, 
                properties=properties
            )
        else:
            raise Exception("노드 생성 실패")
    
    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> GraphRelationship:
        """관계 생성"""
        
        rel_id = str(uuid.uuid4())
        rel_properties = properties or {}
        rel_properties.update({
            "id": rel_id,
            "created_at": datetime.now().isoformat()
        })
        
        query = """
        MATCH (start {id: $start_id}), (end {id: $end_id})
        CREATE (start)-[r:""" + relationship_type + """]->(end)
        SET r = $properties
        RETURN r
        """
        
        params = {
            "start_id": start_node_id,
            "end_id": end_node_id,
            "properties": rel_properties
        }
        
        result = await self.execute_query(query, params)
        
        if result:
            return GraphRelationship(
                id=rel_id,
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                type=relationship_type,
                properties=rel_properties
            )
        else:
            raise Exception("관계 생성 실패")
    
    async def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """노드 검색"""
        
        # WHERE 절 구성
        where_conditions = []
        params = {"limit": limit}
        
        if properties:
            for key, value in properties.items():
                where_conditions.append(f"n.{key} = ${key}")
                params[key] = value
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "true"
        
        # 레이블 문자열 생성
        if labels:
            label_str = ":".join(labels)
            query = f"""
            MATCH (n:{label_str})
            WHERE {where_clause}
            RETURN n
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n
            LIMIT $limit
            """
        
        results = await self.execute_query(query, params)
        
        nodes = []
        for result in results:
            node_data = result["n"]
            nodes.append(GraphNode(
                id=node_data.get("id"),
                labels=list(node_data.labels),
                properties=dict(node_data)
            ))
        
        return nodes
    
    async def find_relationships(
        self,
        start_node_id: Optional[str] = None,
        end_node_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100
    ) -> List[GraphRelationship]:
        """관계 검색"""
        
        # WHERE 절 구성
        where_conditions = []
        params = {"limit": limit}
        
        if start_node_id:
            where_conditions.append("startNode(r).id = $start_id")
            params["start_id"] = start_node_id
        
        if end_node_id:
            where_conditions.append("endNode(r).id = $end_id")
            params["end_id"] = end_node_id
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "true"
        
        # 관계 타입 지정
        if relationship_type:
            query = f"""
            MATCH (start)-[r:{relationship_type}]->(end)
            WHERE {where_clause}
            RETURN r, startNode(r).id as start_id, endNode(r).id as end_id
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (start)-[r]->(end)
            WHERE {where_clause}
            RETURN r, startNode(r).id as start_id, endNode(r).id as end_id, type(r) as rel_type
            LIMIT $limit
            """
        
        results = await self.execute_query(query, params)
        
        relationships = []
        for result in results:
            rel_data = result["r"]
            relationships.append(GraphRelationship(
                id=rel_data.get("id"),
                start_node_id=result["start_id"],
                end_node_id=result["end_id"],
                type=result.get("rel_type", relationship_type),
                properties=dict(rel_data)
            ))
        
        return relationships
    
    async def get_node_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",  # "in", "out", "both"
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """노드의 이웃 노드 조회"""
        
        # 방향 설정
        if direction == "out":
            rel_pattern = "-[r]->"
        elif direction == "in":
            rel_pattern = "<-[r]-"
        else:  # both
            rel_pattern = "-[r]-"
        
        # 관계 타입 필터
        if relationship_types:
            type_filter = "|".join(relationship_types)
            rel_pattern = rel_pattern.replace("[r]", f"[r:{type_filter}]")
        
        query = f"""
        MATCH (start {{id: $node_id}}){rel_pattern}(neighbor)
        RETURN neighbor, r, type(r) as rel_type
        LIMIT 100
        """
        
        params = {"node_id": node_id}
        results = await self.execute_query(query, params)
        
        neighbors = []
        for result in results:
            neighbor_data = result["neighbor"]
            rel_data = result["r"]
            
            neighbors.append({
                "node": GraphNode(
                    id=neighbor_data.get("id"),
                    labels=list(neighbor_data.labels),
                    properties=dict(neighbor_data)
                ),
                "relationship": {
                    "type": result["rel_type"],
                    "properties": dict(rel_data)
                }
            })
        
        return neighbors
    
    async def delete_node(self, node_id: str) -> bool:
        """노드 삭제 (관계 포함)"""
        
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        """
        
        try:
            await self.execute_query(query, {"node_id": node_id})
            return True
        except Exception as e:
            logger.error(f"노드 삭제 실패: {e}")
            return False
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """관계 삭제"""
        
        query = """
        MATCH ()-[r {id: $rel_id}]->()
        DELETE r
        """
        
        try:
            await self.execute_query(query, {"rel_id": relationship_id})
            return True
        except Exception as e:
            logger.error(f"관계 삭제 실패: {e}")
            return False
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """그래프 통계 조회"""
        
        stats_queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_labels": "CALL db.labels() YIELD label RETURN collect(label) as labels",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types",
            "domain_nodes": "MATCH (d:Domain) RETURN count(d) as count",
            "user_nodes": "MATCH (u:User) RETURN count(u) as count",
            "session_nodes": "MATCH (s:Session) RETURN count(s) as count",
            "message_nodes": "MATCH (m:Message) RETURN count(m) as count"
        }
        
        statistics = {}
        
        for stat_name, query in stats_queries.items():
            try:
                result = await self.execute_query(query)
                if result:
                    if stat_name in ["node_labels", "relationship_types"]:
                        statistics[stat_name] = result[0].get("labels" if "labels" in stat_name else "types", [])
                    else:
                        statistics[stat_name] = result[0].get("count", 0)
                else:
                    statistics[stat_name] = 0
            except Exception as e:
                logger.error(f"통계 조회 실패 ({stat_name}): {e}")
                statistics[stat_name] = 0
        
        statistics.update({
            "query_stats": self.query_stats,
            "schema_initialized": self.schema_initialized,
            "connected": self.connected,
            "last_updated": datetime.now().isoformat()
        })
        
        return statistics
    
    def _update_query_stats(self, execution_time: float, success: bool):
        """쿼리 통계 업데이트"""
        
        self.query_stats["total_queries"] += 1
        
        if success:
            # 평균 쿼리 시간 업데이트
            total = self.query_stats["total_queries"]
            current_avg = self.query_stats["avg_query_time"]
            
            self.query_stats["avg_query_time"] = (
                (current_avg * (total - 1) + execution_time) / total
            )
        else:
            self.query_stats["failed_queries"] += 1
    
    async def backup_graph(self, backup_path: str) -> bool:
        """그래프 백업 (APOC 필요)"""
        
        try:
            query = """
            CALL apoc.export.cypher.all($file, {
                format: "cypher-shell",
                useOptimizations: {type: "UNWIND_BATCH", unwindBatchSize: 20}
            })
            """
            
            await self.execute_query(query, {"file": backup_path})
            logger.info(f"그래프 백업 완료: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"그래프 백업 실패: {e}")
            return False
    
    async def clear_graph(self, confirm: bool = False) -> bool:
        """그래프 전체 삭제 (주의!)"""
        
        if not confirm:
            logger.warning("그래프 삭제 위험! confirm=True로 설정 필요")
            return False
        
        try:
            # 모든 노드와 관계 삭제
            await self.execute_query("MATCH (n) DETACH DELETE n")
            
            # 스키마 재초기화
            self.schema_initialized = False
            await self._initialize_schema()
            
            logger.info("그래프 전체 삭제 및 재초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"그래프 삭제 실패: {e}")
            return False