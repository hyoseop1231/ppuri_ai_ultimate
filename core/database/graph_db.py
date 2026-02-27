"""
PPuRI-AI Ultimate - Neo4j Graph Database
지식 그래프 저장 및 순회

Features:
- 엔티티/관계 CRUD
- 그래프 순회 (BFS/DFS)
- 경로 탐색
- 커뮤니티 탐지
- 시각화용 데이터 변환
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Neo4j 드라이버 (선택적)
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j not installed. Run: pip install neo4j")


@dataclass
class GraphDBConfig:
    """Neo4j 설정"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "ppuri"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0


class Neo4jGraphDB:
    """
    Neo4j 그래프 데이터베이스 클라이언트

    기능:
    1. 엔티티/관계 저장 및 조회
    2. 그래프 순회 및 경로 탐색
    3. 유사 엔티티 검색
    4. 시각화용 데이터 변환
    """

    def __init__(self, config: GraphDBConfig):
        self.config = config
        self._driver: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """드라이버 초기화"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available")
            return False

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout
            )

            # 연결 테스트
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run("RETURN 1 AS test")
                await result.single()

            # 인덱스 생성
            await self._create_indexes()

            self._initialized = True
            logger.info("Neo4j connection established")
            return True

        except Exception as e:
            logger.error(f"Neo4j initialization failed: {e}")
            return False

    async def _create_indexes(self):
        """필요한 인덱스 생성"""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_industry IF NOT EXISTS FOR (e:Entity) ON (e.industry)",
            "CREATE INDEX entity_normalized IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description, e.aliases]"
        ]

        async with self._driver.session(database=self.config.database) as session:
            for idx in indexes:
                try:
                    await session.run(idx)
                except Exception as e:
                    logger.debug(f"Index creation skipped: {e}")

    async def close(self):
        """연결 종료"""
        if self._driver:
            await self._driver.close()
            self._initialized = False

    # ==================== 엔티티 CRUD ====================

    async def create_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str = "",
        properties: Optional[Dict] = None,
        aliases: Optional[List[str]] = None,
        industry: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """엔티티 생성"""
        if not self._initialized:
            return False

        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.normalized_name = toLower(trim($name)),
            e.entity_type = $entity_type,
            e.description = $description,
            e.properties = $properties,
            e.aliases = $aliases,
            e.industry = $industry,
            e.embedding = $embedding,
            e.updated_at = datetime()
        ON CREATE SET e.created_at = datetime()
        RETURN e.id
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(
                    query,
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    properties=properties or {},
                    aliases=aliases or [],
                    industry=industry,
                    embedding=embedding
                )
                await result.single()
                return True
        except Exception as e:
            logger.error(f"Entity creation failed: {e}")
            return False

    async def get_entity(self, entity_id: str) -> Optional[Dict]:
        """엔티티 조회"""
        if not self._initialized:
            return None

        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, id=entity_id)
                record = await result.single()
                if record:
                    return dict(record["e"])
                return None
        except Exception as e:
            logger.error(f"Entity retrieval failed: {e}")
            return None

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        industry: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """엔티티 검색 (풀텍스트)"""
        if not self._initialized:
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query)
        YIELD node, score
        WHERE ($entity_type IS NULL OR node.entity_type = $entity_type)
          AND ($industry IS NULL OR node.industry = $industry)
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(
                    cypher,
                    query=query,
                    entity_type=entity_type,
                    industry=industry,
                    limit=limit
                )
                records = await result.data()
                return [{"entity": dict(r["node"]), "score": r["score"]} for r in records]
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    # ==================== 관계 CRUD ====================

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        weight: float = 1.0,
        properties: Optional[Dict] = None
    ) -> bool:
        """관계 생성"""
        if not self._initialized:
            return False

        # 동적 관계 타입 생성
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relation_type}]->(target)
        SET r.description = $description,
            r.weight = $weight,
            r.properties = $properties,
            r.updated_at = datetime()
        ON CREATE SET r.created_at = datetime()
        RETURN r
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    description=description,
                    weight=weight,
                    properties=properties or {}
                )
                await result.single()
                return True
        except Exception as e:
            logger.error(f"Relationship creation failed: {e}")
            return False

    async def get_relationships_for_entities(
        self,
        entity_ids: List[str],
        limit: int = 50
    ) -> List[Dict]:
        """엔티티들의 관계 조회"""
        if not self._initialized:
            return []

        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE e1.id IN $entity_ids OR e2.id IN $entity_ids
        RETURN e1.id as source_id, e1.name as source_name,
               type(r) as relation_type, r.description as description,
               r.weight as weight,
               e2.id as target_id, e2.name as target_name
        LIMIT $limit
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, entity_ids=entity_ids, limit=limit)
                records = await result.data()
                return records
        except Exception as e:
            logger.error(f"Relationship retrieval failed: {e}")
            return []

    # ==================== 그래프 순회 ====================

    async def get_entity_subgraph(
        self,
        entity_name: str,
        depth: int = 2,
        limit: int = 100
    ) -> Dict[str, Any]:
        """엔티티 중심 서브그래프 탐색"""
        if not self._initialized:
            return {"nodes": [], "edges": [], "center": None}

        query = """
        MATCH (center:Entity)
        WHERE center.name = $name OR center.normalized_name = toLower($name)
        WITH center
        CALL apoc.path.subgraphAll(center, {
            maxLevel: $depth,
            limit: $limit
        })
        YIELD nodes, relationships
        RETURN center, nodes, relationships
        """

        # APOC 없는 경우 대체 쿼리
        fallback_query = f"""
        MATCH (center:Entity)
        WHERE center.name = $name OR center.normalized_name = toLower($name)
        OPTIONAL MATCH path = (center)-[*1..{depth}]-(connected:Entity)
        WITH center, collect(DISTINCT connected) as connected_nodes,
             collect(DISTINCT relationships(path)) as all_rels
        RETURN center,
               connected_nodes as nodes,
               [r IN REDUCE(acc = [], rels IN all_rels | acc + rels) | r] as relationships
        LIMIT 1
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                try:
                    result = await session.run(query, name=entity_name, depth=depth, limit=limit)
                except:
                    result = await session.run(fallback_query, name=entity_name)

                record = await result.single()
                if not record:
                    return {"nodes": [], "edges": [], "center": None}

                center = dict(record["center"]) if record["center"] else None
                nodes = [dict(n) for n in (record["nodes"] or [])]
                edges = []

                for rel in (record["relationships"] or []):
                    if rel:
                        edges.append({
                            "source": rel.start_node.get("id"),
                            "target": rel.end_node.get("id"),
                            "type": rel.type,
                            "properties": dict(rel)
                        })

                return {
                    "center": center,
                    "nodes": nodes,
                    "edges": edges
                }

        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": [], "center": None}

    async def find_shortest_path(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 5
    ) -> Optional[Dict]:
        """두 엔티티 간 최단 경로 탐색"""
        if not self._initialized:
            return None

        query = """
        MATCH (source:Entity), (target:Entity)
        WHERE (source.name = $source OR source.normalized_name = toLower($source))
          AND (target.name = $target OR target.normalized_name = toLower($target))
        MATCH path = shortestPath((source)-[*..%d]-(target))
        RETURN path,
               [n IN nodes(path) | {id: n.id, name: n.name, type: n.entity_type}] as nodes,
               [r IN relationships(path) | {type: type(r), description: r.description}] as relationships,
               length(path) as path_length
        """ % max_depth

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, source=source_name, target=target_name)
                record = await result.single()
                if record:
                    return {
                        "nodes": record["nodes"],
                        "relationships": record["relationships"],
                        "length": record["path_length"]
                    }
                return None
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return None

    async def get_related_entities(
        self,
        entity_name: str,
        relation_types: Optional[List[str]] = None,
        direction: str = "both",  # in, out, both
        limit: int = 20
    ) -> List[Dict]:
        """관련 엔티티 조회"""
        if not self._initialized:
            return []

        # 방향에 따른 패턴
        if direction == "out":
            pattern = "(e)-[r]->(related)"
        elif direction == "in":
            pattern = "(e)<-[r]-(related)"
        else:
            pattern = "(e)-[r]-(related)"

        # 관계 타입 필터
        type_filter = ""
        if relation_types:
            types_str = "|".join(relation_types)
            pattern = pattern.replace("[r]", f"[r:{types_str}]")

        query = f"""
        MATCH (e:Entity)
        WHERE e.name = $name OR e.normalized_name = toLower($name)
        MATCH {pattern}
        WHERE related:Entity
        RETURN related.id as id,
               related.name as name,
               related.entity_type as entity_type,
               related.description as description,
               type(r) as relation,
               r.weight as weight
        ORDER BY r.weight DESC
        LIMIT $limit
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, name=entity_name, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Related entities retrieval failed: {e}")
            return []

    # ==================== 분석 기능 ====================

    async def get_entity_statistics(self) -> Dict[str, Any]:
        """엔티티 통계"""
        if not self._initialized:
            return {}

        query = """
        MATCH (e:Entity)
        WITH count(e) as total_entities,
             collect(DISTINCT e.entity_type) as types,
             collect(DISTINCT e.industry) as industries
        MATCH ()-[r]->()
        WITH total_entities, types, industries, count(r) as total_relationships
        RETURN total_entities, types, industries, total_relationships
        """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query)
                record = await result.single()
                if record:
                    return {
                        "total_entities": record["total_entities"],
                        "entity_types": record["types"],
                        "industries": record["industries"],
                        "total_relationships": record["total_relationships"]
                    }
                return {}
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {}

    async def get_top_entities(
        self,
        by: str = "connections",  # connections, mentions
        limit: int = 10,
        industry: Optional[str] = None
    ) -> List[Dict]:
        """상위 엔티티 조회"""
        if not self._initialized:
            return []

        if by == "connections":
            query = """
            MATCH (e:Entity)
            WHERE $industry IS NULL OR e.industry = $industry
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) as connection_count
            ORDER BY connection_count DESC
            LIMIT $limit
            RETURN e.id as id, e.name as name, e.entity_type as type,
                   connection_count as score
            """
        else:
            query = """
            MATCH (e:Entity)
            WHERE $industry IS NULL OR e.industry = $industry
            RETURN e.id as id, e.name as name, e.entity_type as type,
                   coalesce(e.mention_count, 0) as score
            ORDER BY score DESC
            LIMIT $limit
            """

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, industry=industry, limit=limit)
                return await result.data()
        except Exception as e:
            logger.error(f"Top entities retrieval failed: {e}")
            return []

    # ==================== 시각화용 변환 ====================

    async def export_for_visualization(
        self,
        entity_ids: Optional[List[str]] = None,
        industry: Optional[str] = None,
        limit: int = 500
    ) -> Dict[str, Any]:
        """시각화용 데이터 내보내기 (D3.js/Vis.js 호환)"""
        if not self._initialized:
            return {"nodes": [], "links": []}

        if entity_ids:
            query = """
            MATCH (e:Entity)
            WHERE e.id IN $ids
            OPTIONAL MATCH (e)-[r]-(connected:Entity)
            WHERE connected.id IN $ids
            WITH collect(DISTINCT e) + collect(DISTINCT connected) as all_nodes,
                 collect(DISTINCT r) as all_rels
            UNWIND all_nodes as node
            WITH collect(DISTINCT node) as nodes, all_rels
            RETURN nodes,
                   [r IN all_rels |
                    {source: startNode(r).id, target: endNode(r).id,
                     type: type(r), weight: r.weight}] as links
            """
            params = {"ids": entity_ids}
        else:
            query = """
            MATCH (e:Entity)
            WHERE $industry IS NULL OR e.industry = $industry
            WITH e LIMIT $limit
            OPTIONAL MATCH (e)-[r]-(connected:Entity)
            WHERE $industry IS NULL OR connected.industry = $industry
            WITH collect(DISTINCT e) + collect(DISTINCT connected) as all_nodes,
                 collect(DISTINCT r) as all_rels
            UNWIND all_nodes as node
            WITH collect(DISTINCT node) as nodes, all_rels
            RETURN nodes,
                   [r IN all_rels |
                    {source: startNode(r).id, target: endNode(r).id,
                     type: type(r), weight: coalesce(r.weight, 1)}] as links
            LIMIT 1
            """
            params = {"industry": industry, "limit": limit}

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run(query, **params)
                record = await result.single()

                if not record:
                    return {"nodes": [], "links": []}

                nodes = []
                for n in (record["nodes"] or []):
                    nodes.append({
                        "id": n.get("id"),
                        "name": n.get("name"),
                        "type": n.get("entity_type"),
                        "industry": n.get("industry"),
                        "description": n.get("description", "")[:100]
                    })

                return {
                    "nodes": nodes,
                    "links": record["links"] or []
                }

        except Exception as e:
            logger.error(f"Visualization export failed: {e}")
            return {"nodes": [], "links": []}

    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        if not self._initialized:
            return {"status": "not_initialized", "neo4j_available": NEO4J_AVAILABLE}

        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run("RETURN 1")
                await result.single()

            stats = await self.get_entity_statistics()

            return {
                "status": "healthy",
                "neo4j_available": True,
                **stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 싱글톤
_graph_db: Optional[Neo4jGraphDB] = None


async def get_graph_db() -> Neo4jGraphDB:
    """Neo4j 그래프 DB 싱글톤"""
    global _graph_db

    if _graph_db is None:
        config = GraphDBConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "ppuri")
        )

        _graph_db = Neo4jGraphDB(config)
        await _graph_db.initialize()

    return _graph_db
