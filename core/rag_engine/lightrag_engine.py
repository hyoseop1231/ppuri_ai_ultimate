"""
PPuRI-AI Ultimate - LightRAG 엔진

ApeRAG/LightRAG 스타일의 듀얼 레벨 검색 엔진
- Entity Extraction: LLM 기반 엔티티/관계 추출
- Entity Normalization: 중복 엔티티 자동 병합
- Dual-Level Retrieval: Low-Level (Entity) + High-Level (Relations)
- Incremental Updates: 전체 재구축 없이 증분 업데이트

References:
- LightRAG: https://github.com/HKUDS/LightRAG
- ApeRAG: https://github.com/apecloud/ApeRAG
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
import re

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """검색 모드"""
    LOCAL = "local"      # 엔티티 중심 (세부 정보)
    GLOBAL = "global"    # 관계 중심 (전체 맥락)
    HYBRID = "hybrid"    # 둘 다
    NAIVE = "naive"      # 기본 벡터 검색만


class EntityType(Enum):
    """뿌리산업 엔티티 타입"""
    TECHNOLOGY = "기술"
    PROCESS = "공정"
    MATERIAL = "재료"
    DEFECT = "결함"
    EQUIPMENT = "장비"
    STANDARD = "규격"
    PARAMETER = "파라미터"
    PRODUCT = "제품"
    COMPANY = "기관"
    PERSON = "인물"


class RelationType(Enum):
    """관계 타입"""
    USES = "사용"
    CAUSES = "발생원인"
    SOLVES = "해결방법"
    RELATED_TO = "관련기술"
    REQUIRES = "전제조건"
    PRODUCES = "결과"
    PART_OF = "구성요소"
    ALTERNATIVE = "대체"
    IMPROVES = "개선"
    MEASURES = "측정"


@dataclass
class Entity:
    """엔티티 (개체)"""
    id: str
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    canonical_name: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source_chunk_ids: List[str] = field(default_factory=list)
    industry: Optional[str] = None  # casting, mold, welding 등
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "aliases": self.aliases,
            "canonical_name": self.canonical_name,
            "properties": self.properties,
            "industry": self.industry
        }


@dataclass
class Relationship:
    """관계"""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    description: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunk_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "description": self.description,
            "weight": self.weight
        }


@dataclass
class LightRAGConfig:
    """LightRAG 설정"""
    # 엔티티 추출
    entity_types: List[str] = field(default_factory=lambda: [
        "기술", "공정", "재료", "결함", "장비", "규격", "파라미터"
    ])

    # 관계 타입
    relationship_types: List[str] = field(default_factory=lambda: [
        "사용", "발생원인", "해결방법", "관련기술", "전제조건", "결과"
    ])

    # 검색 설정
    top_k_entities: int = 10
    top_k_relationships: int = 20
    top_k_chunks: int = 5
    similarity_threshold: float = 0.7

    # Entity Normalization
    enable_entity_merging: bool = True
    merge_similarity_threshold: float = 0.85

    # 추출 설정
    max_entities_per_chunk: int = 20
    max_relationships_per_chunk: int = 30


@dataclass
class RetrievalResult:
    """검색 결과"""
    entities: List[Entity]
    relationships: List[Relationship]
    chunks: List[Dict[str, Any]]
    query: str
    mode: RetrievalMode
    search_time_ms: float = 0.0

    def get_context(self, max_tokens: int = 4000) -> str:
        """컨텍스트 문자열 생성"""
        context_parts = []

        # 엔티티 정보
        if self.entities:
            entity_info = "## 관련 개념:\n"
            for e in self.entities[:5]:
                entity_info += f"- **{e.name}** ({e.entity_type.value}): {e.description[:200]}\n"
            context_parts.append(entity_info)

        # 관계 정보
        if self.relationships:
            rel_info = "## 관계:\n"
            for r in self.relationships[:5]:
                rel_info += f"- {r.description[:150]}\n"
            context_parts.append(rel_info)

        # 청크 정보
        if self.chunks:
            chunk_info = "## 참조 문서:\n"
            for i, c in enumerate(self.chunks[:3], 1):
                chunk_info += f"[{i}] {c.get('content', '')[:500]}...\n\n"
            context_parts.append(chunk_info)

        return "\n".join(context_parts)


class LightRAGEngine:
    """
    LightRAG 스타일 듀얼 레벨 검색 엔진

    주요 기능:
    1. Entity/Relationship Extraction: LLM으로 지식 그래프 자동 구축
    2. Entity Normalization: 중복 엔티티 병합 (ApeRAG 스타일)
    3. Dual-Level Retrieval: Local (Entity) + Global (Relationship)
    4. Incremental Updates: 실시간 그래프 업데이트

    사용 예시:
    ```python
    engine = LightRAGEngine(config, llm_client, embedding_service, vector_db, graph_db)
    await engine.initialize()

    # 문서 인덱싱
    await engine.index_document(doc_id, text, metadata)

    # 검색
    results = await engine.retrieve(query, mode=RetrievalMode.HYBRID)
    ```
    """

    def __init__(
        self,
        config: LightRAGConfig,
        llm_client,        # OpenRouterClient
        embedding_service, # BGEM3Service
        vector_db,         # Qdrant/ChromaDB
        graph_db           # Neo4j
    ):
        self.config = config
        self.llm = llm_client
        self.embedding = embedding_service
        self.vector_db = vector_db
        self.graph_db = graph_db

        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """엔진 초기화"""
        try:
            # 기존 엔티티/관계 로드 (있다면)
            await self._load_existing_graph()

            self._initialized = True
            logger.info("LightRAG Engine initialized")
            return True

        except Exception as e:
            logger.error(f"LightRAG initialization failed: {e}")
            return False

    async def _load_existing_graph(self):
        """기존 그래프 데이터 로드"""
        try:
            if self.graph_db:
                # Neo4j에서 엔티티 로드
                entities = await self.graph_db.get_all_entities()
                for e in entities:
                    self._entities[e.id] = e

                # 관계 로드
                relationships = await self.graph_db.get_all_relationships()
                for r in relationships:
                    self._relationships[r.id] = r

                logger.info(f"Loaded {len(self._entities)} entities, {len(self._relationships)} relationships")

        except Exception as e:
            logger.warning(f"Could not load existing graph: {e}")

    async def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        문서 인덱싱

        1. 텍스트에서 엔티티/관계 추출
        2. Entity Normalization (중복 병합)
        3. 벡터 DB에 저장
        4. 그래프 DB에 저장
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        logger.info(f"Indexing document: {doc_id}")

        # 1. 엔티티/관계 추출
        entities, relationships = await self.extract_entities_and_relations(
            text=text,
            doc_id=doc_id,
            metadata=metadata
        )

        # 2. Entity Normalization
        if self.config.enable_entity_merging:
            entities = await self._normalize_entities(entities)

        # 3. 임베딩 생성 및 벡터 DB 저장
        for entity in entities:
            if entity.embedding is None:
                embed_result = await self.embedding.embed(
                    f"{entity.name}: {entity.description}"
                )
                entity.embedding = embed_result.dense.tolist()

            # 벡터 DB에 저장
            if self.vector_db:
                await self.vector_db.upsert(
                    collection="entities",
                    id=entity.id,
                    vector=entity.embedding,
                    payload=entity.to_dict()
                )

        # 4. 그래프 DB에 저장
        if self.graph_db:
            for entity in entities:
                await self.graph_db.upsert_entity(entity)

            for rel in relationships:
                await self.graph_db.upsert_relationship(rel)

        # 캐시 업데이트
        for e in entities:
            self._entities[e.id] = e
        for r in relationships:
            self._relationships[r.id] = r

        logger.info(f"Indexed {len(entities)} entities, {len(relationships)} relationships")

        return entities, relationships

    async def extract_entities_and_relations(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """LLM 기반 엔티티/관계 추출"""

        industry = metadata.get("industry", "") if metadata else ""

        prompt = f"""다음 기술 문서에서 엔티티(개체)와 관계를 추출해주세요.
뿌리산업 (주조, 금형, 용접, 소성가공, 표면처리, 열처리) 관련 기술 용어에 집중해주세요.

문서 내용:
{text[:6000]}

엔티티 타입: {', '.join(self.config.entity_types)}
관계 타입: {', '.join(self.config.relationship_types)}

JSON 형식으로 출력해주세요:
{{
  "entities": [
    {{
      "name": "TIG 용접",
      "type": "기술",
      "description": "가스 텅스텐 아크 용접으로 불활성 가스를 사용하여 용접하는 방법",
      "aliases": ["GTAW", "가스 텅스텐 아크 용접"],
      "industry": "welding"
    }}
  ],
  "relationships": [
    {{
      "source": "TIG 용접",
      "target": "텅스텐 혼입",
      "type": "발생원인",
      "description": "TIG 용접 시 전극 접촉으로 텅스텐 혼입 결함이 발생할 수 있음"
    }}
  ]
}}

규칙:
1. 기술 용어는 정확하게 추출
2. 각 엔티티의 description은 명확하고 구체적으로
3. 관계는 실제 문서에서 언급된 것만 추출
4. 최대 {self.config.max_entities_per_chunk}개 엔티티, {self.config.max_relationships_per_chunk}개 관계
"""

        try:
            from core.llm import ModelTier

            response = await self.llm.generate(
                prompt,
                tier=ModelTier.REASONING,
                response_format="json",
                max_tokens=4096
            )

            # JSON 파싱
            content = response.content
            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            # 엔티티 생성
            entities = []
            for e in data.get("entities", []):
                entity_type = self._parse_entity_type(e.get("type", "기술"))
                entity = Entity(
                    id=self._generate_id(e["name"]),
                    name=e["name"],
                    entity_type=entity_type,
                    description=e.get("description", ""),
                    aliases=e.get("aliases", []),
                    industry=e.get("industry", industry),
                    source_chunk_ids=[doc_id]
                )
                entities.append(entity)

            # 관계 생성
            relationships = []
            entity_name_to_id = {e.name: e.id for e in entities}
            # aliases도 매핑에 추가
            for e in entities:
                for alias in e.aliases:
                    entity_name_to_id[alias] = e.id

            for r in data.get("relationships", []):
                source_id = entity_name_to_id.get(r["source"])
                target_id = entity_name_to_id.get(r["target"])

                if source_id and target_id:
                    rel_type = self._parse_relation_type(r.get("type", "관련기술"))
                    relationship = Relationship(
                        id=self._generate_id(f"{source_id}_{target_id}_{rel_type.value}"),
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=rel_type,
                        description=r.get("description", ""),
                        source_chunk_ids=[doc_id]
                    )
                    relationships.append(relationship)

            return entities, relationships

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []

    async def _normalize_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """
        Entity Normalization (ApeRAG 스타일)

        유사한 엔티티를 병합하여 중복 제거
        """
        normalized = []
        merged_ids: Set[str] = set()

        for entity in entities:
            if entity.id in merged_ids:
                continue

            # 기존 엔티티와 유사도 비교
            if self.vector_db and entity.embedding:
                similar = await self.vector_db.search(
                    collection="entities",
                    vector=entity.embedding,
                    limit=3,
                    score_threshold=self.config.merge_similarity_threshold
                )

                if similar and similar[0].get("score", 0) > self.config.merge_similarity_threshold:
                    # 기존 엔티티에 병합
                    existing_id = similar[0].get("id")
                    if existing_id in self._entities:
                        existing = self._entities[existing_id]
                        # aliases 추가
                        existing.aliases.append(entity.name)
                        existing.aliases.extend(entity.aliases)
                        existing.aliases = list(set(existing.aliases))
                        # source_chunk_ids 추가
                        existing.source_chunk_ids.extend(entity.source_chunk_ids)
                        merged_ids.add(entity.id)
                        continue

            # 새 엔티티로 추가
            entity.canonical_name = entity.name
            normalized.append(entity)

        return normalized

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        industry_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        듀얼 레벨 검색

        Args:
            query: 검색 쿼리
            mode: 검색 모드 (LOCAL, GLOBAL, HYBRID, NAIVE)
            industry_filter: 산업 필터 (casting, mold, welding 등)
            top_k: 반환할 결과 수
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        start_time = datetime.now()

        # 쿼리 임베딩
        query_embed = await self.embedding.embed_query(query)

        entities = []
        relationships = []
        chunks = []

        # Local Search (Entity-centric)
        if mode in [RetrievalMode.LOCAL, RetrievalMode.HYBRID]:
            entities = await self._search_entities(
                query_vector=query_embed.dense.tolist(),
                industry_filter=industry_filter,
                top_k=top_k or self.config.top_k_entities
            )

        # Global Search (Relationship-centric)
        if mode in [RetrievalMode.GLOBAL, RetrievalMode.HYBRID]:
            if entities:
                entity_ids = [e.id for e in entities]
                relationships = await self._search_relationships(
                    entity_ids=entity_ids,
                    top_k=top_k or self.config.top_k_relationships
                )

        # Chunk Search (기본 RAG)
        if mode in [RetrievalMode.NAIVE, RetrievalMode.HYBRID]:
            chunks = await self._search_chunks(
                query_vector=query_embed.dense.tolist(),
                industry_filter=industry_filter,
                top_k=top_k or self.config.top_k_chunks
            )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return RetrievalResult(
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            query=query,
            mode=mode,
            search_time_ms=elapsed_ms
        )

    async def _search_entities(
        self,
        query_vector: List[float],
        industry_filter: Optional[str],
        top_k: int
    ) -> List[Entity]:
        """엔티티 검색"""
        try:
            if not self.vector_db:
                return []

            filter_dict = None
            if industry_filter:
                filter_dict = {"industry": industry_filter}

            results = await self.vector_db.search(
                collection="entities",
                vector=query_vector,
                limit=top_k,
                filter=filter_dict
            )

            entities = []
            for r in results:
                entity_id = r.get("id")
                if entity_id in self._entities:
                    entities.append(self._entities[entity_id])
                else:
                    # DB에서 로드
                    payload = r.get("payload", {})
                    entity = Entity(
                        id=entity_id,
                        name=payload.get("name", ""),
                        entity_type=self._parse_entity_type(payload.get("entity_type", "기술")),
                        description=payload.get("description", ""),
                        aliases=payload.get("aliases", []),
                        industry=payload.get("industry")
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    async def _search_relationships(
        self,
        entity_ids: List[str],
        top_k: int
    ) -> List[Relationship]:
        """관계 검색 (그래프 순회)"""
        try:
            if not self.graph_db:
                return []

            relationships = await self.graph_db.get_relationships_for_entities(
                entity_ids=entity_ids,
                limit=top_k
            )

            return relationships

        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            return []

    async def _search_chunks(
        self,
        query_vector: List[float],
        industry_filter: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """청크 검색 (기본 RAG)"""
        try:
            if not self.vector_db:
                return []

            filter_dict = None
            if industry_filter:
                filter_dict = {"industry": industry_filter}

            results = await self.vector_db.search(
                collection="chunks",
                vector=query_vector,
                limit=top_k,
                filter=filter_dict
            )

            return [r.get("payload", {}) for r in results]

        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return []

    def _generate_id(self, name: str) -> str:
        """엔티티/관계 ID 생성"""
        return hashlib.md5(name.encode()).hexdigest()[:16]

    def _parse_entity_type(self, type_str: str) -> EntityType:
        """문자열을 EntityType으로 변환"""
        type_map = {
            "기술": EntityType.TECHNOLOGY,
            "공정": EntityType.PROCESS,
            "재료": EntityType.MATERIAL,
            "결함": EntityType.DEFECT,
            "장비": EntityType.EQUIPMENT,
            "규격": EntityType.STANDARD,
            "파라미터": EntityType.PARAMETER,
            "제품": EntityType.PRODUCT,
            "기관": EntityType.COMPANY,
            "인물": EntityType.PERSON
        }
        return type_map.get(type_str, EntityType.TECHNOLOGY)

    def _parse_relation_type(self, type_str: str) -> RelationType:
        """문자열을 RelationType으로 변환"""
        type_map = {
            "사용": RelationType.USES,
            "발생원인": RelationType.CAUSES,
            "해결방법": RelationType.SOLVES,
            "관련기술": RelationType.RELATED_TO,
            "전제조건": RelationType.REQUIRES,
            "결과": RelationType.PRODUCES,
            "구성요소": RelationType.PART_OF,
            "대체": RelationType.ALTERNATIVE,
            "개선": RelationType.IMPROVES,
            "측정": RelationType.MEASURES
        }
        return type_map.get(type_str, RelationType.RELATED_TO)

    async def get_entity_graph(
        self,
        entity_name: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        엔티티 중심 그래프 탐색

        Returns:
            {
                "center": Entity,
                "nodes": [Entity, ...],
                "edges": [Relationship, ...]
            }
        """
        if not self.graph_db:
            return {"center": None, "nodes": [], "edges": []}

        return await self.graph_db.get_entity_subgraph(
            entity_name=entity_name,
            depth=depth
        )

    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "entity_count": len(self._entities),
            "relationship_count": len(self._relationships)
        }


# 싱글톤 인스턴스
_lightrag_engine: Optional[LightRAGEngine] = None


async def get_lightrag_engine() -> LightRAGEngine:
    """LightRAG 엔진 싱글톤"""
    global _lightrag_engine

    if _lightrag_engine is None:
        from core.llm import get_openrouter_client
        from core.embeddings import get_embedding_service

        llm = await get_openrouter_client()
        embedding = await get_embedding_service()
        config = LightRAGConfig()

        _lightrag_engine = LightRAGEngine(
            config=config,
            llm_client=llm,
            embedding_service=embedding,
            vector_db=None,  # TODO: 벡터 DB 연결
            graph_db=None    # TODO: 그래프 DB 연결
        )
        await _lightrag_engine.initialize()

    return _lightrag_engine
