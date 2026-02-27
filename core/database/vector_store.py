"""
PPuRI-AI Ultimate - Vector Store
pgvector 기반 벡터 검색 엔진

Features:
- 코사인 유사도 검색
- 하이브리드 검색 (Dense + Sparse)
- 산업별 필터링
- 배치 인덱싱
"""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import select, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Document, DocumentChunk, Entity, Relationship, SearchCache
from .connection import get_async_session

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """벡터 스토어 설정"""
    similarity_threshold: float = 0.5
    max_results: int = 10
    use_hybrid_search: bool = True
    sparse_weight: float = 0.3  # Dense vs Sparse 가중치
    cache_ttl_hours: int = 24
    batch_size: int = 100


@dataclass
class SearchResult:
    """검색 결과"""
    chunk_id: str
    document_id: str
    content: str
    score: float
    page_number: Optional[int] = None
    document_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntitySearchResult:
    """엔티티 검색 결과"""
    entity_id: str
    name: str
    entity_type: str
    description: str
    score: float
    related_entities: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    pgvector 기반 벡터 스토어

    기능:
    1. Dense 벡터 검색 (코사인 유사도)
    2. Sparse 벡터 검색 (BM25 스타일)
    3. 하이브리드 검색 (Dense + Sparse)
    4. 엔티티 검색
    5. 검색 결과 캐싱
    """

    def __init__(self, config: VectorStoreConfig, embedding_service):
        self.config = config
        self.embedding = embedding_service
        self._initialized = False

    async def initialize(self) -> bool:
        """벡터 스토어 초기화"""
        try:
            # 인덱스 최적화 확인
            async with get_async_session() as session:
                # IVFFlat 인덱스 설정 확인
                result = await session.execute(text("""
                    SELECT indexname FROM pg_indexes
                    WHERE indexname LIKE '%embedding%'
                """))
                indexes = [row[0] for row in result.fetchall()]
                logger.info(f"Vector indexes found: {indexes}")

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            return False

    async def search_chunks(
        self,
        query: str,
        top_k: int = None,
        industry_filter: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        청크 벡터 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            industry_filter: 산업 필터
            document_ids: 특정 문서들로 제한
            use_cache: 캐시 사용 여부
        """
        top_k = top_k or self.config.max_results

        # 캐시 확인
        if use_cache:
            cache_key = self._make_cache_key(query, industry_filter, document_ids)
            cached = await self._get_cached_results(cache_key)
            if cached:
                return cached[:top_k]

        # 쿼리 임베딩 생성
        query_embedding = await self.embedding.embed_query(query)
        dense_vector = query_embedding.dense_embedding

        async with get_async_session() as session:
            if self.config.use_hybrid_search and query_embedding.sparse_embedding:
                # 하이브리드 검색
                results = await self._hybrid_search(
                    session=session,
                    dense_vector=dense_vector,
                    sparse_vector=query_embedding.sparse_embedding,
                    top_k=top_k,
                    industry_filter=industry_filter,
                    document_ids=document_ids
                )
            else:
                # Dense 검색만
                results = await self._dense_search(
                    session=session,
                    dense_vector=dense_vector,
                    top_k=top_k,
                    industry_filter=industry_filter,
                    document_ids=document_ids
                )

        # 캐시 저장
        if use_cache and results:
            await self._cache_results(cache_key, query, results)

        return results

    async def _dense_search(
        self,
        session: AsyncSession,
        dense_vector: List[float],
        top_k: int,
        industry_filter: Optional[str],
        document_ids: Optional[List[str]]
    ) -> List[SearchResult]:
        """Dense 벡터 검색 (코사인 유사도)"""

        # pgvector 코사인 거리 검색
        vector_str = f"[{','.join(map(str, dense_vector))}]"

        query = f"""
            SELECT
                dc.id,
                dc.document_id,
                dc.content,
                dc.page_number,
                dc.metadata,
                d.title as document_title,
                1 - (dc.embedding_dense <=> '{vector_str}'::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding_dense IS NOT NULL
        """

        # 필터 추가
        conditions = []
        if industry_filter:
            conditions.append(f"d.industry = '{industry_filter}'")
        if document_ids:
            doc_ids_str = ",".join([f"'{d}'" for d in document_ids])
            conditions.append(f"dc.document_id IN ({doc_ids_str})")

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += f"""
            ORDER BY similarity DESC
            LIMIT {top_k}
        """

        result = await session.execute(text(query))
        rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                content=row[2],
                page_number=row[3],
                metadata=row[4] or {},
                document_title=row[5],
                score=float(row[6]) if row[6] else 0.0
            )
            for row in rows
            if row[6] and row[6] >= self.config.similarity_threshold
        ]

    async def _hybrid_search(
        self,
        session: AsyncSession,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        top_k: int,
        industry_filter: Optional[str],
        document_ids: Optional[List[str]]
    ) -> List[SearchResult]:
        """하이브리드 검색 (Dense + Sparse)"""

        # Dense 검색 결과
        dense_results = await self._dense_search(
            session, dense_vector, top_k * 2,
            industry_filter, document_ids
        )

        # Sparse 검색 (BM25 스타일)
        sparse_results = await self._sparse_search(
            session, sparse_vector, top_k * 2,
            industry_filter, document_ids
        )

        # 결과 병합 (RRF - Reciprocal Rank Fusion)
        merged = self._merge_results_rrf(
            dense_results, sparse_results,
            dense_weight=1 - self.config.sparse_weight,
            sparse_weight=self.config.sparse_weight
        )

        return merged[:top_k]

    async def _sparse_search(
        self,
        session: AsyncSession,
        sparse_vector: Dict[str, float],
        top_k: int,
        industry_filter: Optional[str],
        document_ids: Optional[List[str]]
    ) -> List[SearchResult]:
        """Sparse 벡터 검색 (키워드 기반)"""

        # 상위 키워드 추출
        top_keywords = sorted(
            sparse_vector.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        if not top_keywords:
            return []

        # PostgreSQL 텍스트 검색 사용
        keyword_conditions = " OR ".join([
            f"dc.content ILIKE '%{kw[0]}%'"
            for kw in top_keywords
        ])

        query = f"""
            SELECT
                dc.id,
                dc.document_id,
                dc.content,
                dc.page_number,
                dc.metadata,
                d.title as document_title,
                ts_rank_cd(
                    to_tsvector('simple', dc.content),
                    plainto_tsquery('simple', '{' '.join([kw[0] for kw in top_keywords[:5]])}')
                ) as rank_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE ({keyword_conditions})
        """

        # 필터 추가
        conditions = []
        if industry_filter:
            conditions.append(f"d.industry = '{industry_filter}'")
        if document_ids:
            doc_ids_str = ",".join([f"'{d}'" for d in document_ids])
            conditions.append(f"dc.document_id IN ({doc_ids_str})")

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += f"""
            ORDER BY rank_score DESC
            LIMIT {top_k}
        """

        result = await session.execute(text(query))
        rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                content=row[2],
                page_number=row[3],
                metadata=row[4] or {},
                document_title=row[5],
                score=float(row[6]) if row[6] else 0.0
            )
            for row in rows
        ]

    def _merge_results_rrf(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float,
        sparse_weight: float,
        k: int = 60
    ) -> List[SearchResult]:
        """RRF (Reciprocal Rank Fusion) 결과 병합"""
        scores = {}
        results_map = {}

        # Dense 결과 점수
        for rank, result in enumerate(dense_results):
            rrf_score = dense_weight / (k + rank + 1)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + rrf_score
            results_map[result.chunk_id] = result

        # Sparse 결과 점수
        for rank, result in enumerate(sparse_results):
            rrf_score = sparse_weight / (k + rank + 1)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in results_map:
                results_map[result.chunk_id] = result

        # 점수 순 정렬
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id]
            result.score = scores[chunk_id]
            merged.append(result)

        return merged

    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
        industry_filter: Optional[str] = None,
        include_relations: bool = True
    ) -> List[EntitySearchResult]:
        """
        엔티티 벡터 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            entity_types: 엔티티 타입 필터
            industry_filter: 산업 필터
            include_relations: 관계 정보 포함 여부
        """
        query_embedding = await self.embedding.embed_query(query)
        dense_vector = query_embedding.dense_embedding
        vector_str = f"[{','.join(map(str, dense_vector))}]"

        async with get_async_session() as session:
            sql_query = f"""
                SELECT
                    e.id,
                    e.name,
                    e.entity_type,
                    e.description,
                    e.properties,
                    e.importance_score,
                    1 - (e.embedding <=> '{vector_str}'::vector) as similarity
                FROM entities e
                WHERE e.embedding IS NOT NULL
            """

            conditions = []
            if entity_types:
                types_str = ",".join([f"'{t}'" for t in entity_types])
                conditions.append(f"e.entity_type IN ({types_str})")
            if industry_filter:
                conditions.append(f"e.industry = '{industry_filter}'")

            if conditions:
                sql_query += " AND " + " AND ".join(conditions)

            sql_query += f"""
                ORDER BY similarity DESC
                LIMIT {top_k}
            """

            result = await session.execute(text(sql_query))
            rows = result.fetchall()

            entities = []
            for row in rows:
                entity_id = str(row[0])

                # 관계 정보 가져오기
                related = []
                if include_relations:
                    rel_query = """
                        SELECT
                            r.relation_type,
                            e2.name as target_name,
                            e2.entity_type as target_type
                        FROM relationships r
                        JOIN entities e2 ON r.target_id = e2.id
                        WHERE r.source_id = :entity_id
                        LIMIT 5
                    """
                    rel_result = await session.execute(
                        text(rel_query),
                        {"entity_id": entity_id}
                    )
                    for rel_row in rel_result.fetchall():
                        related.append({
                            "relation": rel_row[0],
                            "target_name": rel_row[1],
                            "target_type": rel_row[2]
                        })

                entities.append(EntitySearchResult(
                    entity_id=entity_id,
                    name=row[1],
                    entity_type=row[2],
                    description=row[3] or "",
                    score=float(row[6]) if row[6] else 0.0,
                    related_entities=related,
                    metadata=row[4] or {}
                ))

            return entities

    async def index_chunk(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """청크 인덱싱"""
        # 임베딩 생성
        embedding_result = await self.embedding.embed(content, return_sparse=True)

        async with get_async_session() as session:
            chunk = DocumentChunk(
                document_id=document_id,
                content=content,
                chunk_index=chunk_index,
                page_number=page_number,
                embedding_dense=embedding_result.dense_embedding,
                embedding_sparse=embedding_result.sparse_embedding,
                token_count=len(content.split()),
                metadata=metadata or {}
            )
            session.add(chunk)
            await session.flush()
            chunk_id = str(chunk.id)

        logger.debug(f"Indexed chunk {chunk_index} for document {document_id}")
        return chunk_id

    async def index_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        document_id: Optional[str] = None,
        properties: Optional[Dict] = None,
        industry: Optional[str] = None
    ) -> str:
        """엔티티 인덱싱"""
        # 임베딩 생성
        embed_text = f"{name}: {description}"
        embedding_result = await self.embedding.embed(embed_text)

        # 정규화된 이름
        normalized_name = name.lower().strip()

        async with get_async_session() as session:
            # 기존 엔티티 확인 (중복 방지)
            existing = await session.execute(
                select(Entity).where(
                    and_(
                        Entity.normalized_name == normalized_name,
                        Entity.entity_type == entity_type
                    )
                )
            )
            existing_entity = existing.scalar_one_or_none()

            if existing_entity:
                # 기존 엔티티 업데이트
                existing_entity.mention_count += 1
                if description and len(description) > len(existing_entity.description or ""):
                    existing_entity.description = description
                entity_id = str(existing_entity.id)
            else:
                # 새 엔티티 생성
                entity = Entity(
                    name=name,
                    normalized_name=normalized_name,
                    entity_type=entity_type,
                    description=description,
                    document_id=document_id,
                    embedding=embedding_result.dense_embedding,
                    properties=properties or {},
                    industry=industry
                )
                session.add(entity)
                await session.flush()
                entity_id = str(entity.id)

        return entity_id

    def _make_cache_key(
        self,
        query: str,
        industry_filter: Optional[str],
        document_ids: Optional[List[str]]
    ) -> str:
        """캐시 키 생성"""
        key_parts = [query, industry_filter or ""]
        if document_ids:
            key_parts.append(",".join(sorted(document_ids)))
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """캐시된 결과 가져오기"""
        async with get_async_session() as session:
            result = await session.execute(
                select(SearchCache).where(
                    and_(
                        SearchCache.query_hash == cache_key,
                        SearchCache.expires_at > datetime.utcnow()
                    )
                )
            )
            cache = result.scalar_one_or_none()

            if cache:
                cache.hit_count += 1
                return [SearchResult(**r) for r in cache.results]

        return None

    async def _cache_results(
        self,
        cache_key: str,
        query: str,
        results: List[SearchResult]
    ):
        """검색 결과 캐싱"""
        async with get_async_session() as session:
            cache = SearchCache(
                query_hash=cache_key,
                query_text=query,
                results=[{
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "content": r.content,
                    "score": r.score,
                    "page_number": r.page_number,
                    "document_title": r.document_title,
                    "metadata": r.metadata
                } for r in results],
                result_count=len(results),
                expires_at=datetime.utcnow() + timedelta(hours=self.config.cache_ttl_hours)
            )
            session.add(cache)


# 싱글톤
_vector_store: Optional[VectorStore] = None


async def get_vector_store() -> VectorStore:
    """벡터 스토어 싱글톤"""
    global _vector_store

    if _vector_store is None:
        from core.embeddings import get_embedding_service

        embedding = await get_embedding_service()
        config = VectorStoreConfig()

        _vector_store = VectorStore(config, embedding)
        await _vector_store.initialize()

    return _vector_store
