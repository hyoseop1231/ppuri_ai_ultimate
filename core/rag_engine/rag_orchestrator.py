"""
RAG Orchestrator - 고성능 RAG 시스템 총괄 관리자

Firestarter 영감의 RAG 엔진 핵심 조정자로 문서 처리부터 
검색, 생성까지 전체 파이프라인을 통합 관리하는 시스템.

Features:
- 전체 RAG 파이프라인 조정
- 네임스페이스 기반 검색 최적화
- 다중 검색 전략 통합
- 실시간 성능 모니터링
- 지식 그래프 연동
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """RAG 쿼리"""
    text: str
    namespace: Optional[str] = None
    domain: Optional[str] = None
    search_strategy: str = "hybrid"
    max_results: int = 5
    similarity_threshold: float = 0.7
    include_metadata: bool = True
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGDocument:
    """RAG 문서"""
    id: str
    title: str
    content: str
    namespace: str
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None


@dataclass
class RAGChunk:
    """RAG 청크"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    namespace: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """RAG 검색 결과"""
    chunk: RAGChunk
    score: float
    rank: int
    retrieval_strategy: str
    highlighted_text: Optional[str] = None


@dataclass
class RAGResponse:
    """RAG 응답"""
    query: RAGQuery
    results: List[RAGResult]
    generated_answer: Optional[str] = None
    total_results: int = 0
    processing_time: float = 0.0
    search_metadata: Dict[str, Any] = field(default_factory=dict)


class RAGOrchestrator:
    """
    Firestarter 영감 RAG 시스템 오케스트레이터
    
    고성능 검색과 생성을 통합 관리하여 뿌리산업 특화 
    지식 검색 및 답변 생성을 제공하는 핵심 시스템.
    """
    
    def __init__(
        self,
        config_manager,
        korean_optimizer=None,
        graph_manager=None,
        conversational_engine=None
    ):
        self.config_manager = config_manager
        self.korean_optimizer = korean_optimizer
        self.graph_manager = graph_manager
        self.conversational_engine = conversational_engine
        
        # RAG 설정
        self.rag_config = config_manager.get_rag_config()
        
        # 컴포넌트들 (지연 초기화)
        self.document_processor = None
        self.namespace_manager = None
        self.retrieval_engine = None
        self.chunk_manager = None
        self.search_strategies = None
        
        # 성능 통계
        self.performance_stats = {
            "total_queries": 0,
            "avg_query_time": 0.0,
            "avg_results_per_query": 0.0,
            "cache_hit_rate": 0.0,
            "namespace_usage": {},
            "search_strategy_usage": {}
        }
        
        # 결과 캐시 (간단한 LRU)
        self.result_cache: Dict[str, Tuple[RAGResponse, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        self.max_cache_size = 1000
        
        logger.info("RAG Orchestrator 초기화 완료")
    
    async def initialize(self):
        """RAG 시스템 초기화"""
        
        logger.info("RAG 컴포넌트 초기화 중...")
        
        # 실제 컴포넌트들 초기화
        from .document_processor import DocumentProcessor
        from .chunk_manager import ChunkManager, ChunkingConfig
        from .retrieval_engine import RetrievalEngine, SearchConfig, EmbeddingConfig
        from .namespace_manager import NamespaceManager
        
        # DocumentProcessor 초기화
        self.document_processor = DocumentProcessor(
            self.config_manager, 
            self.korean_optimizer
        )
        
        # ChunkManager 초기화
        chunking_config = ChunkingConfig(
            chunk_size=self.rag_config.get("chunk_size", 1000),
            chunk_overlap=self.rag_config.get("chunk_overlap", 200),
            adaptive_sizing=self.rag_config.get("adaptive_sizing", True),
            use_semantic_chunking=self.rag_config.get("use_semantic_chunking", True)
        )
        self.chunk_manager = ChunkManager(chunking_config, self.korean_optimizer)
        
        # RetrievalEngine 초기화
        search_config = SearchConfig(
            embedding_model=self.rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
            max_results=self.rag_config.get("max_results", 10),
            enable_hybrid_search=self.rag_config.get("enable_hybrid_search", True),
            enable_reranking=self.rag_config.get("enable_reranking", True)
        )
        
        embedding_config = EmbeddingConfig(
            model_name=self.rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
            normalize_embeddings=True,
            device=self.rag_config.get("device", "cpu")
        )
        
        self.retrieval_engine = RetrievalEngine(
            self.config_manager,
            search_config,
            embedding_config,
            self.korean_optimizer
        )
        
        # NamespaceManager 초기화
        self.namespace_manager = NamespaceManager(
            self.config_manager,
            self.korean_optimizer
        )
        
        # 컴포넌트들 초기화 실행
        await self.retrieval_engine.initialize()
        await self.namespace_manager.initialize()
        
        logger.info("✅ RAG 시스템 초기화 완료")
    
    async def process_document(
        self,
        document: RAGDocument,
        namespace: Optional[str] = None
    ) -> bool:
        """문서 처리 및 인덱싱"""
        
        start_time = time.time()
        
        try:
            # 네임스페이스 자동 감지 또는 설정
            if namespace:
                document.namespace = namespace
            elif not document.namespace:
                if self.namespace_manager:
                    detected_ns, confidence = await self.namespace_manager.detect_namespace(
                        document.content, document.title, document.metadata
                    )
                    document.namespace = detected_ns
                    document.metadata["auto_namespace_confidence"] = confidence
                else:
                    document.namespace = "default"
            
            # 1. DocumentProcessor로 문서 처리
            if self.document_processor:
                processed_doc = await self.document_processor.process_text(
                    text=document.content,
                    title=document.title,
                    namespace=document.namespace,
                    custom_metadata=document.metadata
                )
                
                if not processed_doc:
                    logger.error(f"문서 처리 실패: {document.id}")
                    return False
                
                # 처리된 내용으로 업데이트
                document.content = processed_doc.content
                document.metadata.update(processed_doc.metadata)
            
            # 2. ChunkManager로 문서 청킹
            if self.chunk_manager:
                processed_chunks = await self.chunk_manager.chunk_document(
                    content=document.content,
                    document_id=document.id,
                    namespace=document.namespace
                )
                
                if not processed_chunks:
                    logger.error(f"문서 청킹 실패: {document.id}")
                    return False
                
                # RAGChunk 형태로 변환
                chunks = []
                for processed_chunk in processed_chunks:
                    rag_chunk = RAGChunk(
                        id=processed_chunk.id,
                        document_id=document.id,
                        content=processed_chunk.content,
                        chunk_index=processed_chunk.metadata.chunk_index,
                        namespace=document.namespace,
                        metadata={
                            "quality_score": processed_chunk.metadata.quality.overall_score if processed_chunk.metadata.quality else 0.0,
                            "industry_terms": processed_chunk.metadata.industry_terms,
                            "content_length": processed_chunk.metadata.content_length,
                            "sentence_count": processed_chunk.metadata.sentence_count
                        }
                    )
                    chunks.append(rag_chunk)
            else:
                # Fallback: 기본 청킹
                chunks = await self._chunk_document(document)
            
            # 3. RetrievalEngine에 문서 인덱싱
            if self.retrieval_engine:
                documents_to_index = []
                for chunk in chunks:
                    doc_data = {
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": {
                            **chunk.metadata,
                            "document_id": document.id,
                            "document_title": document.title,
                            "chunk_index": chunk.chunk_index,
                            "namespace": document.namespace
                        }
                    }
                    documents_to_index.append(doc_data)
                
                success = await self.retrieval_engine.index_documents(
                    documents_to_index, document.namespace
                )
                
                if not success:
                    logger.error(f"문서 인덱싱 실패: {document.id}")
                    return False
            else:
                # Fallback: 기본 저장
                await self._store_document_and_chunks(document, chunks)
            
            # 4. NamespaceManager에 문서 등록
            if self.namespace_manager:
                await self.namespace_manager.add_document(
                    document_id=document.id,
                    namespace=document.namespace,
                    auto_detect=False,  # 이미 감지됨
                    content=document.content,
                    metadata=document.metadata
                )
            
            # 5. 지식 그래프에 연결 (선택적)
            if self.graph_manager:
                await self._link_to_knowledge_graph(document, chunks)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"문서 처리 완료: {document.id} "
                f"({len(chunks)}개 청크, {processing_time:.2f}초, 네임스페이스: {document.namespace})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"문서 처리 실패 ({document.id}): {e}")
            return False
    
    async def search(
        self,
        query: RAGQuery,
        generate_answer: bool = True
    ) -> RAGResponse:
        """
        RAG 검색 및 답변 생성
        
        Args:
            query: RAG 쿼리
            generate_answer: 답변 생성 여부
            
        Returns:
            RAGResponse: 검색 결과 및 생성된 답변
        """
        
        start_time = time.time()
        
        try:
            # 1. 캐시 확인
            cache_key = self._generate_cache_key(query)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                self._update_performance_stats(query, cached_response, True)
                return cached_response
            
            # 2. 쿼리 전처리
            processed_query = await self._preprocess_query(query)
            
            # 3. 검색 실행
            search_results = await self._execute_search(processed_query)
            
            # 4. 답변 생성 (선택적)
            generated_answer = None
            if generate_answer and search_results:
                generated_answer = await self._generate_answer(processed_query, search_results)
            
            # 5. 응답 구성
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                query=query,
                results=search_results,
                generated_answer=generated_answer,
                total_results=len(search_results),
                processing_time=processing_time,
                search_metadata=self._collect_search_metadata(processed_query, search_results)
            )
            
            # 6. 캐시 저장
            self._cache_response(cache_key, response)
            
            # 7. 통계 업데이트
            self._update_performance_stats(query, response, False)
            
            logger.debug(
                f"RAG 검색 완료: {len(search_results)}개 결과 "
                f"({processing_time:.3f}초)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG 검색 실패: {e}")
            return RAGResponse(
                query=query,
                results=[],
                processing_time=time.time() - start_time,
                search_metadata={"error": str(e)}
            )
    
    async def _chunk_document(self, document: RAGDocument) -> List[RAGChunk]:
        """문서 청킹"""
        
        chunk_size = self.rag_config.get("chunk_size", 1000)
        chunk_overlap = self.rag_config.get("chunk_overlap", 200)
        
        chunks = []
        text = document.content
        
        # 단순 청킹 (실제로는 더 정교한 방법 사용)
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 문장 끝에서 자르기
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = RAGChunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    namespace=document.namespace,
                    metadata={
                        "document_title": document.title,
                        "document_domain": document.domain,
                        "chunk_size": len(chunk_content),
                        **document.metadata
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # 다음 청크 시작 위치 (오버랩 고려)
            start = max(start + chunk_size - chunk_overlap, end)
        
        return chunks
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        
        # 실제로는 sentence-transformers 등 사용
        # 여기서는 더미 임베딩 반환
        import hashlib
        import random
        
        # 텍스트 해시를 시드로 사용하여 일관된 더미 임베딩 생성
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        embedding_dim = 384  # MiniLM 차원
        embedding = [random.uniform(-1, 1) for _ in range(embedding_dim)]
        
        return embedding
    
    async def _store_document_and_chunks(
        self,
        document: RAGDocument,
        chunks: List[RAGChunk]
    ):
        """문서와 청크를 벡터 DB에 저장 (Fallback 메소드)"""
        
        logger.debug(f"Fallback 문서 저장: {document.id} ({len(chunks)}개 청크)")
        
        if self.retrieval_engine:
            # RetrievalEngine이 있으면 사용
            documents_to_index = []
            for chunk in chunks:
                doc_data = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": {
                        **chunk.metadata,
                        "document_id": document.id,
                        "document_title": document.title,
                        "chunk_index": chunk.chunk_index,
                        "namespace": document.namespace
                    }
                }
                documents_to_index.append(doc_data)
            
            success = await self.retrieval_engine.index_documents(
                documents_to_index, document.namespace
            )
            
            if success:
                logger.debug(f"RetrievalEngine 저장 성공: {document.id}")
            else:
                logger.error(f"RetrievalEngine 저장 실패: {document.id}")
        else:
            # RetrievalEngine이 없으면 메모리에만 저장 (개발/테스트용)
            logger.warning(f"RetrievalEngine 없음, 메모리 저장: {document.id}")
            
            # 간단한 메모리 저장소 사용
            if not hasattr(self, '_memory_store'):
                self._memory_store = {}
            
            self._memory_store[document.id] = {
                "document": document,
                "chunks": chunks,
                "stored_at": datetime.now().isoformat()
            }
    
    async def _link_to_knowledge_graph(
        self,
        document: RAGDocument,
        chunks: List[RAGChunk]
    ):
        """지식 그래프에 문서 연결"""
        
        if not self.graph_manager:
            return
        
        try:
            # 문서 노드 생성
            doc_node = await self.graph_manager.create_node(
                labels=["Document"],
                properties={
                    "id": document.id,
                    "title": document.title,
                    "namespace": document.namespace,
                    "domain": document.domain,
                    "chunk_count": len(chunks),
                    "created_at": document.created_at.isoformat()
                }
            )
            
            # 청크 노드들 생성 및 연결
            for chunk in chunks:
                chunk_node = await self.graph_manager.create_node(
                    labels=["Chunk"],
                    properties={
                        "id": chunk.id,
                        "content_preview": chunk.content[:100] + "...",
                        "chunk_index": chunk.chunk_index,
                        "namespace": chunk.namespace
                    }
                )
                
                # 문서-청크 관계 생성
                await self.graph_manager.create_relationship(
                    start_node_id=doc_node.id,
                    end_node_id=chunk_node.id,
                    relationship_type="HAS_CHUNK",
                    properties={"chunk_index": chunk.chunk_index}
                )
            
            logger.debug(f"지식 그래프 연결 완료: {document.id}")
            
        except Exception as e:
            logger.error(f"지식 그래프 연결 실패: {e}")
    
    async def _preprocess_query(self, query: RAGQuery) -> RAGQuery:
        """쿼리 전처리"""
        
        processed_query = query
        
        # 한국어 최적화
        if self.korean_optimizer:
            korean_result = await self.korean_optimizer.process_korean_text(query.text)
            
            # 정규화된 텍스트 사용
            processed_query.text = korean_result.normalized_text
            
            # 뿌리산업 용어 기반 도메인 감지
            if korean_result.industry_terms and not query.domain:
                from collections import Counter
                domains = [
                    self.korean_optimizer.industry_terms[term].category
                    for term in korean_result.industry_terms
                    if term in self.korean_optimizer.industry_terms
                ]
                if domains:
                    processed_query.domain = Counter(domains).most_common(1)[0][0]
            
            # 메타데이터 추가
            processed_query.user_context.update({
                "detected_terms": korean_result.industry_terms,
                "processing_confidence": korean_result.confidence_score
            })
        
        return processed_query
    
    async def _execute_search(self, query: RAGQuery) -> List[RAGResult]:
        """검색 실행"""
        
        if not self.retrieval_engine:
            # Fallback: 더미 검색
            return await self._fallback_search(query)
        
        try:
            # RetrievalEngine을 사용한 실제 검색
            from .retrieval_engine import SearchQuery as EngineSearchQuery
            
            # RAGQuery를 RetrievalEngine의 SearchQuery로 변환
            engine_query = EngineSearchQuery(
                text=query.text,
                namespace=query.namespace,
                filters=query.user_context,
                max_results=query.max_results,
                similarity_threshold=query.similarity_threshold,
                search_type=query.search_strategy
            )
            
            # 검색 실행
            search_response = await self.retrieval_engine.search(engine_query)
            
            # 결과를 RAGResult 형태로 변환
            rag_results = []
            for i, result in enumerate(search_response.results):
                # RetrievalEngine의 SearchResult를 RAGResult로 변환
                rag_chunk = RAGChunk(
                    id=result.id,
                    document_id=result.metadata.get("document_id", result.id),
                    content=result.content,
                    chunk_index=result.metadata.get("chunk_index", 0),
                    namespace=result.metadata.get("namespace", query.namespace or "default"),
                    metadata=result.metadata
                )
                
                rag_result = RAGResult(
                    chunk=rag_chunk,
                    score=result.score,
                    rank=i + 1,
                    retrieval_strategy=result.search_type,
                    highlighted_text=result.content[:200] + "..." if len(result.content) > 200 else result.content
                )
                
                rag_results.append(rag_result)
            
            # NamespaceManager에 검색 기록
            if self.namespace_manager:
                await self.namespace_manager._update_namespace_stats(
                    query.namespace or "default", "search"
                )
            
            return rag_results
            
        except Exception as e:
            logger.error(f"검색 실행 실패: {e}")
            # Fallback으로 더미 검색 실행
            return await self._fallback_search(query)
    
    async def _fallback_search(self, query: RAGQuery) -> List[RAGResult]:
        """Fallback 더미 검색 (RetrievalEngine 없을 때)"""
        
        logger.warning("RetrievalEngine 없음, fallback 검색 사용")
        
        dummy_results = []
        search_strategy = query.search_strategy
        
        # 전략별 결과 개수 조정
        if search_strategy == "hybrid":
            result_count = min(5, query.max_results)
            base_score = 0.9
        elif search_strategy == "semantic":
            result_count = min(3, query.max_results)
            base_score = 0.85
        elif search_strategy == "keyword":
            result_count = min(3, query.max_results)
            base_score = 0.8
        else:
            result_count = min(4, query.max_results)
            base_score = 0.88
        
        for i in range(result_count):
            chunk = RAGChunk(
                id=f"fallback_chunk_{i}",
                document_id=f"fallback_doc_{i}",
                content=f"[Fallback] 뿌리산업 관련 내용 {i}: {query.text}에 대한 정보입니다. "
                       f"실제 검색을 위해서는 RetrievalEngine이 필요합니다.",
                chunk_index=i,
                namespace=query.namespace or "default",
                metadata={
                    "fallback_mode": True,
                    "original_query": query.text,
                    "search_strategy": search_strategy
                }
            )
            
            result = RAGResult(
                chunk=chunk,
                score=base_score - i * 0.1,
                rank=i + 1,
                retrieval_strategy=f"fallback_{search_strategy}",
                highlighted_text=chunk.content[:150] + "..."
            )
            
            dummy_results.append(result)
        
        return dummy_results
    
    def _integrate_search_results(
        self,
        strategy_results: Dict[str, List[RAGResult]],
        max_results: int
    ) -> List[RAGResult]:
        """검색 결과 통합"""
        
        all_results = []
        
        # 모든 전략의 결과 수집
        for strategy, results in strategy_results.items():
            all_results.extend(results)
        
        # 중복 제거 (같은 청크)
        unique_results = {}
        for result in all_results:
            chunk_id = result.chunk.id
            if chunk_id not in unique_results or result.score > unique_results[chunk_id].score:
                unique_results[chunk_id] = result
        
        # 점수 순으로 정렬
        final_results = list(unique_results.values())
        final_results.sort(key=lambda r: r.score, reverse=True)
        
        # 순위 재조정
        for i, result in enumerate(final_results[:max_results]):
            result.rank = i + 1
        
        return final_results[:max_results]
    
    async def _generate_answer(
        self,
        query: RAGQuery,
        search_results: List[RAGResult]
    ) -> str:
        """검색 결과 기반 답변 생성"""
        
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."
        
        # 검색 결과를 컨텍스트로 구성
        context_chunks = []
        for result in search_results[:3]:  # 상위 3개 결과만 사용
            context_chunks.append(f"[출처 {result.rank}] {result.chunk.content}")
        
        context = "\n\n".join(context_chunks)
        
        # 답변 생성 프롬프트 구성
        answer_prompt = f"""다음 검색된 정보를 바탕으로 질문에 답변해주세요:

**질문**: {query.text}

**검색된 정보**:
{context}

**요구사항**:
1. 검색된 정보만을 기반으로 답변하세요
2. 정확하지 않은 정보는 포함하지 마세요  
3. 출처를 명시하여 답변하세요
4. 뿌리산업 전문 용어를 정확히 사용하세요

**답변**:"""
        
        # 대화형 엔진으로 답변 생성
        if self.conversational_engine:
            try:
                # 임시 세션으로 답변 생성
                temp_session = await self.conversational_engine.start_conversation()
                
                async for result in self.conversational_engine.chat(
                    temp_session, answer_prompt, stream=False
                ):
                    generated_answer = result.response
                    break
                
                await self.conversational_engine.end_conversation(temp_session)
                return generated_answer
                
            except Exception as e:
                logger.error(f"답변 생성 실패: {e}")
        
        # 기본 답변 (검색 결과 요약)
        return f"""검색 결과를 바탕으로 답변드리겠습니다:

{search_results[0].chunk.content[:200]}...

자세한 내용은 검색된 {len(search_results)}개의 문서를 참조해주세요."""
    
    def _generate_cache_key(self, query: RAGQuery) -> str:
        """캐시 키 생성"""
        import hashlib
        
        key_data = f"{query.text}:{query.namespace}:{query.domain}:{query.search_strategy}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[RAGResponse]:
        """캐시된 응답 조회"""
        
        if cache_key in self.result_cache:
            response, cached_time = self.result_cache[cache_key]
            
            # TTL 확인
            if datetime.now() - cached_time < self.cache_ttl:
                return response
            else:
                # 만료된 캐시 삭제
                del self.result_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: RAGResponse):
        """응답 캐시 저장"""
        
        # 캐시 크기 제한
        if len(self.result_cache) >= self.max_cache_size:
            # 가장 오래된 항목 삭제
            oldest_key = min(
                self.result_cache.keys(),
                key=lambda k: self.result_cache[k][1]
            )
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = (response, datetime.now())
    
    def _collect_search_metadata(
        self,
        query: RAGQuery,
        results: List[RAGResult]
    ) -> Dict[str, Any]:
        """검색 메타데이터 수집"""
        
        strategies_used = list(set(r.retrieval_strategy for r in results))
        namespaces_found = list(set(r.chunk.namespace for r in results))
        
        return {
            "strategies_used": strategies_used,
            "namespaces_found": namespaces_found,
            "total_results": len(results),
            "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            "query_domain": query.domain,
            "query_namespace": query.namespace
        }
    
    def _update_performance_stats(
        self,
        query: RAGQuery,
        response: RAGResponse,
        from_cache: bool
    ):
        """성능 통계 업데이트"""
        
        self.performance_stats["total_queries"] += 1
        total = self.performance_stats["total_queries"]
        
        # 평균 쿼리 시간 업데이트
        if not from_cache:
            current_avg = self.performance_stats["avg_query_time"]
            self.performance_stats["avg_query_time"] = (
                (current_avg * (total - 1) + response.processing_time) / total
            )
        
        # 평균 결과 수 업데이트
        current_avg_results = self.performance_stats["avg_results_per_query"]
        self.performance_stats["avg_results_per_query"] = (
            (current_avg_results * (total - 1) + len(response.results)) / total
        )
        
        # 캐시 히트율 업데이트
        cache_hits = self.performance_stats.get("cache_hits", 0)
        if from_cache:
            cache_hits += 1
        
        self.performance_stats["cache_hits"] = cache_hits
        self.performance_stats["cache_hit_rate"] = cache_hits / total
        
        # 네임스페이스 사용 통계
        if query.namespace:
            ns_usage = self.performance_stats["namespace_usage"]
            ns_usage[query.namespace] = ns_usage.get(query.namespace, 0) + 1
        
        # 검색 전략 사용 통계
        strategy_usage = self.performance_stats["search_strategy_usage"]
        strategy_usage[query.search_strategy] = strategy_usage.get(query.search_strategy, 0) + 1
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        
        stats = {
            **self.performance_stats,
            "cache_size": len(self.result_cache),
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600,
            "supported_search_strategies": ["hybrid", "semantic", "keyword", "all"],
            "last_updated": datetime.now().isoformat()
        }
        
        # 실제 컴포넌트들의 통계 추가
        try:
            # RetrievalEngine 통계
            if self.retrieval_engine:
                retrieval_stats = self.retrieval_engine.get_search_statistics()
                stats["retrieval_engine"] = retrieval_stats
            
            # ChunkManager 통계
            if self.chunk_manager:
                chunking_stats = self.chunk_manager.get_chunking_statistics()
                stats["chunk_manager"] = chunking_stats
            
            # NamespaceManager 통계
            if self.namespace_manager:
                # 비동기 메소드이므로 기본 정보만 포함
                stats["namespace_manager"] = {
                    "total_namespaces": len(self.namespace_manager.namespaces),
                    "total_documents": len(self.namespace_manager.document_locations),
                    "available_namespaces": list(self.namespace_manager.namespaces.keys())
                }
            
            # DocumentProcessor 통계
            if self.document_processor:
                processor_stats = self.document_processor.get_processing_statistics()
                stats["document_processor"] = processor_stats
            
        except Exception as e:
            logger.error(f"통계 수집 실패: {e}")
            stats["stats_error"] = str(e)
        
        return stats
    
    async def clear_cache(self):
        """캐시 정리"""
        
        cache_size = len(self.result_cache)
        self.result_cache.clear()
        
        logger.info(f"RAG 캐시 정리 완료: {cache_size}개 항목")
    
    async def get_namespace_info(self, namespace: str) -> Dict[str, Any]:
        """네임스페이스 정보 조회"""
        
        # NamespaceManager가 있으면 실제 정보 조회
        if self.namespace_manager:
            namespace_info = self.namespace_manager.get_namespace_info(namespace)
            if namespace_info:
                return namespace_info
        
        # RetrievalEngine에서 컬렉션 정보 조회
        collection_info = {}
        if self.retrieval_engine:
            try:
                collection_info = await self.retrieval_engine.get_collection_info(namespace)
            except Exception as e:
                logger.error(f"컬렉션 정보 조회 실패: {e}")
                collection_info = {"error": str(e)}
        
        # Fallback 정보 구성
        fallback_info = {
            "namespace": namespace,
            "document_count": collection_info.get("document_count", 0),
            "chunk_count": collection_info.get("document_count", 0),  # 대략적 추정
            "last_updated": datetime.now().isoformat(),
            "domains": [],
            "total_size_mb": 0.0,
            "collection_info": collection_info,
            "fallback_mode": True
        }
        
        return fallback_info
    
    async def cleanup(self):
        """리소스 정리"""
        
        # 캐시 정리
        await self.clear_cache()
        
        # 실제 컴포넌트들 정리
        cleanup_tasks = []
        
        # RetrievalEngine 정리
        if self.retrieval_engine:
            cleanup_tasks.append(self.retrieval_engine.cleanup())
        
        # NamespaceManager 정리
        if self.namespace_manager:
            cleanup_tasks.append(self.namespace_manager.cleanup())
        
        # ChunkManager 정리
        if self.chunk_manager:
            cleanup_tasks.append(self.chunk_manager.cleanup())
        
        # DocumentProcessor 정리
        if self.document_processor:
            cleanup_tasks.append(self.document_processor.cleanup())
        
        # 모든 컴포넌트 정리 실행
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"컴포넌트 정리 중 오류: {e}")
        
        # 메모리 저장소 정리
        if hasattr(self, '_memory_store'):
            self._memory_store.clear()
        
        logger.info("RAG Orchestrator 정리 완료")