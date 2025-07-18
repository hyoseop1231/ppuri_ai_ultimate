"""
Retrieval Engine - 고성능 벡터 검색 엔진

ChromaDB 기반 벡터 검색과 하이브리드 검색을 제공하는
실제 동작하는 검색 엔진.

Features:
- ChromaDB 벡터 데이터베이스 연동
- 다중 임베딩 모델 지원
- 하이브리드 검색 (벡터 + 키워드)
- 네임스페이스 기반 격리
- 지능형 리랭킹
- 검색 결과 최적화
- 실시간 성능 모니터링
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import hashlib
from pathlib import Path

# ChromaDB 임포트
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# 임베딩 모델 임포트
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """검색 설정"""
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 10
    similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    rerank_top_k: int = 20
    collection_prefix: str = "ppuri"


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class SearchResult:
    """검색 결과"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    search_type: str = "vector"  # vector, keyword, hybrid


@dataclass
class SearchQuery:
    """검색 쿼리"""
    text: str
    namespace: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 10
    similarity_threshold: float = 0.7
    search_type: str = "hybrid"  # vector, keyword, hybrid


@dataclass
class SearchResponse:
    """검색 응답"""
    query: SearchQuery
    results: List[SearchResult]
    total_found: int
    search_time: float
    embedding_time: float
    rerank_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalEngine:
    """
    고성능 벡터 검색 엔진
    
    ChromaDB를 기반으로 한 실제 동작하는 벡터 검색 시스템.
    뿌리산업 특화 검색 최적화와 하이브리드 검색을 제공.
    """
    
    def __init__(
        self,
        config_manager,
        search_config: Optional[SearchConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        korean_optimizer=None
    ):
        self.config_manager = config_manager
        self.search_config = search_config or SearchConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.korean_optimizer = korean_optimizer
        
        # ChromaDB 클라이언트
        self.chroma_client = None
        self.collections: Dict[str, Any] = {}
        
        # 임베딩 모델
        self.embedding_model = None
        self.embedding_function = None
        
        # 검색 통계
        self.search_stats = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "avg_embedding_time": 0.0,
            "avg_results_count": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "by_search_type": {
                "vector": 0,
                "keyword": 0,
                "hybrid": 0
            },
            "by_namespace": {}
        }
        
        # 결과 캐시
        self.result_cache: Dict[str, Tuple[SearchResponse, datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)
        self.max_cache_size = 1000
        
        # 뿌리산업 특화 키워드
        self.industry_keywords = self._init_industry_keywords()
        
        logger.info("Retrieval Engine 초기화 완료")
    
    def _init_industry_keywords(self) -> Dict[str, List[str]]:
        """뿌리산업 키워드 초기화"""
        
        return {
            "주조": ["주조", "캐스팅", "용탕", "응고", "주형", "다이캐스팅"],
            "금형": ["금형", "다이", "몰드", "프레스", "성형", "사출"],
            "소성가공": ["소성가공", "단조", "압연", "인발", "전조", "압출"],
            "용접": ["용접", "접합", "아크용접", "가스용접", "저항용접"],
            "표면처리": ["표면처리", "도금", "코팅", "양극산화", "침탄"],
            "열처리": ["열처리", "소입", "소성", "담금질", "풀림", "노멀라이징"]
        }
    
    async def initialize(self):
        """검색 엔진 초기화"""
        
        logger.info("검색 엔진 초기화 중...")
        
        # 1. ChromaDB 초기화
        await self._initialize_chromadb()
        
        # 2. 임베딩 모델 초기화
        await self._initialize_embedding_model()
        
        # 3. 컬렉션 설정
        await self._setup_collections()
        
        logger.info("✅ 검색 엔진 초기화 완료")
    
    async def _initialize_chromadb(self):
        """ChromaDB 초기화"""
        
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB가 설치되지 않음. pip install chromadb 실행 필요")
            raise ImportError("ChromaDB 필요")
        
        try:
            # 데이터 디렉토리 설정
            db_path = Path("./vector_db_data")
            db_path.mkdir(exist_ok=True)
            
            # ChromaDB 클라이언트 생성
            self.chroma_client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB 초기화 완료: {db_path}")
            
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            # Fallback to in-memory client
            self.chroma_client = chromadb.EphemeralClient()
            logger.warning("메모리 내 ChromaDB 사용")
    
    async def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        
        if not EMBEDDING_AVAILABLE:
            logger.warning("sentence-transformers 미설치. 기본 임베딩 사용")
            return
        
        try:
            # SentenceTransformer 모델 로드
            self.embedding_model = SentenceTransformer(
                self.embedding_config.model_name,
                device=self.embedding_config.device
            )
            
            # ChromaDB 임베딩 함수 생성
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_config.model_name,
                device=self.embedding_config.device,
                normalize_embeddings=self.embedding_config.normalize_embeddings
            )
            
            logger.info(f"임베딩 모델 로드 완료: {self.embedding_config.model_name}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            # 기본 임베딩 함수 사용
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    async def _setup_collections(self):
        """컬렉션 설정"""
        
        # 기본 컬렉션들 생성
        default_namespaces = ["default", "주조", "금형", "소성가공", "용접", "표면처리", "열처리"]
        
        for namespace in default_namespaces:
            collection_name = f"{self.search_config.collection_prefix}_{namespace}"
            
            try:
                # 기존 컬렉션 확인
                try:
                    collection = self.chroma_client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.debug(f"기존 컬렉션 사용: {collection_name}")
                    
                except Exception:
                    # 새 컬렉션 생성
                    collection = self.chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"namespace": namespace, "created_at": datetime.now().isoformat()}
                    )
                    logger.info(f"새 컬렉션 생성: {collection_name}")
                
                self.collections[namespace] = collection
                
            except Exception as e:
                logger.error(f"컬렉션 설정 실패 ({namespace}): {e}")
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "default"
    ) -> bool:
        """문서 인덱싱"""
        
        if namespace not in self.collections:
            await self._create_collection(namespace)
        
        collection = self.collections[namespace]
        
        try:
            # 문서 데이터 준비
            ids = []
            contents = []
            metadatas = []
            
            for doc in documents:
                doc_id = doc.get("id") or str(time.time())
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # 한국어 최적화
                if self.korean_optimizer and content:
                    result = await self.korean_optimizer.process_korean_text(content)
                    content = result.normalized_text
                    metadata["industry_terms"] = result.industry_terms
                    metadata["confidence"] = result.confidence_score
                
                ids.append(doc_id)
                contents.append(content)
                metadatas.append({
                    **metadata,
                    "namespace": namespace,
                    "indexed_at": datetime.now().isoformat(),
                    "content_length": len(content)
                })
            
            # ChromaDB에 추가
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"문서 인덱싱 완료: {len(documents)}개 ({namespace})")
            return True
            
        except Exception as e:
            logger.error(f"문서 인덱싱 실패 ({namespace}): {e}")
            return False
    
    async def search(
        self,
        query: SearchQuery
    ) -> SearchResponse:
        """
        검색 실행
        
        Args:
            query: 검색 쿼리
            
        Returns:
            SearchResponse: 검색 응답
        """
        
        start_time = time.time()
        
        try:
            # 1. 캐시 확인
            cache_key = self._generate_cache_key(query)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                self._update_search_stats(query, cached_response, True)
                return cached_response
            
            # 2. 검색 실행
            if query.search_type == "vector":
                results = await self._vector_search(query)
            elif query.search_type == "keyword":
                results = await self._keyword_search(query)
            else:  # hybrid
                results = await self._hybrid_search(query)
            
            # 3. 리랭킹 (선택적)
            rerank_start = time.time()
            if self.search_config.enable_reranking and len(results) > 1:
                results = await self._rerank_results(query, results)
            rerank_time = time.time() - rerank_start
            
            # 4. 응답 구성
            search_time = time.time() - start_time
            
            response = SearchResponse(
                query=query,
                results=results[:query.max_results],
                total_found=len(results),
                search_time=search_time,
                embedding_time=getattr(self, '_last_embedding_time', 0.0),
                rerank_time=rerank_time,
                metadata={
                    "cache_used": False,
                    "collection_queried": query.namespace or "default",
                    "search_strategy": query.search_type
                }
            )
            
            # 5. 캐시 저장
            self._cache_response(cache_key, response)
            
            # 6. 통계 업데이트
            self._update_search_stats(query, response, False)
            
            logger.debug(
                f"검색 완료: {len(results)}개 결과 "
                f"({search_time:.3f}초)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_found=0,
                search_time=time.time() - start_time,
                embedding_time=0.0,
                rerank_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def _vector_search(self, query: SearchQuery) -> List[SearchResult]:
        """벡터 검색"""
        
        namespace = query.namespace or "default"
        if namespace not in self.collections:
            return []
        
        collection = self.collections[namespace]
        
        try:
            # 쿼리 임베딩 생성
            embedding_start = time.time()
            
            # 한국어 최적화
            search_text = query.text
            if self.korean_optimizer:
                result = await self.korean_optimizer.process_korean_text(query.text)
                search_text = result.normalized_text
            
            self._last_embedding_time = time.time() - embedding_start
            
            # 벡터 검색 실행
            search_results = collection.query(
                query_texts=[search_text],
                n_results=min(query.max_results * 2, 50),  # 오버페치 후 필터링
                where=self._build_where_clause(query.filters),
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 변환
            results = []
            if search_results["ids"] and search_results["ids"][0]:
                for i, doc_id in enumerate(search_results["ids"][0]):
                    distance = search_results["distances"][0][i]
                    similarity = 1.0 - distance  # ChromaDB는 거리 반환
                    
                    if similarity >= query.similarity_threshold:
                        result = SearchResult(
                            id=doc_id,
                            content=search_results["documents"][0][i],
                            score=similarity,
                            metadata=search_results["metadatas"][0][i],
                            search_type="vector"
                        )
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """키워드 검색"""
        
        namespace = query.namespace or "default"
        if namespace not in self.collections:
            return []
        
        collection = self.collections[namespace]
        
        try:
            # 키워드 추출
            keywords = self._extract_keywords(query.text)
            
            results = []
            
            # 각 키워드로 검색
            for keyword in keywords:
                # ChromaDB의 메타데이터 검색 사용
                search_results = collection.query(
                    query_texts=[keyword],
                    n_results=20,
                    where=self._build_where_clause(query.filters),
                    include=["documents", "metadatas", "distances"]
                )
                
                if search_results["ids"] and search_results["ids"][0]:
                    for i, doc_id in enumerate(search_results["ids"][0]):
                        content = search_results["documents"][0][i]
                        
                        # 키워드 매칭 점수 계산
                        keyword_score = self._calculate_keyword_score(keyword, content)
                        
                        if keyword_score >= query.similarity_threshold:
                            result = SearchResult(
                                id=doc_id,
                                content=content,
                                score=keyword_score,
                                metadata=search_results["metadatas"][0][i],
                                search_type="keyword"
                            )
                            results.append(result)
            
            # 중복 제거 및 점수 정렬
            unique_results = {}
            for result in results:
                if result.id not in unique_results or result.score > unique_results[result.id].score:
                    unique_results[result.id] = result
            
            return sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"키워드 검색 실패: {e}")
            return []
    
    async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """하이브리드 검색 (벡터 + 키워드)"""
        
        try:
            # 벡터 검색과 키워드 검색 병렬 실행
            vector_task = asyncio.create_task(self._vector_search(query))
            keyword_task = asyncio.create_task(self._keyword_search(query))
            
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )
            
            # 예외 처리
            if isinstance(vector_results, Exception):
                logger.error(f"벡터 검색 실패: {vector_results}")
                vector_results = []
            
            if isinstance(keyword_results, Exception):
                logger.error(f"키워드 검색 실패: {keyword_results}")
                keyword_results = []
            
            # 결과 통합
            combined_results = self._combine_search_results(
                vector_results, keyword_results, query
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            return []
    
    def _combine_search_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        query: SearchQuery
    ) -> List[SearchResult]:
        """검색 결과 통합"""
        
        # 가중치 설정
        vector_weight = 0.7
        keyword_weight = 0.3
        
        # 뿌리산업 용어가 있으면 키워드 검색 가중치 증가
        industry_terms = self._detect_industry_terms(query.text)
        if industry_terms:
            vector_weight = 0.6
            keyword_weight = 0.4
        
        # 결과 통합
        combined = {}
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            combined[result.id] = SearchResult(
                id=result.id,
                content=result.content,
                score=result.score * vector_weight,
                metadata=result.metadata,
                search_type="hybrid"
            )
        
        # 키워드 검색 결과 통합
        for result in keyword_results:
            if result.id in combined:
                # 기존 결과와 점수 조합
                combined[result.id].score += result.score * keyword_weight
                combined[result.id].metadata["keyword_boost"] = True
            else:
                # 새 결과 추가
                combined[result.id] = SearchResult(
                    id=result.id,
                    content=result.content,
                    score=result.score * keyword_weight,
                    metadata=result.metadata,
                    search_type="hybrid"
                )
        
        # 점수순 정렬
        results = list(combined.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        
        keywords = []
        
        # 기본 키워드 분할
        words = text.split()
        keywords.extend([w.strip(".,!?") for w in words if len(w) > 2])
        
        # 뿌리산업 특화 키워드 추출
        industry_terms = self._detect_industry_terms(text)
        keywords.extend(industry_terms)
        
        # 중복 제거
        return list(set(keywords))
    
    def _detect_industry_terms(self, text: str) -> List[str]:
        """뿌리산업 용어 감지"""
        
        detected_terms = []
        
        for category, terms in self.industry_keywords.items():
            for term in terms:
                if term in text:
                    detected_terms.append(term)
        
        return detected_terms
    
    def _calculate_keyword_score(self, keyword: str, content: str) -> float:
        """키워드 매칭 점수 계산"""
        
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        # 기본 매칭 점수
        if keyword_lower in content_lower:
            base_score = 0.8
            
            # 정확한 단어 매칭 보너스
            import re
            if re.search(rf'\b{re.escape(keyword_lower)}\b', content_lower):
                base_score += 0.1
            
            # 빈도 보너스
            frequency = content_lower.count(keyword_lower)
            freq_bonus = min(0.1, frequency * 0.02)
            
            # 뿌리산업 용어 보너스
            if keyword in self._detect_industry_terms(content):
                base_score += 0.1
            
            return min(1.0, base_score + freq_bonus)
        
        return 0.0
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ChromaDB where 절 구성"""
        
        if not filters:
            return None
        
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, str):
                where_clause[key] = {"$eq": value}
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                where_clause[key] = value
        
        return where_clause
    
    async def _rerank_results(
        self,
        query: SearchQuery,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """검색 결과 리랭킹"""
        
        if len(results) <= 1:
            return results
        
        try:
            # 1. 뿌리산업 용어 부스팅
            industry_boosted = self._apply_industry_boosting(query, results)
            
            # 2. 컨텍스트 관련성 평가
            context_scored = await self._apply_context_scoring(query, industry_boosted)
            
            # 3. 최종 정렬
            context_scored.sort(key=lambda x: x.score, reverse=True)
            
            return context_scored
            
        except Exception as e:
            logger.error(f"리랭킹 실패: {e}")
            return results
    
    def _apply_industry_boosting(
        self,
        query: SearchQuery,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """뿌리산업 용어 부스팅"""
        
        query_terms = self._detect_industry_terms(query.text)
        
        if not query_terms:
            return results
        
        boosted_results = []
        
        for result in results:
            boost_factor = 1.0
            
            # 결과에서 일치하는 산업 용어 찾기
            result_terms = self._detect_industry_terms(result.content)
            
            # 공통 용어 개수에 따른 부스팅
            common_terms = set(query_terms) & set(result_terms)
            if common_terms:
                boost_factor = 1.0 + (len(common_terms) * 0.1)
            
            # 메타데이터의 산업 용어 확인
            if "industry_terms" in result.metadata:
                meta_terms = result.metadata["industry_terms"]
                if isinstance(meta_terms, list):
                    meta_common = set(query_terms) & set(meta_terms)
                    if meta_common:
                        boost_factor += len(meta_common) * 0.05
            
            # 부스팅 적용
            boosted_result = SearchResult(
                id=result.id,
                content=result.content,
                score=min(1.0, result.score * boost_factor),
                metadata={**result.metadata, "industry_boost": boost_factor},
                search_type=result.search_type
            )
            
            boosted_results.append(boosted_result)
        
        return boosted_results
    
    async def _apply_context_scoring(
        self,
        query: SearchQuery,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """컨텍스트 관련성 점수 적용"""
        
        context_scored = []
        
        for result in results:
            context_score = self._calculate_context_score(query, result)
            
            # 컨텍스트 점수와 기존 점수 조합
            final_score = (result.score * 0.8) + (context_score * 0.2)
            
            context_result = SearchResult(
                id=result.id,
                content=result.content,
                score=min(1.0, final_score),
                metadata={**result.metadata, "context_score": context_score},
                search_type=result.search_type
            )
            
            context_scored.append(context_result)
        
        return context_scored
    
    def _calculate_context_score(
        self,
        query: SearchQuery,
        result: SearchResult
    ) -> float:
        """컨텍스트 관련성 점수 계산"""
        
        score = 0.5  # 기본 점수
        
        # 1. 문서 길이 적절성
        content_length = len(result.content)
        if 200 <= content_length <= 2000:
            score += 0.1
        elif content_length < 100:
            score -= 0.1
        
        # 2. 메타데이터 품질
        metadata = result.metadata
        if "confidence" in metadata:
            confidence = metadata.get("confidence", 0.5)
            score += (confidence - 0.5) * 0.2
        
        # 3. 네임스페이스 일치
        if query.namespace and "namespace" in metadata:
            if metadata["namespace"] == query.namespace:
                score += 0.1
        
        # 4. 최신성 (선택적)
        if "indexed_at" in metadata:
            try:
                indexed_time = datetime.fromisoformat(metadata["indexed_at"])
                age_days = (datetime.now() - indexed_time).days
                if age_days < 30:
                    score += 0.05
            except:
                pass
        
        return max(0.0, min(1.0, score))
    
    async def _create_collection(self, namespace: str):
        """새 컬렉션 생성"""
        
        collection_name = f"{self.search_config.collection_prefix}_{namespace}"
        
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"namespace": namespace, "created_at": datetime.now().isoformat()}
            )
            
            self.collections[namespace] = collection
            logger.info(f"새 컬렉션 생성: {collection_name}")
            
        except Exception as e:
            logger.error(f"컬렉션 생성 실패 ({namespace}): {e}")
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """캐시 키 생성"""
        
        key_data = {
            "text": query.text,
            "namespace": query.namespace,
            "filters": query.filters,
            "search_type": query.search_type,
            "max_results": query.max_results
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[SearchResponse]:
        """캐시된 응답 조회"""
        
        if cache_key in self.result_cache:
            response, cached_time = self.result_cache[cache_key]
            
            if datetime.now() - cached_time < self.cache_ttl:
                self.search_stats["cache_hits"] += 1
                return response
            else:
                del self.result_cache[cache_key]
        
        self.search_stats["cache_misses"] += 1
        return None
    
    def _cache_response(self, cache_key: str, response: SearchResponse):
        """응답 캐시 저장"""
        
        # 캐시 크기 제한
        if len(self.result_cache) >= self.max_cache_size:
            oldest_key = min(
                self.result_cache.keys(),
                key=lambda k: self.result_cache[k][1]
            )
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = (response, datetime.now())
    
    def _update_search_stats(
        self,
        query: SearchQuery,
        response: SearchResponse,
        from_cache: bool
    ):
        """검색 통계 업데이트"""
        
        self.search_stats["total_searches"] += 1
        total = self.search_stats["total_searches"]
        
        if not from_cache:
            # 평균 검색 시간 업데이트
            current_avg = self.search_stats["avg_search_time"]
            self.search_stats["avg_search_time"] = (
                (current_avg * (total - 1) + response.search_time) / total
            )
            
            # 평균 임베딩 시간 업데이트
            current_avg_emb = self.search_stats["avg_embedding_time"]
            self.search_stats["avg_embedding_time"] = (
                (current_avg_emb * (total - 1) + response.embedding_time) / total
            )
        
        # 평균 결과 수 업데이트
        current_avg_results = self.search_stats["avg_results_count"]
        self.search_stats["avg_results_count"] = (
            (current_avg_results * (total - 1) + len(response.results)) / total
        )
        
        # 검색 타입별 통계
        self.search_stats["by_search_type"][query.search_type] += 1
        
        # 네임스페이스별 통계
        namespace = query.namespace or "default"
        if namespace not in self.search_stats["by_namespace"]:
            self.search_stats["by_namespace"][namespace] = 0
        self.search_stats["by_namespace"][namespace] += 1
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        
        cache_total = self.search_stats["cache_hits"] + self.search_stats["cache_misses"]
        cache_hit_rate = (
            self.search_stats["cache_hits"] / cache_total
            if cache_total > 0 else 0.0
        )
        
        return {
            **self.search_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.result_cache),
            "collections_count": len(self.collections),
            "embedding_model": self.embedding_config.model_name,
            "chromadb_available": CHROMADB_AVAILABLE,
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_collection_info(self, namespace: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        
        if namespace not in self.collections:
            return {"error": "컬렉션을 찾을 수 없음"}
        
        collection = self.collections[namespace]
        
        try:
            count = collection.count()
            
            # 샘플 문서 조회
            sample_docs = collection.peek(limit=3)
            
            return {
                "namespace": namespace,
                "document_count": count,
                "sample_documents": sample_docs,
                "collection_name": f"{self.search_config.collection_prefix}_{namespace}",
                "embedding_model": self.embedding_config.model_name
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def clear_cache(self):
        """캐시 정리"""
        
        cache_size = len(self.result_cache)
        self.result_cache.clear()
        
        logger.info(f"검색 캐시 정리 완료: {cache_size}개 항목")
    
    async def delete_documents(
        self,
        document_ids: List[str],
        namespace: str = "default"
    ) -> bool:
        """문서 삭제"""
        
        if namespace not in self.collections:
            return False
        
        collection = self.collections[namespace]
        
        try:
            collection.delete(ids=document_ids)
            logger.info(f"문서 삭제 완료: {len(document_ids)}개 ({namespace})")
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False
    
    async def cleanup(self):
        """검색 엔진 정리"""
        
        # 캐시 정리
        await self.clear_cache()
        
        # 임베딩 모델 정리
        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None
        
        # ChromaDB 연결 정리
        if self.chroma_client:
            # ChromaDB는 자동으로 정리됨
            pass
        
        logger.info("Retrieval Engine 정리 완료")