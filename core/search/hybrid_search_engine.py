"""
PPuRI-AI Ultimate - 하이브리드 검색 엔진
NotebookLM 스타일의 문서 기반 RAG + 실시간 웹 검색 통합

Features:
- 문서 우선 검색 (Source Grounding)
- 실시간 웹 검색 통합
- 검색 결과 융합 및 순위화
- 출처 추적 및 인용 생성
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, AsyncIterator
import aiohttp
import logging

logger = logging.getLogger(__name__)


class SearchSource(Enum):
    """검색 소스 유형"""
    DOCUMENT = "document"      # 업로드된 문서
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Neo4j 지식 그래프
    WEB_SEARCH = "web_search"  # 실시간 웹 검색
    ACADEMIC = "academic"      # 학술 논문 (Semantic Scholar, arXiv)
    PATENT = "patent"          # 특허 (KIPRIS, Google Patents)


@dataclass
class Citation:
    """인용 정보 (NotebookLM 스타일)"""
    id: str
    source_type: SearchSource
    title: str
    content_snippet: str
    url: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_inline_reference(self) -> str:
        """인라인 참조 형식으로 변환 [1], [2] 등"""
        return f"[{self.id}]"

    def to_footnote(self) -> str:
        """각주 형식으로 변환"""
        source_label = {
            SearchSource.DOCUMENT: "문서",
            SearchSource.KNOWLEDGE_GRAPH: "지식그래프",
            SearchSource.WEB_SEARCH: "웹",
            SearchSource.ACADEMIC: "논문",
            SearchSource.PATENT: "특허"
        }.get(self.source_type, "출처")

        if self.url:
            return f"[{self.id}] {self.title} ({source_label}) - {self.url}"
        return f"[{self.id}] {self.title} ({source_label})"


@dataclass
class SearchResult:
    """통합 검색 결과"""
    query: str
    results: List[Citation]
    total_count: int
    search_time_ms: float
    sources_used: List[SearchSource]

    def get_citations_by_source(self, source: SearchSource) -> List[Citation]:
        """소스별 인용 필터링"""
        return [c for c in self.results if c.source_type == source]


class HybridSearchEngine:
    """
    NotebookLM 스타일 하이브리드 검색 엔진

    검색 우선순위:
    1. 업로드된 문서 (Source Grounding - NotebookLM 핵심)
    2. 지식 그래프 (Neo4j)
    3. 웹 검색 (선택적)
    4. 학술/특허 검색 (선택적)
    """

    def __init__(
        self,
        rag_orchestrator,
        graph_manager,
        web_search_enabled: bool = True,
        academic_search_enabled: bool = True
    ):
        self.rag = rag_orchestrator
        self.graph = graph_manager
        self.web_search_enabled = web_search_enabled
        self.academic_search_enabled = academic_search_enabled

        # 웹 검색 API 설정
        self.brave_api_key = None
        self.serper_api_key = None

        # 검색 가중치 (NotebookLM처럼 문서 우선)
        self.source_weights = {
            SearchSource.DOCUMENT: 1.0,        # 최우선
            SearchSource.KNOWLEDGE_GRAPH: 0.9,
            SearchSource.ACADEMIC: 0.8,
            SearchSource.PATENT: 0.7,
            SearchSource.WEB_SEARCH: 0.6,      # 보조적
        }

        self._initialized = False

    async def initialize(self) -> bool:
        """엔진 초기화"""
        try:
            # API 키 로드
            import os
            self.brave_api_key = os.getenv("BRAVE_API_KEY")
            self.serper_api_key = os.getenv("SERPER_API_KEY")

            self._initialized = True
            logger.info("HybridSearchEngine initialized")
            return True
        except Exception as e:
            logger.error(f"HybridSearchEngine initialization failed: {e}")
            return False

    async def search(
        self,
        query: str,
        sources: Optional[List[SearchSource]] = None,
        max_results: int = 10,
        include_web: bool = True,
        industry_filter: Optional[str] = None
    ) -> SearchResult:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            sources: 검색할 소스 목록 (None이면 전체)
            max_results: 최대 결과 수
            include_web: 웹 검색 포함 여부
            industry_filter: 뿌리산업 필터 (casting, mold, welding 등)
        """
        start_time = datetime.now()

        if sources is None:
            sources = [SearchSource.DOCUMENT, SearchSource.KNOWLEDGE_GRAPH]
            if include_web and self.web_search_enabled:
                sources.append(SearchSource.WEB_SEARCH)
            if self.academic_search_enabled:
                sources.extend([SearchSource.ACADEMIC, SearchSource.PATENT])

        # 병렬 검색 실행
        search_tasks = []

        if SearchSource.DOCUMENT in sources:
            search_tasks.append(self._search_documents(query, industry_filter))

        if SearchSource.KNOWLEDGE_GRAPH in sources:
            search_tasks.append(self._search_knowledge_graph(query, industry_filter))

        if SearchSource.WEB_SEARCH in sources and self.web_search_enabled:
            search_tasks.append(self._search_web(query, industry_filter))

        if SearchSource.ACADEMIC in sources:
            search_tasks.append(self._search_academic(query))

        if SearchSource.PATENT in sources:
            search_tasks.append(self._search_patents(query, industry_filter))

        # 병렬 실행 및 결과 수집
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 결과 통합
        citations: List[Citation] = []
        for result in all_results:
            if isinstance(result, Exception):
                logger.warning(f"Search task failed: {result}")
                continue
            if isinstance(result, list):
                citations.extend(result)

        # 가중치 기반 정렬
        citations = self._rank_results(citations, query)

        # 상위 결과만 반환
        citations = citations[:max_results]

        # ID 재할당
        for i, citation in enumerate(citations, 1):
            citation.id = str(i)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return SearchResult(
            query=query,
            results=citations,
            total_count=len(citations),
            search_time_ms=elapsed_ms,
            sources_used=sources
        )

    async def _search_documents(
        self,
        query: str,
        industry_filter: Optional[str] = None
    ) -> List[Citation]:
        """문서 RAG 검색 (Source Grounding)"""
        try:
            # RAG 오케스트레이터 사용
            rag_results = await self.rag.retrieve(
                query=query,
                top_k=5,
                namespace=industry_filter
            )

            citations = []
            for i, result in enumerate(rag_results.get("documents", [])):
                citations.append(Citation(
                    id=f"doc_{i}",
                    source_type=SearchSource.DOCUMENT,
                    title=result.get("title", "문서"),
                    content_snippet=result.get("content", "")[:500],
                    page_number=result.get("page"),
                    confidence=result.get("score", 0.0),
                    metadata=result.get("metadata", {})
                ))

            return citations
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []

    async def _search_knowledge_graph(
        self,
        query: str,
        industry_filter: Optional[str] = None
    ) -> List[Citation]:
        """지식 그래프 검색"""
        try:
            # Neo4j 그래프 검색
            graph_results = await self.graph.search_entities(
                query=query,
                limit=5,
                industry=industry_filter
            )

            citations = []
            for i, result in enumerate(graph_results):
                citations.append(Citation(
                    id=f"kg_{i}",
                    source_type=SearchSource.KNOWLEDGE_GRAPH,
                    title=result.get("name", "지식 노드"),
                    content_snippet=result.get("description", ""),
                    confidence=result.get("relevance", 0.0),
                    metadata={
                        "node_type": result.get("type"),
                        "relationships": result.get("relationships", [])
                    }
                ))

            return citations
        except Exception as e:
            logger.error(f"Knowledge graph search failed: {e}")
            return []

    async def _search_web(
        self,
        query: str,
        industry_filter: Optional[str] = None
    ) -> List[Citation]:
        """실시간 웹 검색"""
        try:
            # 뿌리산업 컨텍스트 추가
            enhanced_query = query
            if industry_filter:
                industry_terms = {
                    "casting": "주조 기술",
                    "mold": "금형 기술",
                    "welding": "용접 기술",
                    "surface": "표면처리 기술",
                    "heat": "열처리 기술",
                    "forming": "소성가공 기술"
                }
                enhanced_query = f"{industry_terms.get(industry_filter, '')} {query}"

            citations = []

            # Brave Search API 사용
            if self.brave_api_key:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "X-Subscription-Token": self.brave_api_key,
                        "Accept": "application/json"
                    }
                    params = {
                        "q": enhanced_query,
                        "count": 5,
                        "search_lang": "ko"
                    }

                    async with session.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers=headers,
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            for i, result in enumerate(data.get("web", {}).get("results", [])):
                                citations.append(Citation(
                                    id=f"web_{i}",
                                    source_type=SearchSource.WEB_SEARCH,
                                    title=result.get("title", ""),
                                    content_snippet=result.get("description", ""),
                                    url=result.get("url"),
                                    confidence=0.7,  # 웹 결과는 기본 신뢰도
                                    metadata={"source": "brave_search"}
                                ))

            return citations
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    async def _search_academic(self, query: str) -> List[Citation]:
        """학술 논문 검색 (Semantic Scholar)"""
        try:
            citations = []

            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "limit": 5,
                    "fields": "title,abstract,url,year,citationCount"
                }

                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for i, paper in enumerate(data.get("data", [])):
                            citations.append(Citation(
                                id=f"paper_{i}",
                                source_type=SearchSource.ACADEMIC,
                                title=paper.get("title", ""),
                                content_snippet=paper.get("abstract", "")[:500] if paper.get("abstract") else "",
                                url=paper.get("url"),
                                confidence=min(0.9, 0.5 + (paper.get("citationCount", 0) / 1000)),
                                metadata={
                                    "year": paper.get("year"),
                                    "citations": paper.get("citationCount")
                                }
                            ))

            return citations
        except Exception as e:
            logger.error(f"Academic search failed: {e}")
            return []

    async def _search_patents(
        self,
        query: str,
        industry_filter: Optional[str] = None
    ) -> List[Citation]:
        """특허 검색 (KIPRIS/Google Patents)"""
        # TODO: KIPRIS API 연동 구현
        # 현재는 플레이스홀더
        return []

    def _rank_results(
        self,
        citations: List[Citation],
        query: str
    ) -> List[Citation]:
        """결과 순위화 (가중치 기반)"""
        for citation in citations:
            # 소스별 가중치 적용
            source_weight = self.source_weights.get(citation.source_type, 0.5)
            citation.confidence *= source_weight

        # 신뢰도 기준 정렬
        return sorted(citations, key=lambda c: c.confidence, reverse=True)

    def format_response_with_citations(
        self,
        response: str,
        citations: List[Citation]
    ) -> Dict[str, Any]:
        """
        NotebookLM 스타일 응답 포맷팅

        Returns:
            {
                "answer": "인라인 인용이 포함된 응답 [1][2]...",
                "citations": [...],
                "footnotes": ["[1] 출처1...", "[2] 출처2..."]
            }
        """
        footnotes = [c.to_footnote() for c in citations]

        return {
            "answer": response,
            "citations": [
                {
                    "id": c.id,
                    "source_type": c.source_type.value,
                    "title": c.title,
                    "snippet": c.content_snippet,
                    "url": c.url,
                    "confidence": c.confidence
                }
                for c in citations
            ],
            "footnotes": footnotes
        }
