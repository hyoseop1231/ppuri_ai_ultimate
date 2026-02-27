"""
PPuRI-AI Ultimate - 외부 검색 커넥터

SurfSense 스타일 외부 소스 통합:
- Tavily: AI 최적화 웹 검색
- Semantic Scholar: 학술 논문 검색
- KIPRIS: 한국 특허 검색
- DuckDuckGo: 무료 웹 검색 (폴백)
- arXiv: 프리프린트 논문 검색
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import aiohttp
import httpx

logger = logging.getLogger(__name__)


class ConnectorType(Enum):
    """커넥터 타입"""
    WEB_SEARCH = "web_search"
    ACADEMIC = "academic"
    PATENT = "patent"
    NEWS = "news"
    CODE = "code"


@dataclass
class SearchResult:
    """검색 결과"""
    title: str
    content: str
    url: Optional[str] = None
    source: str = ""
    connector_type: ConnectorType = ConnectorType.WEB_SEARCH
    score: float = 0.0
    published_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self, index: int) -> Dict[str, Any]:
        """인용 형식으로 변환"""
        return {
            "id": str(index),
            "title": self.title,
            "snippet": self.content[:300],
            "url": self.url,
            "source": self.source,
            "type": self.connector_type.value
        }


class BaseConnector(ABC):
    """커넥터 베이스 클래스"""

    connector_type: ConnectorType = ConnectorType.WEB_SEARCH

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass


class TavilyConnector(BaseConnector):
    """
    Tavily AI 검색 커넥터

    특징:
    - AI 최적화 검색 결과
    - 검색 깊이 조절 (basic/advanced)
    - 답변 포함 옵션
    """

    connector_type = ConnectorType.WEB_SEARCH

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",  # basic or advanced
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        max_results: int = 10,
        include_answer: bool = True,
        include_raw_content: bool = False
    ) -> List[SearchResult]:
        """Tavily 검색"""
        if not self.api_key:
            logger.warning("Tavily API key not set")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "include_answer": include_answer,
                    "include_raw_content": include_raw_content
                }

                if include_domains:
                    payload["include_domains"] = include_domains
                if exclude_domains:
                    payload["exclude_domains"] = exclude_domains

                async with session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Tavily error: {response.status}")
                        return []

                    data = await response.json()

                    results = []

                    # AI 생성 답변 (있다면)
                    if include_answer and data.get("answer"):
                        results.append(SearchResult(
                            title="AI Generated Answer",
                            content=data["answer"],
                            source="tavily_ai",
                            connector_type=ConnectorType.WEB_SEARCH,
                            score=1.0
                        ))

                    # 검색 결과
                    for item in data.get("results", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            url=item.get("url"),
                            source="tavily",
                            connector_type=ConnectorType.WEB_SEARCH,
                            score=item.get("score", 0.0),
                            metadata={
                                "raw_content": item.get("raw_content"),
                                "published_date": item.get("published_date")
                            }
                        ))

                    return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    async def health_check(self) -> bool:
        return bool(self.api_key)


class SemanticScholarConnector(BaseConnector):
    """
    Semantic Scholar 학술 검색 커넥터

    특징:
    - 학술 논문 검색
    - 인용 수 기반 스코어링
    - 저자/연도 필터링
    """

    connector_type = ConnectorType.ACADEMIC

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # 선택적 (rate limit 증가)
        self.base_url = "https://api.semanticscholar.org/graph/v1"

    async def search(
        self,
        query: str,
        fields: str = "title,abstract,url,year,citationCount,authors,venue",
        limit: int = 10,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        open_access_only: bool = False
    ) -> List[SearchResult]:
        """Semantic Scholar 검색"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "query": query,
                    "limit": limit,
                    "fields": fields
                }

                if year_start and year_end:
                    params["year"] = f"{year_start}-{year_end}"
                elif year_start:
                    params["year"] = f"{year_start}-"
                elif year_end:
                    params["year"] = f"-{year_end}"

                if open_access_only:
                    params["openAccessPdf"] = ""

                headers = {}
                if self.api_key:
                    headers["x-api-key"] = self.api_key

                async with session.get(
                    f"{self.base_url}/paper/search",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Semantic Scholar error: {response.status}")
                        return []

                    data = await response.json()

                    results = []
                    for paper in data.get("data", []):
                        # 인용 수 기반 스코어 (로그 스케일)
                        citation_count = paper.get("citationCount", 0)
                        import math
                        score = min(1.0, math.log10(citation_count + 1) / 4)

                        authors = [a.get("name", "") for a in paper.get("authors", [])]

                        results.append(SearchResult(
                            title=paper.get("title", ""),
                            content=paper.get("abstract", "") or "",
                            url=paper.get("url"),
                            source="semantic_scholar",
                            connector_type=ConnectorType.ACADEMIC,
                            score=score,
                            published_date=str(paper.get("year", "")),
                            metadata={
                                "year": paper.get("year"),
                                "citation_count": citation_count,
                                "authors": authors,
                                "venue": paper.get("venue")
                            }
                        ))

                    return results

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    async def health_check(self) -> bool:
        return True  # API key 없이도 사용 가능


class ArxivConnector(BaseConnector):
    """
    arXiv 프리프린트 검색 커넥터
    """

    connector_type = ConnectorType.ACADEMIC

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",  # relevance, lastUpdatedDate, submittedDate
        categories: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """arXiv 검색"""
        try:
            import feedparser

            search_query = f"all:{query}"
            if categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"({search_query}) AND ({cat_query})"

            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return []

                    text = await response.text()

                    # feedparser로 Atom 피드 파싱
                    loop = asyncio.get_event_loop()
                    feed = await loop.run_in_executor(None, feedparser.parse, text)

                    results = []
                    for entry in feed.entries:
                        results.append(SearchResult(
                            title=entry.get("title", "").replace("\n", " "),
                            content=entry.get("summary", "").replace("\n", " "),
                            url=entry.get("link"),
                            source="arxiv",
                            connector_type=ConnectorType.ACADEMIC,
                            score=0.7,  # arXiv는 피어리뷰 전이므로 기본 스코어
                            published_date=entry.get("published", ""),
                            metadata={
                                "authors": [a.get("name") for a in entry.get("authors", [])],
                                "categories": [t.get("term") for t in entry.get("tags", [])],
                                "arxiv_id": entry.get("id", "").split("/")[-1]
                            }
                        ))

                    return results

        except ImportError:
            logger.warning("feedparser not installed for arXiv search")
            return []
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    async def health_check(self) -> bool:
        return True


class DuckDuckGoConnector(BaseConnector):
    """
    DuckDuckGo 검색 커넥터 (무료 폴백)
    """

    connector_type = ConnectorType.WEB_SEARCH

    async def search(
        self,
        query: str,
        max_results: int = 10,
        region: str = "kr-kr"  # 한국어 결과 우선
    ) -> List[SearchResult]:
        """DuckDuckGo 검색"""
        try:
            from duckduckgo_search import DDGS

            loop = asyncio.get_event_loop()

            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.text(
                        query,
                        region=region,
                        max_results=max_results
                    ))

            results_data = await loop.run_in_executor(None, _search)

            results = []
            for item in results_data:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    content=item.get("body", ""),
                    url=item.get("href"),
                    source="duckduckgo",
                    connector_type=ConnectorType.WEB_SEARCH,
                    score=0.6  # 무료 검색이므로 기본 스코어
                ))

            return results

        except ImportError:
            logger.warning("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    async def health_check(self) -> bool:
        return True


class KIPRISConnector(BaseConnector):
    """
    KIPRIS 한국 특허 검색 커넥터
    """

    connector_type = ConnectorType.PATENT

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("KIPRIS_API_KEY")
        self.base_url = "http://plus.kipris.or.kr/openapi/rest"

    async def search(
        self,
        query: str,
        patent_type: str = "patent",  # patent, utility, design
        max_results: int = 10
    ) -> List[SearchResult]:
        """KIPRIS 특허 검색"""
        if not self.api_key:
            logger.warning("KIPRIS API key not set")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "accessKey": self.api_key,
                    "word": query,
                    "numOfRows": max_results,
                    "patent": "true" if patent_type == "patent" else "false",
                    "utility": "true" if patent_type == "utility" else "false",
                    "design": "true" if patent_type == "design" else "false"
                }

                async with session.get(
                    f"{self.base_url}/patentSearch",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return []

                    # XML 응답 파싱
                    text = await response.text()

                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(text)

                    results = []
                    for item in root.findall(".//item"):
                        title = item.findtext("inventionTitle", "")
                        abstract = item.findtext("astrtCont", "")
                        app_num = item.findtext("applicationNumber", "")
                        app_date = item.findtext("applicationDate", "")

                        results.append(SearchResult(
                            title=title,
                            content=abstract,
                            url=f"https://doi.org/10.8080/{app_num}" if app_num else None,
                            source="kipris",
                            connector_type=ConnectorType.PATENT,
                            score=0.8,
                            published_date=app_date,
                            metadata={
                                "application_number": app_num,
                                "application_date": app_date
                            }
                        ))

                    return results

        except Exception as e:
            logger.error(f"KIPRIS search failed: {e}")
            return []

    async def health_check(self) -> bool:
        return bool(self.api_key)


class SearchOrchestrator:
    """
    검색 오케스트레이터

    여러 커넥터를 통합하여 병렬 검색 및 결과 병합
    """

    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}
        self._initialized = False

    def register(self, name: str, connector: BaseConnector):
        """커넥터 등록"""
        self.connectors[name] = connector
        logger.info(f"Registered connector: {name}")

    async def initialize(self):
        """기본 커넥터들 초기화"""
        # Tavily (메인 웹 검색)
        self.register("tavily", TavilyConnector())

        # Semantic Scholar (학술)
        self.register("semantic_scholar", SemanticScholarConnector())

        # arXiv (프리프린트)
        self.register("arxiv", ArxivConnector())

        # DuckDuckGo (무료 폴백)
        self.register("duckduckgo", DuckDuckGoConnector())

        # KIPRIS (특허)
        self.register("kipris", KIPRISConnector())

        self._initialized = True
        logger.info(f"SearchOrchestrator initialized with {len(self.connectors)} connectors")

    async def search(
        self,
        query: str,
        connector_types: Optional[List[ConnectorType]] = None,
        connector_names: Optional[List[str]] = None,
        max_results_per_connector: int = 5,
        timeout_seconds: float = 30.0
    ) -> List[SearchResult]:
        """
        통합 검색

        Args:
            query: 검색 쿼리
            connector_types: 검색할 커넥터 타입 필터
            connector_names: 검색할 커넥터 이름 필터
            max_results_per_connector: 커넥터당 최대 결과 수
            timeout_seconds: 타임아웃
        """
        if not self._initialized:
            await self.initialize()

        # 검색할 커넥터 결정
        connectors_to_use = {}
        for name, connector in self.connectors.items():
            if connector_names and name not in connector_names:
                continue
            if connector_types and connector.connector_type not in connector_types:
                continue
            connectors_to_use[name] = connector

        # 병렬 검색
        async def search_with_timeout(name: str, connector: BaseConnector):
            try:
                return await asyncio.wait_for(
                    connector.search(query, max_results=max_results_per_connector),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Connector {name} timed out")
                return []
            except Exception as e:
                logger.error(f"Connector {name} failed: {e}")
                return []

        tasks = [
            search_with_timeout(name, conn)
            for name, conn in connectors_to_use.items()
        ]

        all_results = await asyncio.gather(*tasks)

        # 결과 병합
        merged = []
        for results in all_results:
            merged.extend(results)

        # 스코어 기준 정렬
        merged.sort(key=lambda x: x.score, reverse=True)

        return merged

    async def search_web(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """웹 검색만"""
        return await self.search(
            query,
            connector_types=[ConnectorType.WEB_SEARCH],
            max_results_per_connector=max_results
        )

    async def search_academic(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """학술 검색만"""
        return await self.search(
            query,
            connector_types=[ConnectorType.ACADEMIC],
            max_results_per_connector=max_results
        )

    async def search_patent(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """특허 검색만"""
        return await self.search(
            query,
            connector_types=[ConnectorType.PATENT],
            max_results_per_connector=max_results
        )

    async def health_check(self) -> Dict[str, bool]:
        """모든 커넥터 헬스 체크"""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = await connector.health_check()
        return results


# 싱글톤
_search_orchestrator: Optional[SearchOrchestrator] = None


async def get_search_orchestrator() -> SearchOrchestrator:
    """검색 오케스트레이터 싱글톤"""
    global _search_orchestrator

    if _search_orchestrator is None:
        _search_orchestrator = SearchOrchestrator()
        await _search_orchestrator.initialize()

    return _search_orchestrator
