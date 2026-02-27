# PPuRI-AI Ultimate 완전 구현 스펙

## 최신 기술 조사 결과 기반 통합 설계서

---

## 1. 기술 스택 최종 선정

### 1.1 LLM 제공자 (OpenRouter 통합)

| 모델 | Model ID | 용도 | 가격 (Input/Output) |
|------|----------|------|---------------------|
| **Gemini 3 Pro Preview** | `google/gemini-3-pro-preview` | 메인 추론, 복잡한 분석 | $2/$12 per M tokens |
| **Gemini 3 Flash Preview** | `google/gemini-3-flash-preview` | 빠른 응답, 실시간 채팅 | 저렴 |
| **Nano Banana Pro** | `google/gemini-3-pro-image-preview` | 이미지 생성/편집 | - |
| **Claude 3.5 Sonnet** | `anthropic/claude-3.5-sonnet` | 코드 생성, 정밀 분석 | - |
| **DeepSeek R1** | `deepseek/deepseek-r1` | 추론 특화, 비용 효율 | 매우 저렴 |
| **Qwen3 235B** | `qwen/qwen3-235b` | 한국어 성능 우수 | - |

```python
# core/llm/openrouter_client.py
OPENROUTER_MODELS = {
    "reasoning": "google/gemini-3-pro-preview",      # 복잡한 추론
    "fast": "google/gemini-3-flash-preview",          # 빠른 응답
    "image": "google/gemini-3-pro-image-preview",     # 이미지 생성
    "code": "anthropic/claude-3.5-sonnet",            # 코드 작성
    "cost_efficient": "deepseek/deepseek-r1",         # 비용 효율
    "korean": "qwen/qwen3-235b-a22b",                 # 한국어 특화
}
```

### 1.2 임베딩 모델

| 모델 | 특징 | 차원 | 컨텍스트 |
|------|------|------|---------|
| **BGE-M3** | 다국어, Dense+Sparse 동시 지원 | 1024 | 8192 |
| **BGE-m3-ko** | 한국어 최적화 버전 | 1024 | 8192 |
| **multilingual-e5-large-instruct** | 다국어 균형 성능 | 1024 | 512 |

**선택: BGE-M3 (기본) + BGE-m3-ko (한국어 특화)**

### 1.3 RAG 아키텍처 (ApeRAG/LightRAG 참조)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PPuRI-AI Ultimate RAG Architecture                   │
│                    (ApeRAG + LightRAG + SurfSense 통합)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    5-Type Hybrid Index System                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐ │   │
│  │  │ Vector  │  │FullText │  │  Graph  │  │ Summary │  │ Vision│ │   │
│  │  │ (Qdrant)│  │(Elastic)│  │ (Neo4j) │  │ (LLM)   │  │(CLIP) │ │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └───┬───┘ │   │
│  │       │            │            │            │            │     │   │
│  │       └────────────┴────────────┼────────────┴────────────┘     │   │
│  │                                 │                               │   │
│  │                    ┌────────────▼────────────┐                  │   │
│  │                    │   Reciprocal Rank Fusion │                  │   │
│  │                    │   + Reranker (Cohere)    │                  │   │
│  │                    └────────────┬────────────┘                  │   │
│  └─────────────────────────────────┼───────────────────────────────┘   │
│                                    │                                   │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                    LightRAG Graph Engine                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐│   │
│  │  │ Entity Extraction → Entity Normalization → Graph Building   ││   │
│  │  │ (LLM-based)         (Merging/Dedup)        (Neo4j)          ││   │
│  │  └─────────────────────────────────────────────────────────────┘│   │
│  │  ┌─────────────────────────────────────────────────────────────┐│   │
│  │  │ Dual-Level Retrieval: Low-Level (Entity) + High-Level (Rel) ││   │
│  │  └─────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    External Search Integration                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ Tavily  │  │Semantic │  │ KIPRIS  │  │ SearxNG │            │   │
│  │  │ (Web)   │  │ Scholar │  │(특허)   │  │(Self-   │            │   │
│  │  │         │  │ (논문)  │  │         │  │ hosted) │            │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 TTS (Audio Overview)

| 솔루션 | 용도 | 라이선스 | 한국어 |
|--------|------|----------|--------|
| **MeloTTS** | 기본 TTS, CPU 최적화 | MIT | 지원 |
| **Dia (Nari Labs)** | 대화형 팟캐스트, 감정 표현 | - | - |
| **VibeVoice** | 장문 멀티스피커 | - | - |
| **Edge-TTS** | 무료 고품질 | MS | 우수 |
| **Kokoro TTS** | 로컬 경량 | 오픈소스 | - |

**선택: MeloTTS (기본) + Dia (팟캐스트) + Edge-TTS (폴백)**

---

## 2. 핵심 모듈 구현 상세

### 2.1 OpenRouter LLM 클라이언트

```python
# core/llm/openrouter_client.py
"""
OpenRouter 통합 LLM 클라이언트
- Gemini 3 Pro/Flash, DeepSeek R1, Claude 3.5 지원
- 자동 모델 라우팅 (비용/품질/속도 최적화)
- Reasoning 모드 지원
"""

import os
import httpx
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
from enum import Enum

class ModelTier(Enum):
    REASONING = "reasoning"      # 복잡한 추론 작업
    FAST = "fast"                # 빠른 응답 필요
    COST_EFFICIENT = "cost"      # 비용 최적화
    KOREAN = "korean"            # 한국어 특화
    CODE = "code"                # 코드 생성
    IMAGE = "image"              # 이미지 생성

@dataclass
class OpenRouterConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "google/gemini-3-flash-preview"

    # 모델 매핑
    model_map: Dict[ModelTier, str] = None

    def __post_init__(self):
        self.model_map = {
            ModelTier.REASONING: "google/gemini-3-pro-preview",
            ModelTier.FAST: "google/gemini-3-flash-preview",
            ModelTier.COST_EFFICIENT: "deepseek/deepseek-r1",
            ModelTier.KOREAN: "qwen/qwen3-235b-a22b",
            ModelTier.CODE: "anthropic/claude-3.5-sonnet",
            ModelTier.IMAGE: "google/gemini-3-pro-image-preview",
        }

class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://ppuri-ai.kitech.re.kr",
                "X-Title": "PPuRI-AI Ultimate"
            },
            timeout=120.0
        )

    async def generate(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FAST,
        reasoning: bool = False,
        reasoning_effort: str = "medium",  # low, medium, high
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """LLM 응답 생성"""

        model = self.config.model_map.get(tier, self.config.default_model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Reasoning 모드 (Gemini 3 Pro 전용)
        if reasoning and "gemini-3-pro" in model:
            payload["reasoning"] = {
                "effort": reasoning_effort
            }

        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": model,
            "usage": data.get("usage", {}),
            "reasoning_details": data["choices"][0]["message"].get("reasoning_details")
        }

    async def generate_stream(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FAST,
        **kwargs
    ) -> AsyncIterator[str]:
        """스트리밍 응답 생성"""

        model = self.config.model_map.get(tier, self.config.default_model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **kwargs
        }

        async with self.client.stream("POST", "/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        import json
                        chunk = json.loads(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]

    async def select_best_model(
        self,
        query: str,
        context: Optional[str] = None
    ) -> ModelTier:
        """쿼리 분석을 통한 최적 모델 자동 선택"""

        # 간단한 휴리스틱 기반 선택
        query_lower = query.lower()

        # 코드 관련
        if any(kw in query_lower for kw in ["코드", "프로그램", "함수", "class", "def", "code"]):
            return ModelTier.CODE

        # 복잡한 추론 필요
        if any(kw in query_lower for kw in ["분석", "비교", "왜", "원인", "설계", "아키텍처"]):
            return ModelTier.REASONING

        # 이미지 관련
        if any(kw in query_lower for kw in ["이미지", "그림", "사진", "그래프", "다이어그램"]):
            return ModelTier.IMAGE

        # 한국어 비중 높음
        korean_chars = sum(1 for c in query if '가' <= c <= '힣')
        if korean_chars / max(len(query), 1) > 0.7:
            return ModelTier.KOREAN

        # 기본: 빠른 응답
        return ModelTier.FAST
```

### 2.2 LightRAG 기반 그래프 검색 엔진

```python
# core/rag_engine/lightrag_engine.py
"""
LightRAG 기반 듀얼 레벨 검색 엔진
- ApeRAG의 Entity Normalization 적용
- 벡터 + 그래프 하이브리드 검색
- 증분 업데이트 지원
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import asyncio

class RetrievalMode(Enum):
    LOCAL = "local"      # 엔티티 중심 (세부 정보)
    GLOBAL = "global"    # 관계 중심 (전체 맥락)
    HYBRID = "hybrid"    # 둘 다

@dataclass
class Entity:
    id: str
    name: str
    type: str  # 기술, 공정, 재료, 결함, 장비 등
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # Entity Normalization을 위한 필드
    aliases: List[str] = field(default_factory=list)
    canonical_name: Optional[str] = None

@dataclass
class Relationship:
    source_id: str
    target_id: str
    type: str  # 사용, 발생원인, 해결방법, 관련기술 등
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LightRAGConfig:
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
    similarity_threshold: float = 0.7

    # Entity Normalization
    enable_entity_merging: bool = True
    merge_similarity_threshold: float = 0.85

class LightRAGEngine:
    """LightRAG 스타일 듀얼 레벨 검색 엔진"""

    def __init__(
        self,
        config: LightRAGConfig,
        vector_db,       # Qdrant
        graph_db,        # Neo4j
        embedding_model, # BGE-M3
        llm_client       # OpenRouter
    ):
        self.config = config
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.embedding = embedding_model
        self.llm = llm_client

    async def extract_entities_and_relations(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """LLM 기반 엔티티/관계 추출"""

        prompt = f"""다음 기술 문서에서 엔티티(개체)와 관계를 추출해주세요.

문서:
{text[:4000]}

엔티티 타입: {', '.join(self.config.entity_types)}
관계 타입: {', '.join(self.config.relationship_types)}

JSON 형식으로 출력:
{{
  "entities": [
    {{"name": "TIG 용접", "type": "기술", "description": "...", "aliases": ["GTAW", "가스 텅스텐 아크 용접"]}}
  ],
  "relationships": [
    {{"source": "TIG 용접", "target": "텅스텐 혼입", "type": "발생원인", "description": "..."}}
  ]
}}
"""

        response = await self.llm.generate(
            prompt,
            tier=ModelTier.REASONING,
            reasoning=True,
            reasoning_effort="medium"
        )

        # JSON 파싱 및 엔티티/관계 객체 생성
        import json
        data = json.loads(response["content"])

        entities = []
        for e in data.get("entities", []):
            entity = Entity(
                id=self._generate_id(e["name"]),
                name=e["name"],
                type=e["type"],
                description=e.get("description", ""),
                aliases=e.get("aliases", [])
            )
            # 임베딩 생성
            entity.embedding = await self.embedding.encode(
                f"{entity.name}: {entity.description}"
            )
            entities.append(entity)

        relationships = []
        for r in data.get("relationships", []):
            rel = Relationship(
                source_id=self._generate_id(r["source"]),
                target_id=self._generate_id(r["target"]),
                type=r["type"],
                properties={"description": r.get("description", "")}
            )
            relationships.append(rel)

        # Entity Normalization (ApeRAG 스타일)
        if self.config.enable_entity_merging:
            entities = await self._normalize_entities(entities)

        return entities, relationships

    async def _normalize_entities(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """엔티티 정규화 (중복 병합)"""

        # 기존 엔티티와 유사도 비교
        normalized = []

        for entity in entities:
            # 벡터 DB에서 유사 엔티티 검색
            similar = await self.vector_db.search(
                collection="entities",
                vector=entity.embedding,
                limit=5,
                score_threshold=self.config.merge_similarity_threshold
            )

            if similar and similar[0].score > self.config.merge_similarity_threshold:
                # 기존 엔티티에 병합
                existing = similar[0]
                existing.aliases.extend(entity.aliases)
                existing.aliases.append(entity.name)
                # canonical_name 유지
            else:
                # 새 엔티티로 추가
                entity.canonical_name = entity.name
                normalized.append(entity)

        return normalized

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        industry_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """듀얼 레벨 검색"""

        query_embedding = await self.embedding.encode(query)

        results = {
            "entities": [],
            "relationships": [],
            "context_chunks": []
        }

        if mode in [RetrievalMode.LOCAL, RetrievalMode.HYBRID]:
            # Low-Level: 엔티티 중심 검색
            entities = await self.vector_db.search(
                collection="entities",
                vector=query_embedding,
                limit=self.config.top_k_entities,
                filter={"industry": industry_filter} if industry_filter else None
            )
            results["entities"] = entities

        if mode in [RetrievalMode.GLOBAL, RetrievalMode.HYBRID]:
            # High-Level: 관계 중심 검색 (그래프 순회)
            if results["entities"]:
                entity_ids = [e.id for e in results["entities"]]

                # Neo4j에서 관련 관계 검색
                cypher = """
                MATCH (e1)-[r]-(e2)
                WHERE e1.id IN $entity_ids
                RETURN e1, r, e2
                LIMIT $limit
                """

                relationships = await self.graph_db.query(
                    cypher,
                    parameters={
                        "entity_ids": entity_ids,
                        "limit": self.config.top_k_relationships
                    }
                )
                results["relationships"] = relationships

        # 원본 청크도 함께 반환
        results["context_chunks"] = await self._get_source_chunks(
            results["entities"]
        )

        return results

    async def incremental_update(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """증분 업데이트 (전체 재구축 없이)"""

        # 새 엔티티/관계 추출
        entities, relationships = await self.extract_entities_and_relations(
            text, metadata
        )

        # 벡터 DB에 추가 (중복 체크 후)
        for entity in entities:
            await self.vector_db.upsert(
                collection="entities",
                id=entity.id,
                vector=entity.embedding,
                payload=entity.__dict__
            )

        # 그래프 DB에 추가
        for rel in relationships:
            await self.graph_db.merge_relationship(rel)

    def _generate_id(self, name: str) -> str:
        """엔티티 ID 생성"""
        import hashlib
        return hashlib.md5(name.encode()).hexdigest()[:12]
```

### 2.3 SurfSense 스타일 외부 소스 커넥터

```python
# core/connectors/external_sources.py
"""
SurfSense 스타일 외부 소스 커넥터
- 웹 검색 (Tavily, SearxNG)
- 학술 검색 (Semantic Scholar, arXiv)
- 특허 검색 (KIPRIS)
- 협업 도구 (Slack, Notion, GitHub)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import aiohttp

class ConnectorType(Enum):
    WEB_SEARCH = "web_search"
    ACADEMIC = "academic"
    PATENT = "patent"
    COLLABORATION = "collaboration"
    STORAGE = "storage"

@dataclass
class SearchResult:
    title: str
    content: str
    url: Optional[str]
    source: str
    connector_type: ConnectorType
    metadata: Dict[str, Any]
    score: float = 0.0

class BaseConnector(ABC):
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

class TavilyConnector(BaseConnector):
    """Tavily AI 검색 (SurfSense 추천)"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",  # basic or advanced
        include_domains: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[SearchResult]:

        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": True
            }

            if include_domains:
                payload["include_domains"] = include_domains

            async with session.post(
                f"{self.base_url}/search",
                json=payload
            ) as response:
                data = await response.json()

                results = []
                for item in data.get("results", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        url=item.get("url"),
                        source="tavily",
                        connector_type=ConnectorType.WEB_SEARCH,
                        metadata={"raw_content": item.get("raw_content")},
                        score=item.get("score", 0.0)
                    ))

                return results

    async def health_check(self) -> bool:
        return True

class SemanticScholarConnector(BaseConnector):
    """Semantic Scholar 학술 검색"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"

    async def search(
        self,
        query: str,
        fields: str = "title,abstract,url,year,citationCount,authors",
        limit: int = 10,
        year_range: Optional[Tuple[int, int]] = None
    ) -> List[SearchResult]:

        async with aiohttp.ClientSession() as session:
            params = {
                "query": query,
                "limit": limit,
                "fields": fields
            }

            if year_range:
                params["year"] = f"{year_range[0]}-{year_range[1]}"

            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key

            async with session.get(
                f"{self.base_url}/paper/search",
                params=params,
                headers=headers
            ) as response:
                data = await response.json()

                results = []
                for paper in data.get("data", []):
                    # 인용수 기반 스코어
                    citation_score = min(1.0, paper.get("citationCount", 0) / 1000)

                    results.append(SearchResult(
                        title=paper.get("title", ""),
                        content=paper.get("abstract", ""),
                        url=paper.get("url"),
                        source="semantic_scholar",
                        connector_type=ConnectorType.ACADEMIC,
                        metadata={
                            "year": paper.get("year"),
                            "citations": paper.get("citationCount"),
                            "authors": [a.get("name") for a in paper.get("authors", [])]
                        },
                        score=citation_score
                    ))

                return results

    async def health_check(self) -> bool:
        return True

class KIPRISConnector(BaseConnector):
    """KIPRIS 한국 특허 검색"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://plus.kipris.or.kr/openapi/rest"

    async def search(
        self,
        query: str,
        patent_type: str = "patent",  # patent, utility, design
        limit: int = 10
    ) -> List[SearchResult]:

        async with aiohttp.ClientSession() as session:
            params = {
                "accessKey": self.api_key,
                "word": query,
                "numOfRows": limit,
                "patent": "true" if patent_type == "patent" else "false",
                "utility": "true" if patent_type == "utility" else "false"
            }

            async with session.get(
                f"{self.base_url}/patentSearch",
                params=params
            ) as response:
                # XML 응답 파싱
                text = await response.text()
                # ... XML 파싱 로직

                results = []
                # 파싱된 특허 데이터를 SearchResult로 변환

                return results

    async def health_check(self) -> bool:
        return True

class ConnectorOrchestrator:
    """커넥터 오케스트레이터 - 다중 소스 통합 검색"""

    def __init__(self):
        self.connectors: Dict[str, BaseConnector] = {}

    def register(self, name: str, connector: BaseConnector):
        self.connectors[name] = connector

    async def search_all(
        self,
        query: str,
        connector_types: Optional[List[ConnectorType]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """모든 등록된 커넥터에서 병렬 검색"""

        tasks = []
        for name, connector in self.connectors.items():
            if connector_types is None or connector.connector_type in connector_types:
                tasks.append(connector.search(query, **kwargs))

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 병합 및 정렬
        merged = []
        for results in all_results:
            if isinstance(results, list):
                merged.extend(results)

        # 스코어 기준 정렬
        merged.sort(key=lambda x: x.score, reverse=True)

        return merged
```

### 2.4 고급 TTS 팟캐스트 생성기

```python
# core/audio/advanced_podcast_generator.py
"""
고급 팟캐스트 생성기
- MeloTTS + Dia 통합
- 감정/비언어 표현 지원
- 한국어 전문용어 발음 최적화
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import asyncio
import tempfile
from pathlib import Path

class SpeakerVoice(Enum):
    # MeloTTS 한국어 음성
    KOREAN_MALE_1 = "ko_KR_male_1"
    KOREAN_FEMALE_1 = "ko_KR_female_1"

    # Edge-TTS 한국어 음성 (폴백)
    INJOON = "ko-KR-InJoonNeural"    # 남성
    SUNHI = "ko-KR-SunHiNeural"      # 여성
    HYUNSU = "ko-KR-HyunsuNeural"    # 남성 (전문가)

@dataclass
class SpeakerProfile:
    name: str
    voice: SpeakerVoice
    role: str  # host, co_host, expert, narrator
    personality: str  # friendly, professional, curious, analytical
    speech_rate: float = 1.0
    pitch_shift: float = 0.0

@dataclass
class DialogueSegment:
    speaker: SpeakerProfile
    text: str
    emotion: str = "neutral"  # neutral, excited, curious, thoughtful, laughing
    nonverbal: Optional[str] = None  # [웃음], [기침], [한숨] 등
    pause_before_ms: int = 0
    pause_after_ms: int = 500

@dataclass
class PodcastEpisode:
    title: str
    description: str
    segments: List[DialogueSegment]
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    duration_seconds: float = 0.0

class AdvancedPodcastGenerator:
    """고급 팟캐스트 생성기"""

    def __init__(
        self,
        llm_client,
        output_dir: str = "./podcasts"
    ):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 기본 스피커 프로필
        self.default_speakers = {
            "host": SpeakerProfile(
                name="민수",
                voice=SpeakerVoice.INJOON,
                role="host",
                personality="friendly",
                speech_rate=1.0
            ),
            "co_host": SpeakerProfile(
                name="지연",
                voice=SpeakerVoice.SUNHI,
                role="co_host",
                personality="curious",
                speech_rate=1.05
            ),
            "expert": SpeakerProfile(
                name="박사님",
                voice=SpeakerVoice.HYUNSU,
                role="expert",
                personality="analytical",
                speech_rate=0.95
            )
        }

        # 뿌리산업 전문용어 발음 사전
        self.pronunciation_dict = {
            # 용접
            "TIG": "티그",
            "MIG": "미그",
            "MAG": "매그",
            "GMAW": "지맥",
            "GTAW": "지탁",
            "SMAW": "스맥",
            "HAZ": "하즈",

            # 표면처리
            "PVD": "피브이디",
            "CVD": "씨브이디",
            "DLC": "디엘씨",

            # 금형
            "CAD": "캐드",
            "CAM": "캠",
            "CAE": "씨에이이",
            "CNC": "씨엔씨",
            "EDM": "이디엠",

            # 주조
            "HPDC": "에이치피디씨",
            "LPDC": "엘피디씨",

            # 일반 기술
            "AI": "에이아이",
            "ML": "엠엘",
            "IoT": "아이오티",
        }

    async def generate_episode(
        self,
        documents: List[Dict[str, Any]],
        title: str,
        style: str = "conversational",  # conversational, educational, interview
        duration_target_minutes: int = 10,
        speakers: Optional[List[SpeakerProfile]] = None
    ) -> PodcastEpisode:
        """팟캐스트 에피소드 생성"""

        # 1. 문서 요약
        summary = await self._summarize_for_podcast(documents)

        # 2. 대화 스크립트 생성
        segments = await self._generate_dialogue(
            summary=summary,
            title=title,
            style=style,
            duration_minutes=duration_target_minutes,
            speakers=speakers or list(self.default_speakers.values())
        )

        # 3. TTS 합성
        audio_segments = await self._synthesize_segments(segments)

        # 4. 오디오 병합
        final_audio, duration = await self._merge_audio(
            audio_segments,
            title
        )

        # 5. 자막 생성
        transcript_path = await self._generate_transcript(
            segments,
            title
        )

        return PodcastEpisode(
            title=title,
            description=summary[:500],
            segments=segments,
            audio_path=str(final_audio),
            transcript_path=str(transcript_path),
            duration_seconds=duration
        )

    async def _generate_dialogue(
        self,
        summary: str,
        title: str,
        style: str,
        duration_minutes: int,
        speakers: List[SpeakerProfile]
    ) -> List[DialogueSegment]:
        """LLM 기반 대화 스크립트 생성"""

        speaker_info = "\n".join([
            f"- {s.name} ({s.role}): {s.personality} 성격"
            for s in speakers
        ])

        style_guide = {
            "conversational": """
친근하고 자연스러운 대화체로 작성합니다.
- 청취자가 옆에서 듣는 듯한 편안한 분위기
- 적절한 추임새와 반응 포함 ("네, 맞아요", "흥미롭네요", "그렇군요")
- 전문 용어는 처음 등장 시 쉽게 풀어서 설명
""",
            "educational": """
교육적이고 체계적인 설명 형식입니다.
- 단계별로 명확하게 설명
- 핵심 개념 강조
- 실제 사례와 예시 포함
""",
            "interview": """
전문가 인터뷰 형식입니다.
- 호스트가 질문, 전문가가 답변
- 깊이 있는 기술 토론
- 현장 경험과 인사이트 공유
"""
        }

        prompt = f"""당신은 기술 팟캐스트 작가입니다.
다음 내용을 바탕으로 {duration_minutes}분 분량의 팟캐스트 대본을 작성해주세요.

제목: {title}
스타일: {style_guide.get(style, style_guide["conversational"])}

출연자:
{speaker_info}

내용 요약:
{summary}

대본 형식 (JSON):
[
  {{
    "speaker": "민수",
    "text": "대사 내용",
    "emotion": "neutral",
    "nonverbal": null
  }},
  {{
    "speaker": "지연",
    "text": "대사 내용",
    "emotion": "curious",
    "nonverbal": "[웃음]"
  }}
]

규칙:
1. 각 대사는 20-30초 내로 읽을 수 있는 길이 (80-120자)
2. emotion: neutral, excited, curious, thoughtful, laughing
3. nonverbal: [웃음], [기침], [한숨], [잠시 생각] 등 (선택)
4. 뿌리산업 전문용어는 처음 등장 시 설명
5. 자연스러운 대화 흐름 유지
"""

        response = await self.llm.generate(
            prompt,
            tier=ModelTier.REASONING,
            reasoning=True
        )

        # JSON 파싱
        import json
        dialogue_data = json.loads(response["content"])

        # SpeakerProfile 매핑
        speaker_map = {s.name: s for s in speakers}

        segments = []
        for item in dialogue_data:
            speaker = speaker_map.get(item["speaker"], speakers[0])

            segments.append(DialogueSegment(
                speaker=speaker,
                text=item["text"],
                emotion=item.get("emotion", "neutral"),
                nonverbal=item.get("nonverbal")
            ))

        return segments

    async def _synthesize_segments(
        self,
        segments: List[DialogueSegment]
    ) -> List[Tuple[DialogueSegment, str]]:
        """TTS 합성"""

        results = []

        for i, segment in enumerate(segments):
            # 발음 최적화
            text = self._optimize_pronunciation(segment.text)

            # 감정에 따른 SSML 조정
            ssml_text = self._apply_emotion_ssml(
                text,
                segment.emotion,
                segment.speaker
            )

            # TTS 합성 (Edge-TTS 사용)
            audio_path = self.output_dir / f"segment_{i:04d}.mp3"

            await self._synthesize_edge_tts(
                text=ssml_text,
                voice=segment.speaker.voice.value,
                rate=segment.speaker.speech_rate,
                output_path=str(audio_path)
            )

            results.append((segment, str(audio_path)))

        return results

    def _optimize_pronunciation(self, text: str) -> str:
        """발음 최적화"""
        for term, pronunciation in self.pronunciation_dict.items():
            # 대소문자 무관 치환
            import re
            text = re.sub(
                re.escape(term),
                pronunciation,
                text,
                flags=re.IGNORECASE
            )
        return text

    def _apply_emotion_ssml(
        self,
        text: str,
        emotion: str,
        speaker: SpeakerProfile
    ) -> str:
        """감정에 따른 SSML 적용"""

        # Edge-TTS는 SSML 일부 지원
        rate_map = {
            "excited": "+10%",
            "curious": "+5%",
            "thoughtful": "-10%",
            "neutral": "+0%"
        }

        pitch_map = {
            "excited": "+10Hz",
            "curious": "+5Hz",
            "thoughtful": "-5Hz",
            "neutral": "+0Hz"
        }

        rate = rate_map.get(emotion, "+0%")
        pitch = pitch_map.get(emotion, "+0Hz")

        # 기본 rate/pitch에 감정 조정 추가
        # (실제 SSML 생성은 TTS 엔진에 맞게 조정)

        return text

    async def _synthesize_edge_tts(
        self,
        text: str,
        voice: str,
        rate: float,
        output_path: str
    ):
        """Edge-TTS 합성"""
        try:
            import edge_tts

            rate_str = f"+{int((rate - 1) * 100)}%" if rate >= 1 else f"{int((rate - 1) * 100)}%"

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate_str
            )
            await communicate.save(output_path)

        except ImportError:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
```

---

## 3. 데이터베이스 스키마

### 3.1 PostgreSQL (pgvector)

```sql
-- 문서 저장
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500),
    content TEXT,
    content_hash VARCHAR(64) UNIQUE,
    file_type VARCHAR(50),
    file_size BIGINT,
    industry VARCHAR(50),  -- casting, mold, welding, etc.
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 청크 저장 (벡터 포함)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT,
    chunk_index INTEGER,
    embedding vector(1024),  -- BGE-M3 차원
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 벡터 인덱스
CREATE INDEX chunks_embedding_idx ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 엔티티 저장
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name VARCHAR(500) UNIQUE,
    entity_type VARCHAR(100),
    description TEXT,
    aliases TEXT[],
    embedding vector(1024),
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 관계 저장
CREATE TABLE relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID REFERENCES entities(id),
    target_entity_id UUID REFERENCES entities(id),
    relationship_type VARCHAR(100),
    weight FLOAT DEFAULT 1.0,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 대화 세션
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 대화 메시지
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20),  -- user, assistant, system
    content TEXT,
    citations JSONB,  -- [{id, source, title, snippet}]
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 팟캐스트 에피소드
CREATE TABLE podcast_episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500),
    description TEXT,
    audio_url VARCHAR(1000),
    transcript TEXT,
    duration_seconds FLOAT,
    source_documents UUID[],
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 3.2 Neo4j (지식 그래프)

```cypher
// 엔티티 노드 생성
CREATE CONSTRAINT entity_canonical_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE;

// 산업별 라벨
CREATE (e:Entity:Welding {
    canonical_name: "TIG 용접",
    entity_type: "기술",
    description: "가스 텅스텐 아크 용접",
    aliases: ["GTAW", "Gas Tungsten Arc Welding"],
    industry: "welding"
});

// 관계 생성
MATCH (e1:Entity {canonical_name: "TIG 용접"})
MATCH (e2:Entity {canonical_name: "텅스텐 혼입"})
CREATE (e1)-[:CAUSES {weight: 0.9, description: "..."}]->(e2);

// 경로 기반 검색
MATCH path = (start:Entity)-[*1..3]-(end:Entity)
WHERE start.canonical_name = "TIG 용접"
RETURN path;

// 커뮤니티 탐지 (GraphRAG 스타일)
CALL gds.louvain.stream('entityGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).canonical_name AS entity, communityId
ORDER BY communityId;
```

---

## 4. API 엔드포인트

### 4.1 주요 API

```yaml
# OpenAPI 3.0 스펙

/api/v1/chat:
  post:
    summary: "대화 (NotebookLM 스타일 인용 포함)"
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              query: string
              session_id: string
              options:
                include_web_search: boolean
                include_academic: boolean
                industry_filter: string
                model_tier: string  # reasoning, fast, korean
    responses:
      200:
        content:
          application/json:
            schema:
              type: object
              properties:
                answer: string
                citations: array
                reasoning_details: object

/api/v1/podcast/generate:
  post:
    summary: "Audio Overview 생성"
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              document_ids: array
              title: string
              style: string  # conversational, educational, interview
              duration_minutes: integer
              speakers: array
    responses:
      200:
        content:
          application/json:
            schema:
              type: object
              properties:
                episode_id: string
                audio_url: string
                transcript_url: string
                duration_seconds: number

/api/v1/output/generate:
  post:
    summary: "구조화된 출력 생성"
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              document_ids: array
              output_format: string  # mind_map, data_table, faq, study_guide
              options: object
    responses:
      200:
        content:
          application/json:
            schema:
              type: object
              properties:
                format: string
                data: object
                export_formats: array

/api/v1/graph/explore:
  get:
    summary: "지식 그래프 탐색"
    parameters:
      - name: entity_name
        in: query
        type: string
      - name: depth
        in: query
        type: integer
      - name: relationship_types
        in: query
        type: array
    responses:
      200:
        content:
          application/json:
            schema:
              type: object
              properties:
                nodes: array
                edges: array
                communities: array
```

---

## 5. 구현 로드맵

### Phase 1: 핵심 인프라 (2주)

```
Week 1:
├── OpenRouter LLM 클라이언트 구현
├── BGE-M3 임베딩 서비스 구현
├── PostgreSQL + pgvector 설정
└── 기본 RAG 파이프라인 구축

Week 2:
├── LightRAG 엔진 구현 (Entity 추출)
├── Entity Normalization 구현
├── Neo4j 그래프 저장소 연동
└── 듀얼 레벨 검색 구현
```

### Phase 2: 외부 소스 통합 (2주)

```
Week 3:
├── Tavily 웹 검색 커넥터
├── Semantic Scholar 학술 검색
├── KIPRIS 특허 검색
└── 하이브리드 검색 오케스트레이터

Week 4:
├── 인용 시스템 구현
├── 검색 결과 Reranking (Cohere)
├── 결과 융합 (RRF)
└── 품질 평가 시스템
```

### Phase 3: Audio & Output (2주)

```
Week 5:
├── MeloTTS/Edge-TTS 통합
├── 대화 스크립트 생성기
├── 감정/비언어 표현 처리
└── 오디오 병합 및 편집

Week 6:
├── Mind Map 생성기 (Mermaid)
├── Data Table 추출기
├── FAQ/Study Guide 생성기
└── 내보내기 (PDF, CSV)
```

### Phase 4: UI/UX & 통합 (2주)

```
Week 7:
├── React 프론트엔드 구현
├── 실시간 스트리밍 UI
├── 인용 하이라이트 UI
└── 오디오 플레이어 UI

Week 8:
├── 지식 그래프 시각화 (D3.js)
├── 성능 최적화
├── 테스트 자동화
└── 배포 (Docker/K8s)
```

---

## 6. 참조 프로젝트

| 프로젝트 | 참조 기능 | GitHub |
|---------|----------|--------|
| **SurfSense** | 외부 소스 통합, Deep Agent | github.com/MODSetter/SurfSense |
| **ApeRAG** | GraphRAG, Entity Normalization | github.com/apecloud/ApeRAG |
| **LightRAG** | 듀얼 레벨 검색, 증분 업데이트 | github.com/HKUDS/LightRAG |
| **Open-Notebook** | 멀티 스피커 팟캐스트 | github.com/lfnovo/open-notebook |
| **WeKnora** | BM25+Dense+GraphRAG 하이브리드 | Tencent 공개 |
| **MeloTTS** | 한국어 TTS | github.com/myshell-ai/MeloTTS |

---

## 7. 성공 지표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 응답 정확도 | 90%+ | 인용 검증률 |
| 응답 속도 | < 3초 | P95 레이턴시 |
| 그래프 품질 | Entity 정확도 85%+ | 샘플 검증 |
| 팟캐스트 품질 | 4.0/5.0+ | 사용자 평가 |
| 비용 효율 | $0.01/query | 평균 비용 |
