"""
PPuRI-AI Ultimate - 통합 채팅 서비스

NotebookLM 스타일의 Source Grounding + 외부 검색 통합
- 문서 기반 RAG (Source Grounding)
- 외부 웹/학술/특허 검색
- 인라인 인용 [1][2] 생성
- 스트리밍 응답
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncIterator, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """검색 모드"""
    DOCUMENTS_ONLY = "documents_only"     # NotebookLM 스타일 (문서만)
    WEB_ENABLED = "web_enabled"           # 문서 + 웹 검색
    FULL_SEARCH = "full_search"           # 문서 + 웹 + 학술 + 특허


@dataclass
class Citation:
    """인용 정보"""
    id: str
    title: str
    content_snippet: str
    source_type: str  # document, web, academic, patent
    url: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_inline_ref(self) -> str:
        """인라인 참조 [1]"""
        return f"[{self.id}]"

    def to_footnote(self) -> str:
        """각주 형식"""
        source_label = {
            "document": "문서",
            "web": "웹",
            "academic": "논문",
            "patent": "특허",
            "knowledge_graph": "지식그래프"
        }.get(self.source_type, "출처")

        footnote = f"[{self.id}] {self.title}"
        if self.page_number:
            footnote += f" (p.{self.page_number})"
        footnote += f" - {source_label}"
        if self.url:
            footnote += f" ({self.url})"
        return footnote


@dataclass
class ChatMessage:
    """채팅 메시지"""
    role: str  # user, assistant, system
    content: str
    citations: List[Citation] = field(default_factory=list)
    reasoning_details: Optional[List[Dict]] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """채팅 응답"""
    answer: str
    citations: List[Citation]
    reasoning_details: Optional[List[Dict]] = None
    search_results_count: int = 0
    search_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    model_used: str = ""


@dataclass
class ChatServiceConfig:
    """채팅 서비스 설정"""
    default_search_mode: SearchMode = SearchMode.WEB_ENABLED
    max_context_tokens: int = 8000
    max_citations: int = 10
    include_reasoning: bool = True
    reasoning_effort: str = "medium"
    temperature: float = 0.7
    industry_filter: Optional[str] = None


class ChatService:
    """
    통합 채팅 서비스

    기능:
    1. Source Grounding: 문서 기반 RAG (NotebookLM 스타일)
    2. 외부 검색: 웹/학술/특허 통합
    3. LightRAG: 엔티티/관계 기반 검색
    4. 인라인 인용: [1][2] 형식 자동 삽입
    5. 스트리밍: 실시간 응답

    사용 예시:
    ```python
    service = ChatService(config, llm_client, lightrag_engine, search_orchestrator)
    await service.initialize()

    # 기본 채팅
    response = await service.chat("TIG 용접 결함 원인은?")

    # 스트리밍
    async for chunk in service.chat_stream("질문"):
        print(chunk, end="")
    ```
    """

    def __init__(
        self,
        config: ChatServiceConfig,
        llm_client,            # OpenRouterClient
        lightrag_engine,       # LightRAGEngine
        search_orchestrator,   # SearchOrchestrator
        embedding_service      # BGEM3Service
    ):
        self.config = config
        self.llm = llm_client
        self.lightrag = lightrag_engine
        self.search = search_orchestrator
        self.embedding = embedding_service

        self._conversation_history: List[ChatMessage] = []
        self._initialized = False

    async def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self._initialized = True
            logger.info("ChatService initialized")
            return True
        except Exception as e:
            logger.error(f"ChatService initialization failed: {e}")
            return False

    async def chat(
        self,
        query: str,
        search_mode: Optional[SearchMode] = None,
        industry_filter: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """
        채팅 응답 생성

        Args:
            query: 사용자 쿼리
            search_mode: 검색 모드
            industry_filter: 산업 필터 (casting, mold, welding 등)
            session_id: 세션 ID (대화 이력 관리용)
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        start_time = datetime.now()
        mode = search_mode or self.config.default_search_mode
        industry = industry_filter or self.config.industry_filter

        # 1. 검색 수행
        search_start = datetime.now()
        context, citations = await self._gather_context(
            query=query,
            mode=mode,
            industry_filter=industry
        )
        search_time = (datetime.now() - search_start).total_seconds() * 1000

        # 2. 프롬프트 구성
        system_prompt = self._build_system_prompt(industry)
        user_prompt = self._build_user_prompt(query, context, citations)

        # 3. LLM 응답 생성
        gen_start = datetime.now()
        from core.llm import ModelTier, ReasoningEffort

        # 쿼리 복잡도에 따른 모델 선택
        tier = await self.llm.select_optimal_model(query)

        response = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            tier=tier,
            reasoning=self.config.include_reasoning,
            reasoning_effort=ReasoningEffort(self.config.reasoning_effort),
            temperature=self.config.temperature,
            max_tokens=4096
        )
        gen_time = (datetime.now() - gen_start).total_seconds() * 1000

        # 4. 응답 후처리 (인용 확인)
        answer = self._process_answer(response.content, citations)

        # 대화 이력에 추가
        self._conversation_history.append(ChatMessage(
            role="user",
            content=query
        ))
        self._conversation_history.append(ChatMessage(
            role="assistant",
            content=answer,
            citations=citations,
            reasoning_details=response.reasoning_details
        ))

        return ChatResponse(
            answer=answer,
            citations=citations,
            reasoning_details=response.reasoning_details,
            search_results_count=len(citations),
            search_time_ms=search_time,
            generation_time_ms=gen_time,
            model_used=response.model
        )

    async def chat_stream(
        self,
        query: str,
        search_mode: Optional[SearchMode] = None,
        industry_filter: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        스트리밍 채팅 응답

        Yields:
            응답 청크 (문자열)
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        mode = search_mode or self.config.default_search_mode
        industry = industry_filter or self.config.industry_filter

        # 1. 검색 수행
        context, citations = await self._gather_context(
            query=query,
            mode=mode,
            industry_filter=industry
        )

        # 2. 인용 정보 먼저 전송 (메타데이터)
        yield f"[CITATIONS_START]{json.dumps([c.__dict__ for c in citations], default=str, ensure_ascii=False)}[CITATIONS_END]"

        # 3. 프롬프트 구성
        system_prompt = self._build_system_prompt(industry)
        user_prompt = self._build_user_prompt(query, context, citations)

        # 4. 스트리밍 응답
        from core.llm import ModelTier

        tier = await self.llm.select_optimal_model(query)

        async for chunk in self.llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            tier=tier,
            temperature=self.config.temperature,
            max_tokens=4096
        ):
            yield chunk

    async def _gather_context(
        self,
        query: str,
        mode: SearchMode,
        industry_filter: Optional[str]
    ) -> Tuple[str, List[Citation]]:
        """컨텍스트 및 인용 수집"""
        all_citations: List[Citation] = []
        context_parts = []

        # 1. LightRAG 검색 (문서 기반)
        try:
            from core.rag_engine.lightrag_engine import RetrievalMode

            lightrag_mode = RetrievalMode.HYBRID
            results = await self.lightrag.retrieve(
                query=query,
                mode=lightrag_mode,
                industry_filter=industry_filter
            )

            # 엔티티 정보
            for i, entity in enumerate(results.entities[:5]):
                all_citations.append(Citation(
                    id=str(len(all_citations) + 1),
                    title=entity.name,
                    content_snippet=entity.description[:300],
                    source_type="knowledge_graph",
                    confidence=0.9,
                    metadata={"entity_type": entity.entity_type.value}
                ))

            # 청크 정보
            for chunk in results.chunks[:3]:
                all_citations.append(Citation(
                    id=str(len(all_citations) + 1),
                    title=chunk.get("title", "문서"),
                    content_snippet=chunk.get("content", "")[:300],
                    source_type="document",
                    page_number=chunk.get("page"),
                    confidence=0.95
                ))

            context_parts.append(results.get_context())

        except Exception as e:
            logger.error(f"LightRAG search failed: {e}")

        # 2. 외부 검색 (모드에 따라)
        if mode in [SearchMode.WEB_ENABLED, SearchMode.FULL_SEARCH]:
            try:
                from core.connectors import ConnectorType

                connector_types = [ConnectorType.WEB_SEARCH]

                if mode == SearchMode.FULL_SEARCH:
                    connector_types.extend([
                        ConnectorType.ACADEMIC,
                        ConnectorType.PATENT
                    ])

                external_results = await self.search.search(
                    query=query,
                    connector_types=connector_types,
                    max_results_per_connector=3
                )

                for result in external_results[:5]:
                    all_citations.append(Citation(
                        id=str(len(all_citations) + 1),
                        title=result.title,
                        content_snippet=result.content[:300],
                        source_type=result.connector_type.value,
                        url=result.url,
                        confidence=result.score,
                        metadata=result.metadata
                    ))

                    context_parts.append(f"[{result.source}] {result.content[:500]}")

            except Exception as e:
                logger.error(f"External search failed: {e}")

        # 인용 수 제한
        all_citations = all_citations[:self.config.max_citations]

        # 컨텍스트 조합
        context = "\n\n---\n\n".join(context_parts)

        # 토큰 제한
        if len(context) > self.config.max_context_tokens * 4:  # 대략적인 문자 수
            context = context[:self.config.max_context_tokens * 4]

        return context, all_citations

    def _build_system_prompt(self, industry: Optional[str] = None) -> str:
        """시스템 프롬프트 구성"""
        industry_context = ""
        if industry:
            industry_names = {
                "casting": "주조",
                "mold": "금형",
                "welding": "용접",
                "forming": "소성가공",
                "surface": "표면처리",
                "heat": "열처리"
            }
            industry_kr = industry_names.get(industry, industry)
            industry_context = f"\n현재 {industry_kr} 산업 관련 질문에 답변 중입니다."

        return f"""당신은 뿌리산업 (주조, 금형, 용접, 소성가공, 표면처리, 열처리) 전문 AI 어시스턴트입니다.
{industry_context}

답변 규칙:
1. 제공된 컨텍스트 정보를 기반으로 정확하게 답변합니다.
2. 답변에 관련 출처를 인라인 인용으로 표시합니다: [1], [2] 등
3. 확실하지 않은 정보는 "~일 수 있습니다", "~로 알려져 있습니다" 등으로 표현합니다.
4. 기술 용어는 처음 등장 시 간단히 설명합니다.
5. 실무에 도움이 되는 구체적인 정보를 제공합니다.

한국어로 답변해주세요."""

    def _build_user_prompt(
        self,
        query: str,
        context: str,
        citations: List[Citation]
    ) -> str:
        """사용자 프롬프트 구성"""
        # 인용 목록 생성
        citation_list = "\n".join([
            f"[{c.id}] {c.title}: {c.content_snippet[:200]}..."
            for c in citations
        ])

        return f"""## 참조 정보:
{context}

## 출처 목록:
{citation_list}

## 질문:
{query}

위 참조 정보를 바탕으로 답변해주세요. 관련 출처는 [1], [2] 형식으로 인용해주세요."""

    def _process_answer(
        self,
        answer: str,
        citations: List[Citation]
    ) -> str:
        """응답 후처리 (인용 검증)"""
        # 인용 번호가 실제 존재하는지 확인
        citation_ids = {c.id for c in citations}

        def validate_citation(match):
            ref_id = match.group(1)
            if ref_id in citation_ids:
                return f"[{ref_id}]"
            return ""  # 존재하지 않는 인용 제거

        # 인용 패턴: [1], [2] 등
        processed = re.sub(r'\[(\d+)\]', validate_citation, answer)

        return processed

    def get_citations_footnotes(self, citations: List[Citation]) -> str:
        """인용 각주 생성"""
        return "\n".join([c.to_footnote() for c in citations])

    def clear_history(self):
        """대화 이력 초기화"""
        self._conversation_history.clear()

    def get_history(self) -> List[ChatMessage]:
        """대화 이력 반환"""
        return self._conversation_history.copy()


# 싱글톤
_chat_service: Optional[ChatService] = None


async def get_chat_service() -> ChatService:
    """채팅 서비스 싱글톤"""
    global _chat_service

    if _chat_service is None:
        # 의존성 초기화
        from core.llm import get_openrouter_client
        from core.embeddings import get_embedding_service
        from core.connectors import get_search_orchestrator

        llm = await get_openrouter_client()
        embedding = await get_embedding_service()
        search = await get_search_orchestrator()

        # LightRAG는 별도 초기화 필요
        # lightrag = ...

        config = ChatServiceConfig()

        _chat_service = ChatService(
            config=config,
            llm_client=llm,
            lightrag_engine=None,  # TODO: 초기화 후 연결
            search_orchestrator=search,
            embedding_service=embedding
        )
        await _chat_service.initialize()

    return _chat_service
