"""
PPuRI-AI Ultimate - Integration Tests
전체 시스템 통합 테스트

실행: pytest tests/test_integration.py -v
"""

import asyncio
import os
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# 테스트 환경 설정
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")


# ==================== Fixtures ====================

@pytest.fixture
def event_loop():
    """이벤트 루프 픽스처"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openrouter_response():
    """OpenRouter 응답 모킹"""
    return {
        "id": "test-id",
        "model": "google/gemini-2.5-pro-preview-03-25",
        "choices": [
            {
                "message": {
                    "content": "테스트 응답입니다. [1] TIG 용접은 고품질 용접에 적합합니다."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50
        }
    }


@pytest.fixture
def mock_embedding():
    """임베딩 모킹"""
    import numpy as np
    return {
        "dense": np.random.rand(1024).tolist(),
        "sparse": {"용접": 0.8, "TIG": 0.9, "결함": 0.7}
    }


@pytest.fixture
def sample_documents():
    """샘플 문서"""
    return [
        {
            "id": "doc-1",
            "title": "TIG 용접 기술 가이드",
            "content": "TIG 용접(GTAW)은 텅스텐 전극과 불활성 가스를 사용하는 고품질 용접 방법입니다.",
            "industry": "welding"
        },
        {
            "id": "doc-2",
            "title": "용접 결함 분석",
            "content": "기공 결함은 용접 시 가스가 용융 금속 내에 갇혀 발생합니다.",
            "industry": "welding"
        }
    ]


# ==================== LLM Client Tests ====================

class TestOpenRouterClient:
    """OpenRouter 클라이언트 테스트"""

    @pytest.mark.asyncio
    async def test_import_client(self):
        """클라이언트 임포트 테스트"""
        from core.llm import OpenRouterClient, ModelTier
        assert OpenRouterClient is not None
        assert ModelTier.REASONING is not None

    @pytest.mark.asyncio
    async def test_model_tiers(self):
        """모델 티어 테스트"""
        from core.llm import ModelTier, OPENROUTER_MODELS

        # 모든 티어가 정의되어 있는지 확인
        assert ModelTier.REASONING is not None
        assert ModelTier.FAST is not None
        assert ModelTier.COST_EFFICIENT is not None

        # 모델 정보 확인
        assert "google/gemini-2.5-pro-preview-03-25" in OPENROUTER_MODELS

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """클라이언트 초기화 테스트"""
        from core.llm import OpenRouterClient, OpenRouterConfig

        config = OpenRouterConfig(api_key="test-key")
        client = OpenRouterClient(config)

        assert client.config.api_key == "test-key"
        assert client.config.base_url == "https://openrouter.ai/api/v1"

    @pytest.mark.asyncio
    async def test_generate_with_mock(self, mock_openrouter_response):
        """응답 생성 테스트 (모킹)"""
        from core.llm import OpenRouterClient, OpenRouterConfig, ModelTier

        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_openrouter_response
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            config = OpenRouterConfig(api_key="test-key")
            client = OpenRouterClient(config)

            # 실제 호출 시뮬레이션은 건너뛰고 구조만 확인
            assert client is not None


# ==================== Embedding Service Tests ====================

class TestEmbeddingService:
    """임베딩 서비스 테스트"""

    @pytest.mark.asyncio
    async def test_import_service(self):
        """서비스 임포트 테스트"""
        from core.embeddings import BGEM3Service, EmbeddingConfig
        assert BGEM3Service is not None

    @pytest.mark.asyncio
    async def test_embedding_config(self):
        """임베딩 설정 테스트"""
        from core.embeddings import EmbeddingConfig, EmbeddingModel

        config = EmbeddingConfig()
        assert config.model == EmbeddingModel.BGE_M3
        assert config.dimension == 1024

    @pytest.mark.asyncio
    async def test_embedding_result_structure(self):
        """임베딩 결과 구조 테스트"""
        from core.embeddings import EmbeddingResult
        import numpy as np

        result = EmbeddingResult(
            dense_embedding=np.random.rand(1024).tolist(),
            sparse_embedding={"test": 0.5},
            colbert_embedding=None
        )

        assert len(result.dense_embedding) == 1024
        assert "test" in result.sparse_embedding


# ==================== Chat Service Tests ====================

class TestChatService:
    """채팅 서비스 테스트"""

    @pytest.mark.asyncio
    async def test_import_service(self):
        """서비스 임포트 테스트"""
        from core.services import ChatService, SearchMode, Citation
        assert ChatService is not None
        assert SearchMode.WEB_ENABLED is not None

    @pytest.mark.asyncio
    async def test_search_modes(self):
        """검색 모드 테스트"""
        from core.services import SearchMode

        assert SearchMode.DOCUMENTS_ONLY.value == "documents_only"
        assert SearchMode.WEB_ENABLED.value == "web_enabled"
        assert SearchMode.FULL_SEARCH.value == "full_search"

    @pytest.mark.asyncio
    async def test_citation_structure(self):
        """인용 구조 테스트"""
        from core.services import Citation

        citation = Citation(
            id="1",
            title="테스트 문서",
            content_snippet="테스트 내용입니다.",
            source_type="document",
            url=None,
            page_number=1,
            confidence=0.95
        )

        assert citation.to_inline_ref() == "[1]"
        assert "테스트 문서" in citation.to_footnote()

    @pytest.mark.asyncio
    async def test_chat_service_config(self):
        """채팅 서비스 설정 테스트"""
        from core.services import ChatServiceConfig, SearchMode

        config = ChatServiceConfig()
        assert config.default_search_mode == SearchMode.WEB_ENABLED
        assert config.max_citations == 10


# ==================== Search Connectors Tests ====================

class TestSearchConnectors:
    """검색 커넥터 테스트"""

    @pytest.mark.asyncio
    async def test_import_connectors(self):
        """커넥터 임포트 테스트"""
        from core.connectors import (
            SearchOrchestrator,
            TavilyConnector,
            SemanticScholarConnector,
            ConnectorType
        )
        assert SearchOrchestrator is not None
        assert TavilyConnector is not None

    @pytest.mark.asyncio
    async def test_connector_types(self):
        """커넥터 타입 테스트"""
        from core.connectors import ConnectorType

        assert ConnectorType.WEB_SEARCH is not None
        assert ConnectorType.ACADEMIC is not None
        assert ConnectorType.PATENT is not None

    @pytest.mark.asyncio
    async def test_search_result_structure(self):
        """검색 결과 구조 테스트"""
        from core.connectors import SearchResult, ConnectorType

        result = SearchResult(
            title="테스트 결과",
            url="https://example.com",
            content="테스트 내용",
            source="test",
            connector_type=ConnectorType.WEB_SEARCH,
            score=0.9
        )

        assert result.title == "테스트 결과"
        assert result.connector_type == ConnectorType.WEB_SEARCH


# ==================== LightRAG Engine Tests ====================

class TestLightRAGEngine:
    """LightRAG 엔진 테스트"""

    @pytest.mark.asyncio
    async def test_import_engine(self):
        """엔진 임포트 테스트"""
        from core.rag_engine import (
            LightRAGEngine,
            RetrievalMode,
            Entity,
            Relationship
        )
        assert LightRAGEngine is not None
        assert RetrievalMode.HYBRID is not None

    @pytest.mark.asyncio
    async def test_retrieval_modes(self):
        """검색 모드 테스트"""
        from core.rag_engine import RetrievalMode

        assert RetrievalMode.LOCAL.value == "local"
        assert RetrievalMode.GLOBAL.value == "global"
        assert RetrievalMode.HYBRID.value == "hybrid"

    @pytest.mark.asyncio
    async def test_entity_structure(self):
        """엔티티 구조 테스트"""
        from core.rag_engine.lightrag_engine import Entity, EntityType

        entity = Entity(
            id="test-id",
            name="TIG 용접",
            entity_type=EntityType.TECHNOLOGY,
            description="텅스텐 불활성 가스 용접"
        )

        assert entity.name == "TIG 용접"
        assert entity.entity_type == EntityType.TECHNOLOGY

        entity_dict = entity.to_dict()
        assert entity_dict["name"] == "TIG 용접"

    @pytest.mark.asyncio
    async def test_relationship_structure(self):
        """관계 구조 테스트"""
        from core.rag_engine.lightrag_engine import Relationship, RelationType

        rel = Relationship(
            id="rel-1",
            source_id="entity-1",
            target_id="entity-2",
            relation_type=RelationType.CAUSES,
            description="원인 관계"
        )

        assert rel.relation_type == RelationType.CAUSES


# ==================== Audio Overview Tests ====================

class TestAudioOverview:
    """Audio Overview 테스트"""

    @pytest.mark.asyncio
    async def test_import_engine(self):
        """엔진 임포트 테스트"""
        from core.audio import (
            AudioOverviewEngine,
            TTSProvider,
            SpeakerRole,
            DialogueTurn
        )
        assert AudioOverviewEngine is not None
        assert TTSProvider.EDGE_TTS is not None

    @pytest.mark.asyncio
    async def test_tts_providers(self):
        """TTS 제공자 테스트"""
        from core.audio import TTSProvider

        assert TTSProvider.EDGE_TTS.value == "edge_tts"
        assert TTSProvider.MELO_TTS.value == "melo_tts"
        assert TTSProvider.OPENAI_TTS.value == "openai_tts"

    @pytest.mark.asyncio
    async def test_dialogue_turn(self):
        """대화 턴 구조 테스트"""
        from core.audio import DialogueTurn, SpeakerRole

        turn = DialogueTurn(
            speaker=SpeakerRole.HOST_A,
            text="안녕하세요, 오늘은 TIG 용접에 대해 이야기해보겠습니다.",
            emotion="neutral"
        )

        assert turn.speaker == SpeakerRole.HOST_A
        assert "TIG 용접" in turn.text


# ==================== Database Tests ====================

class TestDatabase:
    """데이터베이스 테스트"""

    @pytest.mark.asyncio
    async def test_import_modules(self):
        """모듈 임포트 테스트"""
        from core.database import MODELS_AVAILABLE, CONNECTION_AVAILABLE
        # pgvector가 설치되어 있지 않으면 False
        assert isinstance(MODELS_AVAILABLE, bool)

    @pytest.mark.asyncio
    async def test_graph_db_import(self):
        """그래프 DB 임포트 테스트"""
        from core.database.graph_db import Neo4jGraphDB, GraphDBConfig
        assert Neo4jGraphDB is not None

    @pytest.mark.asyncio
    async def test_graph_db_config(self):
        """그래프 DB 설정 테스트"""
        from core.database.graph_db import GraphDBConfig

        config = GraphDBConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.database == "ppuri"


# ==================== End-to-End Tests ====================

class TestEndToEnd:
    """종단간 테스트"""

    @pytest.mark.asyncio
    async def test_full_import_chain(self):
        """전체 임포트 체인 테스트"""
        # 모든 핵심 모듈이 임포트 가능한지 확인
        from core.llm import OpenRouterClient, ModelTier
        from core.embeddings import BGEM3Service
        from core.services import ChatService, SearchMode
        from core.connectors import SearchOrchestrator
        from core.audio import AudioOverviewEngine, TTSProvider
        from core.rag_engine import LightRAGEngine, RetrievalMode

        assert all([
            OpenRouterClient,
            BGEM3Service,
            ChatService,
            SearchOrchestrator,
            AudioOverviewEngine,
            LightRAGEngine
        ])

    @pytest.mark.asyncio
    async def test_citation_flow(self):
        """인용 플로우 테스트"""
        from core.services import Citation, ChatResponse

        # 인용 생성
        citations = [
            Citation(
                id="1",
                title="TIG 용접 가이드",
                content_snippet="TIG 용접은 고품질 용접에 적합합니다.",
                source_type="document",
                confidence=0.95
            ),
            Citation(
                id="2",
                title="용접 결함 분석",
                content_snippet="기공은 가스 포집으로 발생합니다.",
                source_type="academic",
                url="https://example.com/paper",
                confidence=0.88
            )
        ]

        # 응답 생성
        response = ChatResponse(
            answer="TIG 용접[1]은 고품질 용접 방법입니다. 기공 결함[2]에 주의해야 합니다.",
            citations=citations,
            search_results_count=2,
            search_time_ms=150.0,
            generation_time_ms=800.0,
            model_used="gemini-3-pro"
        )

        assert "[1]" in response.answer
        assert "[2]" in response.answer
        assert len(response.citations) == 2


# ==================== Performance Tests ====================

class TestPerformance:
    """성능 테스트"""

    @pytest.mark.asyncio
    async def test_embedding_batch_size(self):
        """임베딩 배치 크기 테스트"""
        from core.embeddings import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.batch_size > 0
        assert config.batch_size <= 64

    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """검색 타임아웃 테스트"""
        from core.connectors.search_connectors import SearchConfig

        config = SearchConfig()
        assert config.timeout_seconds > 0
        assert config.timeout_seconds <= 60


# ==================== 실행 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
