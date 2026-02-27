#!/usr/bin/env python3
"""
PPuRI-AI Ultimate - 메인 실행 파일

뿌리산업 특화 차세대 AI 시스템의 진입점.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from core.config.config_manager import ConfigManager
from ui.web.web_server import WebServer
from ui.ui_orchestrator import UIOrchestrator
from ui.components.chat_interface import ChatInterface
from ui.components.think_visualizer import ThinkVisualizer
from ui.components.knowledge_explorer import KnowledgeExplorer
from ui.components.performance_dashboard import PerformanceDashboard
from ui.components.mcp_monitor import MCPMonitor
from core.kitech_base.conversational_engine import ConversationalEngine
from core.kitech_base.korean_optimizer import KoreanOptimizer
from core.knowledge_graph.graph_manager import GraphManager
from core.rag_engine.rag_orchestrator import RAGOrchestrator
from core.mcp_ecosystem.ecosystem_orchestrator import MCPEcosystemOrchestrator

# NotebookLM-style Ultimate Components
from core.llm import get_openrouter_client
from core.embeddings import get_embedding_service
from core.services import get_chat_service
from core.connectors import get_search_orchestrator
from core.database.connection import init_database, check_database_health


async def main():
    """메인 실행 함수"""
    
    # 로깅 설정 (환경 변수 기반)
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', 'text')
    
    if log_format == 'json':
        import json
        import structlog
        
        # JSON 형태의 구조화된 로깅
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(message)s'
        )
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        # 기본 텍스트 로깅
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    logger = logging.getLogger(__name__)
    logger.info("🏭 PPuRI-AI Ultimate v1.0.0 시작")
    
    try:
        # 1. 설정 관리자 초기화
        config_manager = ConfigManager()
        if not config_manager.initialize():
            logger.error("❌ 설정 초기화 실패")
            return 1
        
        logger.info("✅ 설정 시스템 초기화 완료")
        
        # 2. 서버 설정
        server_config = config_manager.get_server_config()
        
        # 3. 핵심 컴포넌트들 초기화
        logger.info("🧠 핵심 AI 컴포넌트들 초기화 중...")

        # 더미 컴포넌트들 (실제 구현체가 준비될 때까지 사용)
        class DummyComponent:
            async def initialize(self): pass
            async def cleanup(self): pass
            async def health_check(self): return "healthy"

        # NotebookLM-style Ultimate Components 초기화
        try:
            logger.info("🔗 OpenRouter LLM 클라이언트 초기화...")
            llm_client = await get_openrouter_client()
            logger.info("✅ OpenRouter LLM 클라이언트 준비 완료")

            logger.info("🔤 BGE-M3 임베딩 서비스 초기화...")
            embedding_service = await get_embedding_service()
            logger.info("✅ BGE-M3 임베딩 서비스 준비 완료")

            logger.info("🔍 외부 검색 오케스트레이터 초기화...")
            search_orchestrator = await get_search_orchestrator()
            logger.info("✅ 외부 검색 오케스트레이터 준비 완료")

            logger.info("💬 NotebookLM 스타일 채팅 서비스 초기화...")
            chat_service = await get_chat_service()
            logger.info("✅ 채팅 서비스 준비 완료")

            # 데이터베이스 초기화 (선택적)
            db_enabled = os.getenv('DB_ENABLED', 'false').lower() == 'true'
            if db_enabled:
                logger.info("🗄️ PostgreSQL + pgvector 데이터베이스 초기화...")
                await init_database()
                db_health = await check_database_health()
                if db_health.get("status") == "healthy":
                    logger.info(f"✅ 데이터베이스 연결 완료 (pgvector: {db_health.get('pgvector_enabled')})")
                else:
                    logger.warning(f"⚠️ 데이터베이스 상태: {db_health}")

        except Exception as e:
            logger.warning(f"⚠️ Ultimate 컴포넌트 초기화 일부 실패: {e}")
            logger.warning("⚠️ 기존 시스템 컴포넌트로 계속 진행...")

        # 실제 구현체들 초기화
        try:
            # 한국어 최적화기
            korean_optimizer = KoreanOptimizer()
            await korean_optimizer.initialize()
            
            # 지식 그래프 관리자
            graph_manager = GraphManager()
            await graph_manager.initialize()
            
            # RAG 오케스트레이터
            rag_orchestrator = RAGOrchestrator()
            await rag_orchestrator.initialize()
            
            # MCP 생태계 오케스트레이터
            mcp_orchestrator = MCPEcosystemOrchestrator()
            await mcp_orchestrator.initialize()
            
            # 대화 엔진
            conversational_engine = ConversationalEngine(
                korean_optimizer=korean_optimizer,
                rag_orchestrator=rag_orchestrator,
                graph_manager=graph_manager,
                mcp_orchestrator=mcp_orchestrator
            )
            await conversational_engine.initialize()
            
            # Think 블록 매니저 (더미)
            think_block_manager = DummyComponent()
            
            # UI 오케스트레이터
            ui_orchestrator = UIOrchestrator(
                config_manager=config_manager,
                conversational_engine=conversational_engine,
                korean_optimizer=korean_optimizer,
                graph_manager=graph_manager,
                rag_orchestrator=rag_orchestrator,
                mcp_orchestrator=mcp_orchestrator,
                think_block_manager=think_block_manager
            )
            await ui_orchestrator.initialize()
            
            # UI 컴포넌트들
            chat_interface = ChatInterface(ui_orchestrator, korean_optimizer)
            think_visualizer = ThinkVisualizer(ui_orchestrator, korean_optimizer)
            knowledge_explorer = KnowledgeExplorer(ui_orchestrator, graph_manager)
            performance_dashboard = PerformanceDashboard(ui_orchestrator)
            mcp_monitor = MCPMonitor(ui_orchestrator, mcp_orchestrator)
            
            logger.info("✅ 모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            # 오류 발생 시 더미 컴포넌트로 폴백
            ui_orchestrator = DummyComponent()
            chat_interface = DummyComponent()
            think_visualizer = DummyComponent()
            knowledge_explorer = DummyComponent()
            performance_dashboard = DummyComponent()
            mcp_monitor = DummyComponent()
            logger.warning("⚠️ 더미 컴포넌트로 폴백하여 실행 계속")
        
        # 4. 웹 서버 생성 (환경 변수 기반 설정)
        available_port = int(os.getenv('PORT', 8002))  # 기본값 8002
        server_host = os.getenv('HOST', '0.0.0.0')  # 기본값 0.0.0.0
        web_server = WebServer(
            ui_orchestrator=ui_orchestrator,
            chat_interface=chat_interface,
            think_visualizer=think_visualizer,
            knowledge_explorer=knowledge_explorer,
            performance_dashboard=performance_dashboard,
            mcp_monitor=mcp_monitor,
            host=server_host,
            port=available_port
        )
        
        # 5. 서버 시작
        logger.info(f"🌐 웹 서버 시작: http://{server_host}:{available_port}")
        logger.info(f"📖 API 문서: http://localhost:{available_port}/docs")
        logger.info(f"🏭 UI 인터페이스: http://localhost:{available_port}/ui")
        
        await web_server.start_server()
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자 중단 요청")
        return 0
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        return 1


if __name__ == "__main__":
    """CLI 진입점"""
    
    # 배너 출력
    print("""
    🏭 PPuRI-AI Ultimate v2.0.0
    ════════════════════════════════════════════════
    뿌리산업 특화 NotebookLM-스타일 AI 시스템

    📋 시스템 구성:
    ├── 🤖 OpenRouter LLM (Gemini 3 Pro, DeepSeek R1, Claude)
    ├── 🔤 BGE-M3 Embeddings (한국어 최적화)
    ├── 🔍 LightRAG Engine (지식그래프 + 하이브리드 검색)
    ├── 🌐 외부 검색 통합 (Tavily, Semantic Scholar, KIPRIS)
    ├── 🎙️ Audio Overview (TTS 팟캐스트 생성)
    ├── 📊 pgvector 벡터 DB (PostgreSQL)
    └── 💬 Source Grounding (인라인 인용 [1][2])

    🏭 뿌리산업 6개 도메인:
    주조 | 금형 | 소성가공 | 용접 | 표면처리 | 열처리
    ════════════════════════════════════════════════
    """)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        sys.exit(1)