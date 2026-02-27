"""
PPuRI-AI Ultimate - Database Connection
SQLAlchemy 비동기 데이터베이스 연결 관리
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from .models import Base

logger = logging.getLogger(__name__)

# 글로벌 엔진 및 세션 팩토리
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    """데이터베이스 URL 가져오기"""
    # 환경변수에서 가져오기
    db_url = os.getenv("DATABASE_URL")

    if db_url:
        # asyncpg 드라이버로 변환
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        return db_url

    # 기본값 구성
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "ppuri")
    db_password = os.getenv("DB_PASSWORD", "ppuri_secret")
    db_name = os.getenv("DB_NAME", "ppuri_ai")

    return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def get_engine(pool_size: int = 10, max_overflow: int = 20) -> AsyncEngine:
    """비동기 엔진 가져오기 (싱글톤)"""
    global _engine

    if _engine is None:
        database_url = get_database_url()
        logger.info(f"Creating database engine: {database_url.split('@')[-1]}")

        _engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1시간마다 연결 재생성
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """세션 팩토리 가져오기"""
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )

    return _session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """비동기 세션 컨텍스트 매니저"""
    session_factory = get_session_factory()
    session = session_factory()

    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_database(drop_existing: bool = False) -> bool:
    """
    데이터베이스 초기화

    Args:
        drop_existing: True면 기존 테이블 삭제 후 재생성

    Returns:
        성공 여부
    """
    engine = get_engine()

    try:
        async with engine.begin() as conn:
            # pgvector 확장 생성
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")

            if drop_existing:
                logger.warning("Dropping all existing tables...")
                await conn.run_sync(Base.metadata.drop_all)

            # 테이블 생성
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

            # 인덱스 확인
            result = await conn.execute(text("""
                SELECT indexname FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY indexname
            """))
            indexes = [row[0] for row in result.fetchall()]
            logger.info(f"Database indexes: {len(indexes)} indexes created")

        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


async def check_database_health() -> dict:
    """데이터베이스 상태 확인"""
    engine = get_engine()

    try:
        async with engine.begin() as conn:
            # 연결 테스트
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()

            # 테이블 수 확인
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            table_count = result.scalar()

            # pgvector 확인
            result = await conn.execute(text("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """))
            pgvector_enabled = result.scalar()

            # 문서 수 확인
            try:
                result = await conn.execute(text("SELECT COUNT(*) FROM documents"))
                doc_count = result.scalar()
            except:
                doc_count = 0

            # 엔티티 수 확인
            try:
                result = await conn.execute(text("SELECT COUNT(*) FROM entities"))
                entity_count = result.scalar()
            except:
                entity_count = 0

        return {
            "status": "healthy",
            "connected": True,
            "pgvector_enabled": pgvector_enabled,
            "table_count": table_count,
            "document_count": doc_count,
            "entity_count": entity_count
        }

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }


async def close_database():
    """데이터베이스 연결 종료"""
    global _engine, _session_factory

    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connection closed")
