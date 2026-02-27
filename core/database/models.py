"""
PPuRI-AI Ultimate - Database Models
SQLAlchemy + pgvector 기반 데이터 모델

Features:
- 문서 및 청크 관리
- 벡터 임베딩 저장 (pgvector)
- 지식 그래프 (엔티티/관계)
- 채팅 세션 및 메시지
- Audio Overview 저장
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON, Enum, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

# 벡터 차원 (BGE-M3 기본값)
VECTOR_DIM = 1024


class User(Base):
    """사용자"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(100))
    role = Column(String(50), default="user")  # user, admin, expert
    industry = Column(String(50))  # casting, mold, welding, forming, surface, heat
    organization = Column(String(200))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True))

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", lazy="dynamic")
    documents = relationship("Document", back_populates="owner", lazy="dynamic")


class Document(Base):
    """문서"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    summary = Column(Text)
    file_path = Column(String(1000))
    file_type = Column(String(50))  # pdf, docx, txt, html
    file_size = Column(Integer)

    # 분류
    industry = Column(String(50), index=True)  # 뿌리산업 분류
    category = Column(String(100))  # 문서 카테고리
    tags = Column(ARRAY(String(50)))

    # 상태
    status = Column(String(50), default="pending")  # pending, processing, indexed, failed
    indexed_at = Column(DateTime(timezone=True))
    chunk_count = Column(Integer, default=0)
    entity_count = Column(Integer, default=0)

    # 메타데이터
    source_url = Column(String(2000))
    language = Column(String(10), default="ko")
    metadata = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_document_industry", "industry"),
        Index("idx_document_status", "status"),
    )


class DocumentChunk(Base):
    """문서 청크 (벡터 임베딩 포함)"""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # 콘텐츠
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="text")  # text, table, image_caption
    chunk_index = Column(Integer, nullable=False)

    # 위치 정보
    page_number = Column(Integer)
    start_char = Column(Integer)
    end_char = Column(Integer)

    # 벡터 임베딩 (pgvector)
    embedding_dense = Column(Vector(VECTOR_DIM))  # Dense vector
    embedding_sparse = Column(JSONB)  # Sparse vector as JSON

    # 메타데이터
    token_count = Column(Integer)
    metadata = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document", "document_id"),
        Index("idx_chunk_embedding", "embedding_dense", postgresql_using='ivfflat',
              postgresql_ops={'embedding_dense': 'vector_cosine_ops'}),
    )


class Entity(Base):
    """지식 그래프 엔티티"""
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))

    # 엔티티 정보
    name = Column(String(500), nullable=False, index=True)
    normalized_name = Column(String(500), index=True)  # 정규화된 이름 (중복 방지)
    entity_type = Column(String(100), nullable=False)  # material, process, equipment, defect, parameter
    description = Column(Text)

    # 속성
    properties = Column(JSONB, default={})
    aliases = Column(ARRAY(String(200)))  # 다른 이름들

    # 벡터 임베딩
    embedding = Column(Vector(VECTOR_DIM))

    # 점수
    importance_score = Column(Float, default=0.5)
    mention_count = Column(Integer, default=1)

    # 산업 분류
    industry = Column(String(50))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    document = relationship("Document", back_populates="entities")
    source_relations = relationship("Relationship", foreign_keys="Relationship.source_id",
                                    back_populates="source", cascade="all, delete-orphan")
    target_relations = relationship("Relationship", foreign_keys="Relationship.target_id",
                                    back_populates="target", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("normalized_name", "entity_type", name="uq_entity_normalized"),
        Index("idx_entity_type", "entity_type"),
        Index("idx_entity_industry", "industry"),
        Index("idx_entity_embedding", "embedding", postgresql_using='ivfflat',
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )


class Relationship(Base):
    """지식 그래프 관계"""
    __tablename__ = "relationships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)

    # 관계 정보
    relation_type = Column(String(100), nullable=False)  # CAUSES, PREVENTS, REQUIRES, PRODUCES
    description = Column(Text)
    weight = Column(Float, default=1.0)

    # 속성
    properties = Column(JSONB, default={})

    # 벡터 임베딩
    embedding = Column(Vector(VECTOR_DIM))

    # 출처
    source_chunk_id = Column(UUID(as_uuid=True))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    source = relationship("Entity", foreign_keys=[source_id], back_populates="source_relations")
    target = relationship("Entity", foreign_keys=[target_id], back_populates="target_relations")

    __table_args__ = (
        Index("idx_relationship_source", "source_id"),
        Index("idx_relationship_target", "target_id"),
        Index("idx_relationship_type", "relation_type"),
    )


class ChatSession(Base):
    """채팅 세션"""
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # 세션 정보
    title = Column(String(500))
    industry = Column(String(50))  # 산업 필터
    search_mode = Column(String(50), default="web_enabled")  # documents_only, web_enabled, full_search

    # 상태
    is_active = Column(Boolean, default=True)
    message_count = Column(Integer, default=0)

    # 설정
    settings = Column(JSONB, default={})

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan",
                           order_by="ChatMessage.created_at")

    __table_args__ = (
        Index("idx_session_user", "user_id"),
        Index("idx_session_active", "is_active"),
    )


class ChatMessage(Base):
    """채팅 메시지"""
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)

    # 메시지 정보
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)

    # AI 응답 메타데이터
    citations = Column(JSONB, default=[])  # 인용 정보
    reasoning_details = Column(JSONB)  # 추론 과정
    model_used = Column(String(100))

    # 성능 메트릭
    search_time_ms = Column(Float)
    generation_time_ms = Column(Float)
    token_count = Column(Integer)

    # 피드백
    feedback_rating = Column(Integer)  # 1-5
    feedback_text = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("idx_message_session", "session_id"),
        Index("idx_message_created", "created_at"),
    )


class AudioOverview(Base):
    """Audio Overview (팟캐스트)"""
    __tablename__ = "audio_overviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # 기본 정보
    title = Column(String(500), nullable=False)
    summary = Column(Text)

    # 오디오 파일
    audio_path = Column(String(1000))
    audio_format = Column(String(20), default="mp3")
    duration_seconds = Column(Float)
    file_size = Column(Integer)

    # 대본
    transcript = Column(JSONB)  # DialogueTurn 리스트
    transcript_text = Column(Text)  # 텍스트 버전

    # 소스 문서
    source_document_ids = Column(ARRAY(UUID(as_uuid=True)))
    source_document_titles = Column(ARRAY(String(500)))

    # 설정
    style = Column(String(50), default="conversational")
    tts_provider = Column(String(50), default="edge_tts")
    voice_config = Column(JSONB)

    # 상태
    status = Column(String(50), default="pending")  # pending, generating, completed, failed
    error_message = Column(Text)

    # 산업 분류
    industry = Column(String(50))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_audio_user", "user_id"),
        Index("idx_audio_status", "status"),
    )


class SearchCache(Base):
    """검색 결과 캐시"""
    __tablename__ = "search_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 쿼리 정보
    query_hash = Column(String(64), unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    search_mode = Column(String(50))
    industry_filter = Column(String(50))

    # 결과
    results = Column(JSONB, nullable=False)
    result_count = Column(Integer)

    # 메타
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    hit_count = Column(Integer, default=0)

    __table_args__ = (
        Index("idx_cache_expires", "expires_at"),
    )
