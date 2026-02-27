-- PPuRI-AI Ultimate - PostgreSQL + pgvector 초기화 스크립트
-- 실행: docker-compose 시작 시 자동 실행

-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 텍스트 유사도

-- 스키마 생성
CREATE SCHEMA IF NOT EXISTS ppuri;

-- 사용자 테이블
CREATE TABLE IF NOT EXISTS ppuri.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    industry VARCHAR(50),
    organization VARCHAR(200),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ
);

-- 문서 테이블
CREATE TABLE IF NOT EXISTS ppuri.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES ppuri.users(id),
    title VARCHAR(500) NOT NULL,
    content TEXT,
    summary TEXT,
    file_path VARCHAR(1000),
    file_type VARCHAR(50),
    file_size INTEGER,
    industry VARCHAR(50),
    category VARCHAR(100),
    tags TEXT[],
    status VARCHAR(50) DEFAULT 'pending',
    indexed_at TIMESTAMPTZ,
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    source_url VARCHAR(2000),
    language VARCHAR(10) DEFAULT 'ko',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ
);

-- 문서 청크 테이블 (벡터 임베딩 포함)
CREATE TABLE IF NOT EXISTS ppuri.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES ppuri.documents(id) ON DELETE CASCADE NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    embedding_dense vector(1024),  -- BGE-M3 Dense vector
    embedding_sparse JSONB,         -- Sparse vector as JSON
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 엔티티 테이블 (지식 그래프)
CREATE TABLE IF NOT EXISTS ppuri.entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES ppuri.documents(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    normalized_name VARCHAR(500),
    entity_type VARCHAR(100) NOT NULL,
    description TEXT,
    properties JSONB DEFAULT '{}',
    aliases TEXT[],
    embedding vector(1024),
    importance_score FLOAT DEFAULT 0.5,
    mention_count INTEGER DEFAULT 1,
    industry VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ,
    UNIQUE(normalized_name, entity_type)
);

-- 관계 테이블 (지식 그래프)
CREATE TABLE IF NOT EXISTS ppuri.relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES ppuri.entities(id) ON DELETE CASCADE NOT NULL,
    target_id UUID REFERENCES ppuri.entities(id) ON DELETE CASCADE NOT NULL,
    relation_type VARCHAR(100) NOT NULL,
    description TEXT,
    weight FLOAT DEFAULT 1.0,
    properties JSONB DEFAULT '{}',
    embedding vector(1024),
    source_chunk_id UUID,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 채팅 세션 테이블
CREATE TABLE IF NOT EXISTS ppuri.chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES ppuri.users(id),
    title VARCHAR(500),
    industry VARCHAR(50),
    search_mode VARCHAR(50) DEFAULT 'web_enabled',
    is_active BOOLEAN DEFAULT true,
    message_count INTEGER DEFAULT 0,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ,
    last_message_at TIMESTAMPTZ
);

-- 채팅 메시지 테이블
CREATE TABLE IF NOT EXISTS ppuri.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES ppuri.chat_sessions(id) ON DELETE CASCADE NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]',
    reasoning_details JSONB,
    model_used VARCHAR(100),
    search_time_ms FLOAT,
    generation_time_ms FLOAT,
    token_count INTEGER,
    feedback_rating INTEGER,
    feedback_text TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Audio Overview 테이블
CREATE TABLE IF NOT EXISTS ppuri.audio_overviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES ppuri.users(id),
    title VARCHAR(500) NOT NULL,
    summary TEXT,
    audio_path VARCHAR(1000),
    audio_format VARCHAR(20) DEFAULT 'mp3',
    duration_seconds FLOAT,
    file_size INTEGER,
    transcript JSONB,
    transcript_text TEXT,
    source_document_ids UUID[],
    source_document_titles TEXT[],
    style VARCHAR(50) DEFAULT 'conversational',
    tts_provider VARCHAR(50) DEFAULT 'edge_tts',
    voice_config JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    industry VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- 검색 캐시 테이블
CREATE TABLE IF NOT EXISTS ppuri.search_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(64) UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    search_mode VARCHAR(50),
    industry_filter VARCHAR(50),
    results JSONB NOT NULL,
    result_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,
    hit_count INTEGER DEFAULT 0
);

-- 인덱스 생성
-- 문서 인덱스
CREATE INDEX IF NOT EXISTS idx_documents_industry ON ppuri.documents(industry);
CREATE INDEX IF NOT EXISTS idx_documents_status ON ppuri.documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON ppuri.documents(owner_id);

-- 청크 벡터 인덱스 (IVFFlat - 코사인 유사도)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON ppuri.document_chunks
    USING ivfflat (embedding_dense vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON ppuri.document_chunks(document_id);

-- 엔티티 인덱스
CREATE INDEX IF NOT EXISTS idx_entities_name ON ppuri.entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON ppuri.entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON ppuri.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_industry ON ppuri.entities(industry);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON ppuri.entities
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 관계 인덱스
CREATE INDEX IF NOT EXISTS idx_relationships_source ON ppuri.relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON ppuri.relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON ppuri.relationships(relation_type);

-- 세션/메시지 인덱스
CREATE INDEX IF NOT EXISTS idx_sessions_user ON ppuri.chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON ppuri.chat_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_messages_session ON ppuri.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON ppuri.chat_messages(created_at);

-- 캐시 인덱스
CREATE INDEX IF NOT EXISTS idx_cache_hash ON ppuri.search_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON ppuri.search_cache(expires_at);

-- 텍스트 검색 인덱스 (GIN)
CREATE INDEX IF NOT EXISTS idx_documents_content_gin ON ppuri.documents
    USING gin (to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, '')));
CREATE INDEX IF NOT EXISTS idx_chunks_content_gin ON ppuri.document_chunks
    USING gin (to_tsvector('simple', content));
CREATE INDEX IF NOT EXISTS idx_entities_name_gin ON ppuri.entities
    USING gin (to_tsvector('simple', name || ' ' || coalesce(description, '')));

-- 함수: 벡터 유사도 검색
CREATE OR REPLACE FUNCTION ppuri.search_similar_chunks(
    query_embedding vector(1024),
    limit_count INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.5,
    industry_filter VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    page_number INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.content,
        c.page_number,
        (1 - (c.embedding_dense <=> query_embedding))::FLOAT as similarity
    FROM ppuri.document_chunks c
    JOIN ppuri.documents d ON c.document_id = d.id
    WHERE c.embedding_dense IS NOT NULL
      AND (industry_filter IS NULL OR d.industry = industry_filter)
      AND (1 - (c.embedding_dense <=> query_embedding)) >= similarity_threshold
    ORDER BY c.embedding_dense <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- 함수: 엔티티 유사도 검색
CREATE OR REPLACE FUNCTION ppuri.search_similar_entities(
    query_embedding vector(1024),
    limit_count INTEGER DEFAULT 10,
    entity_types VARCHAR[] DEFAULT NULL,
    industry_filter VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    entity_id UUID,
    name VARCHAR,
    entity_type VARCHAR,
    description TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.name,
        e.entity_type,
        e.description,
        (1 - (e.embedding <=> query_embedding))::FLOAT as similarity
    FROM ppuri.entities e
    WHERE e.embedding IS NOT NULL
      AND (entity_types IS NULL OR e.entity_type = ANY(entity_types))
      AND (industry_filter IS NULL OR e.industry = industry_filter)
    ORDER BY e.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- 초기 테스트 데이터 (선택적)
-- INSERT INTO ppuri.users (email, hashed_password, name, role)
-- VALUES ('admin@ppuri.ai', '$2b$12$...', 'Admin', 'admin');

COMMENT ON SCHEMA ppuri IS 'PPuRI-AI Ultimate - 뿌리산업 AI 시스템';
COMMENT ON TABLE ppuri.documents IS '문서 저장소';
COMMENT ON TABLE ppuri.document_chunks IS '문서 청크 및 벡터 임베딩';
COMMENT ON TABLE ppuri.entities IS '지식 그래프 엔티티';
COMMENT ON TABLE ppuri.relationships IS '지식 그래프 관계';
