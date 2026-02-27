"""
PPuRI-AI Ultimate - Database Module
PostgreSQL + pgvector + SQLAlchemy

Note: pgvector 설치 필요 - pip install pgvector
"""

import logging

logger = logging.getLogger(__name__)

# Models (pgvector 필요)
try:
    from .models import (
        Base,
        Document,
        DocumentChunk,
        Entity,
        Relationship,
        ChatSession,
        ChatMessage,
        User,
        AudioOverview
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database models not available: {e}")
    logger.warning("Install pgvector: pip install pgvector")
    Base = None
    Document = None
    DocumentChunk = None
    Entity = None
    Relationship = None
    ChatSession = None
    ChatMessage = None
    User = None
    AudioOverview = None
    MODELS_AVAILABLE = False

# Vector Store
try:
    from .vector_store import (
        VectorStore,
        VectorStoreConfig,
        get_vector_store
    )
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector store not available: {e}")
    VectorStore = None
    VectorStoreConfig = None
    get_vector_store = None
    VECTOR_STORE_AVAILABLE = False

# Connection
try:
    from .connection import (
        get_async_session,
        get_engine,
        init_database,
        check_database_health,
        close_database
    )
    CONNECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database connection not available: {e}")
    get_async_session = None
    get_engine = None
    init_database = None
    check_database_health = None
    close_database = None
    CONNECTION_AVAILABLE = False

__all__ = [
    # Models
    "Base",
    "Document",
    "DocumentChunk",
    "Entity",
    "Relationship",
    "ChatSession",
    "ChatMessage",
    "User",
    "AudioOverview",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "get_vector_store",
    # Connection
    "get_async_session",
    "get_engine",
    "init_database",
    "check_database_health",
    "close_database",
    # Availability flags
    "MODELS_AVAILABLE",
    "VECTOR_STORE_AVAILABLE",
    "CONNECTION_AVAILABLE"
]
