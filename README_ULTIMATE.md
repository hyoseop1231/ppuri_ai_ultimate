# PPuRI-AI Ultimate v2.0.0

> NotebookLM-Style AI System for Korean Root Industries (뿌리산업)

## Overview

PPuRI-AI Ultimate is a comprehensive AI system designed for Korean manufacturing industries (주조, 금형, 소성가공, 용접, 표면처리, 열처리). It combines the best features of NotebookLM, LightRAG, and modern RAG architectures.

## Architecture

```
PPuRI-AI Ultimate
├── Core AI Engine
│   ├── OpenRouter LLM (Gemini 3 Pro, DeepSeek R1, Claude)
│   ├── BGE-M3 Embeddings (Korean-optimized)
│   └── LightRAG Engine (Knowledge Graph + Hybrid Search)
├── External Search
│   ├── Tavily AI Search
│   ├── Semantic Scholar (Academic)
│   ├── KIPRIS (Korean Patents)
│   └── DuckDuckGo (Fallback)
├── Storage
│   ├── PostgreSQL + pgvector (Vector DB)
│   ├── Neo4j (Knowledge Graph)
│   ├── Redis (Cache)
│   └── Qdrant (High-performance Vector)
├── UI Layer
│   ├── Streamlit (NotebookLM-style)
│   └── FastAPI (REST/WebSocket)
└── Audio Overview
    ├── Edge TTS (Free, Korean)
    └── OpenAI TTS (Premium)
```

## Features

### 1. Source Grounding with Inline Citations
- Every response includes inline citations `[1][2]`
- Click to expand source details
- Direct links to original documents

### 2. Knowledge Graph (LightRAG-style)
- Entity extraction and normalization
- Relationship mapping
- Graph-based retrieval
- Community detection

### 3. Hybrid Search
- Dense vector search (BGE-M3)
- Sparse search (BM25)
- Knowledge graph traversal
- RRF (Reciprocal Rank Fusion)

### 4. Audio Overview (NotebookLM Podcast)
- Multi-speaker dialogue generation
- Korean TTS optimization
- Export to MP3/WAV

### 5. Industry-Specific AI
- 6 specialized domains
- Technical terminology handling
- Domain-specific prompts

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - OPENROUTER_API_KEY
# - TAVILY_API_KEY

# 2. Start all services
docker-compose up -d

# 3. Access the UI
# Streamlit: http://localhost:8501
# API Docs: http://localhost:8000/docs
# Neo4j Browser: http://localhost:7474
# Grafana: http://localhost:3000
```

### Option 2: Development Mode

```bash
# 1. Install dependencies
./scripts/start.sh install

# 2. Start development servers
./scripts/start.sh start dev

# This starts:
# - API Server: http://localhost:8002
# - Streamlit UI: http://localhost:8501
```

### Option 3: Individual Components

```bash
# API only
./scripts/start.sh start api

# UI only
./scripts/start.sh start ui
```

## Configuration

### Environment Variables

```env
# LLM Configuration
OPENROUTER_API_KEY=your_key_here
DEFAULT_LLM_PROVIDER=openrouter

# Search Configuration
TAVILY_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql+asyncpg://ppuri:password@localhost:5432/ppuri_ultimate
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379/0

# Features
KOREAN_OPTIMIZATION=true
ENABLE_THINK_BLOCK=true
AUTO_OPTIMIZE_PROMPTS=true
```

### Model Tiers

| Tier | Models | Use Case |
|------|--------|----------|
| ULTRATHINK | DeepSeek R1, o1, Gemini 2.5 Pro | Complex reasoning |
| THINK | Claude Sonnet, Gemini 2.0 | Standard tasks |
| FAST | GPT-4o-mini, Gemini Flash, Qwen | Quick responses |
| COST | Llama 3.3, Gemma, DeepSeek V3 | High volume |

## API Endpoints

### Chat
```
POST /api/v1/chat/sessions          # Create session
POST /api/v1/chat/message           # Send message
GET  /api/v1/chat/stream            # SSE streaming
POST /api/v1/chat/export/{format}   # Export conversation
```

### Audio Overview
```
POST /api/v1/audio/generate         # Generate podcast
GET  /api/v1/audio/{id}             # Get audio file
GET  /api/v1/audio/{id}/transcript  # Get transcript
```

### Documents
```
POST /api/v1/documents/upload       # Upload document
GET  /api/v1/documents/{id}         # Get document
POST /api/v1/documents/search       # Semantic search
```

## Project Structure

```
PPuri_DX+AX/
├── core/
│   ├── llm/
│   │   └── openrouter_client.py    # Multi-model LLM client
│   ├── embeddings/
│   │   └── bge_m3_service.py       # Korean-optimized embeddings
│   ├── rag_engine/
│   │   └── lightrag_engine.py      # Knowledge graph RAG
│   ├── connectors/
│   │   └── search_connectors.py    # External search APIs
│   ├── services/
│   │   └── chat_service.py         # NotebookLM-style chat
│   ├── database/
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── connection.py           # Database connections
│   │   ├── vector_store.py         # pgvector operations
│   │   └── graph_db.py             # Neo4j client
│   └── audio/
│       └── audio_overview_engine.py # Podcast generation
├── api/
│   └── routes/
│       ├── chat.py                 # Chat endpoints
│       └── audio_overview.py       # Audio endpoints
├── ui/
│   └── streamlit/
│       └── app.py                  # Streamlit UI
├── tests/
│   └── test_integration.py         # Integration tests
├── scripts/
│   ├── start.sh                    # Quick start script
│   └── init-db.sql                 # Database initialization
├── docker/
│   ├── Dockerfile                  # Main API image
│   └── Dockerfile.streamlit        # UI image
├── docker-compose.yml              # Full stack deployment
├── requirements.txt                # Python dependencies
└── main.py                         # Application entry point
```

## Testing

```bash
# Run all tests
./scripts/start.sh test

# Run specific test
pytest tests/test_integration.py -v

# With coverage
pytest tests/ --cov=core --cov-report=html
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/ppuri_monitor_2024)

### Available Dashboards
- System Overview
- LLM Performance
- Vector Search Metrics
- Knowledge Graph Statistics

## Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| LLM | OpenRouter + Ollama | Multi-model access |
| Embeddings | BGE-M3 | Korean-optimized vectors |
| Vector DB | pgvector, Qdrant | Similarity search |
| Graph DB | Neo4j | Knowledge graph |
| Cache | Redis | Response caching |
| TTS | Edge TTS, OpenAI | Audio generation |
| UI | Streamlit | Web interface |
| API | FastAPI | REST/WebSocket |
| Monitoring | Prometheus + Grafana | Observability |

## Acknowledgments

This project is inspired by:
- [NotebookLM](https://notebooklm.google.com/) - Source grounding and Audio Overview
- [LightRAG](https://github.com/HKUDS/LightRAG) - Knowledge graph RAG
- [SurfSense](https://github.com/surfsense/surfsense) - Open-source NotebookLM alternative
- [ApeRAG](https://github.com/ape-rag/aperag) - Enterprise GraphRAG platform

## License

MIT License - see LICENSE file for details.

---

**PPuRI-AI Ultimate** - Empowering Korean Manufacturing with AI
