# PPuRI-AI Ultimate DX+AX 개선 스펙안 v1.0

> **작성일**: 2026-01-09
> **프로젝트**: PPuRI-AI Ultimate (뿌리산업 특화 AI 플랫폼)
> **현재 버전**: v1.0.0 (f08ee11)

---

## Executive Summary

PPuRI-AI Ultimate 프로젝트에 대한 철저한 분석 결과, **종합 점수 7.54/10**으로 평가되었습니다.

### 분석 결과 요약

| 영역 | 점수 | 주요 발견 |
|------|:----:|----------|
| 시스템 아키텍처 | 7.5/10 | 모듈화 양호, 순환 의존성 위험 |
| 데이터 아키텍처 | 7.8/10 | 네임스페이스 설계 우수, 캐싱 개선 필요 |
| API 설계 | 7.5/10 | RESTful 준수, WebSocket 재연결 미흡 |
| 보안 아키텍처 | 7.6/10 | 미들웨어 양호, 시크릿 관리 취약 |
| 코드 품질 | 7.0/10 | 31K LOC, 테스트 0%, 문서화 75% |

### 핵심 개선 영역

| 영역 | 현재 상태 | 목표 상태 | 우선순위 |
|------|----------|----------|:--------:|
| 보안 | 하드코딩된 시크릿 | 환경 변수/Secret Manager | **P0** |
| RAG 성능 | 기본 하이브리드 검색 | GraphRAG + CRAG | **P1** |
| 임베딩 | all-MiniLM-L6-v2 | BGE-M3-Korean | **P1** |
| LLM 추론 | Ollama (개발용) | vLLM (프로덕션) | **P2** |
| 프롬프트 최적화 | AdalFlow | DSPy MIPROv2 | **P2** |
| 테스트 | 0% 커버리지 | 80%+ 커버리지 | **P1** |

---

## Part 1: 즉시 조치 필요 사항 (P0)

### 1.1 보안 취약점 해결

#### 1.1.1 시크릿 하드코딩 제거

**현재 문제점 (docker-compose.yml):**
- DATABASE_URL에 평문 비밀번호 포함
- NEO4J_PASSWORD, POSTGRES_PASSWORD 하드코딩
- Redis 인증 없음

**개선안:**
1. `.env` 파일로 시크릿 분리
2. `.env.example` 템플릿 파일 생성
3. Docker Secrets 또는 HashiCorp Vault 도입 검토

**필수 생성 파일: `.env.example`**
```
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname
NEO4J_PASSWORD=your-neo4j-password
JWT_SECRET_KEY=your-256-bit-secret-minimum-32-characters
CSRF_SECRET_KEY=your-csrf-secret
REDIS_PASSWORD=your-redis-password
OLLAMA_BASE_URL=http://localhost:11434
```

#### 1.1.2 JWT 시크릿 강제화

**현재 (api/constants.py:74):**
- 기본값 'your-secret-key-here' 사용

**개선안:**
- 환경 변수 필수화 (기본값 제거)
- 최소 32자 검증
- 시작 시 검증 실패하면 애플리케이션 종료

#### 1.1.3 세션 데이터 직렬화 개선

**현재 (api/middleware/auth.py:296):**
- ast.literal_eval 사용 (보안 위험)

**개선안:**
- JSON 직렬화로 변경 (json.loads/json.dumps)

### 1.2 Deprecated API 수정

**현재 (api/middleware/auth.py:49,80):**
```python
# Python 3.12+에서 deprecated
now = datetime.utcnow()
```

**개선안:**
```python
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
```

---

## Part 2: 기술 스택 현대화 (P1)

### 2.1 임베딩 모델 업그레이드

#### 현재 vs 권장

| 모델 | 차원 | 한국어 성능 | 최대 토큰 |
|------|-----|-----------|----------|
| all-MiniLM-L6-v2 (현재) | 384 | 낮음 | 512 |
| **BGE-M3** (권장) | 1024 | 매우 높음 | 8192 |
| **dragonkue/BGE-m3-ko** (최적) | 1024 | 최고 | 8192 |
| multilingual-e5-large | 1024 | 높음 | 512 |

**마이그레이션 단계:**
1. 새 임베딩 모델로 컬렉션 생성
2. 기존 문서 재인덱싱 (배치 처리)
3. A/B 테스트로 성능 검증
4. 기존 컬렉션 아카이브/삭제

**예상 효과:** 한국어 검색 정확도 30%+ 향상

### 2.2 RAG 아키텍처 고도화

#### 2.2.1 GraphRAG 통합

**Microsoft GraphRAG 파이프라인:**
1. 문서 입력
2. 엔티티/관계 추출 (LLM)
3. 지식 그래프 구축 (Neo4j) - 기존 인프라 활용
4. 커뮤니티 탐지 (Louvain/Leiden)
5. 커뮤니티 요약 사전 생성
6. 검색 시: 글로벌 + 로컬 검색 통합

**신규 모듈 구조:**
```
core/graphrag/
├── __init__.py
├── entity_extractor.py      # LLM 기반 엔티티 추출
├── community_detector.py    # 그래프 커뮤니티 탐지
├── summarizer.py           # 커뮤니티 요약 생성
└── global_search.py        # 글로벌 검색 엔진
```

**PPuRI-AI 적합성:** 매우 높음
- 기존 Neo4j GraphManager 인프라 활용 가능
- 뿌리산업 도메인 지식의 관계형 표현에 이상적

#### 2.2.2 Corrective RAG (CRAG) 적용

**목적:** 검색 결과 신뢰도 검증 및 자가 수정

**작동 원리:**
1. 문서 검색 후 신뢰도 평가
2. 낮은 신뢰도 시 웹 검색 보완
3. 지식 정제 (Knowledge Refinement)
4. 환각(Hallucination) 감소

**신뢰도 임계값:** 0.7 권장

### 2.3 프롬프트 최적화 (AdalFlow → DSPy)

#### DSPy 장점
- Stanford NLP 연구진 개발, Nature 발표
- 프레임워크 오버헤드 최저 (~3.53ms)
- 자동 프롬프트 최적화 (MIPROv2, COPRO)
- 선언적 프로그래밍 패러다임

#### 마이그레이션 계획

**Phase 1: 병행 운영**
- 기존 adalflow_engine 유지 (deprecated 표시)
- 신규 dspy_engine 모듈 생성

**신규 모듈 구조:**
```
core/dspy_engine/
├── __init__.py
├── signatures.py       # DSPy 시그니처 정의
├── modules.py          # DSPy 모듈 정의
├── optimizers.py       # MIPROv2, COPRO 래퍼
└── evaluators.py       # 평가 메트릭
```

**주요 옵티마이저:**
- MIPROv2: Few-shot 없을 때 최적
- BootstrapFewShotWithRandomSearch: 샘플 있을 때 최고 성능
- COPRO: 좌표 하강법 기반 프롬프트 개선

---

## Part 3: 인프라 현대화 (P2)

### 3.1 LLM 추론 엔진 (Ollama → vLLM)

#### 성능 비교

| 지표 | Ollama (현재) | vLLM (권장) | 개선율 |
|-----|-------------|------------|:-----:|
| 처리량 (RPS) | 1-3 | 120-160 | **40x** |
| 동시 요청 | 제한적 | 128+ | **높음** |
| 메모리 효율 | 중간 | PagedAttention | **50% 감소** |
| GPU 활용 | 기본 | 최적화 | **높음** |

#### 마이그레이션 전략

**Phase 1: 병행 운영**
- 개발: Ollama 유지
- 프로덕션: vLLM 배포

**LLM 추상화 계층 구현:**
```
core/llm/
├── __init__.py
├── provider.py         # 추상 베이스 클래스
├── ollama_provider.py  # Ollama 구현
└── vllm_provider.py    # vLLM 구현 (OpenAI 호환)
```

### 3.2 벡터 데이터베이스 최적화

#### 현재: ChromaDB + Qdrant 이중 구성

**권장 전략:**
| 환경 | 권장 DB | 이유 |
|-----|--------|------|
| 개발/테스트 | ChromaDB | 빠른 이터레이션 |
| 프로덕션 | Qdrant | 고성능, 강력한 필터링 |
| 대규모 확장 | Milvus | 빌리언 벡터 지원 |

**Qdrant 최적화 설정:**
- gRPC 프로토콜 사용 (prefer_grpc=True)
- HNSW 인덱스 튜닝 (m=16, ef_construct=100)
- 배치 upsert 사용

### 3.3 한국어 처리 강화

#### KoNLPy → Kiwipiepy 전환

**이유:**
| 라이브러리 | 의존성 | 성능 | 설치 |
|-----------|-------|------|------|
| KoNLPy | Java 필수 | 중간 | 복잡 |
| Kiwipiepy | Pure Python | 빠름 | 간단 |

**구현 계획:**
1. Kiwipiepy 설치 (kiwipiepy>=0.18.0)
2. 뿌리산업 용어 사용자 사전 등록
3. korean_optimizer.py 토큰화 로직 개선

---

## Part 4: 아키텍처 개선 (P2)

### 4.1 코드 구조 정리

#### 루트 레벨 파일 정리

**현재:** 22개 실행/테스트 스크립트가 루트에 산재

**개선안:**
```
PPuri_DX+AX/
├── scripts/
│   ├── server/          # 서버 실행 스크립트
│   ├── test/            # 테스트 스크립트
│   └── utils/           # 유틸리티 스크립트
├── tests/               # 신규 생성
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── main.py              # 유일한 진입점
```

### 4.2 UIOrchestrator 분리

**현재:** 1,333줄의 God Object

**개선안 (단일 책임 원칙 적용):**
```
ui/orchestrator/
├── __init__.py
├── session_manager.py       # 세션 관리 (~300줄)
├── event_dispatcher.py      # 이벤트 발행 (~200줄)
├── websocket_manager.py     # WebSocket 관리 (~300줄)
├── component_registry.py    # 컴포넌트 등록 (~200줄)
└── orchestrator.py          # 통합 오케스트레이터 (~300줄)
```

### 4.3 테스트 인프라 구축

**pytest 설정:**
- asyncio_mode = auto
- 커버리지 리포트 (--cov)
- HTML 리포트 생성

**테스트 유형:**
| 유형 | 커버리지 목표 | 우선순위 |
|-----|-------------|:--------:|
| Unit Tests | 80% | P1 |
| Integration Tests | 60% | P2 |
| E2E Tests | 핵심 시나리오 | P3 |

---

## Part 5: 모니터링 및 운영 (P3)

### 5.1 Observability 스택 활성화

**현재:** docker-compose에 정의됨, profiles로 비활성화

**활성화 계획:**
- Prometheus: 메트릭 수집
- Grafana: 대시보드 시각화
- profiles 제거하여 기본 활성화

### 5.2 분산 추적 (OpenTelemetry)

**신규 모듈:**
```
core/observability/
├── __init__.py
├── tracing.py           # OpenTelemetry 설정
└── metrics.py           # 커스텀 메트릭
```

**필요 라이브러리:**
- opentelemetry-api>=1.20.0
- opentelemetry-sdk>=1.20.0
- opentelemetry-exporter-otlp>=1.20.0

---

## Part 6: 실행 로드맵

### Phase 1: 긴급 보안 패치 (1주)

| 작업 | 담당 | 완료 기준 |
|-----|-----|---------|
| 시크릿 하드코딩 제거 | DevOps | .env 기반 전환 완료 |
| JWT 시크릿 강제화 | Backend | 환경 변수 없으면 시작 실패 |
| Deprecated API 수정 | Backend | Python 3.12 호환 |
| .env.example 생성 | DevOps | 템플릿 파일 커밋 |

### Phase 2: 기술 현대화 (1개월)

| 작업 | 담당 | 완료 기준 |
|-----|-----|---------|
| BGE-M3-Korean 적용 | ML Engineer | 검색 정확도 30% 향상 |
| 테스트 인프라 구축 | QA | 80% 커버리지 달성 |
| 루트 파일 정리 | All | scripts/, tests/ 분리 |
| UIOrchestrator 분리 | Frontend | 5개 모듈로 분리 |

### Phase 3: RAG 고도화 (2개월)

| 작업 | 담당 | 완료 기준 |
|-----|-----|---------|
| GraphRAG 통합 | ML Engineer | Neo4j 연동 완료 |
| CRAG 검증 레이어 | ML Engineer | 신뢰도 평가 자동화 |
| DSPy 마이그레이션 | ML Engineer | AdalFlow 완전 대체 |
| Kiwipiepy 전환 | Backend | KoNLPy 제거 |

### Phase 4: 프로덕션 최적화 (3개월)

| 작업 | 담당 | 완료 기준 |
|-----|-----|---------|
| vLLM 프로덕션 배포 | DevOps | 처리량 10x 향상 |
| Qdrant 최적화 | DevOps | p99 레이턴시 < 100ms |
| 모니터링 활성화 | DevOps | 대시보드 운영 |
| OpenTelemetry 통합 | DevOps | 분산 추적 가능 |

---

## Part 7: 예상 효과

### 정량적 개선

| 지표 | 현재 | 목표 | 개선율 |
|-----|-----|-----|:-----:|
| 한국어 검색 정확도 | ~70% | 90%+ | **+28%** |
| LLM 처리량 (RPS) | 3 | 100+ | **33x** |
| 응답 레이턴시 (p99) | 3s | 1s | **3x** |
| 테스트 커버리지 | 0% | 80% | **+80%** |
| 보안 점수 | 6.5/10 | 9/10 | **+38%** |

### 정성적 개선

- **개발자 경험**: 테스트 자동화, 명확한 코드 구조
- **운영 안정성**: 분산 추적, 실시간 모니터링
- **확장성**: 마이크로서비스 전환 준비 완료
- **보안**: 제로 트러스트 원칙 적용

---

## 부록 A: 신규 의존성 요약

### 추가 (requirements.txt)
```
dspy-ai>=2.5.0               # 프롬프트 자동 최적화
graphrag>=0.3.0              # Microsoft GraphRAG
graspologic>=3.3.0           # 그래프 통계
kiwipiepy>=0.18.0            # 한국어 형태소 분석
opentelemetry-api>=1.20.0    # 분산 추적
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
```

### 제거/교체 (requirements.txt)
```
# konlpy>=0.6.0              # 제거 → kiwipiepy로 대체
# adalflow>=0.2.0            # 점진적 → dspy-ai로 대체
```

## 부록 B: 참고 자료

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [DSPy Documentation](https://dspy.ai/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qdrant Benchmarks](https://qdrant.tech/benchmarks/)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [BGE-M3-Korean](https://huggingface.co/dragonkue/BGE-m3-ko)
- [Kiwipiepy](https://github.com/bab2min/kiwipiepy)
- [LangGraph Multi-Agent](https://www.langchain.com/langgraph)

---

**문서 버전**: 1.0
**작성자**: AI Analysis Team
**검토자**: -
**승인일**: -
