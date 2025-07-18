# PPuRI-AI Ultimate Dockerfile
# KITECH 검증된 멀티스테이지 빌드 패턴 기반

# 🏗️ Build Stage - 의존성 및 라이브러리 빌드
FROM python:3.11-slim as builder

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 빌드
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 🚀 Production Stage - 최적화된 런타임
FROM python:3.11-slim as production

# 런타임 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 비root 사용자 생성 (보안 강화)
RUN groupadd -r ppuri && useradd -r -g ppuri ppuri

# 애플리케이션 디렉토리 설정
WORKDIR /app

# Python 패키지 복사 (builder stage에서)
COPY --from=builder /root/.local /home/ppuri/.local
ENV PATH=/home/ppuri/.local/bin:$PATH

# 애플리케이션 코드 복사
COPY . .

# 권한 설정
RUN chown -R ppuri:ppuri /app
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R ppuri:ppuri /app/data /app/logs /app/config

# KITECH 검증된 환경 변수 설정 (5초 빠른 시작)
ENV PRELOAD_EMBEDDING_MODEL=false
ENV ENABLE_EXTERNAL_ACCESS=true
ENV CORS_ORIGINS=*
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 사용자 전환
USER ppuri

# 헬스체크 (KITECH 패턴)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 애플리케이션 시작
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# 🧪 Development Stage - 개발용 확장 기능
FROM production as development

# 개발 도구 설치
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    mypy \
    jupyter \
    ipython

# 개발용 환경 변수
ENV DEBUG=true
ENV RELOAD=true
ENV LOG_LEVEL=DEBUG

# 개발용 볼륨 마운트 포인트
VOLUME ["/app/data", "/app/logs", "/app/config"]

USER ppuri

# 개발 서버 시작 (핫 리로드)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# 🧪 Testing Stage - 테스트 전용
FROM development as testing

USER root

# 테스트 의존성 추가 설치
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-mock \
    factory-boy \
    faker

USER ppuri

# 테스트 실행
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=html", "--cov-report=term"]

# 📊 Monitoring Stage - 모니터링 기능 강화
FROM production as monitoring

USER root

# 모니터링 도구 설치
RUN pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-exporter-prometheus

USER ppuri

# 모니터링 설정
ENV PROMETHEUS_ENABLED=true
ENV METRICS_PORT=9090
ENV OPENTELEMETRY_ENABLED=true

# 모니터링 포트 추가 노출
EXPOSE 9090

# Labels for better container management
LABEL maintainer="PPuRI-AI Team" \
      version="1.0.0" \
      description="PPuRI-AI Ultimate - 뿌리산업 전용 AI 시스템" \
      architecture="multi-stage" \
      base-pattern="KITECH-verified" \
      optimization="5-second-startup"

# 🎯 최적화 설정
# - 멀티스테이지 빌드로 이미지 크기 최소화
# - KITECH 검증된 5초 시작 최적화
# - 비root 사용자로 보안 강화
# - 헬스체크로 안정성 확보
# - 개발/테스트/모니터링 스테이지 분리