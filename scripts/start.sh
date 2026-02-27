#!/bin/bash
# PPuRI-AI Ultimate - Quick Start Script

set -e

echo "
🏭 PPuRI-AI Ultimate v2.0
══════════════════════════════════════════════════
NotebookLM 스타일 뿌리산업 AI 시스템
"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 함수: 에러 처리
error_exit() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

# 함수: 성공 메시지
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 함수: 정보 메시지
info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 함수: 경고 메시지
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 환경 확인
check_environment() {
    echo "📋 환경 확인 중..."

    # Python 버전 확인
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        success "Python 버전: $PYTHON_VERSION"
    else
        error_exit "Python 3이 설치되어 있지 않습니다"
    fi

    # Docker 확인 (선택적)
    if command -v docker &> /dev/null; then
        success "Docker 설치됨"
    else
        warning "Docker가 설치되어 있지 않습니다 (선택적)"
    fi

    # .env 파일 확인
    if [ -f ".env" ]; then
        success ".env 파일 존재"
    else
        warning ".env 파일이 없습니다. .env.example을 복사하세요"
        if [ -f ".env.example" ]; then
            cp .env.example .env
            success ".env.example -> .env 복사됨"
        fi
    fi
}

# 의존성 설치
install_dependencies() {
    echo ""
    echo "📦 의존성 설치 중..."

    # 가상환경 확인/생성
    if [ ! -d "venv" ]; then
        info "가상환경 생성 중..."
        python3 -m venv venv
        success "가상환경 생성됨"
    fi

    # 가상환경 활성화
    source venv/bin/activate

    # pip 업그레이드
    pip install --upgrade pip -q

    # 의존성 설치
    info "핵심 패키지 설치 중... (약 2-5분 소요)"
    pip install -r requirements.txt -q

    success "의존성 설치 완료"
}

# 서비스 시작
start_services() {
    echo ""
    echo "🚀 서비스 시작 중..."

    case "$1" in
        "docker")
            info "Docker Compose로 전체 스택 시작..."
            docker-compose up -d
            success "Docker 서비스 시작됨"
            echo ""
            echo "📊 서비스 URL:"
            echo "  - API: http://localhost:8000"
            echo "  - UI: http://localhost:8501"
            echo "  - Neo4j: http://localhost:7474"
            echo "  - Grafana: http://localhost:3000"
            ;;

        "api")
            info "API 서버만 시작..."
            source venv/bin/activate
            python main.py &
            success "API 서버 시작됨 (http://localhost:8002)"
            ;;

        "ui")
            info "Streamlit UI 시작..."
            source venv/bin/activate
            streamlit run ui/streamlit/app.py &
            success "Streamlit UI 시작됨 (http://localhost:8501)"
            ;;

        "dev")
            info "개발 모드로 시작 (API + UI)..."
            source venv/bin/activate

            # API 백그라운드 시작
            python main.py &
            API_PID=$!
            success "API 서버 시작됨 (PID: $API_PID)"

            # UI 시작
            streamlit run ui/streamlit/app.py &
            UI_PID=$!
            success "Streamlit UI 시작됨 (PID: $UI_PID)"

            echo ""
            echo "📊 서비스 URL:"
            echo "  - API: http://localhost:8002"
            echo "  - UI: http://localhost:8501"
            echo ""
            echo "종료하려면 Ctrl+C를 누르세요"

            # 프로세스 대기
            wait
            ;;

        *)
            echo "사용법: $0 {docker|api|ui|dev}"
            echo ""
            echo "옵션:"
            echo "  docker  - Docker Compose로 전체 스택 시작"
            echo "  api     - API 서버만 시작"
            echo "  ui      - Streamlit UI만 시작"
            echo "  dev     - 개발 모드 (API + UI)"
            exit 1
            ;;
    esac
}

# 테스트 실행
run_tests() {
    echo ""
    echo "🧪 테스트 실행 중..."
    source venv/bin/activate
    pytest tests/ -v --tb=short
    success "테스트 완료"
}

# 메인 스크립트
main() {
    cd "$(dirname "$0")/.."

    case "$1" in
        "install")
            check_environment
            install_dependencies
            ;;

        "start")
            start_services "${2:-dev}"
            ;;

        "test")
            run_tests
            ;;

        "check")
            check_environment
            ;;

        *)
            echo "사용법: $0 {install|start|test|check}"
            echo ""
            echo "명령어:"
            echo "  install        - 환경 확인 및 의존성 설치"
            echo "  start [mode]   - 서비스 시작 (docker/api/ui/dev)"
            echo "  test           - 테스트 실행"
            echo "  check          - 환경 확인"
            echo ""
            echo "예시:"
            echo "  $0 install              # 최초 설치"
            echo "  $0 start dev            # 개발 모드 시작"
            echo "  $0 start docker         # Docker로 전체 스택 시작"
            exit 1
            ;;
    esac
}

main "$@"
