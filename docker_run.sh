#!/bin/bash
# Docker로 PPuRI-AI Ultimate 서버 실행

echo "🐳 Docker로 PPuRI-AI Ultimate 서버 실행"
echo "=================================================="

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Docker가 설치되지 않았습니다."
    echo "💡 https://docs.docker.com/get-docker/ 에서 Docker를 설치하세요."
    exit 1
fi

# 현재 디렉토리로 이동
cd "$(dirname "$0")"

echo "📁 현재 디렉토리: $(pwd)"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop ppuri-ai-ultimate 2>/dev/null || true
docker rm ppuri-ai-ultimate 2>/dev/null || true

# Docker 이미지 빌드
echo "🔨 Docker 이미지 빌드 중..."
docker build -f Dockerfile.simple -t ppuri-ai-ultimate:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker 이미지 빌드 실패"
    exit 1
fi

# Docker 컨테이너 실행
echo "🚀 Docker 컨테이너 실행 중..."
docker run -d \
    --name ppuri-ai-ultimate \
    --restart unless-stopped \
    -p 8002:8002 \
    -v "$(pwd)/logs:/app/logs" \
    ppuri-ai-ultimate:latest

if [ $? -eq 0 ]; then
    echo "✅ 서버가 성공적으로 시작되었습니다!"
    echo "🌐 브라우저에서 http://localhost:8002 접속하세요"
    echo ""
    echo "📋 유용한 명령어:"
    echo "  docker logs -f ppuri-ai-ultimate     # 로그 확인"
    echo "  docker stop ppuri-ai-ultimate       # 서버 중지"
    echo "  docker start ppuri-ai-ultimate      # 서버 시작"
    echo "  docker restart ppuri-ai-ultimate    # 서버 재시작"
    echo ""
    
    # 5초 후 브라우저에서 열기 (macOS)
    echo "⏰ 5초 후 브라우저를 열겠습니다..."
    sleep 5
    if command -v open &> /dev/null; then
        open http://localhost:8002
    fi
else
    echo "❌ 서버 시작 실패"
    docker logs ppuri-ai-ultimate
    exit 1
fi