#!/bin/bash
# PPuRI-AI Ultimate 서버 원클릭 실행 파일
# 더블클릭으로 실행 가능

echo "🚀 PPuRI-AI Ultimate 서버 시작 중..."
echo "==============================================="

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

echo "📁 현재 디렉토리: $(pwd)"

# Python 가상환경 확인
if [ -d "venv" ]; then
    echo "🐍 Python 가상환경 활성화..."
    source venv/bin/activate
fi

# 의존성 확인 및 설치
echo "📦 의존성 확인 중..."
python3 -m pip install fastapi uvicorn --quiet

# 서버 실행
echo "🚀 서버 실행 시작..."
echo "🌐 브라우저에서 http://localhost:8002 접속하세요!"
echo "💡 종료하려면 Ctrl+C를 누르세요"
echo "==============================================="

python3 simple_stable_server.py

echo "✅ 서버가 종료되었습니다."
read -p "아무 키나 눌러 창을 닫으세요..."