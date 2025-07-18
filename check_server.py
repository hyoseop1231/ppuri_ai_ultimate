#!/usr/bin/env python3
"""
Server Check & Start Script
"""

import sys
import os
from pathlib import Path

# Add project to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

print("🔍 PPuRI-AI Ultimate 서버 체크")
print(f"📁 프로젝트 디렉토리: {project_dir}")
print("=" * 40)

# Check if server file exists
server_file = project_dir / "simple_stable_server.py"
print(f"✅ 서버 파일 존재: {server_file.exists()}")

# Check dependencies
try:
    import fastapi
    print("✅ FastAPI 설치됨")
except ImportError:
    print("❌ FastAPI 설치 필요: pip install fastapi")

try:
    import uvicorn
    print("✅ Uvicorn 설치됨")
except ImportError:
    print("❌ Uvicorn 설치 필요: pip install uvicorn")

print("\n🚀 서버 시작 명령:")
print("python3 simple_stable_server.py")
print("브라우저에서 http://localhost:8002 접속")

# If all is good, try to start server
if server_file.exists():
    try:
        print("\n📦 서버 모듈 로드 시도...")
        import simple_stable_server
        print("✅ 서버 모듈 로드 성공!")
        
        print("\n🚀 서버 실행 중...")
        simple_stable_server.run_server()
        
    except Exception as e:
        print(f"❌ 서버 실행 실패: {e}")
        print("\n수동 실행 필요:")
        print("터미널에서: python3 simple_stable_server.py")