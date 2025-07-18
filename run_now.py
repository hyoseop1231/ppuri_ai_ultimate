#!/usr/bin/env python3
"""
Emergency Server Runner - 즉시 서버 실행
"""

import os
import sys
from pathlib import Path

# 프로젝트 경로 설정
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

print("🚀 PPuRI-AI Ultimate 서버 즉시 실행")
print(f"📁 프로젝트 디렉토리: {project_dir}")
print("=" * 50)

try:
    # 서버 모듈 임포트 및 실행
    import simple_stable_server
    print("✅ 서버 모듈 로드 성공")
    
    # 서버 실행
    simple_stable_server.run_server()
    
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("\n🔧 의존성 설치 필요:")
    print("pip install fastapi uvicorn")
    
except Exception as e:
    print(f"❌ 서버 실행 실패: {e}")
    print("\n🔧 대안:")
    print("1. 터미널에서 직접 실행: python3 simple_stable_server.py")
    print("2. 포트 8002 사용 중인지 확인")