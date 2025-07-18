#!/usr/bin/env python3
"""
Direct Server - 직접 실행 가능한 서버
"""

import os
import sys
import subprocess
from pathlib import Path

def run_server():
    """서버 직접 실행"""
    # 프로젝트 디렉토리
    project_dir = Path(__file__).parent
    
    print("🚀 PPuRI-AI Ultimate 서버 실행 중...")
    print(f"📁 프로젝트 디렉토리: {project_dir}")
    print("=" * 50)
    
    # 서버 실행
    try:
        # simple_stable_server.py 실행
        server_file = project_dir / "simple_stable_server.py"
        if server_file.exists():
            print("✅ simple_stable_server.py 파일 확인됨")
            
            # Python으로 서버 실행
            import simple_stable_server
            print("📦 서버 모듈 로드 성공")
            
            # 서버 실행
            simple_stable_server.run_server()
            
        else:
            print("❌ simple_stable_server.py 파일이 없습니다.")
            
    except Exception as e:
        print(f"❌ 서버 실행 실패: {e}")
        print("\n🔧 대안:")
        print("1. 터미널에서 직접 실행: python3 simple_stable_server.py")
        print("2. 의존성 설치: pip install fastapi uvicorn")
        print("3. 포트 8002 사용 중인지 확인")

if __name__ == "__main__":
    run_server()