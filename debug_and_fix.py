#!/usr/bin/env python3
"""
Debug and Fix - 실행 문제 진단 및 해결
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_environment():
    """Python 환경 체크"""
    print("🐍 Python 환경 체크")
    print("=" * 40)
    print(f"Python 버전: {sys.version}")
    print(f"Python 경로: {sys.executable}")
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"Python path: {sys.path}")

def check_dependencies():
    """의존성 체크"""
    print("\n📦 의존성 체크")
    print("=" * 40)
    
    required_packages = ["fastapi", "uvicorn", "pydantic"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"❌ {package}: 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_core_modules():
    """코어 모듈 체크"""
    print("\n🔧 코어 모듈 체크")
    print("=" * 40)
    
    # 프로젝트 경로 추가
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    modules_to_check = [
        "core.agents.base_agent",
        "core.agents.casting_agent", 
        "core.workflows.base_workflow",
        "core.workflows.analysis_workflow"
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}: 로드 성공")
        except ImportError as e:
            print(f"❌ {module}: 로드 실패 - {e}")

def check_port_availability():
    """포트 사용 가능성 체크"""
    print("\n🌐 포트 8002 체크")
    print("=" * 40)
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8002))
        sock.close()
        
        if result == 0:
            print("❌ 포트 8002가 이미 사용 중입니다")
            print("해결 방법:")
            print("  lsof -i :8002")
            print("  kill -9 [PID]")
            return False
        else:
            print("✅ 포트 8002 사용 가능")
            return True
    except Exception as e:
        print(f"포트 체크 오류: {e}")
        return True

def try_simple_server():
    """간단한 서버로 테스트"""
    print("\n🧪 간단한 서버 테스트")
    print("=" * 40)
    
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        import uvicorn
        
        app = FastAPI(title="Test Server")
        
        @app.get("/", response_class=HTMLResponse)
        def root():
            return """
            <html><body>
                <h1>🎉 서버 작동 테스트 성공!</h1>
                <p>PPuRI-AI Ultimate 기본 서버가 정상 작동합니다.</p>
                <p>시간: """ + str(__import__('datetime').datetime.now()) + """</p>
            </body></html>
            """
        
        @app.get("/test")
        def test():
            return {"status": "OK", "message": "테스트 성공"}
        
        print("✅ FastAPI 앱 생성 성공")
        print("🚀 테스트 서버 시작 중...")
        print("📍 http://localhost:8002 에서 확인하세요")
        
        uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
        
    except ImportError as e:
        print(f"❌ FastAPI 로드 실패: {e}")
        print("해결 방법: pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")

def install_missing_dependencies():
    """누락된 의존성 자동 설치"""
    print("\n🔧 의존성 자동 설치")
    print("=" * 40)
    
    packages = ["fastapi", "uvicorn[standard]", "pydantic"]
    
    for package in packages:
        try:
            print(f"📦 {package} 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 설치 실패: {e}")

def main():
    """메인 진단 실행"""
    print("🔍 PPuRI-AI Ultimate 서버 문제 진단")
    print("=" * 50)
    
    # 1. Python 환경 체크
    check_python_environment()
    
    # 2. 의존성 체크
    deps_ok = check_dependencies()
    
    # 3. 의존성 설치 (필요시)
    if not deps_ok:
        install_missing_dependencies()
    
    # 4. 코어 모듈 체크
    check_core_modules()
    
    # 5. 포트 체크
    port_ok = check_port_availability()
    
    # 6. 간단한 서버 테스트
    print("\n" + "=" * 50)
    print("🎯 진단 완료! 테스트 서버를 시작합니다.")
    print("브라우저에서 http://localhost:8002 를 확인하세요.")
    print("=" * 50)
    
    if port_ok:
        try_simple_server()
    else:
        print("포트 문제를 먼저 해결하세요:")
        print("lsof -i :8002")
        print("kill -9 [PID]")

if __name__ == "__main__":
    main()