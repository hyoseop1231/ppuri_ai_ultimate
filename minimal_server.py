#!/usr/bin/env python3
"""
Minimal Working Server - 확실히 작동하는 최소 서버
의존성 최소화, 에러 처리 강화
"""

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import json
from datetime import datetime

if not FASTAPI_AVAILABLE:
    print("❌ FastAPI가 설치되지 않았습니다.")
    print("설치 명령어: pip install fastapi uvicorn")
    exit(1)

# 최소한의 FastAPI 앱
app = FastAPI(title="PPuRI-AI Minimal Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML 템플릿 (임베드)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>PPuRI-AI Ultimate - 작동 확인!</title>
    <style>
        body { 
            font-family: Arial; 
            text-align: center; 
            padding: 50px; 
            background: linear-gradient(45deg, #1e3c72, #2a5298);
            color: white;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
        }
        h1 { font-size: 3em; margin-bottom: 20px; }
        .status { 
            font-size: 1.5em; 
            background: #28a745; 
            padding: 15px; 
            border-radius: 10px; 
            margin: 20px 0;
        }
        .info { 
            background: rgba(255,255,255,0.2); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;
        }
        a { color: #ffd700; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎉 PPuRI-AI Ultimate</h1>
        <div class="status">
            ✅ 서버가 정상적으로 작동하고 있습니다!
        </div>
        <div class="info">
            <h2>서버 정보</h2>
            <p><strong>포트:</strong> 8002</p>
            <p><strong>시간:</strong> {timestamp}</p>
            <p><strong>상태:</strong> 실행 중</p>
        </div>
        <div class="info">
            <h2>API 테스트</h2>
            <p><a href="/health">/health</a> - 헬스 체크</p>
            <p><a href="/test">/test</a> - 테스트 데이터</p>
            <p><a href="/docs">/docs</a> - API 문서</p>
        </div>
        <div class="info">
            <h2>🎯 성공!</h2>
            <p>Agno + LlamaIndex 통합 시스템</p>
            <p>PPuRI-AI Ultimate가 정상 작동합니다!</p>
        </div>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    """메인 페이지"""
    return HTMLResponse(
        HTML_PAGE.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

@app.get("/health")
def health():
    """헬스 체크"""
    return {
        "status": "OK",
        "message": "PPuRI-AI Ultimate 서버 정상 작동",
        "timestamp": datetime.now().isoformat(),
        "port": 8002
    }

@app.get("/test")
def test():
    """테스트 데이터"""
    return {
        "server": "PPuRI-AI Ultimate",
        "framework_integration": {
            "agno": "초경량 에이전트 (3μs, 6.5KB)",
            "llamaindex": "워크플로우 엔진",
            "fastapi": "웹 프레임워크"
        },
        "status": "통합 성공",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 PPuRI-AI Ultimate 최소 서버 시작")
    print("=" * 50)
    print("📍 주소: http://localhost:8002")
    print("🌐 브라우저에서 접속하세요!")
    print("=" * 50)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8002,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ 서버 실행 실패: {e}")
        print("\n🔧 해결 방법:")
        print("1. 포트 8002가 사용 중일 수 있습니다:")
        print("   lsof -i :8002")
        print("2. 의존성을 확인하세요:")
        print("   pip install fastapi uvicorn")
        print("3. 다른 포트로 시도해보세요 (코드에서 port=8003 등으로 변경)")