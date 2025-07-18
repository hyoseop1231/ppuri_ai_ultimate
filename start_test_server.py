#!/usr/bin/env python3
"""
Start Test Server - PPuRI-AI Ultimate 테스트 서버 시작 스크립트

개선된 API 모듈들을 사용하여 실제 서버를 시작합니다.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_server.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """메인 실행 함수"""
    print("🚀 PPuRI-AI Ultimate 테스트 서버 시작")
    print("=" * 50)
    
    try:
        # 환경 변수 설정
        os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-development")
        os.environ.setdefault("DATABASE_URL", "postgresql://user:password@localhost/test_db")
        os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
        os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
        
        # 테스트 서버 import 및 실행
        from test_server import TestServer
        
        print("✅ 테스트 서버 모듈 로드 성공")
        
        # 서버 생성
        server = TestServer()
        print("✅ 테스트 서버 인스턴스 생성 성공")
        
        # 서버 실행
        print("🌐 서버 시작 중...")
        print("📍 서버 주소: http://localhost:8002")
        print("📖 API 문서: http://localhost:8002/docs")
        print("💡 종료하려면 Ctrl+C를 누르세요")
        print("=" * 50)
        
        await server.start()
        
    except KeyboardInterrupt:
        print("\n⏹️ 서버 종료 신호 수신")
        logger.info("서버 종료 중...")
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        logger.error(f"서버 시작 실패: {e}")
        sys.exit(1)
    finally:
        print("✅ 서버 종료 완료")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 서버 종료됨")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        sys.exit(1)