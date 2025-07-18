"""
Quick Test - PPuRI-AI Ultimate 빠른 테스트

구현된 API 모듈들의 기본 기능을 빠르게 테스트
"""

import asyncio
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_imports():
    """기본 import 테스트"""
    print("🧪 기본 import 테스트 시작")
    
    try:
        # 상수 모듈 테스트
        from api.constants import APIErrors, SecurityConstants
        print("✅ Constants 모듈 import 성공")
        
        # 모델 테스트
        from api.models.responses import ErrorResponse, SuccessResponse
        from api.models.requests import ChatMessageRequest
        print("✅ Models 모듈 import 성공")
        
        # 미들웨어 테스트
        from api.middleware.auth import JWTManager, SessionManager
        from api.middleware.security import SecurityHeadersMiddleware
        print("✅ Middleware 모듈 import 성공")
        
        # 라우터 테스트
        from api.routes.auth import router as auth_router
        from api.routes.sessions import router as sessions_router
        from api.routes.chat import router as chat_router
        print("✅ Routes 모듈 import 성공")
        
        # 데이터베이스 테스트
        from api.database.connection_pool import connection_pool_manager
        print("✅ Database 모듈 import 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        return False


async def test_jwt_manager():
    """JWT 관리자 테스트"""
    print("\n🔐 JWT 관리자 테스트 시작")
    
    try:
        from api.middleware.auth import JWTManager
        
        # JWT 관리자 생성
        jwt_manager = JWTManager()
        
        # 토큰 생성 테스트
        user_id = "test_user"
        access_token = jwt_manager.create_access_token(user_id)
        refresh_token = jwt_manager.create_refresh_token(user_id)
        
        print(f"✅ Access Token 생성 성공: {access_token[:20]}...")
        print(f"✅ Refresh Token 생성 성공: {refresh_token[:20]}...")
        
        # 토큰 검증 테스트
        payload = jwt_manager.verify_token(access_token)
        print(f"✅ Token 검증 성공: {payload['user_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ JWT 관리자 테스트 실패: {e}")
        return False


async def test_response_models():
    """응답 모델 테스트"""
    print("\n📝 응답 모델 테스트 시작")
    
    try:
        from api.models.responses import SuccessResponse, ErrorResponse
        from api.models.exceptions import ValidationException
        
        # 성공 응답 테스트
        success_response = SuccessResponse(
            data={"message": "테스트 성공"},
            request_id="test_123"
        )
        print(f"✅ Success Response 생성 성공: {success_response.status}")
        
        # 에러 응답 테스트
        error_response = ErrorResponse.from_exception(
            ValidationException("테스트 에러"),
            request_id="test_456"
        )
        print(f"✅ Error Response 생성 성공: {error_response.error_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ 응답 모델 테스트 실패: {e}")
        return False


async def test_database_connection():
    """데이터베이스 연결 테스트"""
    print("\n🗄️ 데이터베이스 연결 테스트 시작")
    
    try:
        from api.database.connection_pool import connection_pool_manager
        
        # 연결 풀 초기화
        await connection_pool_manager.initialize()
        print("✅ Connection Pool 초기화 성공")
        
        # 헬스 체크
        health_status = await connection_pool_manager.health_check()
        print(f"✅ Health Check 성공: {health_status}")
        
        # 정리
        await connection_pool_manager.close_all()
        print("✅ Connection Pool 정리 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 연결 테스트 실패: {e}")
        return False


async def test_fastapi_app():
    """FastAPI 앱 테스트"""
    print("\n🌐 FastAPI 앱 테스트 시작")
    
    try:
        from fastapi import FastAPI
        from api.routes.auth import router as auth_router
        from api.routes.sessions import router as sessions_router
        from api.routes.chat import router as chat_router
        
        # FastAPI 앱 생성
        app = FastAPI(title="Test App")
        
        # 라우터 등록
        app.include_router(auth_router)
        app.include_router(sessions_router)
        app.include_router(chat_router)
        
        # 기본 엔드포인트
        @app.get("/")
        async def root():
            return {"message": "Test App Running"}
        
        # 앱 정보 확인
        print(f"✅ FastAPI 앱 생성 성공: {app.title}")
        print(f"✅ 라우터 등록 완료: {len(app.routes)}개 경로")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI 앱 테스트 실패: {e}")
        return False


async def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 PPuRI-AI Ultimate 빠른 테스트 시작")
    print("=" * 50)
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(await test_basic_imports())
    test_results.append(await test_jwt_manager())
    test_results.append(await test_response_models())
    test_results.append(await test_database_connection())
    test_results.append(await test_fastapi_app())
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ 성공: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        print("✨ PPuRI-AI Ultimate 개선 버전이 정상적으로 작동합니다!")
    else:
        print("⚠️ 일부 테스트 실패")
        print("💡 실패한 테스트를 확인하고 문제를 해결하세요.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())