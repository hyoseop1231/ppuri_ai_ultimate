#!/usr/bin/env python3
"""
API Client Test - API 엔드포인트 테스트 클라이언트
"""

import asyncio
import aiohttp
import json
from datetime import datetime

# 서버 URL
BASE_URL = "http://localhost:8002"

async def test_server_health():
    """서버 헬스 체크"""
    print("\n=== 서버 헬스 체크 ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            # 루트 엔드포인트
            async with session.get(f"{BASE_URL}/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 서버 상태: {data['status']}")
                    print(f"   버전: {data['version']}")
                else:
                    print(f"❌ 서버 응답 오류: {response.status}")
            
            # 헬스 체크
            async with session.get(f"{BASE_URL}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 헬스 체크: {data['status']}")
                else:
                    print(f"❌ 헬스 체크 오류: {response.status}")
                    
        except aiohttp.ClientConnectorError:
            print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        except Exception as e:
            print(f"❌ 오류: {e}")

async def test_industrial_ai_endpoints():
    """산업 AI 엔드포인트 테스트"""
    print("\n=== 산업 AI 엔드포인트 테스트 ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. 에이전트 목록 조회
            print("\n1. 사용 가능한 에이전트 조회")
            async with session.get(f"{BASE_URL}/api/industrial-ai/agents") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 에이전트 조회 성공")
                    print(f"   - 전체 에이전트: {data['data']['total_agents']}개")
                    print(f"   - 활성 에이전트: {data['data']['active_agents']}개")
                elif response.status == 401:
                    print("⚠️  인증이 필요합니다. 로그인 후 다시 시도하세요.")
                else:
                    print(f"❌ 에이전트 조회 실패: {response.status}")
            
            # 2. 워크플로우 상태 조회
            print("\n2. 워크플로우 상태 조회")
            async with session.get(f"{BASE_URL}/api/industrial-ai/workflows") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 워크플로우 조회 성공")
                    print(f"   - 워크플로우 이름: {data['data']['workflow_name']}")
                    print(f"   - 상태: {data['data']['status']}")
                else:
                    print(f"❌ 워크플로우 조회 실패: {response.status}")
            
            # 3. 성능 메트릭 조회
            print("\n3. 성능 메트릭 조회")
            async with session.get(f"{BASE_URL}/api/industrial-ai/performance") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 성능 메트릭 조회 성공")
                    print(f"   - 에이전트 생성 시간: {data['data']['agents']['creation_time']}")
                    print(f"   - 메모리 사용량: {data['data']['agents']['memory_usage_per_agent']}")
                else:
                    print(f"❌ 성능 메트릭 조회 실패: {response.status}")
                    
        except Exception as e:
            print(f"❌ 오류: {e}")

async def test_casting_analysis():
    """주조 분석 테스트 (인증 필요)"""
    print("\n=== 주조 분석 테스트 ===")
    
    async with aiohttp.ClientSession() as session:
        try:
            # 먼저 로그인 시도
            print("\n로그인 시도...")
            login_data = {
                "username": "admin_001",
                "password": "admin_pass_001"
            }
            
            async with session.post(
                f"{BASE_URL}/api/auth/login",
                json=login_data
            ) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    access_token = auth_data['data']['access_token']
                    print("✅ 로그인 성공")
                    
                    # 헤더에 토큰 추가
                    headers = {
                        "Authorization": f"Bearer {access_token}"
                    }
                    
                    # 주조 분석 요청
                    print("\n주조 문제 분석 요청...")
                    analysis_data = {
                        "problem_type": "defect_analysis",
                        "description": "주조 제품에서 기공 결함이 발견되었습니다.",
                        "process_data": {
                            "온도": 780,
                            "압력": 300,
                            "주입속도": 1.5
                        }
                    }
                    
                    async with session.post(
                        f"{BASE_URL}/api/industrial-ai/analyze/casting",
                        json=analysis_data,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            print("✅ 주조 분석 성공!")
                            print(f"   - 상태: {result['data']['status']}")
                            print(f"   - 분석된 결함: {len(result['data']['analysis']['detected_defects'])}개")
                            print(f"   - 신뢰도: {result['data']['analysis']['confidence']:.2%}")
                            print(f"   - 솔루션 제안: {len(result['data']['solution']['immediate_actions'])}개")
                        else:
                            print(f"❌ 주조 분석 실패: {response.status}")
                            error_data = await response.text()
                            print(f"   오류: {error_data}")
                else:
                    print(f"❌ 로그인 실패: {response.status}")
                    
        except Exception as e:
            print(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """메인 테스트 실행"""
    print("🚀 PPuRI-AI Ultimate API 테스트")
    print("=" * 50)
    print(f"서버 주소: {BASE_URL}")
    print(f"실행 시간: {datetime.now()}")
    print("=" * 50)
    
    # 테스트 실행
    await test_server_health()
    await test_industrial_ai_endpoints()
    await test_casting_analysis()
    
    print("\n✅ API 테스트 완료!")

if __name__ == "__main__":
    # aiohttp 설치 여부 확인
    try:
        import aiohttp
        asyncio.run(main())
    except ImportError:
        print("❌ aiohttp가 설치되지 않았습니다.")
        print("설치하려면: pip install aiohttp")
        print("\n대신 requests를 사용한 동기 테스트를 실행합니다...")
        
        # requests를 사용한 대체 테스트
        import requests
        
        print("\n=== 간단한 서버 테스트 ===")
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print(f"✅ 서버 응답 성공: {response.json()}")
            else:
                print(f"❌ 서버 응답 실패: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ 서버에 연결할 수 없습니다.")