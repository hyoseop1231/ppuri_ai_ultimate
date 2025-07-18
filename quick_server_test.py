#!/usr/bin/env python3
"""
Quick Server Test - 빠른 서버 테스트
"""

import requests
import json

def test_server():
    """서버 빠른 테스트"""
    BASE_URL = "http://localhost:8002"
    
    print("🚀 서버 빠른 테스트")
    print("=" * 30)
    
    # 1. 서버 상태 확인
    print("1. 서버 상태 확인")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   상태코드: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   서버 상태: {data.get('status', 'unknown')}")
            print(f"   버전: {data.get('version', 'unknown')}")
        else:
            print(f"   응답: {response.text}")
    except Exception as e:
        print(f"   오류: {e}")
    
    # 2. 헬스 체크
    print("\n2. 헬스 체크")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"   상태코드: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   헬스 상태: {data.get('status', 'unknown')}")
        else:
            print(f"   응답: {response.text}")
    except Exception as e:
        print(f"   오류: {e}")
    
    # 3. 테스트 엔드포인트
    print("\n3. 테스트 엔드포인트")
    try:
        response = requests.get(f"{BASE_URL}/api/test")
        print(f"   상태코드: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   메시지: {data.get('message', 'unknown')}")
            modules = data.get('modules', {})
            for module, status in modules.items():
                print(f"   {module}: {status}")
        else:
            print(f"   응답: {response.text}")
    except Exception as e:
        print(f"   오류: {e}")
    
    # 4. 로그인 테스트
    print("\n4. 로그인 테스트")
    try:
        login_data = {
            "username": "admin_001",
            "password": "admin_pass_001"
        }
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        print(f"   상태코드: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   로그인 성공: {data.get('status', 'unknown')}")
            access_token = data.get('data', {}).get('access_token')
            if access_token:
                print(f"   토큰 획득: {access_token[:20]}...")
                
                # 5. 인증된 요청 테스트
                print("\n5. 인증된 요청 테스트")
                headers = {"Authorization": f"Bearer {access_token}"}
                
                # 에이전트 목록 조회
                response = requests.get(f"{BASE_URL}/api/industrial-ai/agents", headers=headers)
                print(f"   에이전트 목록: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    agents_data = data.get('data', {})
                    print(f"   총 에이전트: {agents_data.get('total_agents', 0)}개")
                    print(f"   활성 에이전트: {agents_data.get('active_agents', 0)}개")
                else:
                    print(f"   오류: {response.text}")
                
                # 주조 분석 테스트
                print("\n6. 주조 분석 테스트")
                analysis_data = {
                    "problem_type": "defect_analysis",
                    "description": "기공 결함 발견",
                    "process_data": {
                        "온도": 780,
                        "압력": 300,
                        "주입속도": 1.5
                    }
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/industrial-ai/analyze/casting",
                    json=analysis_data,
                    headers=headers
                )
                print(f"   분석 요청: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    result_data = data.get('data', {})
                    print(f"   분석 상태: {result_data.get('status', 'unknown')}")
                    print(f"   도메인: {result_data.get('domain', 'unknown')}")
                    
                    analysis = result_data.get('analysis', {})
                    print(f"   결함 수: {len(analysis.get('detected_defects', []))}")
                    print(f"   신뢰도: {analysis.get('confidence', 0):.2%}")
                    
                    solution = result_data.get('solution', {})
                    print(f"   솔루션: {len(solution.get('immediate_actions', []))}개 즉시 조치")
                    print(f"   개선율: {solution.get('estimated_improvement', 0)}%")
                    
                else:
                    print(f"   오류: {response.text}")
                    
        else:
            print(f"   로그인 실패: {response.text}")
            
    except Exception as e:
        print(f"   오류: {e}")
    
    print("\n" + "=" * 30)
    print("✅ 테스트 완료!")

if __name__ == "__main__":
    test_server()