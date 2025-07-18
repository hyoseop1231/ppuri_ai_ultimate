#!/usr/bin/env python3
"""
Manual Test Execution - Server Testing
"""

import requests
import json
from datetime import datetime
import traceback

BASE_URL = "http://localhost:8002"

def run_tests():
    """Run all tests manually"""
    
    print("🚀 PPuRI-AI Ultimate Manual Test")
    print("=" * 50)
    print(f"서버 주소: {BASE_URL}")
    print(f"실행 시간: {datetime.now()}")
    print("=" * 50)
    
    # Test 1: Server Status
    print("\n🔍 1. 서버 상태 확인")
    print("-" * 30)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data.get('status', 'unknown')}")
            print(f"   버전: {data.get('version', 'unknown')}")
            print(f"   타임스탬프: {data.get('timestamp', 'unknown')}")
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 포트 8002에 서버가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False
    
    # Test 2: Health Check
    print("\n🏥 2. 헬스 체크")
    print("-" * 30)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 헬스 체크: {data.get('status', 'unknown')}")
            components = data.get('components', {})
            print(f"   API: {components.get('api', 'unknown')}")
            print(f"   미들웨어: {components.get('middleware', 'unknown')}")
        else:
            print(f"❌ 헬스 체크 실패: {response.status_code}")
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")
    
    # Test 3: Industrial AI Endpoints
    print("\n🏭 3. Industrial AI 엔드포인트 테스트")
    print("-" * 30)
    
    try:
        response = requests.get(f"{BASE_URL}/api/industrial-ai/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ 에이전트 목록 조회 성공")
            if 'data' in data:
                agents_data = data['data']
                print(f"   총 에이전트: {agents_data.get('total_agents', 0)}개")
                print(f"   활성 에이전트: {agents_data.get('active_agents', 0)}개")
                
                if 'agents' in agents_data and 'casting' in agents_data['agents']:
                    casting_info = agents_data['agents']['casting']
                    print(f"   주조 에이전트: {casting_info.get('name', 'unknown')} ({casting_info.get('status', 'unknown')})")
                    capabilities = casting_info.get('capabilities', [])
                    print(f"   기능: {', '.join(capabilities)}")
        elif response.status_code == 401:
            print("⚠️ 인증이 필요합니다. 로그인 후 다시 시도하세요.")
        else:
            print(f"❌ 에이전트 목록 조회 실패: {response.status_code}")
            print(f"   응답: {response.text}")
    except Exception as e:
        print(f"❌ Industrial AI 테스트 오류: {e}")
    
    # Test 4: Login and Authentication
    print("\n🔐 4. 로그인 및 인증 테스트")
    print("-" * 30)
    
    login_data = {
        "username": "admin_001",
        "password": "admin_pass_001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data, timeout=5)
        if response.status_code == 200:
            auth_data = response.json()
            if 'data' in auth_data and 'access_token' in auth_data['data']:
                access_token = auth_data['data']['access_token']
                print("✅ 로그인 성공")
                
                # Test authenticated endpoints
                headers = {"Authorization": f"Bearer {access_token}"}
                
                # Test 5: Casting Analysis
                print("\n🔧 5. 주조 분석 테스트")
                print("-" * 30)
                
                analysis_data = {
                    "problem_type": "defect_analysis",
                    "description": "주조 제품에서 기공 결함이 발견되었습니다.",
                    "process_data": {
                        "온도": 780,
                        "압력": 300,
                        "주입속도": 1.5
                    }
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/industrial-ai/analyze/casting",
                    json=analysis_data,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ 주조 분석 성공!")
                    if 'data' in result:
                        data = result['data']
                        print(f"   상태: {data.get('status', 'unknown')}")
                        print(f"   도메인: {data.get('domain', 'unknown')}")
                        
                        if 'analysis' in data:
                            analysis = data['analysis']
                            print(f"   분석 결과:")
                            print(f"      - 결함 수: {len(analysis.get('detected_defects', []))}")
                            print(f"      - 근본 원인: {len(analysis.get('root_causes', []))}")
                            print(f"      - 신뢰도: {analysis.get('confidence', 0):.2%}")
                            
                            # 상세 결과 출력
                            defects = analysis.get('detected_defects', [])
                            if defects:
                                print(f"\n   🔍 발견된 결함:")
                                for defect in defects[:3]:  # 최대 3개만 표시
                                    print(f"      - {defect.get('type', 'unknown')}: {defect.get('location', 'unknown')} ({defect.get('size', 'unknown')})")
                        
                        if 'solution' in data:
                            solution = data['solution']
                            print(f"   솔루션:")
                            print(f"      - 즉시 조치: {len(solution.get('immediate_actions', []))}개")
                            print(f"      - 예상 개선율: {solution.get('estimated_improvement', 0)}%")
                            
                            actions = solution.get('immediate_actions', [])
                            if actions:
                                print(f"\n   ⚡ 즉시 조치사항:")
                                for action in actions[:3]:  # 최대 3개만 표시
                                    print(f"      - {action.get('action', 'unknown')} (긴급도: {action.get('urgency', 'unknown')})")
                else:
                    print(f"❌ 주조 분석 실패: {response.status_code}")
                    print(f"   응답: {response.text}")
                
                # Test 6: Workflow Execution
                print("\n🔄 6. 워크플로우 테스트")
                print("-" * 30)
                
                workflow_data = {
                    "problem_type": "complex_defect",
                    "description": "주조 공정에서 복합적인 품질 문제가 발생했습니다.",
                    "domain": "casting",
                    "priority": "high",
                    "process_data": {
                        "온도": 800,
                        "압력": 150,
                        "주입속도": 2.5
                    }
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/industrial-ai/analyze",
                    json=workflow_data,
                    headers=headers,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ 워크플로우 실행 성공!")
                    if 'data' in result:
                        data = result['data']
                        print(f"   워크플로우 ID: {data.get('workflow_id', 'unknown')}")
                        print(f"   상태: {data.get('status', 'unknown')}")
                        print(f"   실행 시간: {data.get('execution_time', 0):.3f}초")
                else:
                    print(f"❌ 워크플로우 실행 실패: {response.status_code}")
                    print(f"   응답: {response.text}")
                
                # Test 7: Performance Metrics
                print("\n📊 7. 성능 메트릭 조회")
                print("-" * 30)
                
                response = requests.get(f"{BASE_URL}/api/industrial-ai/performance", headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print("✅ 성능 메트릭 조회 성공")
                    if 'data' in data and 'agents' in data['data']:
                        agents = data['data']['agents']
                        print(f"   에이전트 성능:")
                        print(f"      - 생성 시간: {agents.get('creation_time', 'unknown')}")
                        print(f"      - 메모리 사용량: {agents.get('memory_usage_per_agent', 'unknown')}")
                        print(f"      - 총 요청 수: {agents.get('total_requests', 0)}")
                        print(f"      - 평균 응답 시간: {agents.get('average_response_time', 0):.3f}초")
                else:
                    print(f"❌ 성능 메트릭 조회 실패: {response.status_code}")
                    print(f"   응답: {response.text}")
            else:
                print(f"❌ 로그인 응답 형식 오류: {response.text}")
        else:
            print(f"❌ 로그인 실패: {response.status_code}")
            print(f"   응답: {response.text}")
    except Exception as e:
        print(f"❌ 로그인 테스트 오류: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 테스트 완료!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    run_tests()