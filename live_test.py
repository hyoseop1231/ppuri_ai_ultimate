#!/usr/bin/env python3
"""
Live Test - 실행 중인 서버 테스트
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8002"

def test_server_status():
    """서버 상태 확인"""
    print("🔍 서버 상태 확인")
    print("-" * 30)
    
    try:
        # 루트 엔드포인트
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data['status']}")
            print(f"   버전: {data['version']}")
            print(f"   타임스탬프: {data['timestamp']}")
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
        
        # 헬스 체크
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 헬스 체크: {data['status']}")
            print(f"   API: {data['components']['api']}")
            print(f"   미들웨어: {data['components']['middleware']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다.")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False
    
    return True

def test_industrial_ai_public():
    """인증 없이 접근 가능한 엔드포인트 테스트"""
    print("\n🏭 Industrial AI 엔드포인트 테스트")
    print("-" * 30)
    
    # 에이전트 목록 조회 (인증 필요할 수 있음)
    try:
        response = requests.get(f"{BASE_URL}/api/industrial-ai/agents")
        if response.status_code == 200:
            data = response.json()
            print("✅ 에이전트 목록 조회 성공")
            print(f"   총 에이전트: {data['data']['total_agents']}개")
            print(f"   활성 에이전트: {data['data']['active_agents']}개")
            
            # 주조 에이전트 상세 정보
            casting_info = data['data']['agents']['casting']
            print(f"   주조 에이전트: {casting_info['name']} ({casting_info['status']})")
            print(f"   기능: {', '.join(casting_info['capabilities'])}")
            
        elif response.status_code == 401:
            print("⚠️ 인증이 필요합니다. 로그인 후 다시 시도하세요.")
        else:
            print(f"❌ 에이전트 목록 조회 실패: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 오류: {e}")

def login_and_test():
    """로그인 후 테스트"""
    print("\n🔐 로그인 및 인증 테스트")
    print("-" * 30)
    
    # 로그인 시도
    login_data = {
        "username": "admin_001",
        "password": "admin_pass_001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        if response.status_code == 200:
            auth_data = response.json()
            access_token = auth_data['data']['access_token']
            print("✅ 로그인 성공")
            
            # 인증된 요청 헤더
            headers = {"Authorization": f"Bearer {access_token}"}
            
            # 주조 분석 테스트
            print("\n🔧 주조 분석 테스트")
            print("-" * 30)
            
            analysis_data = {
                "problem_type": "defect_analysis",
                "description": "주조 제품에서 기공 결함이 발견되었습니다.",
                "process_data": {
                    "온도": 780,  # 정상 범위 650-750 초과
                    "압력": 300,
                    "주입속도": 1.5
                }
            }
            
            response = requests.post(
                f"{BASE_URL}/api/industrial-ai/analyze/casting",
                json=analysis_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 주조 분석 성공!")
                print(f"   상태: {result['data']['status']}")
                print(f"   도메인: {result['data']['domain']}")
                print(f"   분석 결과:")
                
                analysis = result['data']['analysis']
                print(f"      - 결함 수: {len(analysis['detected_defects'])}")
                print(f"      - 근본 원인: {len(analysis['root_causes'])}")
                print(f"      - 신뢰도: {analysis['confidence']:.2%}")
                
                solution = result['data']['solution']
                print(f"   솔루션:")
                print(f"      - 즉시 조치: {len(solution['immediate_actions'])}개")
                print(f"      - 예상 개선율: {solution['estimated_improvement']}%")
                
                # 상세 결과 출력
                if analysis['detected_defects']:
                    print(f"\n   🔍 발견된 결함:")
                    for defect in analysis['detected_defects']:
                        print(f"      - {defect['type']}: {defect['location']} ({defect['size']})")
                
                if solution['immediate_actions']:
                    print(f"\n   ⚡ 즉시 조치사항:")
                    for action in solution['immediate_actions'][:3]:  # 상위 3개만
                        print(f"      - {action['action']} (긴급도: {action['urgency']})")
                
            else:
                print(f"❌ 주조 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
            
            # 워크플로우 테스트
            print("\n🔄 워크플로우 테스트")
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
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 워크플로우 실행 성공!")
                print(f"   워크플로우 ID: {result['data']['workflow_id']}")
                print(f"   상태: {result['data']['status']}")
                print(f"   실행 시간: {result['data']['execution_time']:.3f}초")
                
            else:
                print(f"❌ 워크플로우 실행 실패: {response.status_code}")
            
            # 성능 메트릭 조회
            print("\n📊 성능 메트릭 조회")
            print("-" * 30)
            
            response = requests.get(f"{BASE_URL}/api/industrial-ai/performance", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("✅ 성능 메트릭 조회 성공")
                
                agents = data['data']['agents']
                print(f"   에이전트 성능:")
                print(f"      - 생성 시간: {agents['creation_time']}")
                print(f"      - 메모리 사용량: {agents['memory_usage_per_agent']}")
                print(f"      - 총 요청 수: {agents['total_requests']}")
                print(f"      - 평균 응답 시간: {agents['average_response_time']:.3f}초")
                
            else:
                print(f"❌ 성능 메트릭 조회 실패: {response.status_code}")
                
        else:
            print(f"❌ 로그인 실패: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 오류: {e}")

def main():
    """메인 테스트 실행"""
    print("🚀 PPuRI-AI Ultimate 라이브 테스트")
    print("=" * 50)
    print(f"서버 주소: {BASE_URL}")
    print(f"실행 시간: {datetime.now()}")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    if not test_server_status():
        return
    
    # 2. 공개 엔드포인트 테스트
    test_industrial_ai_public()
    
    # 3. 로그인 후 전체 테스트
    login_and_test()
    
    print("\n" + "=" * 50)
    print("🎉 라이브 테스트 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()