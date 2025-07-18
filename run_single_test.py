#!/usr/bin/env python3
"""
Single Test Runner - 단일 테스트 실행
shell 환경 문제를 우회하여 단일 테스트 실행
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def run_single_test():
    """단일 테스트 실행"""
    print("🚀 단일 테스트 실행")
    print("=" * 30)
    
    # 1. 기본 import 테스트
    print("1. 기본 import 테스트")
    try:
        from core.agents.casting_agent import CastingExpertAgent
        print("   ✅ CastingExpertAgent import 성공")
    except Exception as e:
        print(f"   ❌ CastingExpertAgent import 실패: {e}")
        return
    
    # 2. 에이전트 생성 테스트
    print("\n2. 에이전트 생성 테스트")
    try:
        agent = CastingExpertAgent()
        print("   ✅ 에이전트 생성 성공")
        print(f"   - 도메인: {agent.domain}")
        print(f"   - 모델 프로바이더: {agent.model_provider}")
    except Exception as e:
        print(f"   ❌ 에이전트 생성 실패: {e}")
        return
    
    # 3. 에이전트 분석 테스트
    print("\n3. 에이전트 분석 테스트")
    try:
        test_data = {
            "problem_type": "defect_analysis", 
            "description": "기공 결함 발견",
            "process_data": {"온도": 780, "압력": 300}
        }
        
        result = await agent.analyze(test_data)
        print("   ✅ 분석 성공")
        print(f"   - 결함 수: {len(result['detected_defects'])}")
        print(f"   - 신뢰도: {result['confidence']:.2%}")
        
    except Exception as e:
        print(f"   ❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 솔루션 생성 테스트
    print("\n4. 솔루션 생성 테스트")
    try:
        solution = await agent.generate_solution(result)
        print("   ✅ 솔루션 생성 성공")
        print(f"   - 즉시 조치: {len(solution['immediate_actions'])}")
        print(f"   - 예상 개선율: {solution['estimated_improvement']}%")
        
    except Exception as e:
        print(f"   ❌ 솔루션 생성 실패: {e}")
        return
    
    # 5. 통합 요청 처리 테스트
    print("\n5. 통합 요청 처리 테스트")
    try:
        final_result = await agent.process_request(test_data)
        print("   ✅ 통합 처리 성공")
        print(f"   - 상태: {final_result['status']}")
        print(f"   - 도메인: {final_result['domain']}")
        print(f"   - 총 요청 수: {final_result['metrics']['total_requests']}")
        
    except Exception as e:
        print(f"   ❌ 통합 처리 실패: {e}")
        return
    
    print("\n✅ 모든 테스트 성공!")

if __name__ == "__main__":
    # 직접 실행
    asyncio.run(run_single_test())