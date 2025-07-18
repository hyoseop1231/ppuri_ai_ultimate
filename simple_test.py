#!/usr/bin/env python3
"""
Simple Test - 간단한 통합 테스트
Shell 환경 문제를 우회하여 직접 테스트 실행
"""

import asyncio
import sys
import os
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_casting_agent():
    """주조 에이전트 테스트"""
    print("\n=== 주조 에이전트 테스트 ===")
    
    try:
        from core.agents.casting_agent import CastingExpertAgent
        print("✅ CastingExpertAgent import 성공")
        
        # 에이전트 생성
        agent = CastingExpertAgent()
        print("✅ 에이전트 생성 성공")
        
        # 테스트 데이터
        test_data = {
            "problem_type": "defect_analysis",
            "description": "주조 제품에서 기공 결함 발견",
            "process_data": {
                "온도": 780,
                "압력": 300,
                "주입속도": 1.5
            }
        }
        
        # 요청 처리
        result = await agent.process_request(test_data)
        print(f"✅ 요청 처리 성공: {result['status']}")
        print(f"   - 분석된 결함: {len(result['analysis']['detected_defects'])}개")
        print(f"   - 신뢰도: {result['analysis']['confidence']:.2%}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def test_workflow():
    """워크플로우 테스트"""
    print("\n=== 워크플로우 테스트 ===")
    
    try:
        from core.workflows.analysis_workflow import IndustrialAnalysisWorkflow
        print("✅ IndustrialAnalysisWorkflow import 성공")
        
        # 워크플로우 생성
        workflow = IndustrialAnalysisWorkflow()
        print("✅ 워크플로우 생성 성공")
        
        # 테스트 데이터
        test_data = {
            "problem_type": "complex_defect",
            "description": "주조 공정에서 복합 품질 문제 발생",
            "domain": "casting",
            "process_data": {
                "온도": 800,
                "압력": 150
            }
        }
        
        # 워크플로우 실행
        result = await workflow.execute(test_data)
        print(f"✅ 워크플로우 실행 성공: {result['status']}")
        print(f"   - 워크플로우 ID: {result['workflow_id']}")
        print(f"   - 실행 시간: {result['execution_time']:.3f}초")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def test_api_routes():
    """API 라우트 테스트"""
    print("\n=== API 라우트 테스트 ===")
    
    try:
        from api.routes.industrial_ai import router
        print("✅ industrial_ai router import 성공")
        print(f"   - 등록된 라우트 수: {len(router.routes)}")
        
        for route in router.routes:
            if hasattr(route, 'path'):
                print(f"   - {route.methods} {route.path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """메인 테스트 실행"""
    print("🚀 PPuRI-AI Ultimate 간단 통합 테스트")
    print("=" * 50)
    print(f"실행 시간: {datetime.now()}")
    print("=" * 50)
    
    # 각 테스트 실행
    await test_casting_agent()
    await test_workflow()
    await test_api_routes()
    
    print("\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())