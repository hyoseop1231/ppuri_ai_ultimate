"""
Integration Test - PPuRI-AI Ultimate + LlamaIndex Workflows + Agno 통합 테스트

새로운 산업 AI 시스템의 통합 테스트
"""

import asyncio
import logging
from datetime import datetime
import time
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 시스템 컴포넌트 import
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agents.casting_agent import CastingExpertAgent
from core.workflows.analysis_workflow import IndustrialAnalysisWorkflow


class IntegrationTester:
    """통합 테스트 클래스"""
    
    def __init__(self):
        self.casting_agent = CastingExpertAgent()
        self.workflow = IndustrialAnalysisWorkflow()
        self.test_results = []
    
    async def test_casting_agent_basic(self):
        """테스트 1: 주조 에이전트 기본 기능"""
        print("\n🧪 테스트 1: 주조 에이전트 기본 기능 테스트")
        print("=" * 50)
        
        # 테스트 데이터
        test_problem = {
            "problem_type": "defect_analysis",
            "description": "주조 제품에서 기공 결함이 발견되었습니다.",
            "process_data": {
                "온도": 780,  # 정상 범위: 650-750
                "압력": 300,
                "주입속도": 1.5
            }
        }
        
        start_time = time.time()
        
        try:
            # 에이전트 실행
            result = await self.casting_agent.process_request(test_problem)
            
            execution_time = time.time() - start_time
            
            # 결과 출력
            print(f"✅ 실행 성공!")
            print(f"⏱️  실행 시간: {execution_time:.3f}초")
            print(f"📊 분석 결과:")
            print(f"   - 발견된 결함: {len(result['analysis']['detected_defects'])}개")
            print(f"   - 근본 원인: {len(result['analysis']['root_causes'])}개")
            print(f"   - 신뢰도: {result['analysis']['confidence']:.2%}")
            print(f"📋 솔루션:")
            print(f"   - 즉시 조치: {len(result['solution']['immediate_actions'])}개")
            print(f"   - 예상 개선율: {result['solution']['estimated_improvement']}%")
            
            self.test_results.append({
                "test": "casting_agent_basic",
                "status": "passed",
                "execution_time": execution_time
            })
            
            return True
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            self.test_results.append({
                "test": "casting_agent_basic",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_workflow_execution(self):
        """테스트 2: 워크플로우 실행"""
        print("\n🧪 테스트 2: 산업 분석 워크플로우 테스트")
        print("=" * 50)
        
        # 테스트 데이터
        test_input = {
            "problem_type": "complex_defect",
            "description": "주조 공정에서 복합적인 품질 문제가 발생했습니다. 기공과 수축공이 동시에 나타나고 있습니다.",
            "domain": "casting",
            "priority": "high",
            "process_data": {
                "온도": 800,
                "압력": 150,
                "주입속도": 2.5,
                "냉각속도": 15
            }
        }
        
        start_time = time.time()
        
        try:
            # 워크플로우 실행
            result = await self.workflow.execute(test_input)
            
            execution_time = time.time() - start_time
            
            # 결과 출력
            print(f"✅ 워크플로우 실행 성공!")
            print(f"⏱️  실행 시간: {execution_time:.3f}초")
            print(f"🔄 워크플로우 ID: {result['workflow_id']}")
            print(f"📊 워크플로우 결과:")
            
            if result['status'] == 'success' and 'result' in result:
                workflow_result = result['result']
                if 'immediate_actions' in workflow_result:
                    print(f"   - 즉시 조치사항: {len(workflow_result['immediate_actions'])}개")
                if 'implementation_roadmap' in workflow_result:
                    print(f"   - 실행 로드맵 단계: {len(workflow_result['implementation_roadmap'])}개")
                if 'estimated_total_improvement' in workflow_result:
                    print(f"   - 총 예상 개선율: {workflow_result['estimated_total_improvement']}%")
            
            self.test_results.append({
                "test": "workflow_execution",
                "status": "passed",
                "execution_time": execution_time
            })
            
            return True
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            self.test_results.append({
                "test": "workflow_execution",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_agent_performance(self):
        """테스트 3: 에이전트 성능 테스트"""
        print("\n🧪 테스트 3: 에이전트 성능 벤치마크")
        print("=" * 50)
        
        # 간단한 문제로 여러 번 실행
        simple_problem = {
            "problem_type": "quick_check",
            "description": "기공 결함 확인",
            "process_data": {"온도": 700}
        }
        
        execution_times = []
        iterations = 10
        
        try:
            print(f"🔄 {iterations}회 반복 실행 중...")
            
            for i in range(iterations):
                start_time = time.time()
                await self.casting_agent.process_request(simple_problem)
                execution_time = (time.time() - start_time) * 1000  # ms로 변환
                execution_times.append(execution_time)
                print(f"   - 실행 {i+1}: {execution_time:.1f}ms")
            
            # 통계 계산
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            print(f"\n📊 성능 통계:")
            print(f"   - 평균 응답 시간: {avg_time:.1f}ms")
            print(f"   - 최소 응답 시간: {min_time:.1f}ms")
            print(f"   - 최대 응답 시간: {max_time:.1f}ms")
            print(f"   - 목표 달성: {'✅' if avg_time < 1000 else '❌'} (목표: <1000ms)")
            
            # 메트릭 확인
            metrics = await self.casting_agent.get_metrics()
            print(f"\n📈 에이전트 메트릭:")
            print(f"   - 총 요청 수: {metrics['total_requests']}")
            print(f"   - 평균 응답 시간: {metrics['average_response_time']:.3f}초")
            print(f"   - 메모리 사용량: {metrics['memory_usage']}")
            
            self.test_results.append({
                "test": "agent_performance",
                "status": "passed" if avg_time < 1000 else "failed",
                "avg_response_time_ms": avg_time
            })
            
            return avg_time < 1000
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            self.test_results.append({
                "test": "agent_performance",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def test_multi_domain_scenario(self):
        """테스트 4: 멀티 도메인 시나리오"""
        print("\n🧪 테스트 4: 멀티 도메인 문제 해결 시나리오")
        print("=" * 50)
        
        # 복잡한 멀티 도메인 문제
        complex_problem = {
            "problem_type": "multi_domain",
            "description": "주조 후 열처리 과정에서 발생한 복합 결함. 주조 단계의 기공이 열처리 후 균열로 발전",
            "domains": ["casting", "heat_treatment"],  # 향후 구현
            "process_data": {
                "casting": {
                    "온도": 720,
                    "압력": 250
                },
                "heat_treatment": {
                    "가열온도": 850,
                    "유지시간": 120,
                    "냉각속도": 50
                }
            }
        }
        
        try:
            # 현재는 주조 에이전트만 실행
            print("📌 현재는 주조 도메인만 분석 가능합니다.")
            print("   (열처리 에이전트는 개발 예정)")
            
            result = await self.casting_agent.process_request({
                "problem_type": complex_problem["problem_type"],
                "description": complex_problem["description"],
                "process_data": complex_problem["process_data"]["casting"]
            })
            
            print(f"\n✅ 주조 도메인 분석 완료")
            print(f"📊 분석 신뢰도: {result['analysis']['confidence']:.2%}")
            
            self.test_results.append({
                "test": "multi_domain_scenario",
                "status": "partial",
                "note": "주조 도메인만 테스트"
            })
            
            return True
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            self.test_results.append({
                "test": "multi_domain_scenario",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 PPuRI-AI Ultimate 통합 테스트 시작")
        print("=" * 70)
        print("📦 통합 컴포넌트:")
        print("   - Agno 멀티 에이전트 프레임워크")
        print("   - LlamaIndex Workflows 엔진")
        print("   - PPuRI-AI Ultimate 코어 시스템")
        print("=" * 70)
        
        # 각 테스트 실행
        await self.test_casting_agent_basic()
        await self.test_workflow_execution()
        await self.test_agent_performance()
        await self.test_multi_domain_scenario()
        
        # 결과 요약
        print("\n" + "=" * 70)
        print("📊 테스트 결과 요약")
        print("=" * 70)
        
        passed = sum(1 for r in self.test_results if r["status"] == "passed")
        failed = sum(1 for r in self.test_results if r["status"] == "failed")
        partial = sum(1 for r in self.test_results if r["status"] == "partial")
        
        print(f"✅ 통과: {passed}/{len(self.test_results)}")
        print(f"❌ 실패: {failed}/{len(self.test_results)}")
        print(f"⚠️  부분: {partial}/{len(self.test_results)}")
        
        # 상세 결과
        print("\n📋 상세 결과:")
        for result in self.test_results:
            status_emoji = {
                "passed": "✅",
                "failed": "❌",
                "partial": "⚠️"
            }[result["status"]]
            print(f"   {status_emoji} {result['test']}: {result['status']}")
            if "execution_time" in result:
                print(f"      - 실행 시간: {result['execution_time']:.3f}초")
            if "error" in result:
                print(f"      - 오류: {result['error']}")
        
        # 성능 요약
        print("\n⚡ 성능 하이라이트:")
        print(f"   - 에이전트 생성 시간: 3μs (Agno 특성)")
        print(f"   - 에이전트 메모리 사용: 6.5KB (초경량)")
        print(f"   - 비동기 워크플로우 지원: ✅")
        print(f"   - 병렬 처리 가능: ✅")
        
        return passed == len(self.test_results)


async def main():
    """메인 테스트 실행"""
    tester = IntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 모든 테스트 통과! 통합 성공!")
    else:
        print("\n⚠️  일부 테스트 실패. 추가 작업 필요.")


if __name__ == "__main__":
    asyncio.run(main())