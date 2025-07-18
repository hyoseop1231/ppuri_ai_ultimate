"""
Industrial Analysis Workflow - 산업 문제 분석 워크플로우

복잡한 뿌리산업 문제를 단계별로 분석하고 해결하는 통합 워크플로우
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .base_workflow import (
    BaseIndustrialWorkflow,
    WorkflowState,
    ProblemSubmissionEvent,
    AnalysisCompleteEvent,
    SolutionGeneratedEvent
)

# LlamaIndex Workflows imports
try:
    from llama_index.core.workflow import (
        Workflow,
        step,
        Context,
        Event,
        StartEvent,
        StopEvent
    )
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # 이미 base_workflow에서 정의됨

# 에이전트 import
from ..agents.casting_agent import CastingExpertAgent

logger = logging.getLogger(__name__)


class IndustrialAnalysisWorkflow(BaseIndustrialWorkflow):
    """산업 문제 분석 통합 워크플로우"""
    
    def __init__(self):
        super().__init__("industrial_analysis_workflow")
        
        # 도메인별 에이전트 초기화
        self.agents = {
            "casting": CastingExpertAgent(),
            # TODO: 다른 도메인 에이전트 추가
            # "molding": MoldingExpertAgent(),
            # "forming": FormingExpertAgent(),
            # "welding": WeldingExpertAgent(),
            # "surface_treatment": SurfaceTreatmentExpertAgent(),
            # "heat_treatment": HeatTreatmentExpertAgent()
        }
        
        # 워크플로우 설정
        self.config = {
            "max_parallel_agents": 5,
            "timeout_seconds": 300,
            "min_confidence_threshold": 0.7
        }
    
    def _create_workflow(self) -> Workflow:
        """LlamaIndex Workflow 생성"""
        if not LLAMAINDEX_AVAILABLE:
            return None
        
        class ActualWorkflow(Workflow):
            @step
            async def problem_intake(self, ctx: Context, ev: StartEvent) -> ProblemSubmissionEvent:
                """Step 1: 문제 접수 및 분류"""
                input_data = ev.input_data
                
                # 문제 데이터 검증 및 전처리
                problem_data = {
                    "problem_type": input_data.get("problem_type", "unknown"),
                    "description": input_data.get("description", ""),
                    "domain_hint": input_data.get("domain", None),
                    "images": input_data.get("images", []),
                    "process_data": input_data.get("process_data", {}),
                    "priority": input_data.get("priority", "normal")
                }
                
                # 도메인 자동 분류 (간단한 키워드 기반)
                if not problem_data["domain_hint"]:
                    problem_data["domain_hint"] = self._classify_domain(problem_data["description"])
                
                logger.info(f"문제 접수: {problem_data['problem_type']} (도메인: {problem_data['domain_hint']})")
                
                return ProblemSubmissionEvent(
                    problem_data=problem_data,
                    priority=problem_data["priority"],
                    request_id=f"req_{datetime.now().timestamp()}"
                )
            
            @step
            async def multi_agent_analysis(self, ctx: Context, ev: ProblemSubmissionEvent) -> AnalysisCompleteEvent:
                """Step 2: 다중 에이전트 병렬 분석"""
                problem_data = ev.problem_data
                domain_hint = problem_data.get("domain_hint")
                
                # 관련 에이전트 선택
                relevant_agents = self._select_relevant_agents(domain_hint, problem_data)
                
                # 병렬 분석 실행
                analysis_tasks = []
                for agent_name, agent in relevant_agents.items():
                    task = agent.analyze(problem_data)
                    analysis_tasks.append((agent_name, task))
                
                # 모든 에이전트 분석 결과 수집
                analysis_results = {}
                for agent_name, task in analysis_tasks:
                    try:
                        result = await task
                        analysis_results[agent_name] = result
                        logger.info(f"{agent_name} 에이전트 분석 완료")
                    except Exception as e:
                        logger.error(f"{agent_name} 에이전트 분석 실패: {e}")
                        analysis_results[agent_name] = {"error": str(e)}
                
                # 결과 통합 및 신뢰도 계산
                combined_analysis = self._combine_analysis_results(analysis_results)
                confidence_score = self._calculate_combined_confidence(analysis_results)
                
                return AnalysisCompleteEvent(
                    analysis_result=combined_analysis,
                    domain_agents_used=list(relevant_agents.keys()),
                    confidence_score=confidence_score
                )
            
            @step
            async def solution_generation(self, ctx: Context, ev: AnalysisCompleteEvent) -> StopEvent:
                """Step 3: 통합 솔루션 생성"""
                analysis_result = ev.analysis_result
                agents_used = ev.domain_agents_used
                
                # 각 에이전트로부터 솔루션 생성
                solution_tasks = []
                for agent_name in agents_used:
                    if agent_name in self.agents:
                        agent = self.agents[agent_name]
                        agent_analysis = analysis_result.get(agent_name, {})
                        task = agent.generate_solution(agent_analysis)
                        solution_tasks.append((agent_name, task))
                
                # 솔루션 수집
                all_solutions = {}
                for agent_name, task in solution_tasks:
                    try:
                        solution = await task
                        all_solutions[agent_name] = solution
                        logger.info(f"{agent_name} 에이전트 솔루션 생성 완료")
                    except Exception as e:
                        logger.error(f"{agent_name} 에이전트 솔루션 생성 실패: {e}")
                        all_solutions[agent_name] = {"error": str(e)}
                
                # 최종 통합 솔루션 생성
                final_solution = self._synthesize_solutions(all_solutions)
                
                return StopEvent(result=final_solution)
            
            def _classify_domain(self, description: str) -> str:
                """텍스트 기반 도메인 분류"""
                description_lower = description.lower()
                
                domain_keywords = {
                    "casting": ["주조", "주물", "용탕", "금형", "기공", "수축"],
                    "molding": ["성형", "사출", "압출", "금형"],
                    "forming": ["단조", "압연", "인발", "프레스"],
                    "welding": ["용접", "접합", "용착", "아크"],
                    "surface_treatment": ["도금", "도장", "표면처리", "코팅"],
                    "heat_treatment": ["열처리", "담금질", "뜨임", "어닐링"]
                }
                
                for domain, keywords in domain_keywords.items():
                    if any(keyword in description_lower for keyword in keywords):
                        return domain
                
                return "casting"  # 기본값
            
            def _select_relevant_agents(self, domain_hint: str, problem_data: Dict) -> Dict:
                """관련 에이전트 선택"""
                selected = {}
                
                # 도메인 힌트가 있으면 해당 에이전트 선택
                if domain_hint and domain_hint in self.agents:
                    selected[domain_hint] = self.agents[domain_hint]
                
                # 추가 관련 에이전트 선택 로직
                # TODO: 더 정교한 선택 로직 구현
                
                # 현재는 주조 에이전트만 반환
                if not selected and "casting" in self.agents:
                    selected["casting"] = self.agents["casting"]
                
                return selected
            
            def _combine_analysis_results(self, results: Dict[str, Dict]) -> Dict:
                """분석 결과 통합"""
                combined = {
                    "detected_issues": [],
                    "root_causes": [],
                    "affected_parameters": [],
                    "severity": "unknown",
                    "confidence_scores": {}
                }
                
                for agent_name, result in results.items():
                    if "error" not in result:
                        # 검출된 문제 통합
                        if "detected_defects" in result:
                            combined["detected_issues"].extend(result["detected_defects"])
                        
                        # 근본 원인 통합
                        if "root_causes" in result:
                            combined["root_causes"].extend(result["root_causes"])
                        
                        # 영향받는 파라미터 통합
                        if "affected_parameters" in result:
                            combined["affected_parameters"].extend(result["affected_parameters"])
                        
                        # 신뢰도 저장
                        if "confidence" in result:
                            combined["confidence_scores"][agent_name] = result["confidence"]
                        
                        # 심각도 업데이트
                        if result.get("severity") == "high":
                            combined["severity"] = "high"
                
                return combined
            
            def _calculate_combined_confidence(self, results: Dict[str, Dict]) -> float:
                """통합 신뢰도 계산"""
                confidences = []
                
                for agent_name, result in results.items():
                    if "confidence" in result and "error" not in result:
                        confidences.append(result["confidence"])
                
                if confidences:
                    return sum(confidences) / len(confidences)
                
                return 0.0
            
            def _synthesize_solutions(self, solutions: Dict[str, Dict]) -> Dict:
                """솔루션 통합"""
                synthesized = {
                    "immediate_actions": [],
                    "parameter_adjustments": {},
                    "long_term_improvements": [],
                    "implementation_roadmap": [],
                    "estimated_total_improvement": 0,
                    "confidence": 0
                }
                
                improvement_scores = []
                
                for agent_name, solution in solutions.items():
                    if "error" not in solution:
                        # 즉시 조치사항 통합
                        if "immediate_actions" in solution:
                            for action in solution["immediate_actions"]:
                                action["source_agent"] = agent_name
                                synthesized["immediate_actions"].append(action)
                        
                        # 파라미터 조정 통합
                        if "parameter_adjustments" in solution:
                            for param, adjustment in solution["parameter_adjustments"].items():
                                synthesized["parameter_adjustments"][f"{agent_name}_{param}"] = adjustment
                        
                        # 장기 개선사항 통합
                        if "long_term_improvements" in solution:
                            for improvement in solution["long_term_improvements"]:
                                improvement["source_agent"] = agent_name
                                synthesized["long_term_improvements"].append(improvement)
                        
                        # 개선율 수집
                        if "estimated_improvement" in solution:
                            improvement_scores.append(solution["estimated_improvement"])
                
                # 실행 로드맵 생성
                synthesized["implementation_roadmap"] = self._create_implementation_roadmap(synthesized)
                
                # 총 개선율 계산
                if improvement_scores:
                    synthesized["estimated_total_improvement"] = sum(improvement_scores) / len(improvement_scores)
                
                # 신뢰도 계산
                synthesized["confidence"] = self._calculate_solution_confidence(solutions)
                
                return synthesized
            
            def _create_implementation_roadmap(self, synthesized: Dict) -> List[Dict]:
                """실행 로드맵 생성"""
                roadmap = []
                
                # Phase 1: 즉시 조치 (0-24시간)
                if synthesized["immediate_actions"]:
                    roadmap.append({
                        "phase": 1,
                        "title": "즉시 조치사항",
                        "timeline": "0-24시간",
                        "actions": synthesized["immediate_actions"][:5]  # 상위 5개
                    })
                
                # Phase 2: 파라미터 조정 (1-7일)
                if synthesized["parameter_adjustments"]:
                    roadmap.append({
                        "phase": 2,
                        "title": "공정 파라미터 최적화",
                        "timeline": "1-7일",
                        "actions": [
                            {
                                "action": f"{param} 조정",
                                "details": adjustment
                            }
                            for param, adjustment in list(synthesized["parameter_adjustments"].items())[:3]
                        ]
                    })
                
                # Phase 3: 장기 개선 (1-6개월)
                if synthesized["long_term_improvements"]:
                    roadmap.append({
                        "phase": 3,
                        "title": "장기 개선 프로젝트",
                        "timeline": "1-6개월",
                        "actions": synthesized["long_term_improvements"][:3]  # 상위 3개
                    })
                
                return roadmap
            
            def _calculate_solution_confidence(self, solutions: Dict[str, Dict]) -> float:
                """솔루션 신뢰도 계산"""
                valid_solutions = [s for s in solutions.values() if "error" not in s]
                
                if not valid_solutions:
                    return 0.0
                
                # 성공적인 솔루션 비율
                success_rate = len(valid_solutions) / len(solutions)
                
                # 평균 개선율이 있는 솔루션 비율
                improvement_rate = len([s for s in valid_solutions if s.get("estimated_improvement", 0) > 0]) / len(solutions)
                
                return (success_rate + improvement_rate) / 2
        
        # 워크플로우 인스턴스에 필요한 속성 바인딩
        workflow = ActualWorkflow()
        workflow.agents = self.agents
        workflow.config = self.config
        
        return workflow
    
    async def _execute_fallback_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback 워크플로우 실행"""
        logger.info("Fallback 워크플로우 실행 중...")
        
        # Step 1: 문제 분류
        self.state.current_step = "problem_classification"
        problem_data = {
            "problem_type": input_data.get("problem_type", "unknown"),
            "description": input_data.get("description", ""),
            "domain": input_data.get("domain", "casting"),
            "process_data": input_data.get("process_data", {})
        }
        
        # Step 2: 에이전트 분석
        self.state.current_step = "agent_analysis"
        if problem_data["domain"] in self.agents:
            agent = self.agents[problem_data["domain"]]
            analysis_result = await agent.analyze(problem_data)
            
            # Step 3: 솔루션 생성
            self.state.current_step = "solution_generation"
            solution = await agent.generate_solution(analysis_result)
            
            return {
                "analysis": analysis_result,
                "solution": solution,
                "workflow_type": "fallback"
            }
        
        return {
            "error": "No suitable agent found",
            "workflow_type": "fallback"
        }