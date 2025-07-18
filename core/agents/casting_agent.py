"""
Casting Expert Agent - 주조 전문 에이전트

주조 공정의 결함 분석, 품질 예측, 공정 최적화를 담당하는 전문 에이전트
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

from .base_agent import BaseIndustrialAgent

logger = logging.getLogger(__name__)


class CastingExpertAgent(BaseIndustrialAgent):
    """주조 전문 에이전트"""
    
    def __init__(self):
        # 주조 전문 도구들
        casting_tools = [
            "metallurgy_calculator",      # 금속학 계산기
            "defect_pattern_analyzer",     # 결함 패턴 분석기
            "casting_simulation_tool",     # 주조 시뮬레이션
            "temperature_optimizer",       # 온도 최적화
            "mold_design_validator"        # 금형 설계 검증
        ]
        
        super().__init__(
            domain="casting",
            model_provider="claude-3-opus",  # 고성능 모델 사용
            reasoning_type="technical_analysis",
            tools=casting_tools,
            knowledge_base="casting_expertise_db"
        )
        
        # 주조 전문 지식 베이스
        self.casting_knowledge = {
            "defect_types": {
                "기공": {
                    "causes": ["가스 용해", "수축", "터빈"],
                    "solutions": ["탈가스 처리", "주입 속도 조절", "금형 설계 개선"]
                },
                "수축공": {
                    "causes": ["응고 수축", "불균일 냉각", "부적절한 라이저"],
                    "solutions": ["라이저 설계 개선", "냉각 속도 조절", "합금 조성 최적화"]
                },
                "균열": {
                    "causes": ["열응력", "급냉", "합금 취성"],
                    "solutions": ["냉각 속도 제어", "응력 완화 열처리", "합금 개선"]
                },
                "편석": {
                    "causes": ["느린 응고", "대류", "중력 편석"],
                    "solutions": ["응고 속도 증가", "전자기 교반", "합금 조성 조절"]
                }
            },
            "process_parameters": {
                "온도": {"범위": [650, 750], "단위": "°C"},
                "압력": {"범위": [100, 500], "단위": "MPa"},
                "주입속도": {"범위": [0.5, 2.0], "단위": "m/s"},
                "냉각속도": {"범위": [1, 10], "단위": "°C/s"}
            }
        }
    
    async def analyze(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """주조 문제 분석"""
        logger.info(f"주조 문제 분석 시작: {problem_data.get('problem_type', 'unknown')}")
        
        analysis_result = {
            "problem_type": problem_data.get("problem_type"),
            "severity": "unknown",
            "detected_defects": [],
            "root_causes": [],
            "affected_parameters": [],
            "confidence": 0.0
        }
        
        try:
            # 1. 결함 유형 식별
            if "defect_images" in problem_data:
                defect_analysis = await self._analyze_defect_images(problem_data["defect_images"])
                analysis_result["detected_defects"] = defect_analysis["defects"]
                analysis_result["severity"] = defect_analysis["severity"]
            
            # 2. 공정 파라미터 분석
            if "process_data" in problem_data:
                parameter_analysis = await self._analyze_process_parameters(problem_data["process_data"])
                analysis_result["affected_parameters"] = parameter_analysis["abnormal_parameters"]
            
            # 3. 근본 원인 추론
            root_causes = await self._infer_root_causes(
                analysis_result["detected_defects"],
                analysis_result["affected_parameters"]
            )
            analysis_result["root_causes"] = root_causes
            
            # 4. 신뢰도 계산
            analysis_result["confidence"] = self._calculate_confidence(analysis_result)
            
            logger.info(f"주조 문제 분석 완료: {len(analysis_result['detected_defects'])}개 결함 발견")
            
        except Exception as e:
            logger.error(f"주조 문제 분석 실패: {e}")
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    async def generate_solution(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """주조 문제 해결책 생성"""
        logger.info("주조 문제 해결책 생성 시작")
        
        solution = {
            "immediate_actions": [],
            "long_term_improvements": [],
            "parameter_adjustments": {},
            "estimated_improvement": 0,
            "implementation_priority": []
        }
        
        try:
            # 1. 즉시 조치사항 생성
            for defect in analysis_result.get("detected_defects", []):
                immediate_actions = self._get_immediate_actions(defect)
                solution["immediate_actions"].extend(immediate_actions)
            
            # 2. 파라미터 조정 권고
            for param in analysis_result.get("affected_parameters", []):
                adjustment = self._calculate_parameter_adjustment(param)
                solution["parameter_adjustments"][param["name"]] = adjustment
            
            # 3. 장기 개선사항
            root_causes = analysis_result.get("root_causes", [])
            solution["long_term_improvements"] = self._generate_long_term_improvements(root_causes)
            
            # 4. 우선순위 설정
            solution["implementation_priority"] = self._prioritize_solutions(solution)
            
            # 5. 예상 개선율 계산
            solution["estimated_improvement"] = self._estimate_improvement(solution)
            
            logger.info(f"주조 문제 해결책 생성 완료: {len(solution['immediate_actions'])}개 즉시 조치")
            
        except Exception as e:
            logger.error(f"주조 문제 해결책 생성 실패: {e}")
            solution["error"] = str(e)
        
        return solution
    
    async def _analyze_defect_images(self, images: List[str]) -> Dict[str, Any]:
        """결함 이미지 분석 (시뮬레이션)"""
        # 실제 구현에서는 컴퓨터 비전 모델 사용
        await asyncio.sleep(0.1)  # 처리 시간 시뮬레이션
        
        # 예시 결과
        return {
            "defects": [
                {"type": "기공", "location": "중심부", "size": "5mm", "count": 3},
                {"type": "수축공", "location": "라이저 근처", "size": "10mm", "count": 1}
            ],
            "severity": "high"
        }
    
    async def _analyze_process_parameters(self, process_data: Dict[str, Any]) -> Dict[str, Any]:
        """공정 파라미터 분석"""
        await asyncio.sleep(0.05)  # 처리 시간 시뮬레이션
        
        abnormal_params = []
        
        for param_name, value in process_data.items():
            if param_name in self.casting_knowledge["process_parameters"]:
                param_info = self.casting_knowledge["process_parameters"][param_name]
                min_val, max_val = param_info["범위"]
                
                if value < min_val or value > max_val:
                    abnormal_params.append({
                        "name": param_name,
                        "current_value": value,
                        "normal_range": param_info["범위"],
                        "unit": param_info["단위"],
                        "deviation": abs(value - (min_val + max_val) / 2)
                    })
        
        return {"abnormal_parameters": abnormal_params}
    
    async def _infer_root_causes(self, defects: List[Dict], parameters: List[Dict]) -> List[Dict]:
        """근본 원인 추론"""
        await asyncio.sleep(0.05)  # 추론 시간 시뮬레이션
        
        root_causes = []
        
        # 결함 기반 원인 추론
        for defect in defects:
            defect_type = defect.get("type")
            if defect_type in self.casting_knowledge["defect_types"]:
                causes = self.casting_knowledge["defect_types"][defect_type]["causes"]
                for cause in causes:
                    root_causes.append({
                        "cause": cause,
                        "related_defect": defect_type,
                        "probability": 0.7,  # 실제로는 AI 모델이 계산
                        "evidence": f"{defect_type} 결함 발견"
                    })
        
        # 파라미터 기반 원인 추론
        for param in parameters:
            if param["name"] == "온도" and param["deviation"] > 50:
                root_causes.append({
                    "cause": "과열 또는 과냉",
                    "related_parameter": "온도",
                    "probability": 0.8,
                    "evidence": f"온도 편차 {param['deviation']}°C"
                })
        
        return root_causes
    
    def _get_immediate_actions(self, defect: Dict) -> List[Dict]:
        """즉시 조치사항 도출"""
        actions = []
        defect_type = defect.get("type")
        
        if defect_type in self.casting_knowledge["defect_types"]:
            solutions = self.casting_knowledge["defect_types"][defect_type]["solutions"]
            for solution in solutions[:2]:  # 상위 2개 솔루션
                actions.append({
                    "action": solution,
                    "target_defect": defect_type,
                    "urgency": "high",
                    "estimated_time": "1-2시간"
                })
        
        return actions
    
    def _calculate_parameter_adjustment(self, param: Dict) -> Dict[str, Any]:
        """파라미터 조정 계산"""
        param_name = param["name"]
        current_value = param["current_value"]
        normal_range = param["normal_range"]
        
        # 목표값은 정상 범위의 중간값
        target_value = (normal_range[0] + normal_range[1]) / 2
        adjustment_amount = target_value - current_value
        
        return {
            "current": current_value,
            "target": target_value,
            "adjustment": adjustment_amount,
            "unit": param["unit"],
            "method": "점진적 조정" if abs(adjustment_amount) > 20 else "즉시 조정"
        }
    
    def _generate_long_term_improvements(self, root_causes: List[Dict]) -> List[Dict]:
        """장기 개선사항 생성"""
        improvements = []
        
        # 원인별 개선사항
        cause_types = set(cause["cause"] for cause in root_causes)
        
        if "가스 용해" in cause_types:
            improvements.append({
                "improvement": "탈가스 시스템 업그레이드",
                "expected_benefit": "기공 결함 70% 감소",
                "investment": "중간",
                "implementation_time": "3-6개월"
            })
        
        if "응고 수축" in cause_types:
            improvements.append({
                "improvement": "라이저 설계 최적화 시스템 도입",
                "expected_benefit": "수축공 결함 80% 감소",
                "investment": "높음",
                "implementation_time": "6-12개월"
            })
        
        return improvements
    
    def _prioritize_solutions(self, solution: Dict[str, Any]) -> List[Dict]:
        """솔루션 우선순위 설정"""
        priorities = []
        
        # 즉시 조치사항 - 최우선
        for idx, action in enumerate(solution["immediate_actions"]):
            priorities.append({
                "priority": idx + 1,
                "type": "immediate",
                "description": action["action"],
                "impact": "high"
            })
        
        # 파라미터 조정 - 중간 우선순위
        for idx, (param, adjustment) in enumerate(solution["parameter_adjustments"].items()):
            priorities.append({
                "priority": len(solution["immediate_actions"]) + idx + 1,
                "type": "parameter",
                "description": f"{param} 조정: {adjustment['adjustment']}{adjustment['unit']}",
                "impact": "medium"
            })
        
        return priorities[:5]  # 상위 5개만 반환
    
    def _calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """분석 신뢰도 계산"""
        confidence = 0.5  # 기본 신뢰도
        
        # 발견된 결함 수에 따라 신뢰도 증가
        defect_count = len(analysis_result.get("detected_defects", []))
        if defect_count > 0:
            confidence += min(defect_count * 0.1, 0.3)
        
        # 비정상 파라미터 수에 따라 신뢰도 증가
        param_count = len(analysis_result.get("affected_parameters", []))
        if param_count > 0:
            confidence += min(param_count * 0.05, 0.2)
        
        return min(confidence, 0.95)  # 최대 95%
    
    def _estimate_improvement(self, solution: Dict[str, Any]) -> float:
        """예상 개선율 계산"""
        improvement = 0.0
        
        # 즉시 조치사항 효과
        improvement += len(solution["immediate_actions"]) * 10
        
        # 파라미터 조정 효과
        improvement += len(solution["parameter_adjustments"]) * 15
        
        # 장기 개선사항 효과
        improvement += len(solution["long_term_improvements"]) * 25
        
        return min(improvement, 85)  # 최대 85% 개선율