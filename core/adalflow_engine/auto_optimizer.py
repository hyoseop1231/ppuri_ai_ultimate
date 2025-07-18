"""
Auto Prompt Optimizer - AdalFlow LLM-AutoDiff 구현

최신 연구 기반 자동 프롬프트 최적화:
- LLM-AutoDiff: 텍스트 기울기 기반 자동 미분
- Learn-to-Reason: 퓨샷 인컨텍스트 학습
- PyTorch 스타일 Parameter 및 Optimizer 클래스
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import torch
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """최적화 모드"""
    PROMPT_ONLY = "prompt"           # 프롬프트만 최적화
    DEMOS_ONLY = "demos"             # 퓨샷 예시만 최적화  
    COMBINED = "combined"            # 프롬프트 + 퓨샷 동시 최적화
    TEMPLATE = "template"            # 템플릿 구조 최적화


@dataclass
class OptimizationResult:
    """최적화 결과"""
    original_prompt: str
    optimized_prompt: str
    performance_gain: float
    optimization_steps: int
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    mode: OptimizationMode = OptimizationMode.PROMPT_ONLY
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "performance_gain": self.performance_gain,
            "optimization_steps": self.optimization_steps,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode.value
        }


class AutoPromptOptimizer:
    """
    AdalFlow 스타일 자동 프롬프트 최적화기
    
    Features:
    - LLM-AutoDiff 기반 텍스트 기울기 계산
    - PyTorch 스타일 Parameter 관리
    - 뿌리산업 특화 최적화
    - 실시간 성능 추적
    """
    
    def __init__(
        self,
        ollama_client,
        base_model: str = "qwen3:30b-a3b",  # KITECH 검증된 모델
        optimizer_model: str = "qwen3:30b-a3b",
        max_steps: int = 10,
        performance_threshold: float = 0.85,
        korean_specialized: bool = True
    ):
        self.ollama_client = ollama_client
        self.base_model = base_model
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.performance_threshold = performance_threshold
        self.korean_specialized = korean_specialized
        
        # PyTorch 스타일 최적화 상태
        self.parameters: Dict[str, Any] = {}
        self.gradients: Dict[str, str] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # 뿌리산업 특화 설정
        self.industry_domains = [
            "주조", "금형", "소성가공", "용접", "표면처리", "열처리"
        ]
        
        # 성능 메트릭
        self.performance_metrics = {
            "accuracy": [],
            "relevance": [],
            "korean_quality": [],
            "technical_accuracy": []
        }
    
    async def optimize_prompt(
        self,
        prompt: str,
        context: Dict[str, Any],
        target_domain: str = "뿌리산업",
        mode: OptimizationMode = OptimizationMode.COMBINED
    ) -> OptimizationResult:
        """
        자동 프롬프트 최적화 실행
        
        Args:
            prompt: 최적화할 원본 프롬프트
            context: 최적화 컨텍스트 (사용자 히스토리, 성공 패턴 등)
            target_domain: 타겟 도메인 (뿌리산업 분야)
            mode: 최적화 모드
            
        Returns:
            OptimizationResult: 최적화 결과
        """
        logger.info(f"프롬프트 자동 최적화 시작: {target_domain} 도메인")
        
        original_prompt = prompt
        current_prompt = prompt
        step = 0
        best_performance = 0.0
        best_prompt = prompt
        
        try:
            # 1. 초기 성능 평가
            initial_performance = await self._evaluate_prompt_performance(
                current_prompt, context, target_domain
            )
            logger.info(f"초기 성능: {initial_performance:.3f}")
            
            # 2. LLM-AutoDiff 최적화 루프
            for step in range(self.max_steps):
                logger.info(f"최적화 스텝 {step + 1}/{self.max_steps}")
                
                # 텍스트 기울기 계산 (LLM-AutoDiff)
                gradient = await self._compute_textual_gradient(
                    current_prompt, context, target_domain
                )
                
                # 프롬프트 업데이트
                updated_prompt = await self._apply_gradient(
                    current_prompt, gradient, mode
                )
                
                # 성능 평가
                performance = await self._evaluate_prompt_performance(
                    updated_prompt, context, target_domain
                )
                
                logger.info(f"스텝 {step + 1} 성능: {performance:.3f}")
                
                # 최적 프롬프트 업데이트
                if performance > best_performance:
                    best_performance = performance
                    best_prompt = updated_prompt
                    current_prompt = updated_prompt
                    
                    # 성능 임계값 달성 시 조기 종료
                    if performance >= self.performance_threshold:
                        logger.info(f"성능 임계값 달성: {performance:.3f}")
                        break
                else:
                    # 성능 개선이 없으면 다른 방향 시도
                    current_prompt = await self._diversify_prompt(
                        current_prompt, gradient
                    )
            
            # 3. 최적화 결과 생성
            performance_gain = (best_performance - initial_performance) * 100
            confidence_score = min(best_performance * 1.2, 1.0)  # 신뢰도 계산
            
            result = OptimizationResult(
                original_prompt=original_prompt,
                optimized_prompt=best_prompt,
                performance_gain=performance_gain,
                optimization_steps=step + 1,
                confidence_score=confidence_score,
                mode=mode
            )
            
            # 히스토리에 저장
            self.optimization_history.append(result)
            
            logger.info(
                f"최적화 완료! 성능 향상: +{performance_gain:.1f}% "
                f"({step + 1} 스텝)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"프롬프트 최적화 실패: {e}")
            # 실패 시 원본 프롬프트 반환
            return OptimizationResult(
                original_prompt=original_prompt,
                optimized_prompt=original_prompt,
                performance_gain=0.0,
                optimization_steps=0,
                confidence_score=0.0,
                mode=mode
            )
    
    async def _compute_textual_gradient(
        self,
        prompt: str,
        context: Dict[str, Any],
        target_domain: str
    ) -> str:
        """
        LLM-AutoDiff: 텍스트 기울기 계산
        
        frozen backward engine LLM을 사용하여 텍스트 기울기 생성.
        현재 프롬프트의 문제점과 개선 방향을 분석.
        """
        
        gradient_prompt = f"""당신은 뿌리산업 전문 프롬프트 최적화 전문가입니다.

현재 프롬프트를 분석하고 개선 방향을 제시해주세요:

**타겟 도메인**: {target_domain}
**사용자 컨텍스트**: {json.dumps(context.get('user_profile', {}), ensure_ascii=False)}
**성공 패턴**: {json.dumps(context.get('success_patterns', []), ensure_ascii=False)}

**현재 프롬프트**:
```
{prompt}
```

다음 관점에서 분석하고 개선 방향을 제시하세요:

1. **한국어 품질**: 자연스러운 한국어 표현인가?
2. **기술 정확성**: 뿌리산업 기술 용어가 정확한가?
3. **사용자 맞춤성**: 사용자 수준과 관심사에 적합한가?
4. **응답 품질**: 원하는 응답을 유도하는가?
5. **구조 최적화**: 프롬프트 구조가 효율적인가?

**개선 방향** (JSON 형식):
{{
    "문제점": ["구체적 문제점 나열"],
    "개선_방향": ["구체적 개선 방향"],
    "추천_수정": "구체적인 수정 제안",
    "예상_효과": "개선 후 예상 효과"
}}"""

        try:
            response = await self.ollama_client.generate({
                "model": self.optimizer_model,
                "prompt": gradient_prompt,
                "format": "json",
                "options": {
                    "temperature": 0.3,  # 안정적인 분석을 위해 낮은 온도
                    "top_p": 0.9,
                    "num_predict": 800
                }
            })
            
            gradient_data = json.loads(response.get('response', '{}'))
            return gradient_data.get('추천_수정', prompt)
            
        except Exception as e:
            logger.error(f"텍스트 기울기 계산 실패: {e}")
            return prompt  # 실패 시 원본 반환
    
    async def _apply_gradient(
        self,
        prompt: str,
        gradient: str,
        mode: OptimizationMode
    ) -> str:
        """
        텍스트 기울기를 적용하여 프롬프트 업데이트
        """
        
        if mode == OptimizationMode.PROMPT_ONLY:
            update_instruction = "프롬프트 지시문만 개선"
        elif mode == OptimizationMode.DEMOS_ONLY:
            update_instruction = "퓨샷 예시만 개선"
        elif mode == OptimizationMode.TEMPLATE:
            update_instruction = "템플릿 구조만 개선"
        else:  # COMBINED
            update_instruction = "프롬프트 전체를 종합적으로 개선"
        
        update_prompt = f"""현재 프롬프트를 개선 방향에 따라 업데이트해주세요.

**업데이트 모드**: {update_instruction}
**개선 방향**: {gradient}

**현재 프롬프트**:
```
{prompt}
```

**요구사항**:
1. 뿌리산업 전문성 강화
2. 한국어 자연스러움 개선
3. 사용자 친화적 표현
4. 구체적이고 실행 가능한 지시

업데이트된 프롬프트만 출력하세요 (추가 설명 없이):"""

        try:
            response = await self.ollama_client.generate({
                "model": self.optimizer_model,
                "prompt": update_prompt,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "num_predict": 1000
                }
            })
            
            updated_prompt = response.get('response', '').strip()
            return updated_prompt if updated_prompt else prompt
            
        except Exception as e:
            logger.error(f"기울기 적용 실패: {e}")
            return prompt
    
    async def _evaluate_prompt_performance(
        self,
        prompt: str,
        context: Dict[str, Any],
        target_domain: str
    ) -> float:
        """
        프롬프트 성능 평가
        
        Returns:
            float: 성능 점수 (0.0 ~ 1.0)
        """
        
        evaluation_prompt = f"""다음 프롬프트의 품질을 평가해주세요:

**평가 프롬프트**:
```
{prompt}
```

**평가 기준**:
1. 한국어 품질 (0-10점)
2. 뿌리산업 전문성 (0-10점) 
3. 명확성 및 구체성 (0-10점)
4. 사용자 친화성 (0-10점)
5. 실행 가능성 (0-10점)

**평가 결과** (JSON 형식):
{{
    "한국어_품질": 점수,
    "전문성": 점수,
    "명확성": 점수,
    "친화성": 점수,
    "실행가능성": 점수,
    "총점": 평균점수,
    "평가_근거": "평가 근거 설명"
}}"""

        try:
            response = await self.ollama_client.generate({
                "model": self.optimizer_model,
                "prompt": evaluation_prompt,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            })
            
            eval_data = json.loads(response.get('response', '{}'))
            total_score = eval_data.get('총점', 5.0)
            
            # 0-10 점수를 0-1 범위로 정규화
            normalized_score = total_score / 10.0
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.error(f"성능 평가 실패: {e}")
            return 0.5  # 기본값
    
    async def _diversify_prompt(self, prompt: str, gradient: str) -> str:
        """
        프롬프트 다양화 (최적화 정체 시)
        """
        diversify_prompt = f"""다음 프롬프트를 다른 접근 방식으로 재작성해주세요:

원본: {prompt}
개선방향: {gradient}

다양한 각도에서 접근하되 핵심 의도는 유지하세요.
창의적이고 혁신적인 표현을 사용하세요.

재작성된 프롬프트:"""

        try:
            response = await self.ollama_client.generate({
                "model": self.optimizer_model,
                "prompt": diversify_prompt,
                "options": {
                    "temperature": 0.7,  # 창의성을 위해 높은 온도
                    "top_p": 0.9,
                    "num_predict": 800
                }
            })
            
            return response.get('response', prompt).strip()
            
        except Exception as e:
            logger.error(f"프롬프트 다양화 실패: {e}")
            return prompt
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        gains = [r.performance_gain for r in self.optimization_history]
        steps = [r.optimization_steps for r in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "avg_performance_gain": sum(gains) / len(gains),
            "max_performance_gain": max(gains),
            "avg_optimization_steps": sum(steps) / len(steps),
            "success_rate": len([g for g in gains if g > 0]) / len(gains),
            "last_optimization": self.optimization_history[-1].timestamp.isoformat()
        }
    
    async def get_personalized_prompt(
        self,
        base_prompt: str,
        user_profile: Dict[str, Any]
    ) -> str:
        """
        사용자 프로필 기반 개인화된 프롬프트 생성
        """
        personalization_prompt = f"""사용자 프로필에 맞춰 프롬프트를 개인화하세요:

**사용자 프로필**:
- 관심 분야: {user_profile.get('interests', '일반')}
- 전문 수준: {user_profile.get('expertise_level', '중급')}
- 선호 스타일: {user_profile.get('preferred_style', '상세')}
- 이전 성공 패턴: {user_profile.get('success_patterns', [])}

**기본 프롬프트**:
{base_prompt}

사용자에게 최적화된 프롬프트로 개선하세요:"""

        try:
            response = await self.ollama_client.generate({
                "model": self.optimizer_model,
                "prompt": personalization_prompt,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            })
            
            return response.get('response', base_prompt).strip()
            
        except Exception as e:
            logger.error(f"개인화 실패: {e}")
            return base_prompt