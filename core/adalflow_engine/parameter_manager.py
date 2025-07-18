"""
Parameter Manager - PyTorch 스타일 파라미터 관리 시스템

AdalFlow LLM-AutoDiff에서 사용되는 모든 파라미터를 관리하고
자동 하이퍼파라미터 튜닝과 학습률 스케줄링을 제공하는 시스템.

Features:
- PyTorch Parameter 스타일 관리
- 자동 하이퍼파라미터 최적화
- 적응형 학습률 스케줄링
- 모델 상태 지속성 관리
"""

import asyncio
import json
import logging
import math
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """옵티마이저 타입"""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


class SchedulerType(Enum):
    """스케줄러 타입"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"


@dataclass
class ParameterConfig:
    """파라미터 설정"""
    name: str
    value: Any
    learnable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step_size: Optional[float] = None
    dtype: str = "float"
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class OptimizerState:
    """옵티마이저 상태"""
    optimizer_type: OptimizerType
    learning_rate: float
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    step_count: int = 0
    
    # Adam 상태
    m: Dict[str, float] = field(default_factory=dict)  # 1차 모멘트
    v: Dict[str, float] = field(default_factory=dict)  # 2차 모멘트
    
    # SGD 상태
    velocity: Dict[str, float] = field(default_factory=dict)


@dataclass
class SchedulerState:
    """스케줄러 상태"""
    scheduler_type: SchedulerType
    base_lr: float
    current_lr: float
    step_count: int = 0
    
    # 스케줄러별 설정
    gamma: float = 0.95  # Exponential, Step
    step_size: int = 1000  # Step
    T_max: int = 10000  # Cosine
    eta_min: float = 1e-6  # Cosine
    last_epoch: int = -1


class ParameterManager:
    """
    PyTorch 스타일 파라미터 관리자
    
    LLM-AutoDiff와 프롬프트 진화 시스템의 모든 파라미터를 
    체계적으로 관리하고 자동 최적화를 수행.
    """
    
    def __init__(
        self,
        save_dir: str = "/app/data/parameters",
        auto_save: bool = True,
        save_interval: int = 100  # 스텝 간격
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # 파라미터 저장소
        self.parameters: Dict[str, ParameterConfig] = {}
        self.parameter_groups: Dict[str, List[str]] = {}
        
        # 옵티마이저 및 스케줄러
        self.optimizer_state: OptimizerState = OptimizerState(
            optimizer_type=OptimizerType.ADAM,
            learning_rate=0.01
        )
        self.scheduler_state: SchedulerState = SchedulerState(
            scheduler_type=SchedulerType.EXPONENTIAL,
            base_lr=0.01,
            current_lr=0.01
        )
        
        # 최적화 히스토리
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        
        # 뿌리산업 특화 파라미터 템플릿
        self._initialize_industry_parameters()
    
    def _initialize_industry_parameters(self):
        """뿌리산업 특화 파라미터 초기화"""
        
        # 프롬프트 최적화 파라미터
        self.register_parameter(ParameterConfig(
            name="prompt_temperature",
            value=0.3,
            min_value=0.1,
            max_value=1.0,
            step_size=0.05,
            description="프롬프트 생성 온도",
            tags=["prompt", "generation"]
        ))
        
        self.register_parameter(ParameterConfig(
            name="prompt_top_p",
            value=0.9,
            min_value=0.1,
            max_value=1.0,
            step_size=0.05,
            description="Top-p 샘플링 파라미터",
            tags=["prompt", "sampling"]
        ))
        
        # 진화 알고리즘 파라미터
        self.register_parameter(ParameterConfig(
            name="mutation_rate",
            value=0.3,
            min_value=0.1,
            max_value=0.8,
            step_size=0.05,
            description="돌연변이 확률",
            tags=["evolution", "genetic"]
        ))
        
        self.register_parameter(ParameterConfig(
            name="crossover_rate",
            value=0.7,
            min_value=0.3,
            max_value=0.9,
            step_size=0.05,
            description="교배 확률",
            tags=["evolution", "genetic"]
        ))
        
        self.register_parameter(ParameterConfig(
            name="elite_ratio",
            value=0.2,
            min_value=0.1,
            max_value=0.5,
            step_size=0.05,
            description="엘리트 비율",
            tags=["evolution", "selection"]
        ))
        
        # 성능 추적 파라미터
        self.register_parameter(ParameterConfig(
            name="learning_rate",
            value=0.01,
            min_value=0.001,
            max_value=0.1,
            step_size=0.001,
            description="메인 학습률",
            tags=["optimization", "learning"]
        ))
        
        # 뿌리산업 도메인 가중치
        industry_domains = ["주조", "금형", "소성가공", "용접", "표면처리", "열처리"]
        for domain in industry_domains:
            self.register_parameter(ParameterConfig(
                name=f"domain_weight_{domain}",
                value=1.0,
                min_value=0.1,
                max_value=2.0,
                step_size=0.1,
                description=f"{domain} 도메인 가중치",
                tags=["domain", "weight", domain]
            ))
        
        # 파라미터 그룹 정의
        self.create_parameter_group("prompt_generation", [
            "prompt_temperature", "prompt_top_p"
        ])
        
        self.create_parameter_group("evolution", [
            "mutation_rate", "crossover_rate", "elite_ratio"
        ])
        
        self.create_parameter_group("domain_weights", [
            f"domain_weight_{d}" for d in industry_domains
        ])
        
        logger.info("뿌리산업 특화 파라미터 초기화 완료")
    
    def register_parameter(self, config: ParameterConfig):
        """파라미터 등록"""
        self.parameters[config.name] = config
        logger.debug(f"파라미터 등록: {config.name} = {config.value}")
    
    def create_parameter_group(self, group_name: str, parameter_names: List[str]):
        """파라미터 그룹 생성"""
        # 유효한 파라미터만 그룹에 추가
        valid_params = [
            name for name in parameter_names 
            if name in self.parameters
        ]
        
        if valid_params:
            self.parameter_groups[group_name] = valid_params
            logger.debug(f"파라미터 그룹 생성: {group_name} ({len(valid_params)}개)")
        else:
            logger.warning(f"파라미터 그룹 {group_name}: 유효한 파라미터가 없음")
    
    def get_parameter(self, name: str) -> Any:
        """파라미터 값 조회"""
        if name not in self.parameters:
            raise KeyError(f"파라미터 '{name}' 없음")
        return self.parameters[name].value
    
    def set_parameter(self, name: str, value: Any):
        """파라미터 값 설정"""
        if name not in self.parameters:
            raise KeyError(f"파라미터 '{name}' 없음")
        
        config = self.parameters[name]
        
        # 값 범위 검증
        if config.min_value is not None and value < config.min_value:
            value = config.min_value
        if config.max_value is not None and value > config.max_value:
            value = config.max_value
        
        config.value = value
        logger.debug(f"파라미터 업데이트: {name} = {value}")
    
    def get_parameter_group(self, group_name: str) -> Dict[str, Any]:
        """파라미터 그룹 값들 조회"""
        if group_name not in self.parameter_groups:
            raise KeyError(f"파라미터 그룹 '{group_name}' 없음")
        
        return {
            name: self.get_parameter(name)
            for name in self.parameter_groups[group_name]
        }
    
    def configure_optimizer(
        self,
        optimizer_type: OptimizerType = OptimizerType.ADAM,
        learning_rate: float = 0.01,
        **kwargs
    ):
        """옵티마이저 설정"""
        self.optimizer_state = OptimizerState(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            **kwargs
        )
        
        # 스케줄러 학습률도 업데이트
        self.scheduler_state.base_lr = learning_rate
        self.scheduler_state.current_lr = learning_rate
        
        logger.info(f"옵티마이저 설정: {optimizer_type.value}, lr={learning_rate}")
    
    def configure_scheduler(
        self,
        scheduler_type: SchedulerType = SchedulerType.EXPONENTIAL,
        **kwargs
    ):
        """스케줄러 설정"""
        self.scheduler_state.scheduler_type = scheduler_type
        
        # 스케줄러별 파라미터 업데이트
        for key, value in kwargs.items():
            if hasattr(self.scheduler_state, key):
                setattr(self.scheduler_state, key, value)
        
        logger.info(f"스케줄러 설정: {scheduler_type.value}")
    
    async def compute_gradients(
        self,
        performance_delta: float,
        parameter_group: Optional[str] = None
    ) -> Dict[str, float]:
        """
        성능 변화를 기반으로 파라미터 기울기 계산
        
        Args:
            performance_delta: 성능 변화량
            parameter_group: 대상 파라미터 그룹 (None이면 모든 파라미터)
            
        Returns:
            Dict[str, float]: 파라미터별 기울기
        """
        gradients = {}
        
        # 대상 파라미터 결정
        if parameter_group and parameter_group in self.parameter_groups:
            target_params = self.parameter_groups[parameter_group]
        else:
            target_params = [
                name for name, config in self.parameters.items()
                if config.learnable
            ]
        
        # 단순한 기울기 추정 (finite difference)
        for param_name in target_params:
            config = self.parameters[param_name]
            
            if config.step_size is None:
                continue
            
            # 기울기 = 성능변화 / 파라미터변화 * 방향성
            # 긍정적 성능 변화 시 같은 방향으로, 부정적 변화 시 반대 방향으로
            gradient = performance_delta / config.step_size
            
            # 파라미터 범위 고려
            current_value = config.value
            if config.min_value is not None and current_value <= config.min_value:
                gradient = max(0, gradient)  # 최소값에서는 증가 방향만
            if config.max_value is not None and current_value >= config.max_value:
                gradient = min(0, gradient)  # 최대값에서는 감소 방향만
            
            gradients[param_name] = gradient
        
        logger.debug(f"기울기 계산 완료: {len(gradients)}개 파라미터")
        return gradients
    
    async def step(
        self,
        gradients: Dict[str, float],
        performance_score: float
    ):
        """
        옵티마이저 스텝 실행
        
        Args:
            gradients: 파라미터별 기울기
            performance_score: 현재 성능 점수
        """
        optimizer = self.optimizer_state
        optimizer.step_count += 1
        
        # 학습률 스케줄링
        await self._update_learning_rate()
        
        # 옵티마이저별 파라미터 업데이트
        if optimizer.optimizer_type == OptimizerType.ADAM:
            await self._adam_step(gradients)
        elif optimizer.optimizer_type == OptimizerType.SGD:
            await self._sgd_step(gradients)
        elif optimizer.optimizer_type == OptimizerType.RMSPROP:
            await self._rmsprop_step(gradients)
        
        # 성능 히스토리 업데이트
        self.performance_history.append(performance_score)
        
        # 최적화 히스토리 기록
        step_info = {
            "step": optimizer.step_count,
            "learning_rate": self.scheduler_state.current_lr,
            "performance": performance_score,
            "gradients": gradients,
            "timestamp": datetime.now().isoformat()
        }
        self.optimization_history.append(step_info)
        
        # 자동 저장
        if self.auto_save and optimizer.step_count % self.save_interval == 0:
            await self.save_state()
        
        logger.debug(f"옵티마이저 스텝 완료: {optimizer.step_count}")
    
    async def _adam_step(self, gradients: Dict[str, float]):
        """Adam 옵티마이저 스텝"""
        optimizer = self.optimizer_state
        lr = self.scheduler_state.current_lr
        
        for param_name, grad in gradients.items():
            if param_name not in self.parameters:
                continue
            
            # 모멘트 초기화
            if param_name not in optimizer.m:
                optimizer.m[param_name] = 0.0
            if param_name not in optimizer.v:
                optimizer.v[param_name] = 0.0
            
            # 모멘트 업데이트
            optimizer.m[param_name] = (
                optimizer.beta1 * optimizer.m[param_name] + 
                (1 - optimizer.beta1) * grad
            )
            optimizer.v[param_name] = (
                optimizer.beta2 * optimizer.v[param_name] + 
                (1 - optimizer.beta2) * grad * grad
            )
            
            # 편향 보정
            m_hat = optimizer.m[param_name] / (1 - optimizer.beta1 ** optimizer.step_count)
            v_hat = optimizer.v[param_name] / (1 - optimizer.beta2 ** optimizer.step_count)
            
            # 파라미터 업데이트
            param_update = lr * m_hat / (math.sqrt(v_hat) + optimizer.eps)
            
            config = self.parameters[param_name]
            new_value = config.value + param_update
            self.set_parameter(param_name, new_value)
    
    async def _sgd_step(self, gradients: Dict[str, float]):
        """SGD 옵티마이저 스텝"""
        optimizer = self.optimizer_state
        lr = self.scheduler_state.current_lr
        
        for param_name, grad in gradients.items():
            if param_name not in self.parameters:
                continue
            
            # 모멘텀 초기화
            if param_name not in optimizer.velocity:
                optimizer.velocity[param_name] = 0.0
            
            # 속도 업데이트 (모멘텀)
            optimizer.velocity[param_name] = (
                optimizer.momentum * optimizer.velocity[param_name] + grad
            )
            
            # 파라미터 업데이트
            param_update = lr * optimizer.velocity[param_name]
            
            config = self.parameters[param_name]
            new_value = config.value + param_update
            self.set_parameter(param_name, new_value)
    
    async def _rmsprop_step(self, gradients: Dict[str, float]):
        """RMSprop 옵티마이저 스텝"""
        optimizer = self.optimizer_state
        lr = self.scheduler_state.current_lr
        
        for param_name, grad in gradients.items():
            if param_name not in self.parameters:
                continue
            
            # 제곱 평균 초기화
            if param_name not in optimizer.v:
                optimizer.v[param_name] = 0.0
            
            # 제곱 평균 업데이트
            optimizer.v[param_name] = (
                0.9 * optimizer.v[param_name] + 0.1 * grad * grad
            )
            
            # 파라미터 업데이트
            param_update = lr * grad / (math.sqrt(optimizer.v[param_name]) + optimizer.eps)
            
            config = self.parameters[param_name]
            new_value = config.value + param_update
            self.set_parameter(param_name, new_value)
    
    async def _update_learning_rate(self):
        """학습률 스케줄링 업데이트"""
        scheduler = self.scheduler_state
        scheduler.step_count += 1
        
        if scheduler.scheduler_type == SchedulerType.CONSTANT:
            return  # 변화 없음
        
        elif scheduler.scheduler_type == SchedulerType.LINEAR:
            # 선형 감소
            decay_factor = max(0.1, 1.0 - scheduler.step_count / 10000)
            scheduler.current_lr = scheduler.base_lr * decay_factor
        
        elif scheduler.scheduler_type == SchedulerType.EXPONENTIAL:
            # 지수적 감소
            scheduler.current_lr = scheduler.base_lr * (scheduler.gamma ** scheduler.step_count)
        
        elif scheduler.scheduler_type == SchedulerType.COSINE:
            # 코사인 어닐링
            scheduler.current_lr = scheduler.eta_min + (scheduler.base_lr - scheduler.eta_min) * \
                (1 + math.cos(math.pi * scheduler.step_count / scheduler.T_max)) / 2
        
        elif scheduler.scheduler_type == SchedulerType.STEP:
            # 스텝 감소
            step_factor = scheduler.step_count // scheduler.step_size
            scheduler.current_lr = scheduler.base_lr * (scheduler.gamma ** step_factor)
        
        # 최소 학습률 제한
        scheduler.current_lr = max(scheduler.current_lr, 1e-6)
    
    async def auto_tune_hyperparameters(
        self,
        performance_history: List[float],
        target_performance: float = 0.9
    ) -> Dict[str, Any]:
        """
        자동 하이퍼파라미터 튜닝
        
        성능 히스토리를 분석하여 파라미터를 자동 조정
        """
        if len(performance_history) < 10:
            return {"message": "충분한 성능 데이터 필요 (최소 10개)"}
        
        current_performance = performance_history[-1]
        recent_trend = sum(performance_history[-5:]) / 5 - sum(performance_history[-10:-5]) / 5
        
        tuning_actions = []
        
        # 성능이 목표에 미달하는 경우
        if current_performance < target_performance:
            
            # 성능이 하락 추세인 경우
            if recent_trend < -0.01:
                # 학습률 감소
                current_lr = self.scheduler_state.current_lr
                new_lr = current_lr * 0.8
                self.scheduler_state.base_lr = new_lr
                self.scheduler_state.current_lr = new_lr
                tuning_actions.append(f"학습률 감소: {current_lr:.6f} → {new_lr:.6f}")
                
                # 돌연변이율 증가 (탐색 강화)
                mutation_rate = self.get_parameter("mutation_rate")
                new_mutation_rate = min(0.8, mutation_rate * 1.2)
                self.set_parameter("mutation_rate", new_mutation_rate)
                tuning_actions.append(f"돌연변이율 증가: {mutation_rate:.3f} → {new_mutation_rate:.3f}")
            
            # 성능이 정체된 경우
            elif abs(recent_trend) < 0.005:
                # 탐색 다양성 증가
                elite_ratio = self.get_parameter("elite_ratio")
                new_elite_ratio = max(0.1, elite_ratio * 0.9)
                self.set_parameter("elite_ratio", new_elite_ratio)
                tuning_actions.append(f"엘리트 비율 감소: {elite_ratio:.3f} → {new_elite_ratio:.3f}")
                
                # 온도 증가 (창의성 강화)
                temperature = self.get_parameter("prompt_temperature")
                new_temperature = min(1.0, temperature * 1.1)
                self.set_parameter("prompt_temperature", new_temperature)
                tuning_actions.append(f"온도 증가: {temperature:.3f} → {new_temperature:.3f}")
        
        # 성능이 양호한 경우 안정화
        else:
            # 온도 감소 (안정성 강화)
            temperature = self.get_parameter("prompt_temperature")
            new_temperature = max(0.1, temperature * 0.95)
            self.set_parameter("prompt_temperature", new_temperature)
            tuning_actions.append(f"온도 안정화: {temperature:.3f} → {new_temperature:.3f}")
        
        logger.info(f"자동 하이퍼파라미터 튜닝: {len(tuning_actions)}개 조정")
        
        return {
            "actions": tuning_actions,
            "current_performance": current_performance,
            "target_performance": target_performance,
            "trend": recent_trend
        }
    
    async def save_state(self, checkpoint_name: Optional[str] = None):
        """파라미터 상태 저장"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        save_path = self.save_dir / f"{checkpoint_name}.pkl"
        
        state = {
            "parameters": self.parameters,
            "parameter_groups": self.parameter_groups,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "optimization_history": self.optimization_history[-1000:],  # 최근 1000개만
            "performance_history": self.performance_history[-1000:],
            "save_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"파라미터 상태 저장: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")
            return None
    
    async def load_state(self, checkpoint_path: str):
        """파라미터 상태 로드"""
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            
            self.parameters = state["parameters"]
            self.parameter_groups = state["parameter_groups"]
            self.optimizer_state = state["optimizer_state"]
            self.scheduler_state = state["scheduler_state"]
            self.optimization_history = state["optimization_history"]
            self.performance_history = state["performance_history"]
            
            logger.info(f"파라미터 상태 로드: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"상태 로드 실패: {e}")
            return False
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """파라미터 요약 정보"""
        learnable_params = [
            name for name, config in self.parameters.items()
            if config.learnable
        ]
        
        return {
            "total_parameters": len(self.parameters),
            "learnable_parameters": len(learnable_params),
            "parameter_groups": len(self.parameter_groups),
            "optimizer": {
                "type": self.optimizer_state.optimizer_type.value,
                "learning_rate": self.scheduler_state.current_lr,
                "step_count": self.optimizer_state.step_count
            },
            "scheduler": {
                "type": self.scheduler_state.scheduler_type.value,
                "base_lr": self.scheduler_state.base_lr,
                "current_lr": self.scheduler_state.current_lr
            },
            "performance": {
                "current": self.performance_history[-1] if self.performance_history else 0.0,
                "best": max(self.performance_history) if self.performance_history else 0.0,
                "avg_recent": sum(self.performance_history[-10:]) / min(10, len(self.performance_history)) if self.performance_history else 0.0
            }
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        if not self.performance_history:
            recommendations.append("성능 데이터가 없습니다. 몇 번의 최적화를 실행해보세요.")
            return recommendations
        
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        # 성능 추세 분석
        if len(recent_performance) >= 5:
            early_avg = sum(recent_performance[:5]) / 5
            later_avg = sum(recent_performance[-5:]) / 5
            
            if later_avg < early_avg * 0.95:
                recommendations.append("성능이 하락 추세입니다. 학습률을 낮추거나 정규화를 강화하세요.")
            elif later_avg > early_avg * 1.05:
                recommendations.append("성능이 향상 중입니다. 현재 설정을 유지하세요.")
            else:
                recommendations.append("성능이 정체되었습니다. 파라미터 탐색 범위를 늘려보세요.")
        
        # 학습률 검사
        current_lr = self.scheduler_state.current_lr
        if current_lr > 0.05:
            recommendations.append("학습률이 높습니다. 안정성을 위해 낮춰보세요.")
        elif current_lr < 0.001:
            recommendations.append("학습률이 낮습니다. 학습 속도가 느릴 수 있습니다.")
        
        # 성능 기반 권장사항
        current_perf = self.performance_history[-1]
        if current_perf < 0.7:
            recommendations.append("전체적인 성능이 낮습니다. 하이퍼파라미터 자동 튜닝을 시도해보세요.")
        
        return recommendations if recommendations else ["현재 설정이 양호합니다."]