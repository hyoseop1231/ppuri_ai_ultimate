"""
Performance Tracker - 성능 추적 및 학습 시스템

실시간 성능 모니터링, 사용자 피드백 수집, 자동 학습을 통해
시스템 성능을 지속적으로 개선하는 시스템.

Features:
- 실시간 성능 메트릭 수집
- 사용자 피드백 기반 학습
- A/B 테스트 자동 실행
- 성능 예측 및 조기 경고
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""
    ACCURACY = "accuracy"              # 정확도
    RESPONSE_TIME = "response_time"    # 응답 시간
    USER_SATISFACTION = "satisfaction" # 사용자 만족도
    TOKEN_EFFICIENCY = "token_efficiency" # 토큰 효율성
    RELEVANCE = "relevance"           # 관련성
    COMPLETENESS = "completeness"     # 완성도


@dataclass
class PerformanceMetric:
    """성능 메트릭"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    session_id: str
    user_id: Optional[str] = None
    prompt_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """사용자 피드백"""
    feedback_id: str
    user_id: str
    session_id: str
    prompt_id: str
    rating: float  # 1-5 점수
    feedback_type: str  # thumbs_up, thumbs_down, detailed
    comments: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B 테스트 결과"""
    test_id: str
    variant_a: str
    variant_b: str
    winner: str
    confidence: float
    sample_size: int
    metrics: Dict[str, float]
    duration: timedelta
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """
    성능 추적 및 학습 시스템
    
    실시간으로 시스템 성능을 모니터링하고 
    사용자 피드백을 수집하여 자동 학습.
    """
    
    def __init__(
        self,
        window_size: int = 1000,           # 메트릭 윈도우 크기
        alert_threshold: float = 0.1,      # 성능 하락 경고 임계값
        learning_rate: float = 0.01,       # 학습률
        ab_test_sample_size: int = 100     # A/B 테스트 최소 샘플 크기
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.learning_rate = learning_rate
        self.ab_test_sample_size = ab_test_sample_size
        
        # 성능 데이터 저장소
        self.metrics_history: Dict[MetricType, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.feedback_history: List[UserFeedback] = []
        self.session_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        
        # A/B 테스트 관리
        self.active_ab_tests: Dict[str, Dict] = {}
        self.ab_test_results: List[ABTestResult] = []
        
        # 성능 기준선
        self.baselines: Dict[MetricType, float] = {
            MetricType.ACCURACY: 0.8,
            MetricType.RESPONSE_TIME: 2.0,  # 초
            MetricType.USER_SATISFACTION: 4.0,  # 5점 만점
            MetricType.TOKEN_EFFICIENCY: 0.7,
            MetricType.RELEVANCE: 0.8,
            MetricType.COMPLETENESS: 0.8
        }
        
        # 학습된 패턴
        self.learned_patterns: Dict[str, Any] = {}
        self.performance_predictors: Dict[str, Any] = {}
    
    async def track_interaction(
        self,
        session_id: str,
        prompt: str,
        response: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        사용자 상호작용 추적
        
        Args:
            session_id: 세션 ID
            prompt: 사용된 프롬프트
            response: 생성된 응답
            user_id: 사용자 ID
            context: 추가 컨텍스트
            
        Returns:
            Dict[str, float]: 계산된 메트릭들
        """
        start_time = time.time()
        
        try:
            # 1. 기본 메트릭 계산
            metrics = await self._calculate_interaction_metrics(
                prompt, response, context or {}
            )
            
            # 2. 메트릭 기록
            timestamp = datetime.now()
            for metric_type, value in metrics.items():
                metric = PerformanceMetric(
                    metric_type=MetricType(metric_type),
                    value=value,
                    timestamp=timestamp,
                    session_id=session_id,
                    user_id=user_id,
                    context=context or {}
                )
                
                self.metrics_history[MetricType(metric_type)].append(metric)
                self.session_metrics[session_id].append(metric)
            
            # 3. 성능 이상 감지
            await self._detect_performance_anomalies(metrics)
            
            # 4. 패턴 학습 업데이트
            await self._update_learned_patterns(session_id, metrics, context or {})
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            logger.debug(f"상호작용 추적 완료: {processing_time:.3f}초")
            
            return metrics
            
        except Exception as e:
            logger.error(f"상호작용 추적 실패: {e}")
            return {}
    
    async def _calculate_interaction_metrics(
        self,
        prompt: str,
        response: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """상호작용 메트릭 계산"""
        
        metrics = {}
        
        try:
            # 응답 시간 (컨텍스트에서 추출)
            if 'response_time' in context:
                metrics['response_time'] = context['response_time']
            
            # 토큰 효율성 계산
            prompt_tokens = len(prompt.split())
            response_tokens = len(response.split())
            if prompt_tokens > 0:
                token_efficiency = response_tokens / (prompt_tokens + response_tokens)
                metrics['token_efficiency'] = token_efficiency
            
            # 응답 완성도 (간단한 휴리스틱)
            completeness = min(1.0, len(response) / 500)  # 500자 기준
            metrics['completeness'] = completeness
            
            # 뿌리산업 관련성 (키워드 기반)
            industry_keywords = [
                "주조", "금형", "소성가공", "용접", "표면처리", "열처리",
                "기술", "공정", "소재", "품질", "생산", "제조"
            ]
            
            response_lower = response.lower()
            keyword_matches = sum(
                1 for keyword in industry_keywords 
                if keyword in response_lower
            )
            relevance = min(1.0, keyword_matches / 3)  # 3개 이상이면 최대점
            metrics['relevance'] = relevance
            
        except Exception as e:
            logger.error(f"메트릭 계산 실패: {e}")
        
        return metrics
    
    async def record_user_feedback(
        self,
        session_id: str,
        prompt_id: str,
        rating: float,
        feedback_type: str,
        user_id: Optional[str] = None,
        comments: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        사용자 피드백 기록
        
        Returns:
            str: 피드백 ID
        """
        
        feedback_id = f"fb_{int(time.time())}_{len(self.feedback_history)}"
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id or "anonymous",
            session_id=session_id,
            prompt_id=prompt_id,
            rating=rating,
            feedback_type=feedback_type,
            comments=comments,
            context=context or {}
        )
        
        self.feedback_history.append(feedback)
        
        # 사용자 만족도 메트릭 업데이트
        satisfaction_metric = PerformanceMetric(
            metric_type=MetricType.USER_SATISFACTION,
            value=rating,
            timestamp=feedback.timestamp,
            session_id=session_id,
            user_id=user_id,
            prompt_id=prompt_id,
            context=context or {}
        )
        
        self.metrics_history[MetricType.USER_SATISFACTION].append(satisfaction_metric)
        
        # 피드백 기반 학습
        await self._learn_from_feedback(feedback)
        
        logger.info(f"사용자 피드백 기록: {rating}/5.0 ({feedback_type})")
        return feedback_id
    
    async def _learn_from_feedback(self, feedback: UserFeedback):
        """피드백 기반 학습"""
        
        try:
            # 1. 긍정/부정 패턴 학습
            pattern_key = f"feedback_{feedback.feedback_type}"
            
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "positive_contexts": [],
                    "negative_contexts": [],
                    "avg_rating": 0.0,
                    "sample_count": 0
                }
            
            pattern = self.learned_patterns[pattern_key]
            
            # 평균 평점 업데이트
            pattern["sample_count"] += 1
            pattern["avg_rating"] = (
                (pattern["avg_rating"] * (pattern["sample_count"] - 1) + feedback.rating) /
                pattern["sample_count"]
            )
            
            # 컨텍스트 분류
            if feedback.rating >= 4.0:  # 긍정적 피드백
                pattern["positive_contexts"].append(feedback.context)
            elif feedback.rating <= 2.0:  # 부정적 피드백
                pattern["negative_contexts"].append(feedback.context)
            
            # 2. 세션별 성능 패턴 학습
            session_metrics = self.session_metrics.get(feedback.session_id, [])
            if session_metrics:
                session_pattern = {
                    "feedback_rating": feedback.rating,
                    "avg_response_time": statistics.mean([
                        m.value for m in session_metrics 
                        if m.metric_type == MetricType.RESPONSE_TIME
                    ]) if any(m.metric_type == MetricType.RESPONSE_TIME for m in session_metrics) else 0,
                    "avg_relevance": statistics.mean([
                        m.value for m in session_metrics 
                        if m.metric_type == MetricType.RELEVANCE
                    ]) if any(m.metric_type == MetricType.RELEVANCE for m in session_metrics) else 0
                }
                
                # 성능 예측 모델 업데이트
                await self._update_performance_predictor(session_pattern)
            
        except Exception as e:
            logger.error(f"피드백 학습 실패: {e}")
    
    async def _update_performance_predictor(self, session_pattern: Dict[str, float]):
        """성능 예측 모델 업데이트"""
        
        try:
            # 간단한 선형 회귀 모델 업데이트
            if "satisfaction_predictor" not in self.performance_predictors:
                self.performance_predictors["satisfaction_predictor"] = {
                    "response_time_weight": 0.0,
                    "relevance_weight": 0.0,
                    "bias": 0.0,
                    "samples": []
                }
            
            predictor = self.performance_predictors["satisfaction_predictor"]
            predictor["samples"].append(session_pattern)
            
            # 최근 100개 샘플만 유지
            if len(predictor["samples"]) > 100:
                predictor["samples"] = predictor["samples"][-100:]
            
            # 가중치 업데이트 (간단한 경사하강법)
            if len(predictor["samples"]) >= 10:
                await self._update_predictor_weights(predictor)
            
        except Exception as e:
            logger.error(f"성능 예측 모델 업데이트 실패: {e}")
    
    async def _update_predictor_weights(self, predictor: Dict[str, Any]):
        """예측 모델 가중치 업데이트"""
        
        samples = predictor["samples"]
        
        for sample in samples[-10:]:  # 최근 10개 샘플로 학습
            # 현재 예측
            predicted = (
                predictor["response_time_weight"] * sample["avg_response_time"] +
                predictor["relevance_weight"] * sample["avg_relevance"] +
                predictor["bias"]
            )
            
            # 실제 값과의 오차
            error = sample["feedback_rating"] - predicted
            
            # 가중치 업데이트
            predictor["response_time_weight"] += self.learning_rate * error * sample["avg_response_time"]
            predictor["relevance_weight"] += self.learning_rate * error * sample["avg_relevance"]
            predictor["bias"] += self.learning_rate * error
    
    async def _detect_performance_anomalies(self, current_metrics: Dict[str, float]):
        """성능 이상 감지"""
        
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            try:
                metric_type = MetricType(metric_name)
                baseline = self.baselines.get(metric_type)
                
                if baseline is None:
                    continue
                
                # 이상 감지 로직
                if metric_type == MetricType.RESPONSE_TIME:
                    # 응답 시간이 기준선의 2배 이상
                    if current_value > baseline * 2:
                        anomalies.append(f"응답 시간 이상: {current_value:.2f}초 (기준: {baseline:.2f}초)")
                
                elif metric_type in [MetricType.ACCURACY, MetricType.RELEVANCE, MetricType.COMPLETENESS]:
                    # 품질 메트릭이 기준선 - 임계값 미만
                    if current_value < baseline - self.alert_threshold:
                        anomalies.append(f"{metric_name} 품질 하락: {current_value:.3f} (기준: {baseline:.3f})")
                
                elif metric_type == MetricType.USER_SATISFACTION:
                    # 사용자 만족도가 기준선 - 임계값 미만
                    if current_value < baseline - 1.0:  # 1점 이상 하락
                        anomalies.append(f"사용자 만족도 하락: {current_value:.1f}/5.0 (기준: {baseline:.1f}/5.0)")
                
            except Exception as e:
                logger.error(f"이상 감지 실패 ({metric_name}): {e}")
        
        # 이상 감지 시 로그
        if anomalies:
            logger.warning(f"성능 이상 감지: {', '.join(anomalies)}")
            
            # 자동 복구 액션 트리거 (필요시)
            await self._trigger_recovery_actions(anomalies)
    
    async def _trigger_recovery_actions(self, anomalies: List[str]):
        """자동 복구 액션 트리거"""
        
        # 예시 복구 액션들
        recovery_actions = []
        
        for anomaly in anomalies:
            if "응답 시간" in anomaly:
                recovery_actions.append("프롬프트 최적화 강화")
                recovery_actions.append("캐시 정리")
            
            elif "품질 하락" in anomaly:
                recovery_actions.append("프롬프트 재진화")
                recovery_actions.append("모델 파라미터 조정")
            
            elif "만족도 하락" in anomaly:
                recovery_actions.append("사용자 피드백 분석")
                recovery_actions.append("개인화 강화")
        
        if recovery_actions:
            logger.info(f"복구 액션 실행: {', '.join(set(recovery_actions))}")
    
    async def start_ab_test(
        self,
        test_name: str,
        variant_a: str,
        variant_b: str,
        target_metric: MetricType = MetricType.USER_SATISFACTION
    ) -> str:
        """A/B 테스트 시작"""
        
        test_id = f"ab_{test_name}_{int(time.time())}"
        
        self.active_ab_tests[test_id] = {
            "test_name": test_name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "target_metric": target_metric,
            "start_time": datetime.now(),
            "samples_a": [],
            "samples_b": [],
            "current_variant": "a"  # 교대로 할당
        }
        
        logger.info(f"A/B 테스트 시작: {test_name} (ID: {test_id})")
        return test_id
    
    async def get_ab_test_variant(self, test_id: str) -> Optional[str]:
        """A/B 테스트 변형 선택"""
        
        if test_id not in self.active_ab_tests:
            return None
        
        test = self.active_ab_tests[test_id]
        
        # 교대로 변형 할당
        if test["current_variant"] == "a":
            test["current_variant"] = "b"
            return test["variant_a"]
        else:
            test["current_variant"] = "a"
            return test["variant_b"]
    
    async def record_ab_test_result(
        self,
        test_id: str,
        variant: str,
        metric_value: float
    ):
        """A/B 테스트 결과 기록"""
        
        if test_id not in self.active_ab_tests:
            return
        
        test = self.active_ab_tests[test_id]
        
        if variant == test["variant_a"]:
            test["samples_a"].append(metric_value)
        elif variant == test["variant_b"]:
            test["samples_b"].append(metric_value)
        
        # 충분한 샘플이 모이면 테스트 완료
        if (len(test["samples_a"]) >= self.ab_test_sample_size and 
            len(test["samples_b"]) >= self.ab_test_sample_size):
            
            await self._complete_ab_test(test_id)
    
    async def _complete_ab_test(self, test_id: str):
        """A/B 테스트 완료 및 분석"""
        
        test = self.active_ab_tests[test_id]
        
        samples_a = test["samples_a"]
        samples_b = test["samples_b"]
        
        # 기본 통계 계산
        mean_a = statistics.mean(samples_a)
        mean_b = statistics.mean(samples_b)
        
        # 간단한 통계적 유의성 검정 (t-test 근사)
        if len(samples_a) > 1 and len(samples_b) > 1:
            std_a = statistics.stdev(samples_a)
            std_b = statistics.stdev(samples_b)
            
            # 효과 크기 계산
            effect_size = abs(mean_a - mean_b) / ((std_a + std_b) / 2)
            confidence = min(0.95, effect_size * 0.3)  # 간단한 신뢰도 추정
        else:
            confidence = 0.5
        
        # 승자 결정
        winner = test["variant_a"] if mean_a > mean_b else test["variant_b"]
        
        # 결과 저장
        result = ABTestResult(
            test_id=test_id,
            variant_a=test["variant_a"],
            variant_b=test["variant_b"],
            winner=winner,
            confidence=confidence,
            sample_size=len(samples_a) + len(samples_b),
            metrics={
                "mean_a": mean_a,
                "mean_b": mean_b,
                "effect_size": abs(mean_a - mean_b)
            },
            duration=datetime.now() - test["start_time"]
        )
        
        self.ab_test_results.append(result)
        del self.active_ab_tests[test_id]
        
        logger.info(
            f"A/B 테스트 완료: {test['test_name']} - "
            f"승자: {winner} (신뢰도: {confidence:.2f})"
        )
    
    def get_performance_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """성능 요약 통계"""
        
        if time_window is None:
            time_window = timedelta(hours=24)  # 기본 24시간
        
        cutoff_time = datetime.now() - time_window
        summary = {}
        
        # 각 메트릭별 통계
        for metric_type, metrics in self.metrics_history.items():
            recent_metrics = [
                m for m in metrics 
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[metric_type.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "baseline": self.baselines.get(metric_type, 0.0)
                }
                
                if len(values) > 1:
                    summary[metric_type.value]["std"] = statistics.stdev(values)
        
        # 전체 통계
        summary["overview"] = {
            "total_interactions": sum(
                s["count"] for s in summary.values() 
                if isinstance(s, dict) and "count" in s
            ),
            "active_ab_tests": len(self.active_ab_tests),
            "completed_ab_tests": len(self.ab_test_results),
            "learned_patterns": len(self.learned_patterns),
            "feedback_count": len([
                f for f in self.feedback_history 
                if f.timestamp >= cutoff_time
            ])
        }
        
        return summary
    
    async def predict_user_satisfaction(
        self,
        response_time: float,
        relevance: float
    ) -> float:
        """사용자 만족도 예측"""
        
        if "satisfaction_predictor" not in self.performance_predictors:
            return 3.5  # 기본값
        
        predictor = self.performance_predictors["satisfaction_predictor"]
        
        predicted = (
            predictor["response_time_weight"] * response_time +
            predictor["relevance_weight"] * relevance +
            predictor["bias"]
        )
        
        # 1-5 범위로 클램핑
        return max(1.0, min(5.0, predicted))
    
    def get_optimization_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        
        recommendations = []
        
        # 최근 성능 데이터 분석
        recent_summary = self.get_performance_summary(timedelta(hours=6))
        
        for metric_name, stats in recent_summary.items():
            if metric_name == "overview" or not isinstance(stats, dict):
                continue
            
            baseline = stats.get("baseline", 0.0)
            current_mean = stats.get("mean", 0.0)
            
            # 메트릭별 권장사항
            if metric_name == "response_time" and current_mean > baseline * 1.5:
                recommendations.append("응답 시간 최적화: 프롬프트 길이 단축 또는 모델 파라미터 조정 필요")
            
            elif metric_name == "user_satisfaction" and current_mean < baseline - 0.5:
                recommendations.append("사용자 만족도 개선: 개인화 강화 또는 응답 품질 향상 필요")
            
            elif metric_name == "relevance" and current_mean < baseline - 0.2:
                recommendations.append("관련성 개선: 도메인 특화 키워드 강화 또는 컨텍스트 이해 향상 필요")
        
        # 패턴 기반 권장사항
        if len(self.learned_patterns) > 0:
            avg_satisfaction = statistics.mean([
                p.get("avg_rating", 3.0) 
                for p in self.learned_patterns.values()
                if "avg_rating" in p
            ])
            
            if avg_satisfaction < 3.5:
                recommendations.append("학습 패턴 분석: 부정적 피드백 패턴 개선 필요")
        
        return recommendations if recommendations else ["현재 성능이 양호합니다."]