"""
Performance Dashboard - 실시간 성능 모니터링 대시보드

시스템의 모든 컴포넌트 성능을 실시간으로 모니터링하고
시각화하여 최적화 인사이트를 제공하는 대시보드.

Features:
- 실시간 성능 메트릭 수집
- 컴포넌트별 상세 분석
- 경고 및 알림 시스템
- 성능 트렌드 분석
- 자동 최적화 제안
- 한국어 최적화 리포트
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import json
import statistics
import uuid

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """메트릭 데이터 포인트"""
    timestamp: datetime
    value: float
    metric_name: str
    component: str
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """성능 경고"""
    id: str
    component: str
    metric_name: str
    severity: str  # low, medium, high, critical
    message: str
    threshold_value: float
    current_value: float
    created_at: datetime
    acknowledged: bool = False


@dataclass
class ComponentHealth:
    """컴포넌트 헬스 상태"""
    component: str
    status: str  # healthy, warning, error, unknown
    score: float  # 0.0 ~ 1.0
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemOptimization:
    """시스템 최적화 제안"""
    id: str
    category: str  # performance, memory, network, etc.
    priority: str  # low, medium, high
    title: str
    description: str
    impact_estimate: str
    implementation_effort: str
    created_at: datetime


class PerformanceDashboard:
    """
    실시간 성능 모니터링 대시보드
    
    시스템의 모든 컴포넌트를 실시간으로 모니터링하고
    성능 분석 및 최적화 제안을 제공하는 대시보드.
    """
    
    def __init__(
        self,
        ui_orchestrator,
        korean_optimizer=None,
        metric_retention_hours: int = 24
    ):
        self.ui_orchestrator = ui_orchestrator
        self.korean_optimizer = korean_optimizer
        self.metric_retention_hours = metric_retention_hours
        
        # 메트릭 저장소
        self.metrics_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)  # 최대 10,000개 데이터 포인트
        )
        
        # 컴포넌트 등록
        self.registered_components = self._initialize_components()
        
        # 경고 시스템
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # 헬스 체크
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # 최적화 제안
        self.optimization_suggestions: List[SystemOptimization] = []
        
        # 실시간 통계
        self.dashboard_stats = {
            "total_metrics_collected": 0,
            "active_components": 0,
            "total_alerts": 0,
            "avg_system_health": 0.0,
            "last_updated": datetime.now()
        }
        
        # 모니터링 태스크
        self.monitoring_tasks: List[asyncio.Task] = []
        
        logger.info("Performance Dashboard 초기화 완료")
    
    def _initialize_components(self) -> Dict[str, Dict[str, Any]]:
        """모니터링 컴포넌트 초기화"""
        
        return {
            "ui_orchestrator": {
                "name": "UI 오케스트레이터",
                "metrics": ["active_sessions", "websocket_connections", "response_time"],
                "critical": True
            },
            "conversational_engine": {
                "name": "대화 엔진",
                "metrics": ["response_time", "think_blocks_per_response", "success_rate"],
                "critical": True
            },
            "rag_orchestrator": {
                "name": "RAG 시스템",
                "metrics": ["search_time", "cache_hit_rate", "document_count"],
                "critical": True
            },
            "graph_manager": {
                "name": "지식 그래프",
                "metrics": ["query_time", "node_count", "relationship_count"],
                "critical": True
            },
            "mcp_orchestrator": {
                "name": "MCP 생태계",
                "metrics": ["tool_execution_time", "success_rate", "evolution_count"],
                "critical": False
            },
            "korean_optimizer": {
                "name": "한국어 최적화",
                "metrics": ["processing_time", "confidence_score", "industry_term_matches"],
                "critical": False
            },
            "knowledge_explorer": {
                "name": "지식 탐색기",
                "metrics": ["graph_load_time", "filter_processing_time", "visualization_time"],
                "critical": False
            },
            "chat_interface": {
                "name": "채팅 인터페이스",
                "metrics": ["message_processing_time", "stream_latency", "user_satisfaction"],
                "critical": True
            },
            "think_visualizer": {
                "name": "사고 시각화",
                "metrics": ["block_creation_time", "update_frequency", "visualization_load"],
                "critical": False
            }
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """경고 임계값 초기화"""
        
        return {
            "response_time": {
                "warning": 2.0,    # 2초
                "critical": 5.0    # 5초
            },
            "memory_usage": {
                "warning": 0.8,    # 80%
                "critical": 0.95   # 95%
            },
            "cpu_usage": {
                "warning": 0.7,    # 70%
                "critical": 0.9    # 90%
            },
            "error_rate": {
                "warning": 0.05,   # 5%
                "critical": 0.1    # 10%
            },
            "success_rate": {
                "warning": 0.9,    # 90% 이하
                "critical": 0.8    # 80% 이하
            },
            "active_sessions": {
                "warning": 100,    # 100개 세션
                "critical": 200    # 200개 세션
            }
        }
    
    async def start_monitoring(self):
        """모니터링 시작"""
        
        # 메트릭 수집 태스크
        self.monitoring_tasks.append(
            asyncio.create_task(self._metric_collection_loop())
        )
        
        # 헬스 체크 태스크
        self.monitoring_tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        
        # 경고 시스템 태스크
        self.monitoring_tasks.append(
            asyncio.create_task(self._alert_monitoring_loop())
        )
        
        # 최적화 제안 태스크
        self.monitoring_tasks.append(
            asyncio.create_task(self._optimization_analysis_loop())
        )
        
        # 데이터 정리 태스크
        self.monitoring_tasks.append(
            asyncio.create_task(self._data_cleanup_loop())
        )
        
        logger.info("성능 모니터링 시작")
    
    async def _metric_collection_loop(self):
        """메트릭 수집 루프"""
        
        while True:
            try:
                # 각 컴포넌트에서 메트릭 수집
                for component_name, component_info in self.registered_components.items():
                    await self._collect_component_metrics(component_name)
                
                # 시스템 메트릭 수집
                await self._collect_system_metrics()
                
                # 통계 업데이트
                self._update_dashboard_stats()
                
                await asyncio.sleep(5)  # 5초마다
                
            except Exception as e:
                logger.error(f"메트릭 수집 오류: {e}")
                await asyncio.sleep(10)
    
    async def _collect_component_metrics(self, component_name: str):
        """컴포넌트별 메트릭 수집"""
        
        timestamp = datetime.now()
        
        try:
            if component_name == "ui_orchestrator":
                await self._collect_ui_orchestrator_metrics(timestamp)
            elif component_name == "conversational_engine":
                await self._collect_conversational_engine_metrics(timestamp)
            elif component_name == "rag_orchestrator":
                await self._collect_rag_metrics(timestamp)
            elif component_name == "graph_manager":
                await self._collect_graph_metrics(timestamp)
            elif component_name == "mcp_orchestrator":
                await self._collect_mcp_metrics(timestamp)
            elif component_name == "korean_optimizer":
                await self._collect_korean_optimizer_metrics(timestamp)
            # 추가 컴포넌트들...
                
        except Exception as e:
            logger.error(f"{component_name} 메트릭 수집 실패: {e}")
    
    async def _collect_ui_orchestrator_metrics(self, timestamp: datetime):
        """UI 오케스트레이터 메트릭 수집"""
        
        if not self.ui_orchestrator:
            return
        
        # 활성 세션 수
        active_sessions = len(self.ui_orchestrator.active_sessions)
        self._add_metric_point(MetricPoint(
            timestamp=timestamp,
            value=active_sessions,
            metric_name="active_sessions",
            component="ui_orchestrator",
            unit="count"
        ))
        
        # WebSocket 연결 수
        websocket_connections = sum(
            len(conns) for conns in self.ui_orchestrator.websocket_connections.values()
        )
        self._add_metric_point(MetricPoint(
            timestamp=timestamp,
            value=websocket_connections,
            metric_name="websocket_connections",
            component="ui_orchestrator",
            unit="count"
        ))
        
        # 메시지 스트림 크기
        total_messages = sum(
            len(stream) for stream in self.ui_orchestrator.message_streams.values()
        )
        self._add_metric_point(MetricPoint(
            timestamp=timestamp,
            value=total_messages,
            metric_name="total_messages",
            component="ui_orchestrator",
            unit="count"
        ))
    
    async def _collect_conversational_engine_metrics(self, timestamp: datetime):
        """대화 엔진 메트릭 수집"""
        
        # 대화 엔진에 통계 메소드가 있다고 가정
        if hasattr(self.ui_orchestrator, 'conversational_engine'):
            engine = self.ui_orchestrator.conversational_engine
            
            # 더미 메트릭 (실제로는 엔진에서 제공)
            self._add_metric_point(MetricPoint(
                timestamp=timestamp,
                value=1.2,  # 평균 응답 시간
                metric_name="response_time",
                component="conversational_engine",
                unit="seconds"
            ))
            
            self._add_metric_point(MetricPoint(
                timestamp=timestamp,
                value=2.3,  # 평균 THINK 블록 수
                metric_name="think_blocks_per_response",
                component="conversational_engine",
                unit="count"
            ))
            
            self._add_metric_point(MetricPoint(
                timestamp=timestamp,
                value=0.95,  # 성공률
                metric_name="success_rate",
                component="conversational_engine",
                unit="ratio"
            ))
    
    async def _collect_rag_metrics(self, timestamp: datetime):
        """RAG 시스템 메트릭 수집"""
        
        if hasattr(self.ui_orchestrator, 'rag_orchestrator'):
            rag = self.ui_orchestrator.rag_orchestrator
            
            if hasattr(rag, 'get_performance_statistics'):
                stats = rag.get_performance_statistics()
                
                self._add_metric_point(MetricPoint(
                    timestamp=timestamp,
                    value=stats.get("avg_query_time", 0.0),
                    metric_name="search_time",
                    component="rag_orchestrator",
                    unit="seconds"
                ))
                
                self._add_metric_point(MetricPoint(
                    timestamp=timestamp,
                    value=stats.get("cache_hit_rate", 0.0),
                    metric_name="cache_hit_rate",
                    component="rag_orchestrator",
                    unit="ratio"
                ))
    
    async def _collect_graph_metrics(self, timestamp: datetime):
        """지식 그래프 메트릭 수집"""
        
        if hasattr(self.ui_orchestrator, 'graph_manager'):
            graph = self.ui_orchestrator.graph_manager
            
            # 더미 메트릭 (실제로는 그래프 매니저에서 제공)
            self._add_metric_point(MetricPoint(
                timestamp=timestamp,
                value=0.8,  # 평균 쿼리 시간
                metric_name="query_time",
                component="graph_manager",
                unit="seconds"
            ))
    
    async def _collect_mcp_metrics(self, timestamp: datetime):
        """MCP 생태계 메트릭 수집"""
        
        if hasattr(self.ui_orchestrator, 'mcp_orchestrator'):
            mcp = self.ui_orchestrator.mcp_orchestrator
            
            if hasattr(mcp, 'get_ecosystem_status'):
                status = await mcp.get_ecosystem_status()
                
                self._add_metric_point(MetricPoint(
                    timestamp=timestamp,
                    value=status.get("avg_success_rate", 0.0),
                    metric_name="success_rate",
                    component="mcp_orchestrator",
                    unit="ratio"
                ))
                
                self._add_metric_point(MetricPoint(
                    timestamp=timestamp,
                    value=status.get("active_tools", 0),
                    metric_name="active_tools",
                    component="mcp_orchestrator",
                    unit="count"
                ))
    
    async def _collect_korean_optimizer_metrics(self, timestamp: datetime):
        """한국어 최적화 메트릭 수집"""
        
        if self.korean_optimizer:
            # 더미 메트릭 (실제로는 Korean Optimizer에서 제공)
            self._add_metric_point(MetricPoint(
                timestamp=timestamp,
                value=0.3,  # 평균 처리 시간
                metric_name="processing_time",
                component="korean_optimizer",
                unit="seconds"
            ))
            
            self._add_metric_point(MetricPoint(
                timestamp=timestamp,
                value=0.92,  # 평균 신뢰도
                metric_name="confidence_score",
                component="korean_optimizer",
                unit="ratio"
            ))
    
    async def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        
        import psutil
        timestamp = datetime.now()
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        self._add_metric_point(MetricPoint(
            timestamp=timestamp,
            value=cpu_percent / 100.0,
            metric_name="cpu_usage",
            component="system",
            unit="ratio"
        ))
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        self._add_metric_point(MetricPoint(
            timestamp=timestamp,
            value=memory.percent / 100.0,
            metric_name="memory_usage",
            component="system",
            unit="ratio"
        ))
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        self._add_metric_point(MetricPoint(
            timestamp=timestamp,
            value=disk.percent / 100.0,
            metric_name="disk_usage",
            component="system",
            unit="ratio"
        ))
    
    def _add_metric_point(self, metric_point: MetricPoint):
        """메트릭 포인트 추가"""
        
        metric_key = f"{metric_point.component}:{metric_point.metric_name}"
        self.metrics_buffer[metric_key].append(metric_point)
        
        # 통계 업데이트
        self.dashboard_stats["total_metrics_collected"] += 1
    
    async def _health_check_loop(self):
        """헬스 체크 루프"""
        
        while True:
            try:
                for component_name in self.registered_components:
                    health = await self._check_component_health(component_name)
                    self.component_health[component_name] = health
                
                await asyncio.sleep(30)  # 30초마다
                
            except Exception as e:
                logger.error(f"헬스 체크 오류: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_health(self, component_name: str) -> ComponentHealth:
        """컴포넌트 헬스 체크"""
        
        try:
            # 최근 메트릭 기반 헬스 점수 계산
            health_score = await self._calculate_health_score(component_name)
            
            # 상태 결정
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.6:
                status = "warning"
            elif health_score >= 0.4:
                status = "error"
            else:
                status = "unknown"
            
            # 이슈 및 추천사항 생성
            issues = await self._detect_component_issues(component_name)
            recommendations = await self._generate_component_recommendations(component_name, issues)
            
            return ComponentHealth(
                component=component_name,
                status=status,
                score=health_score,
                last_check=datetime.now(),
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"{component_name} 헬스 체크 실패: {e}")
            return ComponentHealth(
                component=component_name,
                status="unknown",
                score=0.0,
                last_check=datetime.now(),
                issues=[f"헬스 체크 실패: {str(e)}"]
            )
    
    async def _calculate_health_score(self, component_name: str) -> float:
        """헬스 점수 계산"""
        
        component_info = self.registered_components.get(component_name, {})
        metric_names = component_info.get("metrics", [])
        
        if not metric_names:
            return 0.5  # 기본 점수
        
        scores = []
        
        for metric_name in metric_names:
            metric_key = f"{component_name}:{metric_name}"
            
            if metric_key in self.metrics_buffer:
                recent_points = list(self.metrics_buffer[metric_key])[-10:]  # 최근 10개
                
                if recent_points:
                    # 메트릭별 점수 계산
                    metric_score = self._calculate_metric_score(metric_name, recent_points)
                    scores.append(metric_score)
        
        return statistics.mean(scores) if scores else 0.5
    
    def _calculate_metric_score(self, metric_name: str, points: List[MetricPoint]) -> float:
        """메트릭별 점수 계산"""
        
        if not points:
            return 0.5
        
        recent_values = [p.value for p in points]
        avg_value = statistics.mean(recent_values)
        
        # 메트릭 타입별 점수 계산
        if metric_name in ["response_time", "query_time", "processing_time"]:
            # 응답 시간: 낮을수록 좋음
            if avg_value < 1.0:
                return 1.0
            elif avg_value < 2.0:
                return 0.8
            elif avg_value < 5.0:
                return 0.6
            else:
                return 0.2
        
        elif metric_name in ["success_rate", "cache_hit_rate", "confidence_score"]:
            # 비율: 높을수록 좋음
            return avg_value
        
        elif metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            # 사용률: 적당할수록 좋음
            if avg_value < 0.7:
                return 1.0
            elif avg_value < 0.85:
                return 0.8
            elif avg_value < 0.95:
                return 0.4
            else:
                return 0.1
        
        else:
            # 기본: 안정성 기반 (변동 적을수록 좋음)
            if len(recent_values) > 1:
                stdev = statistics.stdev(recent_values)
                stability = max(0.0, 1.0 - stdev / avg_value if avg_value > 0 else 0.0)
                return stability
            else:
                return 0.5
    
    async def _detect_component_issues(self, component_name: str) -> List[str]:
        """컴포넌트 이슈 탐지"""
        
        issues = []
        
        # 메트릭 기반 이슈 탐지
        for metric_name, thresholds in self.alert_thresholds.items():
            metric_key = f"{component_name}:{metric_name}"
            
            if metric_key in self.metrics_buffer:
                recent_points = list(self.metrics_buffer[metric_key])[-5:]  # 최근 5개
                
                if recent_points:
                    avg_value = statistics.mean([p.value for p in recent_points])
                    
                    if metric_name in ["response_time", "cpu_usage", "memory_usage"]:
                        # 높을수록 나쁨
                        if avg_value > thresholds.get("critical", float('inf')):
                            issues.append(f"{metric_name}이 임계치를 초과했습니다 ({avg_value:.2f})")
                        elif avg_value > thresholds.get("warning", float('inf')):
                            issues.append(f"{metric_name}이 경고 수준입니다 ({avg_value:.2f})")
                    
                    elif metric_name in ["success_rate"]:
                        # 낮을수록 나쁨
                        if avg_value < thresholds.get("critical", 0):
                            issues.append(f"{metric_name}이 임계치 이하입니다 ({avg_value:.2f})")
                        elif avg_value < thresholds.get("warning", 0):
                            issues.append(f"{metric_name}이 경고 수준입니다 ({avg_value:.2f})")
        
        return issues
    
    async def _generate_component_recommendations(
        self,
        component_name: str,
        issues: List[str]
    ) -> List[str]:
        """컴포넌트 추천사항 생성"""
        
        recommendations = []
        
        # 이슈 기반 추천사항
        for issue in issues:
            if "response_time" in issue:
                recommendations.append("응답 시간 최적화를 위해 캐싱이나 비동기 처리를 고려해보세요.")
            elif "memory_usage" in issue:
                recommendations.append("메모리 사용량 최적화를 위해 가비지 컬렉션이나 데이터 정리를 확인해보세요.")
            elif "success_rate" in issue:
                recommendations.append("성공률 향상을 위해 에러 처리 로직을 검토해보세요.")
        
        # 컴포넌트별 일반 추천사항
        if component_name == "rag_orchestrator":
            recommendations.append("문서 인덱싱 최적화 및 검색 알고리즘 튜닝을 고려해보세요.")
        elif component_name == "graph_manager":
            recommendations.append("그래프 쿼리 최적화 및 인덱스 설정을 확인해보세요.")
        elif component_name == "mcp_orchestrator":
            recommendations.append("도구 실행 성능 모니터링 및 진화 전략을 검토해보세요.")
        
        return recommendations
    
    async def _alert_monitoring_loop(self):
        """경고 모니터링 루프"""
        
        while True:
            try:
                await self._check_and_generate_alerts()
                await asyncio.sleep(10)  # 10초마다
                
            except Exception as e:
                logger.error(f"경고 모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    async def _check_and_generate_alerts(self):
        """경고 확인 및 생성"""
        
        for metric_key, metric_buffer in self.metrics_buffer.items():
            if not metric_buffer:
                continue
            
            component, metric_name = metric_key.split(":", 1)
            recent_points = list(metric_buffer)[-3:]  # 최근 3개
            
            if len(recent_points) >= 3:
                avg_value = statistics.mean([p.value for p in recent_points])
                
                # 경고 조건 확인
                alert = self._check_alert_conditions(
                    component, metric_name, avg_value
                )
                
                if alert:
                    self.active_alerts[alert.id] = alert
                    logger.warning(f"경고 생성: {alert.message}")
    
    def _check_alert_conditions(
        self,
        component: str,
        metric_name: str,
        value: float
    ) -> Optional[PerformanceAlert]:
        """경고 조건 확인"""
        
        thresholds = self.alert_thresholds.get(metric_name, {})
        
        if not thresholds:
            return None
        
        severity = None
        threshold_value = 0.0
        
        # 임계치 확인
        if metric_name in ["response_time", "cpu_usage", "memory_usage", "error_rate"]:
            # 높을수록 나쁨
            if value > thresholds.get("critical", float('inf')):
                severity = "critical"
                threshold_value = thresholds["critical"]
            elif value > thresholds.get("warning", float('inf')):
                severity = "warning"
                threshold_value = thresholds["warning"]
        
        elif metric_name in ["success_rate"]:
            # 낮을수록 나쁨
            if value < thresholds.get("critical", 0):
                severity = "critical"
                threshold_value = thresholds["critical"]
            elif value < thresholds.get("warning", 1):
                severity = "warning"
                threshold_value = thresholds["warning"]
        
        if severity:
            alert_id = f"{component}_{metric_name}_{severity}"
            
            # 이미 존재하는 경고인지 확인
            if alert_id in self.active_alerts:
                return None
            
            return PerformanceAlert(
                id=alert_id,
                component=component,
                metric_name=metric_name,
                severity=severity,
                message=f"{component}의 {metric_name}이 {severity} 수준입니다 ({value:.2f})",
                threshold_value=threshold_value,
                current_value=value,
                created_at=datetime.now()
            )
        
        return None
    
    async def _optimization_analysis_loop(self):
        """최적화 분석 루프"""
        
        while True:
            try:
                await self._analyze_and_suggest_optimizations()
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                logger.error(f"최적화 분석 오류: {e}")
                await asyncio.sleep(600)
    
    async def _analyze_and_suggest_optimizations(self):
        """최적화 분석 및 제안"""
        
        # 시스템 전체 분석
        system_analysis = await self._analyze_system_performance()
        
        # 최적화 제안 생성
        new_suggestions = await self._generate_optimization_suggestions(system_analysis)
        
        # 기존 제안과 중복 제거
        for suggestion in new_suggestions:
            if not any(s.title == suggestion.title for s in self.optimization_suggestions):
                self.optimization_suggestions.append(suggestion)
        
        # 오래된 제안 정리 (7일 이상)
        cutoff_date = datetime.now() - timedelta(days=7)
        self.optimization_suggestions = [
            s for s in self.optimization_suggestions
            if s.created_at > cutoff_date
        ]
    
    async def _analyze_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 분석"""
        
        analysis = {
            "overall_health": 0.0,
            "bottlenecks": [],
            "resource_usage": {},
            "performance_trends": {},
            "critical_issues": []
        }
        
        # 전체 헬스 점수 계산
        if self.component_health:
            health_scores = [h.score for h in self.component_health.values()]
            analysis["overall_health"] = statistics.mean(health_scores)
        
        # 병목 지점 탐지
        for component_name, health in self.component_health.items():
            if health.score < 0.6:
                analysis["bottlenecks"].append({
                    "component": component_name,
                    "score": health.score,
                    "issues": health.issues
                })
        
        # 리소스 사용량 분석
        system_metrics = ["cpu_usage", "memory_usage", "disk_usage"]
        for metric in system_metrics:
            metric_key = f"system:{metric}"
            if metric_key in self.metrics_buffer:
                recent_points = list(self.metrics_buffer[metric_key])[-20:]  # 최근 20개
                if recent_points:
                    avg_value = statistics.mean([p.value for p in recent_points])
                    analysis["resource_usage"][metric] = avg_value
        
        return analysis
    
    async def _generate_optimization_suggestions(
        self,
        analysis: Dict[str, Any]
    ) -> List[SystemOptimization]:
        """최적화 제안 생성"""
        
        suggestions = []
        
        # 전체 헬스가 낮은 경우
        if analysis["overall_health"] < 0.7:
            suggestions.append(SystemOptimization(
                id=str(uuid.uuid4()),
                category="performance",
                priority="high",
                title="시스템 전체 성능 최적화",
                description="전체 시스템 헬스가 낮습니다. 주요 컴포넌트들의 성능 검토가 필요합니다.",
                impact_estimate="30-50% 성능 향상 예상",
                implementation_effort="높음",
                created_at=datetime.now()
            ))
        
        # 메모리 사용량이 높은 경우
        memory_usage = analysis["resource_usage"].get("memory_usage", 0)
        if memory_usage > 0.8:
            suggestions.append(SystemOptimization(
                id=str(uuid.uuid4()),
                category="memory",
                priority="high",
                title="메모리 사용량 최적화",
                description=f"메모리 사용량이 {memory_usage*100:.1f}%로 높습니다. 캐시 정리 및 메모리 관리 최적화가 필요합니다.",
                impact_estimate="20-30% 메모리 절약 예상",
                implementation_effort="중간",
                created_at=datetime.now()
            ))
        
        # 병목 지점이 있는 경우
        for bottleneck in analysis["bottlenecks"]:
            suggestions.append(SystemOptimization(
                id=str(uuid.uuid4()),
                category="performance",
                priority="medium",
                title=f"{bottleneck['component']} 성능 최적화",
                description=f"{bottleneck['component']} 컴포넌트의 성능이 저하되었습니다.",
                impact_estimate="10-20% 성능 향상 예상",
                implementation_effort="중간",
                created_at=datetime.now()
            ))
        
        return suggestions
    
    async def _data_cleanup_loop(self):
        """데이터 정리 루프"""
        
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.metric_retention_hours)
                
                # 오래된 메트릭 데이터 정리
                for metric_key, metric_buffer in self.metrics_buffer.items():
                    # 최근 데이터만 유지
                    filtered_points = deque([
                        point for point in metric_buffer
                        if point.timestamp > cutoff_time
                    ], maxlen=metric_buffer.maxlen)
                    
                    self.metrics_buffer[metric_key] = filtered_points
                
                # 오래된 경고 정리
                old_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if (datetime.now() - alert.created_at).total_seconds() > 3600  # 1시간
                ]
                
                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(3600)  # 1시간마다
                
            except Exception as e:
                logger.error(f"데이터 정리 오류: {e}")
                await asyncio.sleep(3600)
    
    def _update_dashboard_stats(self):
        """대시보드 통계 업데이트"""
        
        self.dashboard_stats.update({
            "active_components": len([
                h for h in self.component_health.values()
                if h.status in ["healthy", "warning"]
            ]),
            "total_alerts": len(self.active_alerts),
            "avg_system_health": statistics.mean([
                h.score for h in self.component_health.values()
            ]) if self.component_health else 0.0,
            "last_updated": datetime.now()
        })
    
    async def get_real_time_dashboard_data(
        self,
        time_range_minutes: int = 60
    ) -> Dict[str, Any]:
        """실시간 대시보드 데이터 조회"""
        
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        
        dashboard_data = {
            "overview": self.dashboard_stats,
            "component_health": {
                name: {
                    "status": health.status,
                    "score": health.score,
                    "last_check": health.last_check.isoformat(),
                    "issues": health.issues,
                    "recommendations": health.recommendations
                }
                for name, health in self.component_health.items()
            },
            "active_alerts": [
                {
                    "id": alert.id,
                    "component": alert.component,
                    "metric": alert.metric_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "created_at": alert.created_at.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in self.active_alerts.values()
            ],
            "metrics_timeseries": {},
            "optimization_suggestions": [
                {
                    "id": opt.id,
                    "category": opt.category,
                    "priority": opt.priority,
                    "title": opt.title,
                    "description": opt.description,
                    "impact": opt.impact_estimate,
                    "effort": opt.implementation_effort,
                    "created_at": opt.created_at.isoformat()
                }
                for opt in self.optimization_suggestions[-10:]  # 최근 10개
            ]
        }
        
        # 시계열 메트릭 데이터
        for metric_key, metric_buffer in self.metrics_buffer.items():
            recent_points = [
                point for point in metric_buffer
                if point.timestamp > cutoff_time
            ]
            
            if recent_points:
                dashboard_data["metrics_timeseries"][metric_key] = [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "unit": point.unit
                    }
                    for point in recent_points[-100:]  # 최대 100개 포인트
                ]
        
        return dashboard_data
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """경고 승인"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"경고 승인: {alert_id}")
            return True
        
        return False
    
    async def dismiss_alert(self, alert_id: str) -> bool:
        """경고 해제"""
        
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"경고 해제: {alert_id}")
            return True
        
        return False
    
    async def get_component_detailed_metrics(
        self,
        component_name: str,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """컴포넌트 상세 메트릭 조회"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        component_metrics = {}
        
        for metric_key, metric_buffer in self.metrics_buffer.items():
            if metric_key.startswith(f"{component_name}:"):
                metric_name = metric_key.split(":", 1)[1]
                
                recent_points = [
                    point for point in metric_buffer
                    if point.timestamp > cutoff_time
                ]
                
                if recent_points:
                    values = [p.value for p in recent_points]
                    
                    component_metrics[metric_name] = {
                        "current": values[-1] if values else 0,
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                        "trend": "stable",  # 실제로는 트렌드 분석 필요
                        "data_points": [
                            {
                                "timestamp": p.timestamp.isoformat(),
                                "value": p.value
                            }
                            for p in recent_points[-200:]  # 최대 200개
                        ]
                    }
        
        return {
            "component": component_name,
            "time_range_hours": time_range_hours,
            "metrics": component_metrics,
            "health": self.component_health.get(component_name, None).__dict__ if component_name in self.component_health else None
        }
    
    async def export_performance_report(
        self,
        format: str = "json",
        time_range_hours: int = 24
    ) -> Optional[str]:
        """성능 리포트 내보내기"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        if format == "json":
            report_data = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "time_range_hours": time_range_hours,
                    "total_components": len(self.registered_components)
                },
                "summary": self.dashboard_stats,
                "component_health": {
                    name: health.__dict__
                    for name, health in self.component_health.items()
                },
                "alerts": [
                    alert.__dict__
                    for alert in self.active_alerts.values()
                ],
                "optimizations": [
                    opt.__dict__
                    for opt in self.optimization_suggestions
                ]
            }
            
            return json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
        
        elif format == "markdown":
            lines = ["# PPuRI-AI Ultimate 성능 리포트\n"]
            lines.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append(f"**분석 기간**: 최근 {time_range_hours}시간\n\n")
            
            # 시스템 개요
            lines.append("## 시스템 개요\n")
            lines.append(f"- **전체 헬스 점수**: {self.dashboard_stats['avg_system_health']:.2f}/1.0\n")
            lines.append(f"- **활성 컴포넌트**: {self.dashboard_stats['active_components']}개\n")
            lines.append(f"- **활성 경고**: {self.dashboard_stats['total_alerts']}개\n\n")
            
            # 컴포넌트 헬스
            lines.append("## 컴포넌트 상태\n")
            for name, health in self.component_health.items():
                status_emoji = {"healthy": "✅", "warning": "⚠️", "error": "❌", "unknown": "❓"}.get(health.status, "❓")
                lines.append(f"- **{name}**: {status_emoji} {health.status} (점수: {health.score:.2f})\n")
            
            lines.append("\n")
            
            # 경고
            if self.active_alerts:
                lines.append("## 활성 경고\n")
                for alert in self.active_alerts.values():
                    severity_emoji = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(alert.severity, "⚪")
                    lines.append(f"- {severity_emoji} **{alert.component}**: {alert.message}\n")
                lines.append("\n")
            
            # 최적화 제안
            if self.optimization_suggestions:
                lines.append("## 최적화 제안\n")
                for opt in self.optimization_suggestions[-5:]:  # 최근 5개
                    priority_emoji = {"low": "🔵", "medium": "🟡", "high": "🔴"}.get(opt.priority, "⚪")
                    lines.append(f"- {priority_emoji} **{opt.title}**: {opt.description}\n")
            
            return "".join(lines)
        
        return None
    
    async def stop_monitoring(self):
        """모니터링 중단"""
        
        # 모든 모니터링 태스크 취소
        for task in self.monitoring_tasks:
            task.cancel()
        
        # 태스크 완료 대기
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("성능 모니터링 중단")
    
    async def cleanup(self):
        """Performance Dashboard 정리"""
        
        # 모니터링 중단
        await self.stop_monitoring()
        
        # 데이터 정리
        self.metrics_buffer.clear()
        self.active_alerts.clear()
        self.component_health.clear()
        self.optimization_suggestions.clear()
        
        logger.info("Performance Dashboard 정리 완료")