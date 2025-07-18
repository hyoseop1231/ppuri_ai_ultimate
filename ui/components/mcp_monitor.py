"""
MCP Monitor - MCP 도구 생태계 모니터링 컴포넌트

자동 진화하는 MCP 도구 생태계를 실시간으로 모니터링하고
도구 성능, 진화 과정, 네트워크 상태를 시각화하는 컴포넌트.

Features:
- 실시간 도구 상태 모니터링
- 진화 과정 추적 및 시각화
- 도구 네트워크 관계 표시
- 성능 메트릭 분석
- 자동 최적화 제안
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
import json
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ToolStatus:
    """도구 상태"""
    tool_name: str
    category: str
    status: str  # active, idle, evolving, error
    performance_score: float
    usage_count: int
    success_rate: float
    avg_execution_time: float
    last_used: Optional[datetime] = None
    last_evolved: Optional[datetime] = None


@dataclass
class EvolutionEvent:
    """진화 이벤트"""
    event_id: str
    tool_name: str
    strategy: str
    timestamp: datetime
    performance_before: float
    performance_after: float
    description: str
    success: bool


@dataclass
class ToolRelationship:
    """도구 간 관계"""
    source_tool: str
    target_tool: str
    relationship_type: str  # depends_on, enhances, competes_with, collaborates
    strength: float
    frequency: int


@dataclass
class EcosystemAlert:
    """생태계 경고"""
    alert_id: str
    alert_type: str  # performance_degradation, evolution_failed, tool_conflict
    severity: str  # low, medium, high, critical
    tool_name: str
    message: str
    created_at: datetime
    resolved: bool = False


class MCPMonitor:
    """
    MCP 도구 생태계 모니터링
    
    자동 진화하는 MCP 도구들의 상태, 성능, 관계를 
    실시간으로 추적하고 분석하는 모니터링 시스템.
    """
    
    def __init__(
        self,
        ui_orchestrator,
        mcp_orchestrator,
        korean_optimizer=None
    ):
        self.ui_orchestrator = ui_orchestrator
        self.mcp_orchestrator = mcp_orchestrator
        self.korean_optimizer = korean_optimizer
        
        # 도구 상태 추적
        self.tool_status: Dict[str, ToolStatus] = {}
        self.tool_relationships: Dict[str, List[ToolRelationship]] = defaultdict(list)
        
        # 진화 추적
        self.evolution_history: deque = deque(maxlen=1000)
        self.active_evolutions: Dict[str, Dict[str, Any]] = {}
        
        # 성능 메트릭
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # 경고 시스템
        self.active_alerts: Dict[str, EcosystemAlert] = {}
        
        # 생태계 통계
        self.ecosystem_stats = {
            "total_tools": 0,
            "active_tools": 0,
            "evolving_tools": 0,
            "avg_performance": 0.0,
            "total_evolutions": 0,
            "successful_evolutions": 0,
            "last_updated": datetime.now()
        }
        
        # 모니터링 설정
        self.monitoring_config = {
            "performance_threshold": 0.7,
            "evolution_threshold": 0.8,
            "alert_cooldown": 300,  # 5분
            "relationship_strength_threshold": 0.5
        }
        
        # 모니터링 태스크
        self.monitoring_tasks: List[asyncio.Task] = []
        
        logger.info("MCP Monitor 초기화 완료")
    
    async def start_monitoring(self):
        """모니터링 시작"""
        
        # 도구 상태 모니터링
        self.monitoring_tasks.append(
            asyncio.create_task(self._tool_status_monitoring_loop())
        )
        
        # 진화 추적
        self.monitoring_tasks.append(
            asyncio.create_task(self._evolution_tracking_loop())
        )
        
        # 관계 분석
        self.monitoring_tasks.append(
            asyncio.create_task(self._relationship_analysis_loop())
        )
        
        # 경고 시스템
        self.monitoring_tasks.append(
            asyncio.create_task(self._alert_monitoring_loop())
        )
        
        # 생태계 분석
        self.monitoring_tasks.append(
            asyncio.create_task(self._ecosystem_analysis_loop())
        )
        
        logger.info("MCP Monitor 모니터링 시작")
    
    async def _tool_status_monitoring_loop(self):
        """도구 상태 모니터링 루프"""
        
        while True:
            try:
                # MCP 오케스트레이터에서 상태 수집
                ecosystem_status = await self.mcp_orchestrator.get_ecosystem_status()
                
                # 도구별 상태 업데이트
                await self._update_tool_status(ecosystem_status)
                
                # 성능 메트릭 수집
                await self._collect_performance_metrics()
                
                await asyncio.sleep(10)  # 10초마다
                
            except Exception as e:
                logger.error(f"도구 상태 모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    async def _update_tool_status(self, ecosystem_status: Dict[str, Any]):
        """도구 상태 업데이트"""
        
        tools_by_category = ecosystem_status.get("tools_by_category", {})
        top_performing_tools = ecosystem_status.get("top_performing_tools", [])
        
        # 기존 도구 상태 초기화
        active_tools = set()
        
        # 카테고리별 도구 처리
        for category, tool_names in tools_by_category.items():
            for tool_name in tool_names:
                active_tools.add(tool_name)
                
                # 성능 정보 찾기
                performance_info = next(
                    (tool for tool in top_performing_tools if tool["name"] == tool_name),
                    {"performance_score": 0.5, "usage_count": 0, "success_rate": 1.0}
                )
                
                # 상태 결정
                performance_score = performance_info.get("performance_score", 0.5)
                status = self._determine_tool_status(tool_name, performance_score)
                
                # 도구 상태 업데이트 또는 생성
                if tool_name in self.tool_status:
                    tool_status = self.tool_status[tool_name]
                    tool_status.status = status
                    tool_status.performance_score = performance_score
                    tool_status.usage_count = performance_info.get("usage_count", 0)
                    tool_status.success_rate = performance_info.get("success_rate", 1.0)
                    tool_status.last_used = datetime.now()  # 실제로는 마지막 사용 시간
                else:
                    self.tool_status[tool_name] = ToolStatus(
                        tool_name=tool_name,
                        category=category,
                        status=status,
                        performance_score=performance_score,
                        usage_count=performance_info.get("usage_count", 0),
                        success_rate=performance_info.get("success_rate", 1.0),
                        avg_execution_time=1.0,  # 기본값
                        last_used=datetime.now()
                    )
        
        # 비활성 도구 마킹
        for tool_name in self.tool_status:
            if tool_name not in active_tools:
                self.tool_status[tool_name].status = "idle"
        
        # 통계 업데이트
        self._update_ecosystem_stats()
    
    def _determine_tool_status(self, tool_name: str, performance_score: float) -> str:
        """도구 상태 결정"""
        
        # 진화 중인지 확인
        if tool_name in self.active_evolutions:
            return "evolving"
        
        # 성능 기반 상태 결정
        if performance_score < 0.3:
            return "error"
        elif performance_score < 0.7:
            return "idle"
        else:
            return "active"
    
    async def _collect_performance_metrics(self):
        """성능 메트릭 수집"""
        
        timestamp = datetime.now()
        
        for tool_name, tool_status in self.tool_status.items():
            # 성능 히스토리에 추가
            self.performance_history[tool_name].append({
                "timestamp": timestamp,
                "performance_score": tool_status.performance_score,
                "success_rate": tool_status.success_rate,
                "execution_time": tool_status.avg_execution_time,
                "usage_count": tool_status.usage_count
            })
    
    async def _evolution_tracking_loop(self):
        """진화 추적 루프"""
        
        while True:
            try:
                # 진화 이벤트 수집
                recent_evolutions = await self._collect_evolution_events()
                
                # 진화 이벤트 처리
                for event in recent_evolutions:
                    await self._process_evolution_event(event)
                
                await asyncio.sleep(30)  # 30초마다
                
            except Exception as e:
                logger.error(f"진화 추적 오류: {e}")
                await asyncio.sleep(60)
    
    async def _collect_evolution_events(self) -> List[EvolutionEvent]:
        """진화 이벤트 수집"""
        
        try:
            # MCP 오케스트레이터에서 최근 진화 정보 조회
            ecosystem_status = await self.mcp_orchestrator.get_ecosystem_status()
            recent_evolutions = ecosystem_status.get("recent_evolutions", [])
            
            evolution_events = []
            
            for evolution_data in recent_evolutions:
                event = EvolutionEvent(
                    event_id=str(uuid.uuid4()),
                    tool_name=evolution_data.get("tool_name", ""),
                    strategy=evolution_data.get("strategy", ""),
                    timestamp=datetime.fromisoformat(evolution_data.get("timestamp", datetime.now().isoformat())),
                    performance_before=evolution_data.get("performance_before", 0.0),
                    performance_after=evolution_data.get("performance_after", 0.0),
                    description=f"{evolution_data.get('strategy', '')} 전략으로 진화",
                    success=evolution_data.get("performance_after", 0) > evolution_data.get("performance_before", 0)
                )
                
                evolution_events.append(event)
            
            return evolution_events
            
        except Exception as e:
            logger.error(f"진화 이벤트 수집 실패: {e}")
            return []
    
    async def _process_evolution_event(self, event: EvolutionEvent):
        """진화 이벤트 처리"""
        
        # 진화 히스토리에 추가
        self.evolution_history.append(event)
        
        # 도구 상태 업데이트
        if event.tool_name in self.tool_status:
            tool_status = self.tool_status[event.tool_name]
            tool_status.last_evolved = event.timestamp
            tool_status.performance_score = event.performance_after
        
        # 진화 성공/실패에 따른 처리
        if event.success:
            logger.info(f"도구 진화 성공: {event.tool_name} ({event.strategy})")
            self.ecosystem_stats["successful_evolutions"] += 1
        else:
            logger.warning(f"도구 진화 실패: {event.tool_name} ({event.strategy})")
            
            # 진화 실패 경고 생성
            await self._create_evolution_failure_alert(event)
        
        self.ecosystem_stats["total_evolutions"] += 1
    
    async def _relationship_analysis_loop(self):
        """관계 분석 루프"""
        
        while True:
            try:
                # 도구 간 관계 분석
                await self._analyze_tool_relationships()
                
                await asyncio.sleep(120)  # 2분마다
                
            except Exception as e:
                logger.error(f"관계 분석 오류: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_tool_relationships(self):
        """도구 간 관계 분석"""
        
        # 사용 패턴 기반 관계 분석
        usage_patterns = await self._analyze_usage_patterns()
        
        # 성능 상관관계 분석
        performance_correlations = await self._analyze_performance_correlations()
        
        # 진화 영향 분석
        evolution_impacts = await self._analyze_evolution_impacts()
        
        # 관계 업데이트
        await self._update_tool_relationships(
            usage_patterns, performance_correlations, evolution_impacts
        )
    
    async def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """사용 패턴 분석"""
        
        patterns = {}
        
        # 동시 사용 빈도 분석
        co_usage = defaultdict(int)
        
        # 실제로는 세션 데이터에서 분석해야 함
        # 여기서는 더미 데이터
        for tool1 in self.tool_status:
            for tool2 in self.tool_status:
                if tool1 != tool2:
                    # 동일 카테고리면 높은 상관관계
                    if (self.tool_status[tool1].category == 
                        self.tool_status[tool2].category):
                        co_usage[f"{tool1}:{tool2}"] = 5
                    else:
                        co_usage[f"{tool1}:{tool2}"] = 1
        
        patterns["co_usage"] = co_usage
        return patterns
    
    async def _analyze_performance_correlations(self) -> Dict[str, Any]:
        """성능 상관관계 분석"""
        
        correlations = {}
        
        # 성능 점수 간 상관관계 계산 (간단한 버전)
        tools = list(self.tool_status.keys())
        
        for i, tool1 in enumerate(tools):
            for tool2 in tools[i+1:]:
                perf1 = self.tool_status[tool1].performance_score
                perf2 = self.tool_status[tool2].performance_score
                
                # 간단한 상관관계 (실제로는 더 정교한 계산 필요)
                correlation = abs(perf1 - perf2)  # 차이가 작을수록 높은 상관관계
                correlations[f"{tool1}:{tool2}"] = 1.0 - correlation
        
        return correlations
    
    async def _analyze_evolution_impacts(self) -> Dict[str, Any]:
        """진화 영향 분석"""
        
        impacts = {}
        
        # 최근 진화 이벤트들의 영향 분석
        recent_evolutions = [
            event for event in self.evolution_history
            if (datetime.now() - event.timestamp).total_seconds() < 3600  # 1시간 내
        ]
        
        for event in recent_evolutions:
            # 진화한 도구가 다른 도구들에 미친 영향 분석
            evolved_tool = event.tool_name
            
            for other_tool in self.tool_status:
                if other_tool != evolved_tool:
                    # 진화 전후 성능 변화 영향 계산
                    impact_score = abs(event.performance_after - event.performance_before) * 0.1
                    impacts[f"{evolved_tool}:{other_tool}"] = impact_score
        
        return impacts
    
    async def _update_tool_relationships(
        self,
        usage_patterns: Dict[str, Any],
        performance_correlations: Dict[str, Any],
        evolution_impacts: Dict[str, Any]
    ):
        """도구 관계 업데이트"""
        
        # 기존 관계 정리
        self.tool_relationships.clear()
        
        # 사용 패턴 기반 관계
        co_usage = usage_patterns.get("co_usage", {})
        for pair, frequency in co_usage.items():
            if frequency > 3:  # 임계값
                tool1, tool2 = pair.split(":")
                
                relationship = ToolRelationship(
                    source_tool=tool1,
                    target_tool=tool2,
                    relationship_type="collaborates",
                    strength=min(1.0, frequency / 10.0),
                    frequency=frequency
                )
                
                self.tool_relationships[tool1].append(relationship)
        
        # 성능 상관관계 기반 관계
        for pair, correlation in performance_correlations.items():
            if correlation > self.monitoring_config["relationship_strength_threshold"]:
                tool1, tool2 = pair.split(":")
                
                relationship = ToolRelationship(
                    source_tool=tool1,
                    target_tool=tool2,
                    relationship_type="enhances" if correlation > 0.8 else "depends_on",
                    strength=correlation,
                    frequency=0
                )
                
                # 중복 체크 후 추가
                existing = any(r.target_tool == tool2 for r in self.tool_relationships[tool1])
                if not existing:
                    self.tool_relationships[tool1].append(relationship)
    
    async def _alert_monitoring_loop(self):
        """경고 모니터링 루프"""
        
        while True:
            try:
                # 성능 저하 경고
                await self._check_performance_alerts()
                
                # 진화 실패 경고
                await self._check_evolution_alerts()
                
                # 도구 충돌 경고
                await self._check_conflict_alerts()
                
                await asyncio.sleep(60)  # 1분마다
                
            except Exception as e:
                logger.error(f"경고 모니터링 오류: {e}")
                await asyncio.sleep(120)
    
    async def _check_performance_alerts(self):
        """성능 저하 경고 확인"""
        
        threshold = self.monitoring_config["performance_threshold"]
        
        for tool_name, tool_status in self.tool_status.items():
            if tool_status.performance_score < threshold:
                alert_id = f"performance_{tool_name}"
                
                # 쿨다운 체크
                if alert_id in self.active_alerts:
                    last_alert = self.active_alerts[alert_id]
                    if (datetime.now() - last_alert.created_at).total_seconds() < self.monitoring_config["alert_cooldown"]:
                        continue
                
                alert = EcosystemAlert(
                    alert_id=alert_id,
                    alert_type="performance_degradation",
                    severity="medium" if tool_status.performance_score > 0.5 else "high",
                    tool_name=tool_name,
                    message=f"{tool_name} 도구의 성능이 {tool_status.performance_score:.2f}로 저하되었습니다",
                    created_at=datetime.now()
                )
                
                self.active_alerts[alert_id] = alert
                logger.warning(f"성능 경고: {alert.message}")
    
    async def _check_evolution_alerts(self):
        """진화 관련 경고 확인"""
        
        # 최근 진화 실패들 확인
        recent_failures = [
            event for event in self.evolution_history
            if (not event.success and 
                (datetime.now() - event.timestamp).total_seconds() < 1800)  # 30분 내
        ]
        
        # 연속 실패 도구 탐지
        failure_counts = Counter(event.tool_name for event in recent_failures)
        
        for tool_name, failure_count in failure_counts.items():
            if failure_count >= 3:  # 연속 3회 실패
                alert_id = f"evolution_failure_{tool_name}"
                
                if alert_id not in self.active_alerts:
                    alert = EcosystemAlert(
                        alert_id=alert_id,
                        alert_type="evolution_failed",
                        severity="high",
                        tool_name=tool_name,
                        message=f"{tool_name} 도구가 연속 {failure_count}회 진화에 실패했습니다",
                        created_at=datetime.now()
                    )
                    
                    self.active_alerts[alert_id] = alert
                    logger.error(f"진화 실패 경고: {alert.message}")
    
    async def _check_conflict_alerts(self):
        """도구 충돌 경고 확인"""
        
        # 같은 카테고리 내 성능 격차 확인
        categories = defaultdict(list)
        
        for tool_name, tool_status in self.tool_status.items():
            categories[tool_status.category].append((tool_name, tool_status.performance_score))
        
        for category, tools in categories.items():
            if len(tools) > 1:
                performance_scores = [score for _, score in tools]
                max_score = max(performance_scores)
                min_score = min(performance_scores)
                
                # 성능 격차가 큰 경우
                if max_score - min_score > 0.5:
                    worst_tool = min(tools, key=lambda x: x[1])[0]
                    alert_id = f"conflict_{category}_{worst_tool}"
                    
                    if alert_id not in self.active_alerts:
                        alert = EcosystemAlert(
                            alert_id=alert_id,
                            alert_type="tool_conflict",
                            severity="medium",
                            tool_name=worst_tool,
                            message=f"{category} 카테고리 내 도구들 간 성능 격차가 발생했습니다",
                            created_at=datetime.now()
                        )
                        
                        self.active_alerts[alert_id] = alert
    
    async def _create_evolution_failure_alert(self, event: EvolutionEvent):
        """진화 실패 경고 생성"""
        
        alert_id = f"evolution_fail_{event.tool_name}_{event.timestamp.strftime('%H%M%S')}"
        
        alert = EcosystemAlert(
            alert_id=alert_id,
            alert_type="evolution_failed",
            severity="medium",
            tool_name=event.tool_name,
            message=f"{event.tool_name} 도구의 {event.strategy} 진화가 실패했습니다",
            created_at=event.timestamp
        )
        
        self.active_alerts[alert_id] = alert
    
    async def _ecosystem_analysis_loop(self):
        """생태계 분석 루프"""
        
        while True:
            try:
                # 생태계 전체 분석
                await self._analyze_ecosystem_health()
                
                # 최적화 제안 생성
                await self._generate_optimization_recommendations()
                
                await asyncio.sleep(300)  # 5분마다
                
            except Exception as e:
                logger.error(f"생태계 분석 오류: {e}")
                await asyncio.sleep(600)
    
    async def _analyze_ecosystem_health(self):
        """생태계 헬스 분석"""
        
        if not self.tool_status:
            return
        
        # 전체 성능 평균
        total_performance = sum(tool.performance_score for tool in self.tool_status.values())
        avg_performance = total_performance / len(self.tool_status)
        
        # 활성 도구 수
        active_tools = sum(1 for tool in self.tool_status.values() if tool.status == "active")
        
        # 진화 중인 도구 수
        evolving_tools = sum(1 for tool in self.tool_status.values() if tool.status == "evolving")
        
        # 통계 업데이트
        self.ecosystem_stats.update({
            "total_tools": len(self.tool_status),
            "active_tools": active_tools,
            "evolving_tools": evolving_tools,
            "avg_performance": avg_performance,
            "last_updated": datetime.now()
        })
    
    async def _generate_optimization_recommendations(self):
        """최적화 추천사항 생성"""
        
        recommendations = []
        
        # 성능이 낮은 도구들 식별
        low_performance_tools = [
            tool_name for tool_name, tool_status in self.tool_status.items()
            if tool_status.performance_score < 0.6
        ]
        
        if low_performance_tools:
            recommendations.append({
                "type": "performance_improvement",
                "priority": "high",
                "affected_tools": low_performance_tools,
                "recommendation": "성능이 낮은 도구들의 최적화가 필요합니다",
                "action": "해당 도구들의 진화 전략을 검토하고 수동 최적화를 고려하세요"
            })
        
        # 사용되지 않는 도구들 식별
        unused_tools = [
            tool_name for tool_name, tool_status in self.tool_status.items()
            if tool_status.usage_count == 0 and 
               (datetime.now() - (tool_status.last_used or datetime.now())).total_seconds() > 86400  # 1일
        ]
        
        if unused_tools:
            recommendations.append({
                "type": "resource_optimization",
                "priority": "medium",
                "affected_tools": unused_tools,
                "recommendation": "사용되지 않는 도구들을 정리할 수 있습니다",
                "action": "장기간 미사용 도구들의 제거 또는 비활성화를 고려하세요"
            })
        
        # 진화 실패가 많은 도구들
        failed_evolution_tools = [
            event.tool_name for event in self.evolution_history
            if not event.success and 
               (datetime.now() - event.timestamp).total_seconds() < 86400  # 1일 내
        ]
        
        failed_counts = Counter(failed_evolution_tools)
        problematic_tools = [tool for tool, count in failed_counts.items() if count >= 2]
        
        if problematic_tools:
            recommendations.append({
                "type": "evolution_strategy",
                "priority": "high",
                "affected_tools": problematic_tools,
                "recommendation": "진화 전략을 재검토해야 하는 도구들이 있습니다",
                "action": "진화 임계값 및 전략을 조정하거나 수동 개입을 고려하세요"
            })
        
        return recommendations
    
    def _update_ecosystem_stats(self):
        """생태계 통계 업데이트"""
        
        if not self.tool_status:
            return
        
        active_count = sum(1 for tool in self.tool_status.values() if tool.status == "active")
        evolving_count = sum(1 for tool in self.tool_status.values() if tool.status == "evolving")
        avg_performance = sum(tool.performance_score for tool in self.tool_status.values()) / len(self.tool_status)
        
        self.ecosystem_stats.update({
            "total_tools": len(self.tool_status),
            "active_tools": active_count,
            "evolving_tools": evolving_count,
            "avg_performance": avg_performance,
            "last_updated": datetime.now()
        })
    
    async def get_ecosystem_overview(self) -> Dict[str, Any]:
        """생태계 개요 조회"""
        
        return {
            "statistics": self.ecosystem_stats,
            "tool_distribution": self._get_tool_distribution(),
            "performance_summary": self._get_performance_summary(),
            "evolution_summary": self._get_evolution_summary(),
            "relationship_summary": self._get_relationship_summary(),
            "active_alerts": len(self.active_alerts),
            "health_score": self._calculate_ecosystem_health_score()
        }
    
    def _get_tool_distribution(self) -> Dict[str, Any]:
        """도구 분포 정보"""
        
        by_category = defaultdict(int)
        by_status = defaultdict(int)
        
        for tool_status in self.tool_status.values():
            by_category[tool_status.category] += 1
            by_status[tool_status.status] += 1
        
        return {
            "by_category": dict(by_category),
            "by_status": dict(by_status)
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약"""
        
        if not self.tool_status:
            return {"avg": 0, "min": 0, "max": 0, "distribution": {}}
        
        performances = [tool.performance_score for tool in self.tool_status.values()]
        
        # 성능 구간별 분포
        distribution = {"high": 0, "medium": 0, "low": 0}
        for perf in performances:
            if perf >= 0.8:
                distribution["high"] += 1
            elif perf >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return {
            "avg": sum(performances) / len(performances),
            "min": min(performances),
            "max": max(performances),
            "distribution": distribution
        }
    
    def _get_evolution_summary(self) -> Dict[str, Any]:
        """진화 요약"""
        
        if not self.evolution_history:
            return {"total": 0, "successful": 0, "success_rate": 0, "recent_count": 0}
        
        total_evolutions = len(self.evolution_history)
        successful_evolutions = sum(1 for event in self.evolution_history if event.success)
        
        # 최근 24시간 진화 수
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_evolutions = sum(
            1 for event in self.evolution_history 
            if event.timestamp > recent_cutoff
        )
        
        return {
            "total": total_evolutions,
            "successful": successful_evolutions,
            "success_rate": successful_evolutions / total_evolutions if total_evolutions > 0 else 0,
            "recent_count": recent_evolutions
        }
    
    def _get_relationship_summary(self) -> Dict[str, Any]:
        """관계 요약"""
        
        total_relationships = sum(len(rels) for rels in self.tool_relationships.values())
        
        relationship_types = defaultdict(int)
        for relationships in self.tool_relationships.values():
            for rel in relationships:
                relationship_types[rel.relationship_type] += 1
        
        return {
            "total_relationships": total_relationships,
            "by_type": dict(relationship_types),
            "avg_connections": total_relationships / len(self.tool_status) if self.tool_status else 0
        }
    
    def _calculate_ecosystem_health_score(self) -> float:
        """생태계 헬스 점수 계산"""
        
        if not self.tool_status:
            return 0.0
        
        # 성능 점수 (40%)
        avg_performance = self.ecosystem_stats["avg_performance"]
        performance_score = avg_performance * 0.4
        
        # 활성도 점수 (30%)
        activity_ratio = self.ecosystem_stats["active_tools"] / self.ecosystem_stats["total_tools"]
        activity_score = activity_ratio * 0.3
        
        # 진화 성공률 (20%)
        evolution_success_rate = (
            self.ecosystem_stats["successful_evolutions"] / 
            max(1, self.ecosystem_stats["total_evolutions"])
        )
        evolution_score = evolution_success_rate * 0.2
        
        # 경고 패널티 (10%)
        alert_penalty = min(0.1, len(self.active_alerts) * 0.02)
        
        return min(1.0, performance_score + activity_score + evolution_score + (0.1 - alert_penalty))
    
    async def get_tool_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """도구 상세 정보 조회"""
        
        if tool_name not in self.tool_status:
            return None
        
        tool_status = self.tool_status[tool_name]
        
        # 성능 히스토리
        performance_history = list(self.performance_history.get(tool_name, []))
        
        # 진화 히스토리
        evolution_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "strategy": event.strategy,
                "success": event.success,
                "performance_before": event.performance_before,
                "performance_after": event.performance_after,
                "description": event.description
            }
            for event in self.evolution_history
            if event.tool_name == tool_name
        ]
        
        # 관계 정보
        relationships = [
            {
                "target": rel.target_tool,
                "type": rel.relationship_type,
                "strength": rel.strength,
                "frequency": rel.frequency
            }
            for rel in self.tool_relationships.get(tool_name, [])
        ]
        
        return {
            "status": tool_status.__dict__,
            "performance_history": performance_history[-50:],  # 최근 50개
            "evolution_events": evolution_events,
            "relationships": relationships,
            "alerts": [
                alert.__dict__ for alert in self.active_alerts.values()
                if alert.tool_name == tool_name
            ]
        }
    
    async def get_network_visualization_data(self) -> Dict[str, Any]:
        """네트워크 시각화 데이터 조회"""
        
        nodes = []
        edges = []
        
        # 노드 생성 (도구들)
        for tool_name, tool_status in self.tool_status.items():
            node = {
                "id": tool_name,
                "label": tool_name,
                "category": tool_status.category,
                "status": tool_status.status,
                "performance": tool_status.performance_score,
                "size": 20 + tool_status.performance_score * 30,
                "color": self._get_status_color(tool_status.status),
                "metadata": {
                    "usage_count": tool_status.usage_count,
                    "success_rate": tool_status.success_rate,
                    "last_used": tool_status.last_used.isoformat() if tool_status.last_used else None
                }
            }
            nodes.append(node)
        
        # 엣지 생성 (관계들)
        for source_tool, relationships in self.tool_relationships.items():
            for rel in relationships:
                edge = {
                    "id": f"{source_tool}_{rel.target_tool}",
                    "source": source_tool,
                    "target": rel.target_tool,
                    "type": rel.relationship_type,
                    "strength": rel.strength,
                    "weight": rel.strength * 5,
                    "color": self._get_relationship_color(rel.relationship_type),
                    "style": "solid" if rel.strength > 0.7 else "dashed"
                }
                edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "force",
            "metadata": {
                "total_tools": len(nodes),
                "total_relationships": len(edges),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _get_status_color(self, status: str) -> str:
        """상태별 색상 반환"""
        
        color_map = {
            "active": "#27AE60",      # 초록
            "idle": "#F39C12",        # 주황
            "evolving": "#3498DB",    # 파랑
            "error": "#E74C3C"        # 빨강
        }
        
        return color_map.get(status, "#95A5A6")
    
    def _get_relationship_color(self, relationship_type: str) -> str:
        """관계 타입별 색상 반환"""
        
        color_map = {
            "collaborates": "#2ECC71",
            "enhances": "#3498DB",
            "depends_on": "#9B59B6",
            "competes_with": "#E74C3C"
        }
        
        return color_map.get(relationship_type, "#95A5A6")
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """경고 승인"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"MCP 경고 승인: {alert_id}")
            return True
        
        return False
    
    async def dismiss_alert(self, alert_id: str) -> bool:
        """경고 해제"""
        
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"MCP 경고 해제: {alert_id}")
            return True
        
        return False
    
    async def trigger_manual_evolution(self, tool_name: str, strategy: str) -> bool:
        """수동 진화 트리거"""
        
        if tool_name not in self.tool_status:
            return False
        
        try:
            # MCP 오케스트레이터를 통해 수동 진화 실행
            # 실제 구현에서는 MCP 오케스트레이터의 메소드 호출
            logger.info(f"수동 진화 트리거: {tool_name} ({strategy})")
            
            # 진화 중 상태로 변경
            self.tool_status[tool_name].status = "evolving"
            
            return True
            
        except Exception as e:
            logger.error(f"수동 진화 실패: {e}")
            return False
    
    async def export_ecosystem_report(self, format: str = "json") -> Optional[str]:
        """생태계 리포트 내보내기"""
        
        if format == "json":
            report_data = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "ecosystem_version": "1.0.0"
                },
                "overview": await self.get_ecosystem_overview(),
                "tools": {
                    name: status.__dict__
                    for name, status in self.tool_status.items()
                },
                "evolution_history": [
                    event.__dict__ for event in list(self.evolution_history)[-50:]
                ],
                "active_alerts": [
                    alert.__dict__ for alert in self.active_alerts.values()
                ]
            }
            
            return json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
        
        return None
    
    async def stop_monitoring(self):
        """모니터링 중단"""
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("MCP Monitor 모니터링 중단")
    
    async def cleanup(self):
        """MCP Monitor 정리"""
        
        await self.stop_monitoring()
        
        # 데이터 정리
        self.tool_status.clear()
        self.tool_relationships.clear()
        self.evolution_history.clear()
        self.performance_history.clear()
        self.active_alerts.clear()
        
        logger.info("MCP Monitor 정리 완료")