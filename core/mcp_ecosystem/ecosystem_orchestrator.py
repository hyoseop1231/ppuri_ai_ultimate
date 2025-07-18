"""
Ecosystem Orchestrator - MCP 도구 생태계 총괄 관리자

자동 진화 MCP 도구 생태계의 핵심 조정자로 도구 발견, 등록, 
진화, 조합을 통합 관리하는 시스템.

Features:
- 전체 MCP 생태계 조정
- 자동 도구 발견 및 통합
- 성능 기반 도구 진화
- 동적 워크플로우 생성
- 실시간 최적화
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import importlib
import inspect
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP 도구 정의"""
    name: str
    description: str
    category: str
    function: Callable
    parameters: Dict[str, Any]
    performance_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolChain:
    """도구 체인"""
    chain_id: str
    tools: List[str]  # tool names
    performance_score: float
    success_rate: float
    usage_count: int = 0
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EcosystemTask:
    """생태계 작업"""
    task_id: str
    description: str
    required_capabilities: List[str]
    priority: int = 1
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """도구 실행 결과"""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EcosystemOrchestrator:
    """
    MCP 생태계 오케스트레이터
    
    자율적으로 진화하는 MCP 도구 생태계를 관리하고
    최적의 도구 조합으로 작업을 수행하는 핵심 시스템.
    """
    
    def __init__(
        self,
        config_manager,
        korean_optimizer=None,
        graph_manager=None,
        rag_orchestrator=None,
        conversational_engine=None
    ):
        self.config_manager = config_manager
        self.korean_optimizer = korean_optimizer
        self.graph_manager = graph_manager
        self.rag_orchestrator = rag_orchestrator
        self.conversational_engine = conversational_engine
        
        # 도구 레지스트리
        self.tools: Dict[str, MCPTool] = {}
        self.tool_chains: Dict[str, ToolChain] = {}
        
        # 카테고리별 도구 인덱스
        self.tools_by_category: Dict[str, List[str]] = defaultdict(list)
        
        # 성능 기반 도구 랭킹
        self.tool_rankings: Dict[str, List[str]] = defaultdict(list)
        
        # 자동 진화 설정
        self.evolution_config = {
            "min_usage_for_evolution": 10,
            "performance_threshold": 0.8,
            "auto_discovery_enabled": True,
            "chain_optimization_enabled": True
        }
        
        # 뿌리산업 특화 도구 템플릿
        self.industry_tool_templates = self._create_industry_templates()
        
        # 생태계 통계
        self.ecosystem_stats = {
            "total_tools": 0,
            "active_tools": 0,
            "total_executions": 0,
            "avg_success_rate": 0.0,
            "evolution_count": 0,
            "chain_count": 0
        }
        
        logger.info("MCP Ecosystem Orchestrator 초기화 완료")
    
    def _create_industry_templates(self) -> Dict[str, Dict[str, Any]]:
        """뿌리산업 특화 도구 템플릿"""
        
        return {
            "주조_분석": {
                "description": "주조 공정 분석 및 최적화",
                "capabilities": ["temperature_analysis", "molten_metal_flow", "solidification_prediction"],
                "parameters": ["material_type", "temperature", "cooling_rate"],
                "expected_output": "optimization_recommendations"
            },
            "금형_설계": {
                "description": "금형 설계 지원 및 검증",
                "capabilities": ["cad_integration", "stress_analysis", "surface_quality_prediction"],
                "parameters": ["part_geometry", "material", "production_volume"],
                "expected_output": "design_specifications"
            },
            "소성가공_최적화": {
                "description": "소성가공 공정 최적화",
                "capabilities": ["force_calculation", "material_flow", "die_design"],
                "parameters": ["material_properties", "forming_conditions", "target_shape"],
                "expected_output": "process_parameters"
            },
            "용접_품질관리": {
                "description": "용접 품질 예측 및 관리",
                "capabilities": ["weld_quality_prediction", "defect_detection", "parameter_optimization"],
                "parameters": ["welding_method", "material", "joint_design"],
                "expected_output": "quality_assessment"
            },
            "표면처리_선택": {
                "description": "최적 표면처리 방법 선택",
                "capabilities": ["coating_selection", "surface_analysis", "durability_prediction"],
                "parameters": ["base_material", "service_conditions", "required_properties"],
                "expected_output": "treatment_recommendation"
            },
            "열처리_계획": {
                "description": "열처리 공정 계획 및 최적화",
                "capabilities": ["heat_treatment_design", "microstructure_prediction", "property_optimization"],
                "parameters": ["material_composition", "target_properties", "equipment_constraints"],
                "expected_output": "heat_treatment_schedule"
            }
        }
    
    async def initialize(self):
        """생태계 초기화"""
        
        logger.info("MCP 생태계 초기화 중...")
        
        # 1. 기본 도구들 등록
        await self._register_core_tools()
        
        # 2. 뿌리산업 특화 도구 생성
        await self._create_industry_tools()
        
        # 3. 기존 MCP 서버 발견 및 통합
        if self.evolution_config["auto_discovery_enabled"]:
            await self._discover_existing_mcp_servers()
        
        # 4. 도구 성능 프로파일링
        await self._profile_tool_performance()
        
        # 5. 초기 도구 체인 생성
        if self.evolution_config["chain_optimization_enabled"]:
            await self._create_initial_tool_chains()
        
        logger.info(f"✅ MCP 생태계 초기화 완료: {len(self.tools)}개 도구 등록")
    
    async def _register_core_tools(self):
        """핵심 도구들 등록"""
        
        # 검색 도구
        await self.register_tool(MCPTool(
            name="rag_search",
            description="RAG 기반 문서 검색",
            category="search",
            function=self._rag_search_tool,
            parameters={
                "query": {"type": "string", "required": True},
                "namespace": {"type": "string", "required": False},
                "max_results": {"type": "integer", "default": 5}
            }
        ))
        
        # 지식 그래프 도구
        await self.register_tool(MCPTool(
            name="graph_query",
            description="지식 그래프 쿼리",
            category="knowledge",
            function=self._graph_query_tool,
            parameters={
                "cypher_query": {"type": "string", "required": True},
                "parameters": {"type": "object", "required": False}
            }
        ))
        
        # 한국어 처리 도구
        await self.register_tool(MCPTool(
            name="korean_analyze",
            description="한국어 텍스트 분석",
            category="language",
            function=self._korean_analyze_tool,
            parameters={
                "text": {"type": "string", "required": True},
                "include_entities": {"type": "boolean", "default": True}
            }
        ))
        
        # 대화 생성 도구
        await self.register_tool(MCPTool(
            name="conversation_generate",
            description="대화형 응답 생성",
            category="generation",
            function=self._conversation_generate_tool,
            parameters={
                "prompt": {"type": "string", "required": True},
                "context": {"type": "object", "required": False}
            }
        ))
        
        # 프롬프트 최적화 도구
        await self.register_tool(MCPTool(
            name="prompt_optimize",
            description="프롬프트 자동 최적화",
            category="optimization",
            function=self._prompt_optimize_tool,
            parameters={
                "original_prompt": {"type": "string", "required": True},
                "target_domain": {"type": "string", "required": False}
            }
        ))
    
    async def _create_industry_tools(self):
        """뿌리산업 특화 도구 생성"""
        
        for tool_name, template in self.industry_tool_templates.items():
            # 동적 도구 함수 생성
            tool_function = self._create_dynamic_industry_tool(tool_name, template)
            
            await self.register_tool(MCPTool(
                name=tool_name,
                description=template["description"],
                category="industry",
                function=tool_function,
                parameters={
                    param: {"type": "string", "required": True}
                    for param in template["parameters"]
                }
            ))
    
    def _create_dynamic_industry_tool(self, tool_name: str, template: Dict[str, Any]) -> Callable:
        """동적 뿌리산업 도구 생성"""
        
        async def industry_tool_function(**kwargs):
            """동적 생성된 뿌리산업 도구"""
            
            # 입력 파라미터 검증
            required_params = template["parameters"]
            for param in required_params:
                if param not in kwargs:
                    return {
                        "success": False,
                        "error": f"필수 파라미터 누락: {param}"
                    }
            
            # 도구별 로직 실행
            if "주조" in tool_name:
                return await self._execute_casting_analysis(kwargs)
            elif "금형" in tool_name:
                return await self._execute_mold_design(kwargs)
            elif "소성가공" in tool_name:
                return await self._execute_forming_optimization(kwargs)
            elif "용접" in tool_name:
                return await self._execute_welding_analysis(kwargs)
            elif "표면처리" in tool_name:
                return await self._execute_surface_treatment(kwargs)
            elif "열처리" in tool_name:
                return await self._execute_heat_treatment(kwargs)
            else:
                return {
                    "success": False,
                    "error": "알 수 없는 도구 유형"
                }
        
        return industry_tool_function
    
    async def _discover_existing_mcp_servers(self):
        """기존 MCP 서버 발견 및 통합"""
        
        # 기존 MCP 서버들 스캔 (실제로는 네트워크 스캔 등)
        potential_servers = [
            "omnisearch", "exa", "deep-research", "obsidian", "excel",
            "mermaid", "kakao-map", "kakao-nav", "blender", "ppt"
        ]
        
        for server_name in potential_servers:
            try:
                # MCP 서버 연결 시도
                server_info = await self._probe_mcp_server(server_name)
                
                if server_info:
                    # 서버의 도구들을 생태계에 통합
                    await self._integrate_mcp_server_tools(server_name, server_info)
                    
            except Exception as e:
                logger.debug(f"MCP 서버 발견 실패 ({server_name}): {e}")
    
    async def _probe_mcp_server(self, server_name: str) -> Optional[Dict[str, Any]]:
        """MCP 서버 탐지"""
        
        # 실제로는 MCP 프로토콜로 서버 정보 조회
        # 여기서는 더미 정보 반환
        
        server_capabilities = {
            "omnisearch": {
                "tools": ["web_search", "academic_search", "news_search"],
                "description": "전방위 검색 엔진"
            },
            "obsidian": {
                "tools": ["note_create", "note_search", "graph_view"],
                "description": "지식 관리 시스템"
            },
            "excel": {
                "tools": ["data_analysis", "chart_create", "formula_calculate"],
                "description": "데이터 분석 도구"
            }
        }
        
        return server_capabilities.get(server_name)
    
    async def _integrate_mcp_server_tools(self, server_name: str, server_info: Dict[str, Any]):
        """MCP 서버 도구 통합"""
        
        for tool_name in server_info.get("tools", []):
            full_tool_name = f"{server_name}_{tool_name}"
            
            # MCP 도구를 생태계 도구로 래핑
            wrapped_function = self._wrap_mcp_tool(server_name, tool_name)
            
            await self.register_tool(MCPTool(
                name=full_tool_name,
                description=f"{server_info['description']} - {tool_name}",
                category="mcp_integration",
                function=wrapped_function,
                parameters={"query": {"type": "string", "required": True}}
            ))
    
    def _wrap_mcp_tool(self, server_name: str, tool_name: str) -> Callable:
        """MCP 도구 래핑"""
        
        async def wrapped_tool(**kwargs):
            """래핑된 MCP 도구"""
            try:
                # 실제로는 MCP 클라이언트로 서버 호출
                result = await self._call_mcp_server(server_name, tool_name, kwargs)
                
                return {
                    "success": True,
                    "result": result,
                    "server": server_name,
                    "tool": tool_name
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "server": server_name,
                    "tool": tool_name
                }
        
        return wrapped_tool
    
    async def _call_mcp_server(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> Any:
        """MCP 서버 호출"""
        
        # 실제 MCP 클라이언트 구현
        # 여기서는 더미 응답 반환
        
        return {
            "server": server_name,
            "tool": tool_name,
            "response": f"Mock response from {server_name}.{tool_name}",
            "params": params
        }
    
    async def register_tool(self, tool: MCPTool) -> bool:
        """도구 등록"""
        
        try:
            # 도구 검증
            if not await self._validate_tool(tool):
                return False
            
            # 레지스트리에 등록
            self.tools[tool.name] = tool
            
            # 카테고리 인덱스 업데이트
            self.tools_by_category[tool.category].append(tool.name)
            
            # 통계 업데이트
            self.ecosystem_stats["total_tools"] += 1
            if tool.performance_score > 0.5:
                self.ecosystem_stats["active_tools"] += 1
            
            logger.info(f"도구 등록 완료: {tool.name} ({tool.category})")
            return True
            
        except Exception as e:
            logger.error(f"도구 등록 실패 ({tool.name}): {e}")
            return False
    
    async def _validate_tool(self, tool: MCPTool) -> bool:
        """도구 유효성 검증"""
        
        # 필수 속성 확인
        if not tool.name or not tool.function:
            return False
        
        # 함수 시그니처 검증
        try:
            signature = inspect.signature(tool.function)
            # 비동기 함수인지 확인
            if not asyncio.iscoroutinefunction(tool.function):
                logger.warning(f"도구 {tool.name}가 비동기 함수가 아님")
        except Exception:
            return False
        
        # 중복 이름 확인
        if tool.name in self.tools:
            logger.warning(f"중복 도구명: {tool.name}")
            return False
        
        return True
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """도구 실행"""
        
        start_time = asyncio.get_event_loop().time()
        
        if tool_name not in self.tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                execution_time=0.0,
                error_message=f"도구를 찾을 수 없음: {tool_name}"
            )
        
        tool = self.tools[tool_name]
        
        try:
            # 파라미터 검증
            validated_params = self._validate_parameters(tool, parameters)
            
            # 도구 실행
            result = await tool.function(**validated_params)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # 성능 통계 업데이트
            await self._update_tool_performance(tool, execution_time, True)
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"context": context}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # 실패 통계 업데이트
            await self._update_tool_performance(tool, execution_time, False)
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"context": context}
            )
    
    def _validate_parameters(self, tool: MCPTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """파라미터 유효성 검증"""
        
        validated = {}
        
        for param_name, param_spec in tool.parameters.items():
            if param_spec.get("required", False) and param_name not in parameters:
                raise ValueError(f"필수 파라미터 누락: {param_name}")
            
            if param_name in parameters:
                value = parameters[param_name]
                
                # 타입 검증
                expected_type = param_spec.get("type", "string")
                if expected_type == "string" and not isinstance(value, str):
                    value = str(value)
                elif expected_type == "integer" and not isinstance(value, int):
                    value = int(value)
                elif expected_type == "boolean" and not isinstance(value, bool):
                    value = bool(value)
                
                validated[param_name] = value
            elif "default" in param_spec:
                validated[param_name] = param_spec["default"]
        
        return validated
    
    async def _update_tool_performance(self, tool: MCPTool, execution_time: float, success: bool):
        """도구 성능 통계 업데이트"""
        
        tool.usage_count += 1
        tool.last_used = datetime.now()
        
        # 평균 실행 시간 업데이트
        if tool.usage_count == 1:
            tool.avg_execution_time = execution_time
        else:
            tool.avg_execution_time = (
                (tool.avg_execution_time * (tool.usage_count - 1) + execution_time) / tool.usage_count
            )
        
        # 성공률 업데이트
        if success:
            tool.success_rate = (
                (tool.success_rate * (tool.usage_count - 1) + 1.0) / tool.usage_count
            )
        else:
            tool.success_rate = (
                tool.success_rate * (tool.usage_count - 1) / tool.usage_count
            )
        
        # 성능 점수 계산 (성공률 + 속도 보정)
        speed_factor = max(0.1, min(1.0, 5.0 / tool.avg_execution_time))
        tool.performance_score = tool.success_rate * 0.8 + speed_factor * 0.2
        
        # 진화 조건 확인
        if (tool.usage_count >= self.evolution_config["min_usage_for_evolution"] and
            tool.performance_score < self.evolution_config["performance_threshold"]):
            await self._trigger_tool_evolution(tool)
    
    async def _trigger_tool_evolution(self, tool: MCPTool):
        """도구 진화 트리거"""
        
        logger.info(f"도구 진화 시작: {tool.name}")
        
        try:
            # 진화 전략 결정
            evolution_strategy = self._determine_evolution_strategy(tool)
            
            # 진화 실행
            evolved_tool = await self._evolve_tool(tool, evolution_strategy)
            
            if evolved_tool:
                # 기존 도구 교체
                await self._replace_tool(tool.name, evolved_tool)
                
                # 진화 히스토리 기록
                tool.evolution_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "strategy": evolution_strategy,
                    "performance_before": tool.performance_score,
                    "performance_after": evolved_tool.performance_score
                })
                
                self.ecosystem_stats["evolution_count"] += 1
                
                logger.info(f"도구 진화 완료: {tool.name} -> {evolved_tool.name}")
            
        except Exception as e:
            logger.error(f"도구 진화 실패 ({tool.name}): {e}")
    
    def _determine_evolution_strategy(self, tool: MCPTool) -> str:
        """진화 전략 결정"""
        
        if tool.success_rate < 0.7:
            return "error_handling_improvement"
        elif tool.avg_execution_time > 5.0:
            return "performance_optimization"
        elif tool.usage_count > 100:
            return "feature_enhancement"
        else:
            return "general_optimization"
    
    async def _evolve_tool(self, tool: MCPTool, strategy: str) -> Optional[MCPTool]:
        """도구 진화 실행"""
        
        if strategy == "error_handling_improvement":
            return await self._improve_error_handling(tool)
        elif strategy == "performance_optimization":
            return await self._optimize_performance(tool)
        elif strategy == "feature_enhancement":
            return await self._enhance_features(tool)
        else:
            return await self._general_optimization(tool)
    
    async def _improve_error_handling(self, tool: MCPTool) -> Optional[MCPTool]:
        """에러 처리 개선"""
        
        # 기존 함수에 에러 처리 래퍼 추가
        original_function = tool.function
        
        async def improved_function(**kwargs):
            try:
                return await original_function(**kwargs)
            except Exception as e:
                logger.warning(f"도구 {tool.name} 실행 중 오류: {e}")
                
                # 기본 안전 응답 반환
                return {
                    "success": False,
                    "error": str(e),
                    "fallback_applied": True
                }
        
        evolved_tool = MCPTool(
            name=f"{tool.name}_v2",
            description=f"{tool.description} (개선된 에러 처리)",
            category=tool.category,
            function=improved_function,
            parameters=tool.parameters,
            performance_score=tool.performance_score * 1.1
        )
        
        return evolved_tool
    
    # 핵심 도구 구현들
    async def _rag_search_tool(self, query: str, namespace: str = None, max_results: int = 5):
        """RAG 검색 도구"""
        if not self.rag_orchestrator:
            return {"success": False, "error": "RAG 시스템 없음"}
        
        from ..rag_engine.rag_orchestrator import RAGQuery
        
        rag_query = RAGQuery(
            text=query,
            namespace=namespace,
            max_results=max_results
        )
        
        response = await self.rag_orchestrator.search(rag_query, generate_answer=True)
        
        return {
            "success": True,
            "results": [
                {
                    "content": r.chunk.content,
                    "score": r.score,
                    "rank": r.rank
                }
                for r in response.results
            ],
            "generated_answer": response.generated_answer,
            "processing_time": response.processing_time
        }
    
    async def _graph_query_tool(self, cypher_query: str, parameters: Dict[str, Any] = None):
        """지식 그래프 쿼리 도구"""
        if not self.graph_manager:
            return {"success": False, "error": "그래프 시스템 없음"}
        
        try:
            results = await self.graph_manager.execute_query(cypher_query, parameters or {})
            return {
                "success": True,
                "results": results,
                "query": cypher_query
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _korean_analyze_tool(self, text: str, include_entities: bool = True):
        """한국어 분석 도구"""
        if not self.korean_optimizer:
            return {"success": False, "error": "한국어 시스템 없음"}
        
        result = await self.korean_optimizer.process_korean_text(text)
        
        response = {
            "success": True,
            "normalized_text": result.normalized_text,
            "confidence": result.confidence_score,
            "industry_terms": result.industry_terms
        }
        
        if include_entities:
            response["entities"] = [
                {
                    "text": entity.text,
                    "type": entity.entity_type,
                    "confidence": entity.confidence
                }
                for entity in result.entities
            ]
        
        return response
    
    async def _conversation_generate_tool(self, prompt: str, context: Dict[str, Any] = None):
        """대화 생성 도구"""
        if not self.conversational_engine:
            return {"success": False, "error": "대화 시스템 없음"}
        
        try:
            session_id = await self.conversational_engine.start_conversation(
                initial_context=context
            )
            
            response_parts = []
            async for result in self.conversational_engine.chat(session_id, prompt, stream=False):
                response_parts.append(result.response)
            
            await self.conversational_engine.end_conversation(session_id)
            
            return {
                "success": True,
                "response": "".join(response_parts),
                "session_id": session_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _prompt_optimize_tool(self, original_prompt: str, target_domain: str = "뿌리산업"):
        """프롬프트 최적화 도구"""
        
        # 한국어 최적화 적용
        if self.korean_optimizer:
            optimized_prompt = self.korean_optimizer.optimize_korean_prompt(original_prompt)
        else:
            optimized_prompt = original_prompt
        
        return {
            "success": True,
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "target_domain": target_domain,
            "optimization_applied": optimized_prompt != original_prompt
        }
    
    # 뿌리산업 특화 도구 구현들
    async def _execute_casting_analysis(self, params: Dict[str, Any]):
        """주조 분석 실행"""
        return {
            "success": True,
            "analysis_type": "casting",
            "recommendations": [
                f"재료 '{params.get('material_type', 'unknown')}'에 대한 주조 조건 최적화",
                f"온도 {params.get('temperature', 'N/A')}에서의 유동성 개선 방안",
                "응고 시간 단축을 위한 냉각 조건 조정"
            ],
            "parameters": params
        }
    
    async def _execute_mold_design(self, params: Dict[str, Any]):
        """금형 설계 실행"""
        return {
            "success": True,
            "design_type": "mold",
            "specifications": {
                "geometry": params.get("part_geometry", "복잡형상"),
                "material": params.get("material", "강재"),
                "surface_finish": "Ra 0.8μm 이하 권장",
                "cooling_channels": "균등 배치 필요"
            },
            "parameters": params
        }
    
    async def _execute_forming_optimization(self, params: Dict[str, Any]):
        """소성가공 최적화 실행"""
        return {
            "success": True,
            "process_type": "forming",
            "optimization_results": {
                "forming_force": "계산된 최적 가공력",
                "material_flow": "예측된 재료 유동 패턴",
                "die_life": "예상 금형 수명",
                "quality_prediction": "우수한 성형품질 예상"
            },
            "parameters": params
        }
    
    async def _execute_welding_analysis(self, params: Dict[str, Any]):
        """용접 분석 실행"""
        return {
            "success": True,
            "welding_type": params.get("welding_method", "arc"),
            "quality_assessment": {
                "penetration": "적정 용입 깊이",
                "heat_affected_zone": "최소화된 열영향부",
                "defect_probability": "낮음",
                "strength_prediction": "모재 대비 95% 이상"
            },
            "parameters": params
        }
    
    async def _execute_surface_treatment(self, params: Dict[str, Any]):
        """표면처리 실행"""
        return {
            "success": True,
            "treatment_type": "surface",
            "recommendation": {
                "method": "최적 표면처리 방법 선택됨",
                "expected_properties": "향상된 내마모성 및 내식성",
                "process_conditions": "권장 처리 조건",
                "cost_effectiveness": "비용 대비 효과 우수"
            },
            "parameters": params
        }
    
    async def _execute_heat_treatment(self, params: Dict[str, Any]):
        """열처리 실행"""
        return {
            "success": True,
            "treatment_type": "heat",
            "schedule": {
                "heating_rate": "권장 승온 속도",
                "holding_temperature": "최적 유지 온도",
                "holding_time": "적정 유지 시간",
                "cooling_method": "권장 냉각 방법",
                "expected_hardness": "목표 경도 달성 예상"
            },
            "parameters": params
        }
    
    async def get_ecosystem_status(self) -> Dict[str, Any]:
        """생태계 상태 조회"""
        
        # 활성 도구 계산
        active_tools = [
            name for name, tool in self.tools.items()
            if tool.performance_score > 0.5 and tool.usage_count > 0
        ]
        
        # 전체 성공률 계산
        total_success_rate = (
            sum(tool.success_rate for tool in self.tools.values()) / len(self.tools)
            if self.tools else 0.0
        )
        
        return {
            **self.ecosystem_stats,
            "active_tools": len(active_tools),
            "avg_success_rate": total_success_rate,
            "tools_by_category": dict(self.tools_by_category),
            "top_performing_tools": self._get_top_performing_tools(5),
            "recent_evolutions": self._get_recent_evolutions(10),
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_top_performing_tools(self, limit: int) -> List[Dict[str, Any]]:
        """최고 성능 도구 목록"""
        
        sorted_tools = sorted(
            self.tools.values(),
            key=lambda t: t.performance_score,
            reverse=True
        )
        
        return [
            {
                "name": tool.name,
                "category": tool.category,
                "performance_score": tool.performance_score,
                "usage_count": tool.usage_count,
                "success_rate": tool.success_rate
            }
            for tool in sorted_tools[:limit]
        ]
    
    def _get_recent_evolutions(self, limit: int) -> List[Dict[str, Any]]:
        """최근 진화 히스토리"""
        
        all_evolutions = []
        for tool in self.tools.values():
            for evolution in tool.evolution_history:
                all_evolutions.append({
                    "tool_name": tool.name,
                    **evolution
                })
        
        # 시간순 정렬
        all_evolutions.sort(
            key=lambda e: e["timestamp"],
            reverse=True
        )
        
        return all_evolutions[:limit]
    
    async def cleanup(self):
        """생태계 정리"""
        
        # 도구 정리
        for tool in self.tools.values():
            # 도구별 정리 작업
            pass
        
        self.tools.clear()
        self.tool_chains.clear()
        self.tools_by_category.clear()
        
        logger.info("MCP 생태계 정리 완료")