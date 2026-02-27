"""
PPuRI-AI Ultimate - OpenRouter LLM 클라이언트

OpenRouter를 통한 다양한 LLM 모델 통합 지원:
- Google Gemini 3 Pro/Flash Preview
- Anthropic Claude 3.5 Sonnet
- DeepSeek R1 (비용 효율)
- Qwen3 (한국어 특화)

Features:
- 자동 모델 라우팅 (품질/비용/속도 최적화)
- Reasoning 모드 지원 (Gemini 3 Pro)
- 스트리밍 응답
- 에러 핸들링 및 재시도
"""

import os
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, AsyncIterator, List, Union
from enum import Enum
from datetime import datetime

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """모델 등급 (용도별 최적화)"""
    REASONING = "reasoning"      # 복잡한 추론 작업 (Gemini 3 Pro)
    FAST = "fast"                # 빠른 응답 (Gemini 3 Flash)
    COST_EFFICIENT = "cost"      # 비용 최적화 (DeepSeek R1)
    KOREAN = "korean"            # 한국어 특화 (Qwen3)
    CODE = "code"                # 코드 생성 (Claude 3.5)
    IMAGE = "image"              # 이미지 생성 (Nano Banana Pro)
    BALANCED = "balanced"        # 균형 (기본)


class ReasoningEffort(Enum):
    """Gemini 3 Pro Reasoning 모드 강도"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelInfo:
    """모델 정보"""
    id: str
    name: str
    context_window: int
    input_price: float   # per 1M tokens
    output_price: float  # per 1M tokens
    supports_reasoning: bool = False
    supports_vision: bool = False
    supports_tools: bool = False


# OpenRouter 모델 카탈로그 (2025년 1월 기준)
OPENROUTER_MODELS: Dict[str, ModelInfo] = {
    # Google Gemini 3 시리즈
    "google/gemini-3-pro-preview": ModelInfo(
        id="google/gemini-3-pro-preview",
        name="Gemini 3 Pro Preview",
        context_window=1_000_000,
        input_price=2.0,
        output_price=12.0,
        supports_reasoning=True,
        supports_vision=True,
        supports_tools=True
    ),
    "google/gemini-3-flash-preview": ModelInfo(
        id="google/gemini-3-flash-preview",
        name="Gemini 3 Flash Preview",
        context_window=1_000_000,
        input_price=0.10,
        output_price=0.40,
        supports_reasoning=True,
        supports_vision=True,
        supports_tools=True
    ),
    "google/gemini-3-pro-image-preview": ModelInfo(
        id="google/gemini-3-pro-image-preview",
        name="Nano Banana Pro (Image)",
        context_window=32_000,
        input_price=0.50,
        output_price=2.0,
        supports_vision=True
    ),

    # Anthropic Claude
    "anthropic/claude-3.5-sonnet": ModelInfo(
        id="anthropic/claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        context_window=200_000,
        input_price=3.0,
        output_price=15.0,
        supports_vision=True,
        supports_tools=True
    ),

    # DeepSeek
    "deepseek/deepseek-r1": ModelInfo(
        id="deepseek/deepseek-r1",
        name="DeepSeek R1",
        context_window=64_000,
        input_price=0.14,
        output_price=2.19,
        supports_reasoning=True
    ),

    # Qwen (한국어 성능 우수)
    "qwen/qwen3-235b-a22b": ModelInfo(
        id="qwen/qwen3-235b-a22b",
        name="Qwen3 235B",
        context_window=128_000,
        input_price=0.14,
        output_price=0.60
    ),
}


@dataclass
class OpenRouterConfig:
    """OpenRouter 클라이언트 설정"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "google/gemini-3-flash-preview"
    timeout: float = 120.0

    # 모델 티어 매핑
    tier_model_map: Dict[ModelTier, str] = field(default_factory=lambda: {
        ModelTier.REASONING: "google/gemini-3-pro-preview",
        ModelTier.FAST: "google/gemini-3-flash-preview",
        ModelTier.COST_EFFICIENT: "deepseek/deepseek-r1",
        ModelTier.KOREAN: "qwen/qwen3-235b-a22b",
        ModelTier.CODE: "anthropic/claude-3.5-sonnet",
        ModelTier.IMAGE: "google/gemini-3-pro-image-preview",
        ModelTier.BALANCED: "google/gemini-3-flash-preview",
    })

    # 사이트 정보 (OpenRouter 리더보드용)
    site_url: str = "https://ppuri-ai.kitech.re.kr"
    site_name: str = "PPuRI-AI Ultimate"

    def get_model_for_tier(self, tier: ModelTier) -> str:
        return self.tier_model_map.get(tier, self.default_model)


@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    usage: Dict[str, int]
    reasoning_details: Optional[List[Dict]] = None
    finish_reason: str = "stop"
    latency_ms: float = 0.0


class OpenRouterClient:
    """
    OpenRouter LLM 클라이언트

    사용 예시:
    ```python
    client = OpenRouterClient(OpenRouterConfig(api_key="..."))
    await client.initialize()

    # 기본 생성
    response = await client.generate("TIG 용접 결함 원인은?")

    # 추론 모드
    response = await client.generate(
        "복잡한 분석 요청",
        tier=ModelTier.REASONING,
        reasoning=True,
        reasoning_effort=ReasoningEffort.HIGH
    )

    # 스트리밍
    async for chunk in client.generate_stream("질문"):
        print(chunk, end="")
    ```
    """

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        self.config = config or OpenRouterConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """클라이언트 초기화"""
        try:
            if not self.config.api_key:
                raise ValueError("OPENROUTER_API_KEY is required")

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.site_name,
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(self.config.timeout)
            )

            # 연결 테스트
            await self._health_check()

            self._initialized = True
            logger.info("OpenRouterClient initialized successfully")
            return True

        except Exception as e:
            logger.error(f"OpenRouterClient initialization failed: {e}")
            return False

    async def cleanup(self):
        """리소스 정리"""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    async def _health_check(self) -> bool:
        """연결 상태 확인"""
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.BALANCED,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        reasoning: bool = False,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        response_format: Optional[str] = None,  # "json" for JSON mode
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        LLM 응답 생성

        Args:
            prompt: 사용자 프롬프트
            tier: 모델 등급
            model: 특정 모델 ID (tier 오버라이드)
            system_prompt: 시스템 프롬프트
            messages: 대화 히스토리 (prompt 대신 사용)
            reasoning: Reasoning 모드 활성화 (Gemini 3 Pro)
            reasoning_effort: Reasoning 강도
            max_tokens: 최대 출력 토큰
            temperature: 샘플링 온도
            response_format: "json"이면 JSON 출력 강제
            tools: 함수 호출 도구 목록
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        start_time = datetime.now()

        # 모델 선택
        selected_model = model or self.config.get_model_for_tier(tier)
        model_info = OPENROUTER_MODELS.get(selected_model)

        # 메시지 구성
        if messages:
            final_messages = messages
        else:
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.append({"role": "user", "content": prompt})

        # 요청 페이로드
        payload = {
            "model": selected_model,
            "messages": final_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Reasoning 모드 (Gemini 3 Pro 전용)
        if reasoning and model_info and model_info.supports_reasoning:
            payload["reasoning"] = {
                "effort": reasoning_effort.value
            }

        # JSON 모드
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        # 도구 사용
        if tools and model_info and model_info.supports_tools:
            payload["tools"] = tools

        # 추가 파라미터
        payload.update(kwargs)

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            choice = data["choices"][0]

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            return LLMResponse(
                content=choice["message"]["content"],
                model=selected_model,
                usage=data.get("usage", {}),
                reasoning_details=choice["message"].get("reasoning_details"),
                finish_reason=choice.get("finish_reason", "stop"),
                latency_ms=latency_ms
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.BALANCED,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        스트리밍 응답 생성

        사용 예시:
        ```python
        async for chunk in client.generate_stream("질문"):
            print(chunk, end="", flush=True)
        ```
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        selected_model = model or self.config.get_model_for_tier(tier)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": True,
            **kwargs
        }

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            raise

    async def select_optimal_model(
        self,
        query: str,
        context: Optional[str] = None,
        prefer_cost: bool = False
    ) -> ModelTier:
        """
        쿼리 분석을 통한 최적 모델 자동 선택

        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            prefer_cost: True면 비용 효율 우선
        """
        query_lower = query.lower()

        # 비용 효율 우선 모드
        if prefer_cost:
            return ModelTier.COST_EFFICIENT

        # 코드 관련 키워드
        code_keywords = ["코드", "프로그램", "함수", "class", "def", "code", "python", "javascript"]
        if any(kw in query_lower for kw in code_keywords):
            return ModelTier.CODE

        # 복잡한 추론 필요 키워드
        reasoning_keywords = ["분석", "비교", "원인", "왜", "설계", "아키텍처", "전략", "최적화", "문제"]
        if any(kw in query_lower for kw in reasoning_keywords):
            return ModelTier.REASONING

        # 이미지 관련
        image_keywords = ["이미지", "그림", "사진", "다이어그램", "시각화", "그래프"]
        if any(kw in query_lower for kw in image_keywords):
            return ModelTier.IMAGE

        # 한국어 비중 계산
        korean_chars = sum(1 for c in query if '가' <= c <= '힣')
        total_chars = len(query.replace(" ", ""))
        korean_ratio = korean_chars / max(total_chars, 1)

        # 한국어 70% 이상이면 한국어 특화 모델
        if korean_ratio > 0.7:
            return ModelTier.KOREAN

        # 기본: 빠른 응답
        return ModelTier.FAST

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """비용 추정 (USD)"""
        model_info = OPENROUTER_MODELS.get(model)
        if not model_info:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * model_info.input_price
        output_cost = (output_tokens / 1_000_000) * model_info.output_price

        return input_cost + output_cost

    def get_available_models(self) -> Dict[str, ModelInfo]:
        """사용 가능한 모델 목록"""
        return OPENROUTER_MODELS.copy()


# 싱글톤 인스턴스
_client_instance: Optional[OpenRouterClient] = None


async def get_openrouter_client() -> OpenRouterClient:
    """OpenRouter 클라이언트 싱글톤 획득"""
    global _client_instance

    if _client_instance is None:
        _client_instance = OpenRouterClient()
        await _client_instance.initialize()

    return _client_instance
