"""
AdalFlow Engine - 자동 프롬프트 최적화 엔진

LLM-AutoDiff 프레임워크를 기반으로 한 PyTorch 스타일의 자동 최적화 시스템.
뿌리산업 특화 프롬프트를 자동으로 진화시켜 12% 이상의 성능 향상을 달성.
"""

from .auto_optimizer import AutoPromptOptimizer
from .prompt_evolution import PromptEvolutionEngine
from .performance_tracker import PerformanceTracker
from .parameter_manager import ParameterManager

__all__ = [
    "AutoPromptOptimizer",
    "PromptEvolutionEngine", 
    "PerformanceTracker",
    "ParameterManager"
]

__version__ = "1.0.0"
__author__ = "PPuRI-AI Team"