"""
PPuRI-AI Ultimate - BGE-M3 임베딩 서비스

Features:
- Dense + Sparse 임베딩 동시 생성 (하이브리드 검색용)
- 한국어 최적화 (BGE-m3-ko 지원)
- 배치 처리 및 캐싱
- 8192 토큰 컨텍스트 지원
"""

import os
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """지원 임베딩 모델"""
    BGE_M3 = "BAAI/bge-m3"
    BGE_M3_KO = "upskyy/bge-m3-ko"  # 한국어 최적화
    MULTILINGUAL_E5 = "intfloat/multilingual-e5-large-instruct"


@dataclass
class EmbeddingConfig:
    """임베딩 서비스 설정"""
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"  # auto, cpu, cuda, mps
    max_length: int = 8192
    batch_size: int = 32
    normalize: bool = True
    use_fp16: bool = True
    cache_embeddings: bool = True
    cache_dir: str = "./.embedding_cache"


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    dense: np.ndarray  # Dense 벡터 (1024차원)
    sparse: Optional[Dict[int, float]] = None  # Sparse 벡터 (token_id: weight)
    colbert: Optional[np.ndarray] = None  # ColBERT 토큰 벡터
    text: str = ""
    model: str = ""


class BGEM3Service:
    """
    BGE-M3 임베딩 서비스

    특징:
    - Multi-Functionality: Dense, Sparse, ColBERT 동시 지원
    - Multi-Linguality: 100+ 언어 지원
    - Multi-Granularity: 문장~문서 수준 임베딩

    사용 예시:
    ```python
    service = BGEM3Service()
    await service.initialize()

    # 단일 텍스트 임베딩
    result = await service.embed("TIG 용접 기술")

    # 배치 임베딩
    results = await service.embed_batch(["텍스트1", "텍스트2"])

    # 하이브리드 검색용 (Dense + Sparse)
    result = await service.embed("쿼리", return_sparse=True)
    ```
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._cache: Dict[str, EmbeddingResult] = {}

    async def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            # 비동기 컨텍스트에서 동기 초기화 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)

            self._initialized = True
            logger.info(f"BGE-M3 Service initialized: {self.config.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 Service: {e}")
            return False

    def _load_model(self):
        """모델 로드 (동기)"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # 디바이스 선택
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.config.device

            logger.info(f"Loading model on device: {device}")

            # 모델 로드
            self._model = SentenceTransformer(
                self.config.model_name,
                device=device,
                trust_remote_code=True
            )

            # FP16 설정
            if self.config.use_fp16 and device in ["cuda", "mps"]:
                self._model.half()

            logger.info(f"Model loaded: {self.config.model_name}")

        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self._use_fallback = True

    async def embed(
        self,
        text: str,
        return_sparse: bool = False,
        return_colbert: bool = False
    ) -> EmbeddingResult:
        """
        단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트
            return_sparse: Sparse 벡터 반환 여부
            return_colbert: ColBERT 벡터 반환 여부
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        # 캐시 확인
        cache_key = self._get_cache_key(text, return_sparse, return_colbert)
        if self.config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]

        # 임베딩 생성
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._embed_sync,
            text,
            return_sparse,
            return_colbert
        )

        # 캐시 저장
        if self.config.cache_embeddings:
            self._cache[cache_key] = result

        return result

    def _embed_sync(
        self,
        text: str,
        return_sparse: bool,
        return_colbert: bool
    ) -> EmbeddingResult:
        """동기 임베딩 (내부용)"""
        try:
            # BGE-M3는 encode 메서드로 다양한 출력 지원
            if hasattr(self._model, 'encode'):
                # sentence-transformers 스타일
                dense = self._model.encode(
                    text,
                    normalize_embeddings=self.config.normalize,
                    convert_to_numpy=True
                )

                sparse = None
                colbert = None

                # BGE-M3 전용 기능 (FlagEmbedding 라이브러리 사용 시)
                if return_sparse and hasattr(self._model, 'encode_sparse'):
                    sparse = self._model.encode_sparse(text)

                if return_colbert and hasattr(self._model, 'encode_colbert'):
                    colbert = self._model.encode_colbert(text)

                return EmbeddingResult(
                    dense=np.array(dense),
                    sparse=sparse,
                    colbert=colbert,
                    text=text,
                    model=self.config.model_name
                )

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # 폴백: 제로 벡터 반환
            return EmbeddingResult(
                dense=np.zeros(1024),
                text=text,
                model=self.config.model_name
            )

    async def embed_batch(
        self,
        texts: List[str],
        return_sparse: bool = False,
        show_progress: bool = False
    ) -> List[EmbeddingResult]:
        """
        배치 임베딩

        Args:
            texts: 텍스트 리스트
            return_sparse: Sparse 벡터 반환 여부
            show_progress: 진행률 표시 여부
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        results = []

        # 배치 처리
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                None,
                self._embed_batch_sync,
                batch,
                return_sparse
            )

            results.extend(batch_results)

            if show_progress:
                progress = min(i + self.config.batch_size, len(texts))
                logger.info(f"Embedding progress: {progress}/{len(texts)}")

        return results

    def _embed_batch_sync(
        self,
        texts: List[str],
        return_sparse: bool
    ) -> List[EmbeddingResult]:
        """배치 동기 임베딩 (내부용)"""
        try:
            # 배치 인코딩
            dense_vectors = self._model.encode(
                texts,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                batch_size=len(texts)
            )

            results = []
            for i, text in enumerate(texts):
                results.append(EmbeddingResult(
                    dense=np.array(dense_vectors[i]),
                    sparse=None,  # 배치에서는 sparse 비활성화
                    text=text,
                    model=self.config.model_name
                ))

            return results

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [
                EmbeddingResult(dense=np.zeros(1024), text=t, model=self.config.model_name)
                for t in texts
            ]

    async def embed_query(self, query: str) -> EmbeddingResult:
        """
        쿼리 임베딩 (검색용 최적화)

        E5 계열 모델의 경우 "query: " 접두사 추가
        """
        # BGE-M3는 접두사 불필요, E5는 필요
        if "e5" in self.config.model_name.lower():
            query = f"query: {query}"

        return await self.embed(query, return_sparse=True)

    async def embed_document(self, document: str) -> EmbeddingResult:
        """
        문서 임베딩 (인덱싱용)

        E5 계열 모델의 경우 "passage: " 접두사 추가
        """
        if "e5" in self.config.model_name.lower():
            document = f"passage: {document}"

        return await self.embed(document)

    def compute_similarity(
        self,
        embedding1: EmbeddingResult,
        embedding2: EmbeddingResult,
        method: str = "cosine"
    ) -> float:
        """
        유사도 계산

        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            method: cosine, dot, euclidean
        """
        v1 = embedding1.dense
        v2 = embedding2.dense

        if method == "cosine":
            # 정규화된 벡터의 경우 dot product와 동일
            return float(np.dot(v1, v2))

        elif method == "dot":
            return float(np.dot(v1, v2))

        elif method == "euclidean":
            return float(-np.linalg.norm(v1 - v2))

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _get_cache_key(
        self,
        text: str,
        return_sparse: bool,
        return_colbert: bool
    ) -> str:
        """캐시 키 생성"""
        content = f"{text}_{return_sparse}_{return_colbert}_{self.config.model_name}"
        return hashlib.md5(content.encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        if not self._initialized:
            return {"status": "not_initialized", "model": None}

        try:
            # 간단한 테스트 임베딩
            test_result = await self.embed("테스트")
            return {
                "status": "healthy",
                "model": self.config.model_name,
                "embedding_dim": len(test_result.dense),
                "cache_size": len(self._cache)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def cleanup(self):
        """리소스 정리"""
        self._model = None
        self._cache.clear()
        self._initialized = False


# 싱글톤 인스턴스
_embedding_service: Optional[BGEM3Service] = None


async def get_embedding_service() -> BGEM3Service:
    """임베딩 서비스 싱글톤 획득"""
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = BGEM3Service()
        await _embedding_service.initialize()

    return _embedding_service
