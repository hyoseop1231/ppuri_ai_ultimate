"""
Chunk Manager - 지능형 문서 청킹 관리자

문서를 의미 있는 청크로 분할하고 관리하는 실제 시스템.
뿌리산업 특화 청킹 전략과 최적화를 제공.

Features:
- 의미 기반 지능형 청킹
- 뿌리산업 특화 경계 감지
- 청크 크기 자동 최적화
- 오버랩 관리
- 메타데이터 추출
- 청크 품질 평가
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import json

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """청킹 설정"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    adaptive_sizing: bool = True
    preserve_boundaries: bool = True
    use_semantic_chunking: bool = True
    quality_threshold: float = 0.7


@dataclass
class ChunkQuality:
    """청크 품질 메트릭"""
    coherence_score: float  # 응집성 점수
    completeness_score: float  # 완성도 점수
    boundary_score: float  # 경계 점수
    size_score: float  # 크기 점수
    overall_score: float  # 전체 점수
    issues: List[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """청크 메타데이터"""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_position: int
    end_position: int
    content_length: int
    sentence_count: int
    paragraph_count: int
    industry_terms: List[str] = field(default_factory=list)
    entity_count: int = 0
    quality: Optional[ChunkQuality] = None
    processing_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessedChunk:
    """처리된 청크"""
    id: str
    content: str
    original_content: str
    normalized_content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    namespace: str = "default"


class ChunkManager:
    """
    지능형 청킹 관리자
    
    문서를 의미 있는 청크로 분할하고 품질을 관리하는
    실제 동작하는 청킹 시스템.
    """
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        korean_optimizer=None
    ):
        self.config = config or ChunkingConfig()
        self.korean_optimizer = korean_optimizer
        
        # 청킹 통계
        self.chunking_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "avg_chunks_per_doc": 0.0,
            "avg_quality_score": 0.0,
            "processing_time_total": 0.0
        }
        
        # 뿌리산업 특화 경계 패턴
        self.industry_boundary_patterns = self._init_boundary_patterns()
        
        # 문장 분할기 패턴
        self.sentence_patterns = self._init_sentence_patterns()
        
        logger.info("Chunk Manager 초기화 완료")
    
    def _init_boundary_patterns(self) -> Dict[str, List[str]]:
        """뿌리산업 특화 경계 패턴 초기화"""
        
        return {
            "주조": [
                r"(?:주조|캐스팅)\s*(?:공정|과정|단계)",
                r"(?:용탕|응고)\s*(?:특성|조건)",
                r"(?:주형|몰드)\s*(?:설계|제작)"
            ],
            "금형": [
                r"(?:금형|다이)\s*(?:설계|제작|가공)",
                r"(?:프레스|성형)\s*(?:공정|조건)",
                r"(?:표면|치수)\s*(?:정밀도|품질)"
            ],
            "소성가공": [
                r"(?:소성가공|단조|압연)\s*(?:공정|조건)",
                r"(?:변형|응력)\s*(?:해석|분석)",
                r"(?:재료|금속)\s*(?:유동|변형)"
            ],
            "용접": [
                r"(?:용접|접합)\s*(?:공정|방법|조건)",
                r"(?:열영향부|HAZ)\s*(?:특성|제어)",
                r"(?:용접부|접합부)\s*(?:품질|검사)"
            ],
            "표면처리": [
                r"(?:표면처리|코팅|도금)\s*(?:공정|방법)",
                r"(?:표면|피막)\s*(?:특성|품질)",
                r"(?:내식성|내마모성)\s*(?:평가|개선)"
            ],
            "열처리": [
                r"(?:열처리|소입|소성)\s*(?:공정|조건)",
                r"(?:조직|미세구조)\s*(?:변화|제어)",
                r"(?:경도|강도)\s*(?:측정|평가)"
            ]
        }
    
    def _init_sentence_patterns(self) -> List[str]:
        """문장 분할 패턴 초기화"""
        
        return [
            r'[.!?]+\s+',  # 기본 문장 끝
            r'[.!?]+\n',   # 줄바꿈과 함께
            r'\n\s*\n',    # 단락 구분
            r':\s*\n',     # 콜론 후 줄바꿈
            r';\s*\n',     # 세미콜론 후 줄바꿈
        ]
    
    async def chunk_document(
        self,
        content: str,
        document_id: str,
        namespace: str = "default",
        custom_config: Optional[ChunkingConfig] = None
    ) -> List[ProcessedChunk]:
        """
        문서 청킹
        
        Args:
            content: 문서 내용
            document_id: 문서 ID
            namespace: 네임스페이스
            custom_config: 커스텀 청킹 설정
            
        Returns:
            List[ProcessedChunk]: 처리된 청크 목록
        """
        
        start_time = asyncio.get_event_loop().time()
        config = custom_config or self.config
        
        try:
            # 1. 한국어 전처리
            normalized_content = await self._preprocess_content(content)
            
            # 2. 청킹 전략 결정
            chunking_strategy = self._determine_chunking_strategy(normalized_content, config)
            
            # 3. 청크 분할
            raw_chunks = await self._split_into_chunks(
                normalized_content, chunking_strategy
            )
            
            # 4. 청크 후처리 및 품질 평가
            processed_chunks = []
            for i, chunk_content in enumerate(raw_chunks):
                processed_chunk = await self._process_chunk(
                    chunk_content=chunk_content,
                    original_content=content,
                    chunk_index=i,
                    document_id=document_id,
                    namespace=namespace
                )
                
                if processed_chunk:
                    processed_chunks.append(processed_chunk)
            
            # 5. 청크 품질 최적화
            if config.adaptive_sizing:
                processed_chunks = await self._optimize_chunk_quality(processed_chunks)
            
            # 6. 통계 업데이트
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_chunking_stats(len(processed_chunks), processing_time)
            
            logger.info(
                f"문서 청킹 완료: {document_id} "
                f"({len(processed_chunks)}개 청크, {processing_time:.2f}초)"
            )
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"문서 청킹 실패 ({document_id}): {e}")
            return []
    
    async def _preprocess_content(self, content: str) -> str:
        """내용 전처리"""
        
        # 한국어 최적화 적용
        if self.korean_optimizer:
            result = await self.korean_optimizer.process_korean_text(content)
            return result.normalized_text
        
        # 기본 정규화
        normalized = content
        
        # 연속 공백 정리
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # 줄바꿈 정리
        normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', normalized)
        
        return normalized.strip()
    
    def _determine_chunking_strategy(
        self,
        content: str,
        config: ChunkingConfig
    ) -> Dict[str, Any]:
        """청킹 전략 결정"""
        
        content_length = len(content)
        sentence_count = len(re.findall(r'[.!?]+', content))
        paragraph_count = len(re.findall(r'\n\s*\n', content))
        
        # 뿌리산업 도메인 감지
        detected_domain = self._detect_industry_domain(content)
        
        strategy = {
            "method": "adaptive" if config.adaptive_sizing else "fixed",
            "base_chunk_size": config.chunk_size,
            "overlap_size": config.chunk_overlap,
            "preserve_boundaries": config.preserve_boundaries,
            "use_semantic": config.use_semantic_chunking,
            "detected_domain": detected_domain,
            "content_stats": {
                "length": content_length,
                "sentences": sentence_count,
                "paragraphs": paragraph_count
            }
        }
        
        # 도메인별 청킹 크기 조정
        if detected_domain:
            domain_adjustments = {
                "주조": {"size_multiplier": 1.2, "overlap_multiplier": 1.1},
                "금형": {"size_multiplier": 1.1, "overlap_multiplier": 1.0},
                "소성가공": {"size_multiplier": 1.3, "overlap_multiplier": 1.2},
                "용접": {"size_multiplier": 1.0, "overlap_multiplier": 1.1},
                "표면처리": {"size_multiplier": 1.1, "overlap_multiplier": 1.0},
                "열처리": {"size_multiplier": 1.2, "overlap_multiplier": 1.1}
            }
            
            if detected_domain in domain_adjustments:
                adj = domain_adjustments[detected_domain]
                strategy["base_chunk_size"] = int(
                    strategy["base_chunk_size"] * adj["size_multiplier"]
                )
                strategy["overlap_size"] = int(
                    strategy["overlap_size"] * adj["overlap_multiplier"]
                )
        
        return strategy
    
    def _detect_industry_domain(self, content: str) -> Optional[str]:
        """뿌리산업 도메인 감지"""
        
        domain_scores = {}
        
        for domain, patterns in self.industry_boundary_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
            
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    async def _split_into_chunks(
        self,
        content: str,
        strategy: Dict[str, Any]
    ) -> List[str]:
        """내용을 청크로 분할"""
        
        if strategy["use_semantic"]:
            return await self._semantic_chunking(content, strategy)
        else:
            return await self._fixed_chunking(content, strategy)
    
    async def _semantic_chunking(
        self,
        content: str,
        strategy: Dict[str, Any]
    ) -> List[str]:
        """의미 기반 청킹"""
        
        # 문장 단위로 분할
        sentences = self._split_sentences(content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        target_size = strategy["base_chunk_size"]
        overlap_size = strategy["overlap_size"]
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # 청크 크기 체크
            if current_length + sentence_length > target_size and current_chunk:
                # 현재 청크 완성
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text) >= strategy.get("min_chunk_size", 100):
                    chunks.append(chunk_text)
                
                # 다음 청크 시작 (오버랩 고려)
                overlap_sentences = self._calculate_overlap_sentences(
                    current_chunk, overlap_size
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # 뿌리산업 특화 경계 체크
            if strategy["preserve_boundaries"] and strategy["detected_domain"]:
                if self._is_semantic_boundary(sentence, strategy["detected_domain"]):
                    if current_chunk and current_length >= strategy.get("min_chunk_size", 100):
                        chunk_text = " ".join(current_chunk).strip()
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= strategy.get("min_chunk_size", 100):
                chunks.append(chunk_text)
        
        return chunks
    
    async def _fixed_chunking(
        self,
        content: str,
        strategy: Dict[str, Any]
    ) -> List[str]:
        """고정 크기 청킹"""
        
        chunks = []
        chunk_size = strategy["base_chunk_size"]
        overlap_size = strategy["overlap_size"]
        
        start = 0
        while start < len(content):
            end = start + chunk_size
            
            # 문장 경계에서 자르기 (preserve_boundaries가 True인 경우)
            if strategy["preserve_boundaries"] and end < len(content):
                # 가장 가까운 문장 끝 찾기
                sentence_end = self._find_nearest_sentence_boundary(
                    content, start, end
                )
                if sentence_end > start:
                    end = sentence_end
            
            chunk_content = content[start:end].strip()
            
            if chunk_content and len(chunk_content) >= strategy.get("min_chunk_size", 100):
                chunks.append(chunk_content)
            
            # 다음 청크 시작 위치 (오버랩 고려)
            start = max(start + chunk_size - overlap_size, end)
        
        return chunks
    
    def _split_sentences(self, content: str) -> List[str]:
        """문장 분할"""
        
        sentences = []
        
        # 복합 패턴으로 분할
        current_pos = 0
        
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                if match.start() > current_pos:
                    sentence = content[current_pos:match.end()].strip()
                    if sentence:
                        sentences.append(sentence)
                    current_pos = match.end()
        
        # 남은 내용 처리
        if current_pos < len(content):
            remaining = content[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    def _calculate_overlap_sentences(
        self,
        sentences: List[str],
        overlap_size: int
    ) -> List[str]:
        """오버랩 문장 계산"""
        
        if not sentences:
            return []
        
        overlap_sentences = []
        total_length = 0
        
        # 뒤에서부터 오버랩 크기만큼 수집
        for sentence in reversed(sentences):
            if total_length + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                total_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _is_semantic_boundary(self, sentence: str, domain: str) -> bool:
        """의미적 경계 판단"""
        
        if domain not in self.industry_boundary_patterns:
            return False
        
        patterns = self.industry_boundary_patterns[domain]
        
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        return False
    
    def _find_nearest_sentence_boundary(
        self,
        content: str,
        start: int,
        preferred_end: int
    ) -> int:
        """가장 가까운 문장 경계 찾기"""
        
        # 선호하는 끝 지점 근처에서 문장 끝 찾기
        search_window = min(200, len(content) - preferred_end)
        search_content = content[preferred_end:preferred_end + search_window]
        
        # 문장 끝 패턴 찾기
        sentence_ends = []
        for pattern in [r'[.!?]+\s', r'[.!?]+\n', r'\n\s*\n']:
            for match in re.finditer(pattern, search_content):
                sentence_ends.append(preferred_end + match.end())
        
        if sentence_ends:
            return min(sentence_ends)  # 가장 가까운 문장 끝
        
        return preferred_end
    
    async def _process_chunk(
        self,
        chunk_content: str,
        original_content: str,
        chunk_index: int,
        document_id: str,
        namespace: str
    ) -> Optional[ProcessedChunk]:
        """청크 처리"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            
            # 한국어 정규화
            normalized_content = chunk_content
            industry_terms = []
            
            if self.korean_optimizer:
                result = await self.korean_optimizer.process_korean_text(chunk_content)
                normalized_content = result.normalized_text
                industry_terms = result.industry_terms
            
            # 메타데이터 생성
            start_pos = original_content.find(chunk_content[:50])  # 대략적 위치
            end_pos = start_pos + len(chunk_content) if start_pos >= 0 else -1
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                start_position=max(0, start_pos),
                end_position=max(0, end_pos),
                content_length=len(chunk_content),
                sentence_count=len(re.findall(r'[.!?]+', chunk_content)),
                paragraph_count=len(re.findall(r'\n\s*\n', chunk_content)),
                industry_terms=industry_terms,
                entity_count=len(industry_terms),
                processing_time=asyncio.get_event_loop().time() - start_time
            )
            
            # 청크 품질 평가
            quality = await self._evaluate_chunk_quality(chunk_content, metadata)
            metadata.quality = quality
            
            # 품질 임계값 체크
            if quality.overall_score < self.config.quality_threshold:
                logger.debug(f"청크 품질 미달: {chunk_id} (점수: {quality.overall_score:.2f})")
                # 품질이 낮아도 일단 포함 (후에 최적화에서 처리)
            
            return ProcessedChunk(
                id=chunk_id,
                content=chunk_content,
                original_content=chunk_content,
                normalized_content=normalized_content,
                metadata=metadata,
                namespace=namespace
            )
            
        except Exception as e:
            logger.error(f"청크 처리 실패 ({chunk_index}): {e}")
            return None
    
    async def _evaluate_chunk_quality(
        self,
        content: str,
        metadata: ChunkMetadata
    ) -> ChunkQuality:
        """청크 품질 평가"""
        
        issues = []
        
        # 1. 응집성 점수 (문장 간 연결성)
        coherence_score = self._calculate_coherence_score(content)
        if coherence_score < 0.5:
            issues.append("낮은 응집성")
        
        # 2. 완성도 점수 (문장 완결성)
        completeness_score = self._calculate_completeness_score(content)
        if completeness_score < 0.6:
            issues.append("불완전한 문장")
        
        # 3. 경계 점수 (적절한 시작/끝)
        boundary_score = self._calculate_boundary_score(content)
        if boundary_score < 0.7:
            issues.append("부적절한 경계")
        
        # 4. 크기 점수 (적절한 크기)
        size_score = self._calculate_size_score(metadata.content_length)
        if size_score < 0.5:
            issues.append("부적절한 크기")
        
        # 전체 점수 (가중 평균)
        overall_score = (
            coherence_score * 0.3 +
            completeness_score * 0.3 +
            boundary_score * 0.2 +
            size_score * 0.2
        )
        
        return ChunkQuality(
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            boundary_score=boundary_score,
            size_score=size_score,
            overall_score=overall_score,
            issues=issues
        )
    
    def _calculate_coherence_score(self, content: str) -> float:
        """응집성 점수 계산"""
        
        sentences = self._split_sentences(content)
        if len(sentences) <= 1:
            return 1.0
        
        # 문장 간 연결 단어 확인
        connection_words = [
            "그러나", "하지만", "따라서", "그래서", "또한", "또는", "즉",
            "예를 들어", "구체적으로", "결과적으로", "반면에", "이에 따라"
        ]
        
        connection_count = 0
        for word in connection_words:
            connection_count += content.count(word)
        
        # 문장 수 대비 연결성 점수
        connection_ratio = connection_count / len(sentences)
        return min(1.0, connection_ratio * 2)  # 최대 1.0
    
    def _calculate_completeness_score(self, content: str) -> float:
        """완성도 점수 계산"""
        
        # 문장 완결성 체크
        sentences = self._split_sentences(content)
        complete_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence[-1] in '.!?':
                complete_sentences += 1
        
        if not sentences:
            return 0.0
        
        completeness = complete_sentences / len(sentences)
        
        # 시작과 끝이 자연스러운지 확인
        starts_naturally = not content.strip().startswith((',', '.', '!', '?'))
        ends_naturally = content.strip().endswith(('.', '!', '?', '\n'))
        
        boundary_bonus = 0.0
        if starts_naturally:
            boundary_bonus += 0.1
        if ends_naturally:
            boundary_bonus += 0.1
        
        return min(1.0, completeness + boundary_bonus)
    
    def _calculate_boundary_score(self, content: str) -> float:
        """경계 점수 계산"""
        
        content = content.strip()
        if not content:
            return 0.0
        
        score = 0.5  # 기본 점수
        
        # 좋은 시작 패턴
        good_starts = [
            r'^[A-Z가-힣]',  # 대문자나 한글로 시작
            r'^\d+\.',       # 번호로 시작
            r'^-\s',         # 리스트 항목
            r'^[가-힣]+은|는|이|가'  # 한국어 주어
        ]
        
        for pattern in good_starts:
            if re.match(pattern, content):
                score += 0.2
                break
        
        # 좋은 끝 패턴
        good_ends = [
            r'[.!?]$',       # 문장 부호로 끝
            r'다\.$',        # 한국어 어미
            r'음\.$',        # 한국어 어미
            r'니다\.$'       # 한국어 어미
        ]
        
        for pattern in good_ends:
            if re.search(pattern, content):
                score += 0.2
                break
        
        # 나쁜 패턴 체크
        bad_patterns = [
            r'^[,.]',        # 구두점으로 시작
            r'[^.!?]$',      # 문장 부호 없이 끝 (단, 줄바꿈은 제외)
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, content) and not content.endswith('\n'):
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_size_score(self, content_length: int) -> float:
        """크기 점수 계산"""
        
        min_size = self.config.min_chunk_size
        max_size = self.config.max_chunk_size
        optimal_size = self.config.chunk_size
        
        if content_length < min_size:
            return content_length / min_size * 0.5
        elif content_length > max_size:
            excess = content_length - max_size
            penalty = min(0.5, excess / optimal_size)
            return 1.0 - penalty
        else:
            # 최적 크기 근처일수록 높은 점수
            distance_from_optimal = abs(content_length - optimal_size)
            score = 1.0 - (distance_from_optimal / optimal_size) * 0.3
            return max(0.5, score)
    
    async def _optimize_chunk_quality(
        self,
        chunks: List[ProcessedChunk]
    ) -> List[ProcessedChunk]:
        """청크 품질 최적화"""
        
        if not chunks:
            return chunks
        
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # 품질이 좋으면 그대로 유지
            if current_chunk.metadata.quality.overall_score >= self.config.quality_threshold:
                optimized_chunks.append(current_chunk)
                i += 1
                continue
            
            # 품질이 낮으면 최적화 시도
            optimized = await self._try_optimize_chunk(chunks, i)
            
            if optimized:
                optimized_chunks.extend(optimized["chunks"])
                i = optimized["next_index"]
            else:
                # 최적화 실패시 원본 유지
                optimized_chunks.append(current_chunk)
                i += 1
        
        logger.debug(f"청크 품질 최적화: {len(chunks)} -> {len(optimized_chunks)}")
        return optimized_chunks
    
    async def _try_optimize_chunk(
        self,
        chunks: List[ProcessedChunk],
        index: int
    ) -> Optional[Dict[str, Any]]:
        """청크 최적화 시도"""
        
        current_chunk = chunks[index]
        quality = current_chunk.metadata.quality
        
        # 최적화 전략 결정
        strategies = []
        
        if quality.size_score < 0.5:
            if current_chunk.metadata.content_length < self.config.min_chunk_size:
                strategies.append("merge_next")
            elif current_chunk.metadata.content_length > self.config.max_chunk_size:
                strategies.append("split_chunk")
        
        if quality.boundary_score < 0.6:
            strategies.append("adjust_boundaries")
        
        if quality.completeness_score < 0.6:
            strategies.append("complete_sentences")
        
        # 전략 실행
        for strategy in strategies:
            result = await self._apply_optimization_strategy(chunks, index, strategy)
            if result:
                return result
        
        return None
    
    async def _apply_optimization_strategy(
        self,
        chunks: List[ProcessedChunk],
        index: int,
        strategy: str
    ) -> Optional[Dict[str, Any]]:
        """최적화 전략 적용"""
        
        if strategy == "merge_next" and index + 1 < len(chunks):
            # 다음 청크와 병합
            current = chunks[index]
            next_chunk = chunks[index + 1]
            
            merged_content = current.content + " " + next_chunk.content
            
            # 병합된 청크 생성
            merged_chunk = await self._process_chunk(
                chunk_content=merged_content,
                original_content=merged_content,
                chunk_index=current.metadata.chunk_index,
                document_id=current.metadata.document_id,
                namespace=current.namespace
            )
            
            if merged_chunk and merged_chunk.metadata.quality.overall_score > current.metadata.quality.overall_score:
                return {
                    "chunks": [merged_chunk],
                    "next_index": index + 2
                }
        
        elif strategy == "split_chunk":
            # 청크를 반으로 나누기
            current = chunks[index]
            mid_point = len(current.content) // 2
            
            # 문장 경계에서 나누기
            split_point = self._find_nearest_sentence_boundary(
                current.content, 0, mid_point
            )
            
            if split_point > 0 and split_point < len(current.content):
                first_part = current.content[:split_point].strip()
                second_part = current.content[split_point:].strip()
                
                # 두 개의 새 청크 생성
                first_chunk = await self._process_chunk(
                    chunk_content=first_part,
                    original_content=first_part,
                    chunk_index=current.metadata.chunk_index,
                    document_id=current.metadata.document_id,
                    namespace=current.namespace
                )
                
                second_chunk = await self._process_chunk(
                    chunk_content=second_part,
                    original_content=second_part,
                    chunk_index=current.metadata.chunk_index + 1,
                    document_id=current.metadata.document_id,
                    namespace=current.namespace
                )
                
                if (first_chunk and second_chunk and
                    first_chunk.metadata.quality.overall_score > current.metadata.quality.overall_score):
                    return {
                        "chunks": [first_chunk, second_chunk],
                        "next_index": index + 1
                    }
        
        return None
    
    def _update_chunking_stats(self, chunk_count: int, processing_time: float):
        """청킹 통계 업데이트"""
        
        self.chunking_stats["total_documents"] += 1
        self.chunking_stats["total_chunks"] += chunk_count
        self.chunking_stats["processing_time_total"] += processing_time
        
        # 평균 계산
        total_docs = self.chunking_stats["total_documents"]
        total_chunks = self.chunking_stats["total_chunks"]
        
        self.chunking_stats["avg_chunks_per_doc"] = total_chunks / total_docs
        self.chunking_stats["avg_processing_time"] = (
            self.chunking_stats["processing_time_total"] / total_docs
        )
    
    def get_chunking_statistics(self) -> Dict[str, Any]:
        """청킹 통계 조회"""
        
        return {
            **self.chunking_stats,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "adaptive_sizing": self.config.adaptive_sizing,
                "quality_threshold": self.config.quality_threshold
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def rechunk_with_config(
        self,
        chunks: List[ProcessedChunk],
        new_config: ChunkingConfig
    ) -> List[ProcessedChunk]:
        """새 설정으로 재청킹"""
        
        # 원본 문서 복원
        original_content = " ".join(chunk.original_content for chunk in chunks)
        document_id = chunks[0].metadata.document_id if chunks else "rechunk"
        namespace = chunks[0].namespace if chunks else "default"
        
        # 새 설정으로 청킹
        return await self.chunk_document(
            content=original_content,
            document_id=document_id,
            namespace=namespace,
            custom_config=new_config
        )
    
    async def cleanup(self):
        """Chunk Manager 정리"""
        
        # 통계 리셋
        self.chunking_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0.0,
            "avg_chunks_per_doc": 0.0,
            "avg_quality_score": 0.0,
            "processing_time_total": 0.0
        }
        
        logger.info("Chunk Manager 정리 완료")