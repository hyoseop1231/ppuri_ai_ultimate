"""
Korean Language Optimizer - 한국어 특화 최적화 시스템

KITECH RAG 챗봇에서 검증된 한국어 처리 최적화를 구현하여
뿌리산업 전문 용어와 한국어 자연어 처리를 최적화하는 시스템.

Features:
- 뿌리산업 전문 용어 처리
- 한국어 형태소 분석 최적화
- 한국어 임베딩 특화
- 자연스러운 한국어 응답 생성
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class KoreanProcessingResult:
    """한국어 처리 결과"""
    original_text: str
    normalized_text: str
    tokens: List[str]
    entities: List[Dict[str, Any]]
    industry_terms: List[str]
    confidence_score: float
    processing_time: float


@dataclass
class IndustryTermPattern:
    """뿌리산업 용어 패턴"""
    term: str
    category: str  # 주조, 금형, 소성가공, 용접, 표면처리, 열처리
    variations: List[str] = field(default_factory=list)
    context_keywords: List[str] = field(default_factory=list)
    importance: float = 1.0


class KoreanLanguageOptimizer:
    """
    한국어 특화 최적화 시스템
    
    KITECH에서 검증된 한국어 처리 패턴을 기반으로
    뿌리산업 전문 용어와 한국어 자연어를 최적화.
    """
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.korean_config = config_manager.get_korean_config()
        
        # 뿌리산업 전문 용어 사전
        self.industry_terms: Dict[str, IndustryTermPattern] = {}
        self._initialize_industry_terms()
        
        # 한국어 정규화 패턴
        self.normalization_patterns = self._create_normalization_patterns()
        
        # 한국어 불용어 (선택적)
        self.korean_stopwords = self._load_korean_stopwords()
        
        # 처리 통계
        self.processing_stats = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "term_detection_rate": 0.0
        }
        
        logger.info("Korean Language Optimizer 초기화 완료")
    
    def _initialize_industry_terms(self):
        """뿌리산업 전문 용어 초기화"""
        
        # === 주조 (Casting) ===
        casting_terms = [
            IndustryTermPattern(
                term="주조",
                category="주조",
                variations=["캐스팅", "casting", "鑄造"],
                context_keywords=["용탕", "주형", "응고", "사형"],
                importance=2.0
            ),
            IndustryTermPattern(
                term="용탕",
                category="주조", 
                variations=["용융금속", "molten metal"],
                context_keywords=["온도", "유동성", "주입"],
                importance=1.8
            ),
            IndustryTermPattern(
                term="사형주조",
                category="주조",
                variations=["sand casting", "砂型鑄造"],
                context_keywords=["모래", "점토", "바인더"],
                importance=1.5
            ),
            IndustryTermPattern(
                term="정밀주조",
                category="주조",
                variations=["투자주조", "investment casting", "로스트왁스"],
                context_keywords=["왁스", "세라믹", "정밀도"],
                importance=1.7
            )
        ]
        
        # === 금형 (Mold/Die) ===
        mold_terms = [
            IndustryTermPattern(
                term="금형",
                category="금형",
                variations=["몰드", "다이", "mold", "die", "金型"],
                context_keywords=["프레스", "사출", "성형"],
                importance=2.0
            ),
            IndustryTermPattern(
                term="사출금형",
                category="금형",
                variations=["injection mold", "射出金型"],
                context_keywords=["플라스틱", "수지", "게이트"],
                importance=1.8
            ),
            IndustryTermPattern(
                term="프레스금형",
                category="금형",
                variations=["press die", "프레스다이"],
                context_keywords=["판금", "성형", "펀치"],
                importance=1.7
            ),
            IndustryTermPattern(
                term="표면조도",
                category="금형",
                variations=["surface roughness", "Ra"],
                context_keywords=["마감", "품질", "측정"],
                importance=1.5
            )
        ]
        
        # === 소성가공 (Plastic Working) ===
        plastic_working_terms = [
            IndustryTermPattern(
                term="소성가공",
                category="소성가공",
                variations=["plastic working", "塑性加工"],
                context_keywords=["변형", "성형", "압력"],
                importance=2.0
            ),
            IndustryTermPattern(
                term="단조",
                category="소성가공",
                variations=["forging", "鍛造"],
                context_keywords=["해머", "프레스", "열간"],
                importance=1.8
            ),
            IndustryTermPattern(
                term="압연",
                category="소성가공",
                variations=["rolling", "圧延"],
                context_keywords=["롤", "판재", "두께"],
                importance=1.7
            ),
            IndustryTermPattern(
                term="압출",
                category="소성가공",
                variations=["extrusion", "押出"],
                context_keywords=["다이", "형재", "연속"],
                importance=1.6
            )
        ]
        
        # === 용접 (Welding) ===
        welding_terms = [
            IndustryTermPattern(
                term="용접",
                category="용접",
                variations=["welding", "溶接"],
                context_keywords=["접합", "용융", "전극"],
                importance=2.0
            ),
            IndustryTermPattern(
                term="아크용접",
                category="용접",
                variations=["arc welding", "아크"],
                context_keywords=["전기", "아크", "봉"],
                importance=1.8
            ),
            IndustryTermPattern(
                term="TIG용접",
                category="용접",
                variations=["TIG", "tungsten inert gas", "아르곤"],
                context_keywords=["텅스텐", "아르곤", "정밀"],
                importance=1.7
            ),
            IndustryTermPattern(
                term="레이저용접",
                category="용접",
                variations=["laser welding", "레이저"],
                context_keywords=["빔", "정밀", "자동화"],
                importance=1.6
            )
        ]
        
        # === 표면처리 (Surface Treatment) ===
        surface_terms = [
            IndustryTermPattern(
                term="표면처리",
                category="표면처리",
                variations=["surface treatment", "表面處理"],
                context_keywords=["코팅", "도금", "경화"],
                importance=2.0
            ),
            IndustryTermPattern(
                term="도금",
                category="표면처리",
                variations=["plating", "전기도금", "鍍金"],
                context_keywords=["전해", "니켈", "크롬"],
                importance=1.8
            ),
            IndustryTermPattern(
                term="양극산화",
                category="표면처리",
                variations=["anodizing", "아노다이징"],
                context_keywords=["알루미늄", "산화막", "착색"],
                importance=1.6
            ),
            IndustryTermPattern(
                term="PVD코팅",
                category="표면처리",
                variations=["PVD", "physical vapor deposition"],
                context_keywords=["진공", "증착", "박막"],
                importance=1.5
            )
        ]
        
        # === 열처리 (Heat Treatment) ===
        heat_treatment_terms = [
            IndustryTermPattern(
                term="열처리",
                category="열처리",
                variations=["heat treatment", "熱處理"],
                context_keywords=["온도", "시간", "조직"],
                importance=2.0
            ),
            IndustryTermPattern(
                term="담금질",
                category="열처리",
                variations=["quenching", "焼入れ"],
                context_keywords=["급냉", "경화", "마르텐사이트"],
                importance=1.8
            ),
            IndustryTermPattern(
                term="뜨임",
                category="열처리",
                variations=["tempering", "焼戻し"],
                context_keywords=["인성", "응력", "완화"],
                importance=1.7
            ),
            IndustryTermPattern(
                term="소둔",
                category="열처리",
                variations=["annealing", "燒鈍"],
                context_keywords=["연화", "응력제거", "재결정"],
                importance=1.6
            )
        ]
        
        # 모든 용어를 사전에 등록
        all_terms = (
            casting_terms + mold_terms + plastic_working_terms + 
            welding_terms + surface_terms + heat_treatment_terms
        )
        
        for term_pattern in all_terms:
            self.industry_terms[term_pattern.term] = term_pattern
            
            # 변형어도 등록
            for variation in term_pattern.variations:
                if variation not in self.industry_terms:
                    self.industry_terms[variation] = term_pattern
        
        logger.info(f"뿌리산업 전문 용어 {len(all_terms)}개 등록 완료")
    
    def _create_normalization_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """한국어 정규화 패턴 생성"""
        
        patterns = [
            # 공백 정규화
            (re.compile(r'\s+'), ' '),
            
            # 특수문자 정리
            (re.compile(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]'), ' '),
            
            # 연속된 같은 문자 정리
            (re.compile(r'(.)\1{2,}'), r'\1\1'),
            
            # 숫자+단위 정규화
            (re.compile(r'(\d+)\s*(mm|cm|m|kg|ton|도|℃|%|시간|분|초)'), r'\1\2'),
            
            # 영문+한글 혼용 정리
            (re.compile(r'([a-zA-Z]+)\s*([가-힣]+)'), r'\1 \2'),
            
            # 괄호 내용 정리
            (re.compile(r'\([^)]*\)'), ''),
            
            # 뿌리산업 약어 정규화
            (re.compile(r'(?i)cnc'), 'CNC'),
            (re.compile(r'(?i)cad'), 'CAD'),
            (re.compile(r'(?i)cam'), 'CAM'),
            (re.compile(r'(?i)tig'), 'TIG'),
            (re.compile(r'(?i)mig'), 'MIG'),
            (re.compile(r'(?i)pvd'), 'PVD'),
            (re.compile(r'(?i)cvd'), 'CVD'),
        ]
        
        return patterns
    
    def _load_korean_stopwords(self) -> set:
        """한국어 불용어 로드"""
        
        # 기본 한국어 불용어
        basic_stopwords = {
            '이', '그', '저', '것', '들', '의', '가', '에', '을', '를', '은', '는',
            '과', '와', '도', '만', '까지', '부터', '에서', '로', '으로', '에게',
            '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열',
            '있다', '없다', '되다', '하다', '이다', '아니다', '같다', '다르다',
            '크다', '작다', '많다', '적다', '좋다', '나쁘다', '새롭다', '오래되다'
        }
        
        # 뿌리산업 관련 일반 용어 (불용어가 아님)
        industry_preserve = {
            '기술', '공정', '소재', '품질', '생산', '제조', '가공', '처리',
            '온도', '압력', '시간', '속도', '정밀도', '표면', '내부', '외부'
        }
        
        # 불용어에서 산업 용어 제외
        if self.korean_config.get("use_stopwords", True):
            return basic_stopwords - industry_preserve
        else:
            return set()
    
    async def process_korean_text(
        self, 
        text: str,
        include_normalization: bool = True,
        include_entity_extraction: bool = True
    ) -> KoreanProcessingResult:
        """
        한국어 텍스트 종합 처리
        
        Args:
            text: 처리할 한국어 텍스트
            include_normalization: 정규화 포함 여부
            include_entity_extraction: 개체명 추출 포함 여부
            
        Returns:
            KoreanProcessingResult: 처리 결과
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            original_text = text
            processed_text = text
            
            # 1. 텍스트 정규화
            if include_normalization:
                processed_text = self.normalize_korean_text(processed_text)
            
            # 2. 토큰화 (간단한 공백 기반)
            tokens = self.tokenize_korean_text(processed_text)
            
            # 3. 뿌리산업 전문 용어 추출
            industry_terms = self.extract_industry_terms(processed_text)
            
            # 4. 개체명 추출 (산업 용어 중심)
            entities = []
            if include_entity_extraction:
                entities = self.extract_entities(processed_text, industry_terms)
            
            # 5. 신뢰도 점수 계산
            confidence_score = self.calculate_confidence_score(
                original_text, processed_text, industry_terms
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 통계 업데이트
            self._update_processing_stats(processing_time, len(industry_terms) > 0)
            
            result = KoreanProcessingResult(
                original_text=original_text,
                normalized_text=processed_text,
                tokens=tokens,
                entities=entities,
                industry_terms=industry_terms,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
            logger.debug(f"한국어 텍스트 처리 완료: {processing_time:.3f}초")
            return result
            
        except Exception as e:
            logger.error(f"한국어 텍스트 처리 실패: {e}")
            return KoreanProcessingResult(
                original_text=text,
                normalized_text=text,
                tokens=[],
                entities=[],
                industry_terms=[],
                confidence_score=0.0,
                processing_time=asyncio.get_event_loop().time() - start_time
            )
    
    def normalize_korean_text(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        
        if not self.korean_config.get("normalization", True):
            return text
        
        normalized = text.strip()
        
        # 정규화 패턴 적용
        for pattern, replacement in self.normalization_patterns:
            normalized = pattern.sub(replacement, normalized)
        
        # 추가 한국어 특화 정규화
        
        # 조사 정규화 (은/는, 이/가, 을/를)
        normalized = re.sub(r'([가-힣])은\s+', r'\1는 ', normalized)
        normalized = re.sub(r'([가-힣])이\s+', r'\1가 ', normalized)
        normalized = re.sub(r'([가-힣])을\s+', r'\1를 ', normalized)
        
        # 경어 정규화
        normalized = re.sub(r'습니다\s*\.?', '다.', normalized)
        normalized = re.sub(r'입니다\s*\.?', '이다.', normalized)
        
        # 최종 정리
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def tokenize_korean_text(self, text: str) -> List[str]:
        """한국어 토큰화"""
        
        # 간단한 공백 기반 토큰화 (실제로는 KoNLPy 등 사용 권장)
        tokens = text.split()
        
        # 불용어 제거
        if self.korean_stopwords:
            tokens = [token for token in tokens if token not in self.korean_stopwords]
        
        # 최소 길이 필터
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def extract_industry_terms(self, text: str) -> List[str]:
        """뿌리산업 전문 용어 추출"""
        
        found_terms = []
        text_lower = text.lower()
        
        # 등록된 용어 검색
        for term, pattern in self.industry_terms.items():
            # 용어 직접 매칭
            if term in text or term.lower() in text_lower:
                found_terms.append(term)
                continue
            
            # 변형어 매칭
            for variation in pattern.variations:
                if variation in text or variation.lower() in text_lower:
                    found_terms.append(term)
                    break
        
        # 중복 제거 및 중요도 순 정렬
        unique_terms = list(set(found_terms))
        unique_terms.sort(key=lambda t: self.industry_terms[t].importance, reverse=True)
        
        return unique_terms
    
    def extract_entities(
        self, 
        text: str, 
        industry_terms: List[str]
    ) -> List[Dict[str, Any]]:
        """개체명 추출 (산업 용어 중심)"""
        
        entities = []
        
        # 산업 용어를 개체로 등록
        for term in industry_terms:
            pattern = self.industry_terms[term]
            
            # 텍스트에서 용어 위치 찾기
            start_pos = text.find(term)
            if start_pos != -1:
                entities.append({
                    "text": term,
                    "label": f"INDUSTRY_{pattern.category.upper()}",
                    "start": start_pos,
                    "end": start_pos + len(term),
                    "confidence": pattern.importance / 2.0,
                    "category": pattern.category
                })
        
        # 숫자+단위 패턴 추출
        unit_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|kg|ton|도|℃|%|시간|분|초)')
        for match in unit_pattern.finditer(text):
            entities.append({
                "text": match.group(0),
                "label": "MEASUREMENT",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9,
                "value": float(match.group(1)),
                "unit": match.group(2)
            })
        
        return entities
    
    def calculate_confidence_score(
        self, 
        original_text: str, 
        processed_text: str, 
        industry_terms: List[str]
    ) -> float:
        """텍스트 처리 신뢰도 점수 계산"""
        
        score = 0.5  # 기본 점수
        
        # 산업 용어 발견 여부
        if industry_terms:
            score += 0.3 * min(1.0, len(industry_terms) / 3)
        
        # 한국어 비율
        korean_chars = len(re.findall(r'[가-힣]', original_text))
        total_chars = len(original_text.replace(' ', ''))
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            score += 0.2 * korean_ratio
        
        # 텍스트 길이 적정성
        if 10 <= len(original_text) <= 1000:
            score += 0.1
        
        # 정규화 효과
        if len(processed_text) < len(original_text):
            normalization_effect = (len(original_text) - len(processed_text)) / len(original_text)
            score += 0.1 * min(1.0, normalization_effect * 5)
        
        return min(1.0, score)
    
    def _update_processing_stats(self, processing_time: float, term_found: bool):
        """처리 통계 업데이트"""
        
        self.processing_stats["total_processed"] += 1
        
        # 평균 처리 시간 업데이트
        current_avg = self.processing_stats["avg_processing_time"]
        total = self.processing_stats["total_processed"]
        
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # 용어 탐지율 업데이트
        current_rate = self.processing_stats["term_detection_rate"]
        if term_found:
            self.processing_stats["term_detection_rate"] = (
                (current_rate * (total - 1) + 1.0) / total
            )
        else:
            self.processing_stats["term_detection_rate"] = (
                current_rate * (total - 1) / total
            )
    
    def optimize_korean_prompt(self, prompt: str) -> str:
        """한국어 프롬프트 최적화"""
        
        # 1. 기본 정규화
        optimized = self.normalize_korean_text(prompt)
        
        # 2. 뿌리산업 용어 강조
        industry_terms = self.extract_industry_terms(optimized)
        for term in industry_terms:
            # 용어를 강조 표시로 감싸기
            optimized = optimized.replace(term, f"**{term}**")
        
        # 3. 한국어 문체 자연스럽게 개선
        style_improvements = [
            (r'~에 대해서 알려주세요', '에 대해 설명해주세요'),
            (r'~를 설명해주세요', '에 대해 자세히 알려주세요'),
            (r'어떻게 해야 하나요', '방법을 알려주세요'),
            (r'뭐가 좋나요', '무엇이 권장되나요'),
            (r'어떤 것', '어떤 방법')
        ]
        
        for pattern, replacement in style_improvements:
            optimized = re.sub(pattern, replacement, optimized)
        
        # 4. 구체성 향상
        if any(domain in optimized for domain in ['주조', '금형', '소성가공', '용접', '표면처리', '열처리']):
            optimized += "\n\n구체적인 기술적 세부사항과 실무적 관점에서 답변해주세요."
        
        return optimized.strip()
    
    def get_industry_term_suggestions(self, partial_term: str) -> List[Dict[str, Any]]:
        """부분 용어로 뿌리산업 용어 제안"""
        
        suggestions = []
        partial_lower = partial_term.lower()
        
        for term, pattern in self.industry_terms.items():
            # 부분 매칭
            if partial_lower in term.lower():
                suggestions.append({
                    "term": term,
                    "category": pattern.category,
                    "importance": pattern.importance,
                    "variations": pattern.variations[:3],  # 상위 3개만
                    "match_score": len(partial_term) / len(term)
                })
        
        # 매칭 점수 순으로 정렬
        suggestions.sort(key=lambda x: x["match_score"], reverse=True)
        
        return suggestions[:10]  # 상위 10개만 반환
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        
        return {
            **self.processing_stats,
            "registered_terms": len(self.industry_terms),
            "korean_config": self.korean_config,
            "last_updated": datetime.now().isoformat()
        }