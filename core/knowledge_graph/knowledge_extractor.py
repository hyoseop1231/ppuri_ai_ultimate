"""
Knowledge Extractor - 대화에서 지식 추출 시스템

대화 내용에서 지식 개념, 개체명, 관계를 자동으로 추출하여
지식 그래프를 구축하는 시스템.

Features:
- 대화 내용 자동 분석
- 뿌리산업 개념 추출
- 개체명 인식 (NER)
- 관계 추출 및 분류
- 지식 신뢰도 평가
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """추출된 개체"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    domain: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """추출된 관계"""
    subject: str
    predicate: str
    object: str
    confidence: float
    context: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedConcept:
    """추출된 개념"""
    name: str
    concept_type: str
    description: str
    domain: str
    confidence: float
    related_terms: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeExtractionResult:
    """지식 추출 결과"""
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    concepts: List[ExtractedConcept]
    confidence_score: float
    processing_time: float
    source_metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeExtractor:
    """
    대화 기반 지식 추출기
    
    대화 내용을 분석하여 뿌리산업 관련 지식을 
    자동으로 추출하고 구조화하는 시스템.
    """
    
    def __init__(self, korean_optimizer=None):
        self.korean_optimizer = korean_optimizer
        
        # 뿌리산업 특화 패턴
        self.industry_patterns = self._create_industry_patterns()
        self.entity_patterns = self._create_entity_patterns()
        self.relation_patterns = self._create_relation_patterns()
        
        # 개념 분류 체계
        self.concept_taxonomy = self._create_concept_taxonomy()
        
        # 처리 통계
        self.extraction_stats = {
            "total_extractions": 0,
            "avg_entities_per_text": 0.0,
            "avg_relations_per_text": 0.0,
            "avg_confidence": 0.0
        }
        
        logger.info("Knowledge Extractor 초기화 완료")
    
    def _create_industry_patterns(self) -> Dict[str, List[str]]:
        """뿌리산업 특화 패턴 정의"""
        
        return {
            "주조": [
                r"(?:용탕|molten metal)(?:\s*의?)?\s*(?:온도|temperature|유동성|점도)",
                r"(?:주형|mold)(?:\s*의?)?\s*(?:설계|design|제작|manufacturing)",
                r"(?:응고|solidification)(?:\s*의?)?\s*(?:과정|process|속도|rate)",
                r"(?:사형|sand\s*casting)(?:\s*의?)?\s*(?:조건|condition|방법|method)",
                r"(?:정밀주조|investment\s*casting)(?:\s*의?)?\s*(?:공정|process|특징|feature)"
            ],
            "금형": [
                r"(?:프레스|press)(?:\s*의?)?\s*(?:성형|forming|압력|pressure)",
                r"(?:사출|injection)(?:\s*의?)?\s*(?:조건|condition|온도|temperature)",
                r"(?:표면조도|surface\s*roughness)(?:\s*의?)?\s*(?:측정|measurement|개선|improvement)",
                r"(?:금형|die|mold)(?:\s*의?)?\s*(?:수명|life|내구성|durability)",
                r"(?:게이트|gate)(?:\s*의?)?\s*(?:설계|design|위치|location)"
            ],
            "소성가공": [
                r"(?:단조|forging)(?:\s*의?)?\s*(?:온도|temperature|압력|pressure)",
                r"(?:압연|rolling)(?:\s*의?)?\s*(?:두께|thickness|속도|speed)",
                r"(?:압출|extrusion)(?:\s*의?)?\s*(?:비|ratio|형상|shape)",
                r"(?:소성변형|plastic\s*deformation)(?:\s*의?)?\s*(?:한계|limit|특성|property)",
                r"(?:가공경화|work\s*hardening)(?:\s*의?)?\s*(?:효과|effect|정도|degree)"
            ],
            "용접": [
                r"(?:아크용접|arc\s*welding)(?:\s*의?)?\s*(?:전류|current|전압|voltage)",
                r"(?:TIG|tungsten\s*inert\s*gas)(?:\s*의?)?\s*(?:조건|condition|가스|gas)",
                r"(?:용접부|weld\s*zone)(?:\s*의?)?\s*(?:품질|quality|강도|strength)",
                r"(?:입열량|heat\s*input)(?:\s*의?)?\s*(?:제어|control|영향|effect)",
                r"(?:레이저용접|laser\s*welding)(?:\s*의?)?\s*(?:장점|advantage|한계|limitation)"
            ],
            "표면처리": [
                r"(?:도금|plating)(?:\s*의?)?\s*(?:두께|thickness|균일성|uniformity)",
                r"(?:양극산화|anodizing)(?:\s*의?)?\s*(?:조건|condition|색상|color)",
                r"(?:PVD|physical\s*vapor\s*deposition)(?:\s*의?)?\s*(?:공정|process|특성|property)",
                r"(?:코팅|coating)(?:\s*의?)?\s*(?:밀착성|adhesion|내구성|durability)",
                r"(?:전처리|pretreatment)(?:\s*의?)?\s*(?:중요성|importance|방법|method)"
            ],
            "열처리": [
                r"(?:담금질|quenching)(?:\s*의?)?\s*(?:온도|temperature|속도|rate)",
                r"(?:뜨임|tempering)(?:\s*의?)?\s*(?:온도|temperature|시간|time)",
                r"(?:소둔|annealing)(?:\s*의?)?\s*(?:효과|effect|조건|condition)",
                r"(?:경화|hardening)(?:\s*의?)?\s*(?:깊이|depth|정도|degree)",
                r"(?:조직|microstructure)(?:\s*의?)?\s*(?:변화|change|관찰|observation)"
            ]
        }
    
    def _create_entity_patterns(self) -> Dict[str, List[str]]:
        """개체명 인식 패턴 정의"""
        
        return {
            "MATERIAL": [
                r"(?:탄소강|carbon\s*steel|합금강|alloy\s*steel|스테인리스강|stainless\s*steel)",
                r"(?:알루미늄|aluminum|구리|copper|황동|brass|청동|bronze)",
                r"(?:주철|cast\s*iron|연철|wrought\s*iron|티타늄|titanium)",
                r"(?:세라믹|ceramic|플라스틱|plastic|고무|rubber)"
            ],
            "EQUIPMENT": [
                r"(?:용광로|blast\s*furnace|전기로|electric\s*furnace)",
                r"(?:프레스|press|해머|hammer|밀링머신|milling\s*machine)",
                r"(?:선반|lathe|드릴|drill|보링머신|boring\s*machine)",
                r"(?:용접기|welding\s*machine|절단기|cutting\s*machine)"
            ],
            "PROCESS": [
                r"(?:열간가공|hot\s*working|냉간가공|cold\s*working)",
                r"(?:기계가공|machining|표면가공|surface\s*treatment)",
                r"(?:조립|assembly|검사|inspection|품질관리|quality\s*control)"
            ],
            "MEASUREMENT": [
                r"(\d+(?:\.\d+)?)\s*(?:mm|센티미터|cm|미터|m)",
                r"(\d+(?:\.\d+)?)\s*(?:도|℃|°C|온도)",
                r"(\d+(?:\.\d+)?)\s*(?:MPa|GPa|kgf|톤|ton)",
                r"(\d+(?:\.\d+)?)\s*(?:%|퍼센트|percent)"
            ],
            "STANDARD": [
                r"(?:KS|JIS|ISO|ASTM|DIN)\s*[A-Z0-9\-]+",
                r"(?:한국산업표준|일본공업규격|국제표준화기구)",
                r"(?:품질기준|quality\s*standard|기술기준|technical\s*standard)"
            ]
        }
    
    def _create_relation_patterns(self) -> Dict[str, List[str]]:
        """관계 추출 패턴 정의"""
        
        return {
            "INFLUENCES": [
                r"(.+?)(?:이|가)\s*(.+?)(?:에|를|을)\s*(?:영향을\s*준다|영향한다|좌우한다)",
                r"(.+?)(?:은|는)\s*(.+?)(?:에|를|을)\s*(?:결정한다|좌우한다|변화시킨다)",
                r"(.+?)(?:이|가)\s*(?:높을수록|낮을수록|증가하면|감소하면)\s*(.+?)(?:이|가)\s*(?:증가|감소|향상|저하)"
            ],
            "REQUIRES": [
                r"(.+?)(?:을|를)\s*(?:위해서는|하기\s*위해서는)\s*(.+?)(?:이|가)\s*(?:필요하다|요구된다)",
                r"(.+?)(?:에는|에서는)\s*(.+?)(?:을|를)\s*(?:사용한다|적용한다|이용한다)",
                r"(.+?)(?:공정에서는|과정에서는)\s*(.+?)(?:이|가)\s*(?:중요하다|핵심이다)"
            ],
            "CAUSES": [
                r"(.+?)(?:이|가)\s*(.+?)(?:을|를)\s*(?:야기한다|발생시킨다|초래한다|원인이다)",
                r"(.+?)(?:때문에|으로\s*인해|로\s*인해)\s*(.+?)(?:이|가)\s*(?:발생한다|나타난다)",
                r"(.+?)(?:문제는|결함은)\s*(.+?)(?:에서|로부터)\s*(?:비롯된다|기인한다)"
            ],
            "IMPROVES": [
                r"(.+?)(?:이|가)\s*(.+?)(?:을|를)\s*(?:개선한다|향상시킨다|증진한다)",
                r"(.+?)(?:을|를)\s*(?:통해|이용해|사용해)\s*(.+?)(?:을|를)\s*(?:개선|향상|증진)할\s*수\s*있다",
                r"(.+?)(?:은|는)\s*(.+?)(?:에)\s*(?:효과적이다|유용하다|도움이\s*된다)"
            ],
            "MEASURES": [
                r"(.+?)(?:의)\s*(.+?)(?:이|가)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|도|℃|MPa|%)",
                r"(.+?)(?:에서)\s*(.+?)(?:을|를)\s*(\d+(?:\.\d+)?)\s*(?:으로|로)\s*(?:측정|설정|조정)"
            ]
        }
    
    def _create_concept_taxonomy(self) -> Dict[str, List[str]]:
        """개념 분류 체계 정의"""
        
        return {
            "TECHNOLOGY": ["기술", "공법", "방법", "기법", "과정", "공정"],
            "MATERIAL": ["소재", "재료", "물질", "합금", "강재", "금속"],
            "EQUIPMENT": ["장비", "기계", "설비", "도구", "기구", "장치"],
            "PROPERTY": ["특성", "성질", "물성", "강도", "경도", "연성"],
            "STANDARD": ["기준", "규격", "표준", "품질", "등급", "규정"],
            "DEFECT": ["결함", "불량", "문제", "오류", "이상", "손상"],
            "PARAMETER": ["조건", "변수", "파라미터", "인자", "요소", "요인"]
        }
    
    async def extract_knowledge(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> KnowledgeExtractionResult:
        """
        텍스트에서 지식 추출
        
        Args:
            text: 분석할 텍스트
            context: 추가 컨텍스트 정보
            
        Returns:
            KnowledgeExtractionResult: 추출된 지식
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. 한국어 전처리
            processed_text = text
            if self.korean_optimizer:
                korean_result = await self.korean_optimizer.process_korean_text(text)
                processed_text = korean_result.normalized_text
            
            # 2. 개체명 추출
            entities = await self._extract_entities(processed_text)
            
            # 3. 관계 추출
            relations = await self._extract_relations(processed_text, entities)
            
            # 4. 개념 추출
            concepts = await self._extract_concepts(processed_text, entities)
            
            # 5. 신뢰도 계산
            confidence_score = self._calculate_extraction_confidence(
                entities, relations, concepts, text
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 통계 업데이트
            self._update_extraction_stats(entities, relations, confidence_score)
            
            result = KnowledgeExtractionResult(
                entities=entities,
                relations=relations,
                concepts=concepts,
                confidence_score=confidence_score,
                processing_time=processing_time,
                source_metadata=context or {}
            )
            
            logger.debug(
                f"지식 추출 완료: {len(entities)}개 개체, "
                f"{len(relations)}개 관계, {len(concepts)}개 개념"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"지식 추출 실패: {e}")
            return KnowledgeExtractionResult(
                entities=[],
                relations=[],
                concepts=[],
                confidence_score=0.0,
                processing_time=asyncio.get_event_loop().time() - start_time
            )
    
    async def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """개체명 추출"""
        
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # 신뢰도 계산 (패턴 복잡도 기반)
                    confidence = min(1.0, len(pattern) / 50 + 0.3)
                    
                    # 도메인 감지
                    domain = self._detect_domain(entity_text)
                    
                    entities.append(ExtractedEntity(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        domain=domain
                    ))
        
        # 중복 제거 및 정렬
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        return entities
    
    async def _extract_relations(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        """관계 추출"""
        
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    
                    if len(groups) >= 2:
                        subject = groups[0].strip()
                        obj = groups[1].strip() if len(groups) >= 2 else ""
                        
                        # 측정값 관계의 경우
                        if relation_type == "MEASURES" and len(groups) >= 3:
                            predicate = f"measures_{groups[2]}"
                        else:
                            predicate = relation_type.lower()
                        
                        # 신뢰도 계산
                        confidence = self._calculate_relation_confidence(
                            subject, predicate, obj, match.group(0)
                        )
                        
                        relations.append(ExtractedRelation(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            confidence=confidence,
                            context=match.group(0),
                            relation_type=relation_type
                        ))
        
        return relations
    
    async def _extract_concepts(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[ExtractedConcept]:
        """개념 추출"""
        
        concepts = []
        
        # 뿌리산업 패턴 기반 개념 추출
        for domain, patterns in self.industry_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    concept_text = match.group(0)
                    
                    # 개념 유형 분류
                    concept_type = self._classify_concept(concept_text)
                    
                    # 설명 추출 (주변 텍스트 활용)
                    description = self._extract_concept_description(
                        text, match.start(), match.end()
                    )
                    
                    # 관련 용어 추출
                    related_terms = self._find_related_terms(concept_text, entities)
                    
                    # 신뢰도 계산
                    confidence = self._calculate_concept_confidence(
                        concept_text, description, related_terms
                    )
                    
                    concepts.append(ExtractedConcept(
                        name=concept_text,
                        concept_type=concept_type,
                        description=description,
                        domain=domain,
                        confidence=confidence,
                        related_terms=related_terms
                    ))
        
        # 중복 제거
        concepts = self._deduplicate_concepts(concepts)
        
        return concepts
    
    def _detect_domain(self, entity_text: str) -> Optional[str]:
        """개체의 도메인 감지"""
        
        entity_lower = entity_text.lower()
        
        # 도메인별 키워드 매칭
        domain_keywords = {
            "주조": ["주조", "용탕", "주형", "casting", "molten"],
            "금형": ["금형", "프레스", "사출", "mold", "die"],
            "소성가공": ["단조", "압연", "압출", "forging", "rolling"],
            "용접": ["용접", "아크", "TIG", "welding", "arc"],
            "표면처리": ["도금", "코팅", "표면", "plating", "coating"],
            "열처리": ["열처리", "담금질", "뜨임", "heat", "quenching"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in entity_lower for keyword in keywords):
                return domain
        
        return None
    
    def _classify_concept(self, concept_text: str) -> str:
        """개념 유형 분류"""
        
        concept_lower = concept_text.lower()
        
        for concept_type, keywords in self.concept_taxonomy.items():
            if any(keyword in concept_lower for keyword in keywords):
                return concept_type
        
        return "GENERAL"
    
    def _extract_concept_description(
        self, 
        text: str, 
        start_pos: int, 
        end_pos: int
    ) -> str:
        """개념 설명 추출"""
        
        # 주변 문맥에서 설명 추출 (전후 100자)
        context_start = max(0, start_pos - 100)
        context_end = min(len(text), end_pos + 100)
        
        context = text[context_start:context_end]
        
        # 설명 패턴 매칭
        description_patterns = [
            r"(?:이란|란|는|은)\s*(.{10,100})(?:\.|이다|다)",
            r"(?:의\s*의미는|를\s*말한다|를\s*의미한다)\s*(.{10,100})",
            r"(?:정의|설명|특징):\s*(.{10,100})"
        ]
        
        for pattern in description_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1).strip()
        
        # 기본 설명 (주변 텍스트 일부)
        return context[max(0, 50-start_pos):end_pos-start_pos+50].strip()
    
    def _find_related_terms(
        self, 
        concept_text: str, 
        entities: List[ExtractedEntity]
    ) -> List[str]:
        """관련 용어 찾기"""
        
        related_terms = []
        
        # 동일 도메인의 개체들 중 관련성 높은 것들 선택
        concept_domain = self._detect_domain(concept_text)
        
        for entity in entities:
            if entity.domain == concept_domain and entity.text != concept_text:
                # 유사도 기반 관련성 판단 (간단한 휴리스틱)
                if self._calculate_term_similarity(concept_text, entity.text) > 0.3:
                    related_terms.append(entity.text)
        
        return related_terms[:5]  # 최대 5개까지
    
    def _calculate_term_similarity(self, term1: str, term2: str) -> float:
        """용어 간 유사도 계산"""
        
        # 단순 문자열 유사도 (Jaccard)
        set1 = set(term1.lower())
        set2 = set(term2.lower())
        
        intersection = set1 & set2
        union = set1 | set2
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_relation_confidence(
        self, 
        subject: str, 
        predicate: str, 
        obj: str,
        context: str
    ) -> float:
        """관계 신뢰도 계산"""
        
        confidence = 0.5  # 기본값
        
        # 주어와 목적어의 개체 인식 여부
        if len(subject.strip()) > 2:
            confidence += 0.2
        if len(obj.strip()) > 2:
            confidence += 0.2
        
        # 관계 명시성
        if any(keyword in context.lower() for keyword in ["때문에", "으로 인해", "영향"]):
            confidence += 0.2
        
        # 수치가 포함된 경우
        if re.search(r'\d+', context):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_concept_confidence(
        self, 
        concept_text: str, 
        description: str, 
        related_terms: List[str]
    ) -> float:
        """개념 신뢰도 계산"""
        
        confidence = 0.3  # 기본값
        
        # 개념명의 명확성
        if len(concept_text) >= 3:
            confidence += 0.2
        
        # 설명의 존재 여부
        if len(description) > 10:
            confidence += 0.3
        
        # 관련 용어의 개수
        confidence += min(0.2, len(related_terms) * 0.05)
        
        return min(1.0, confidence)
    
    def _calculate_extraction_confidence(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation], 
        concepts: List[ExtractedConcept],
        text: str
    ) -> float:
        """전체 추출 신뢰도 계산"""
        
        if not entities and not relations and not concepts:
            return 0.0
        
        # 각 요소별 평균 신뢰도
        entity_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0
        relation_confidence = sum(r.confidence for r in relations) / len(relations) if relations else 0
        concept_confidence = sum(c.confidence for c in concepts) / len(concepts) if concepts else 0
        
        # 가중 평균
        weights = [0.3, 0.4, 0.3]  # 개체, 관계, 개념
        confidences = [entity_confidence, relation_confidence, concept_confidence]
        
        overall_confidence = sum(w * c for w, c in zip(weights, confidences))
        
        # 텍스트 길이 보정
        text_length_factor = min(1.0, len(text) / 500)  # 500자 기준
        
        return overall_confidence * text_length_factor
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """개체 중복 제거"""
        
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            entity_key = f"{entity.text.lower()}:{entity.entity_type}"
            
            if entity_key not in seen_texts:
                seen_texts.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_concepts(self, concepts: List[ExtractedConcept]) -> List[ExtractedConcept]:
        """개념 중복 제거"""
        
        unique_concepts = []
        seen_names = set()
        
        for concept in concepts:
            concept_key = f"{concept.name.lower()}:{concept.domain}"
            
            if concept_key not in seen_names:
                seen_names.add(concept_key)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _update_extraction_stats(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
        confidence: float
    ):
        """추출 통계 업데이트"""
        
        self.extraction_stats["total_extractions"] += 1
        total = self.extraction_stats["total_extractions"]
        
        # 평균 개체 수 업데이트
        current_avg_entities = self.extraction_stats["avg_entities_per_text"]
        self.extraction_stats["avg_entities_per_text"] = (
            (current_avg_entities * (total - 1) + len(entities)) / total
        )
        
        # 평균 관계 수 업데이트
        current_avg_relations = self.extraction_stats["avg_relations_per_text"]
        self.extraction_stats["avg_relations_per_text"] = (
            (current_avg_relations * (total - 1) + len(relations)) / total
        )
        
        # 평균 신뢰도 업데이트
        current_avg_confidence = self.extraction_stats["avg_confidence"]
        self.extraction_stats["avg_confidence"] = (
            (current_avg_confidence * (total - 1) + confidence) / total
        )
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """추출 통계 조회"""
        
        return {
            **self.extraction_stats,
            "supported_entity_types": list(self.entity_patterns.keys()),
            "supported_relation_types": list(self.relation_patterns.keys()),
            "supported_domains": list(self.industry_patterns.keys()),
            "concept_taxonomy": self.concept_taxonomy,
            "last_updated": datetime.now().isoformat()
        }
    
    async def extract_knowledge_batch(
        self, 
        texts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[KnowledgeExtractionResult]:
        """배치 지식 추출"""
        
        results = []
        
        for i, text in enumerate(texts):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = await self.extract_knowledge(text, context)
            results.append(result)
        
        logger.info(f"배치 지식 추출 완료: {len(texts)}개 텍스트")
        return results