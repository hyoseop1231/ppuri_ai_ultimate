"""
Relationship Builder - 지식 관계 구축 시스템

추출된 지식 요소들 간의 관계를 분석하여 의미있는 연결을 
생성하고 지식 그래프를 풍부하게 만드는 시스템.

Features:
- 자동 관계 탐지 및 생성
- 의미적 유사도 기반 연결
- 대화 맥락 기반 관계 구축
- 시간적 관계 모델링
- 관계 강도 및 신뢰도 계산
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import math

from .knowledge_extractor import ExtractedEntity, ExtractedRelation, ExtractedConcept
from .graph_manager import GraphManager, GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


@dataclass
class RelationshipScore:
    """관계 점수"""
    strength: float  # 관계 강도 (0.0 ~ 1.0)
    confidence: float  # 신뢰도 (0.0 ~ 1.0) 
    evidence_count: int  # 증거 개수
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """대화 맥락"""
    session_id: str
    user_id: Optional[str]
    message_sequence: int
    timestamp: datetime
    domain: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipCandidate:
    """관계 후보"""
    source_id: str
    target_id: str
    relation_type: str
    score: RelationshipScore
    evidence: List[str] = field(default_factory=list)
    context: Optional[ConversationContext] = None


class RelationshipBuilder:
    """
    지식 관계 구축기
    
    추출된 지식 요소들을 분석하여 의미있는 관계를 탐지하고
    지식 그래프에서 연결을 생성하는 시스템.
    """
    
    def __init__(self, graph_manager: GraphManager, korean_optimizer=None):
        self.graph_manager = graph_manager
        self.korean_optimizer = korean_optimizer
        
        # 관계 유형 정의
        self.relationship_types = self._define_relationship_types()
        
        # 유사도 임계값
        self.similarity_thresholds = {
            "semantic": 0.7,
            "textual": 0.6,
            "contextual": 0.5,
            "temporal": 0.8
        }
        
        # 관계 강도 계산 가중치
        self.strength_weights = {
            "co_occurrence": 0.3,
            "semantic_similarity": 0.4,
            "temporal_proximity": 0.2,
            "context_relevance": 0.1
        }
        
        # 관계 구축 통계
        self.building_stats = {
            "total_relationships_built": 0,
            "avg_relationship_strength": 0.0,
            "relationship_types_count": defaultdict(int),
            "avg_processing_time": 0.0
        }
        
        logger.info("Relationship Builder 초기화 완료")
    
    def _define_relationship_types(self) -> Dict[str, Dict[str, Any]]:
        """관계 유형 정의"""
        
        return {
            # === 뿌리산업 특화 관계 ===
            "USES_IN": {
                "description": "~에서 사용됨",
                "reverse": "USED_BY",
                "domains": ["주조", "금형", "소성가공", "용접", "표면처리", "열처리"],
                "patterns": [r"(.+)에서\s*(.+)을\s*사용", r"(.+)에\s*(.+)을\s*적용"]
            },
            "AFFECTS": {
                "description": "영향을 줌",
                "reverse": "AFFECTED_BY", 
                "patterns": [r"(.+)이\s*(.+)에\s*영향", r"(.+)이\s*(.+)을\s*좌우"]
            },
            "REQUIRES": {
                "description": "필요로 함",
                "reverse": "REQUIRED_BY",
                "patterns": [r"(.+)에는\s*(.+)이\s*필요", r"(.+)을\s*위해\s*(.+)이\s*요구"]
            },
            "CAUSES": {
                "description": "원인이 됨", 
                "reverse": "CAUSED_BY",
                "patterns": [r"(.+)이\s*(.+)을\s*야기", r"(.+)때문에\s*(.+)이\s*발생"]
            },
            "IMPROVES": {
                "description": "개선함",
                "reverse": "IMPROVED_BY",
                "patterns": [r"(.+)이\s*(.+)을\s*개선", r"(.+)으로\s*(.+)을\s*향상"]
            },
            "MEASURES": {
                "description": "측정함",
                "reverse": "MEASURED_BY", 
                "patterns": [r"(.+)의\s*(.+)을\s*측정", r"(.+)에서\s*(.+)을\s*평가"]
            },
            
            # === 일반적 관계 ===
            "RELATED_TO": {
                "description": "관련됨",
                "reverse": "RELATED_TO",
                "patterns": [r"(.+)와\s*(.+)은\s*관련", r"(.+)과\s*(.+)의\s*연관"]
            },
            "PART_OF": {
                "description": "부분임",
                "reverse": "HAS_PART",
                "patterns": [r"(.+)은\s*(.+)의\s*일부", r"(.+)에\s*(.+)이\s*포함"]
            },
            "SIMILAR_TO": {
                "description": "유사함",
                "reverse": "SIMILAR_TO",
                "patterns": [r"(.+)과\s*(.+)은\s*유사", r"(.+)와\s*(.+)는\s*비슷"]
            },
            
            # === 시간적 관계 ===
            "PRECEDES": {
                "description": "선행함",
                "reverse": "FOLLOWS",
                "patterns": [r"(.+)\s*다음에\s*(.+)", r"(.+)\s*후에\s*(.+)"]
            },
            "CONCURRENT": {
                "description": "동시에 발생",
                "reverse": "CONCURRENT",
                "patterns": [r"(.+)과\s*(.+)이\s*동시에", r"(.+)와\s*(.+)를\s*함께"]
            },
            
            # === 대화 맥락 관계 ===
            "DISCUSSED_WITH": {
                "description": "함께 논의됨",
                "reverse": "DISCUSSED_WITH",
                "context_based": True
            },
            "MENTIONED_AFTER": {
                "description": "이후에 언급됨", 
                "reverse": "MENTIONED_BEFORE",
                "context_based": True
            },
            "CO_OCCURS": {
                "description": "동시 출현",
                "reverse": "CO_OCCURS",
                "context_based": True
            }
        }
    
    async def build_relationships(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
        concepts: List[ExtractedConcept],
        context: Optional[ConversationContext] = None
    ) -> List[RelationshipCandidate]:
        """
        지식 요소들 간의 관계 구축
        
        Args:
            entities: 추출된 개체들
            relations: 추출된 관계들
            concepts: 추출된 개념들
            context: 대화 맥락
            
        Returns:
            List[RelationshipCandidate]: 구축된 관계 후보들
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            relationship_candidates = []
            
            # 1. 명시적 관계 구축 (추출된 관계 기반)
            explicit_candidates = await self._build_explicit_relationships(
                relations, entities, concepts, context
            )
            relationship_candidates.extend(explicit_candidates)
            
            # 2. 암시적 관계 구축 (유사도 기반)
            implicit_candidates = await self._build_implicit_relationships(
                entities, concepts, context
            )
            relationship_candidates.extend(implicit_candidates)
            
            # 3. 맥락 기반 관계 구축
            if context:
                contextual_candidates = await self._build_contextual_relationships(
                    entities, concepts, context
                )
                relationship_candidates.extend(contextual_candidates)
            
            # 4. 관계 후보 필터링 및 점수 계산
            filtered_candidates = await self._filter_and_score_relationships(
                relationship_candidates
            )
            
            # 5. 통계 업데이트
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_building_stats(filtered_candidates, processing_time)
            
            logger.debug(
                f"관계 구축 완료: {len(filtered_candidates)}개 관계 "
                f"({processing_time:.3f}초)"
            )
            
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"관계 구축 실패: {e}")
            return []
    
    async def _build_explicit_relationships(
        self,
        relations: List[ExtractedRelation],
        entities: List[ExtractedEntity],
        concepts: List[ExtractedConcept],
        context: Optional[ConversationContext]
    ) -> List[RelationshipCandidate]:
        """명시적 관계 구축"""
        
        candidates = []
        
        for relation in relations:
            # 주어와 목적어에 해당하는 노드 찾기
            source_node = await self._find_or_create_node(
                relation.subject, entities, concepts
            )
            target_node = await self._find_or_create_node(
                relation.object, entities, concepts
            )
            
            if source_node and target_node:
                # 관계 점수 계산
                score = RelationshipScore(
                    strength=relation.confidence,
                    confidence=relation.confidence,
                    evidence_count=1
                )
                
                candidates.append(RelationshipCandidate(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    relation_type=relation.relation_type,
                    score=score,
                    evidence=[relation.context],
                    context=context
                ))
        
        return candidates
    
    async def _build_implicit_relationships(
        self,
        entities: List[ExtractedEntity],
        concepts: List[ExtractedConcept],
        context: Optional[ConversationContext]
    ) -> List[RelationshipCandidate]:
        """암시적 관계 구축 (유사도 기반)"""
        
        candidates = []
        all_items = entities + concepts
        
        # 모든 조합에 대해 유사도 계산
        for i, item1 in enumerate(all_items):
            for item2 in all_items[i+1:]:
                similarity = await self._calculate_semantic_similarity(item1, item2)
                
                if similarity > self.similarity_thresholds["semantic"]:
                    # 관계 유형 결정
                    relation_type = self._determine_implicit_relation_type(
                        item1, item2, similarity
                    )
                    
                    # 노드 ID 결정
                    source_id = getattr(item1, 'text', getattr(item1, 'name', ''))
                    target_id = getattr(item2, 'text', getattr(item2, 'name', ''))
                    
                    if source_id and target_id:
                        score = RelationshipScore(
                            strength=similarity,
                            confidence=similarity * 0.8,  # 암시적이므로 낮은 신뢰도
                            evidence_count=1
                        )
                        
                        candidates.append(RelationshipCandidate(
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=relation_type,
                            score=score,
                            evidence=[f"유사도: {similarity:.3f}"],
                            context=context
                        ))
        
        return candidates
    
    async def _build_contextual_relationships(
        self,
        entities: List[ExtractedEntity],
        concepts: List[ExtractedConcept],
        context: ConversationContext
    ) -> List[RelationshipCandidate]:
        """맥락 기반 관계 구축"""
        
        candidates = []
        all_items = entities + concepts
        
        # 동일 세션/시간대의 요소들 간 관계
        for i, item1 in enumerate(all_items):
            for item2 in all_items[i+1:]:
                # 동시 출현 관계
                co_occurrence_score = self._calculate_co_occurrence_score(
                    item1, item2, context
                )
                
                if co_occurrence_score > self.similarity_thresholds["contextual"]:
                    source_id = getattr(item1, 'text', getattr(item1, 'name', ''))
                    target_id = getattr(item2, 'text', getattr(item2, 'name', ''))
                    
                    if source_id and target_id:
                        score = RelationshipScore(
                            strength=co_occurrence_score,
                            confidence=co_occurrence_score,
                            evidence_count=1
                        )
                        
                        candidates.append(RelationshipCandidate(
                            source_id=source_id,
                            target_id=target_id,
                            relation_type="DISCUSSED_WITH",
                            score=score,
                            evidence=[f"세션 {context.session_id}에서 동시 논의"],
                            context=context
                        ))
        
        return candidates
    
    async def _find_or_create_node(
        self,
        text: str,
        entities: List[ExtractedEntity],
        concepts: List[ExtractedConcept]
    ) -> Optional[GraphNode]:
        """텍스트에 해당하는 노드 찾기 또는 생성"""
        
        # 기존 노드 검색
        existing_nodes = await self.graph_manager.find_nodes(
            properties={"name": text}
        )
        
        if existing_nodes:
            return existing_nodes[0]
        
        # 엔티티에서 찾기
        for entity in entities:
            if entity.text == text:
                return await self.graph_manager.create_node(
                    labels=["Entity", entity.entity_type],
                    properties={
                        "name": entity.text,
                        "type": entity.entity_type,
                        "domain": entity.domain,
                        "confidence": entity.confidence
                    }
                )
        
        # 개념에서 찾기
        for concept in concepts:
            if concept.name == text:
                return await self.graph_manager.create_node(
                    labels=["Concept", concept.concept_type],
                    properties={
                        "name": concept.name,
                        "type": concept.concept_type,
                        "domain": concept.domain,
                        "description": concept.description,
                        "confidence": concept.confidence
                    }
                )
        
        # 새 노드 생성 (일반)
        return await self.graph_manager.create_node(
            labels=["General"],
            properties={"name": text}
        )
    
    async def _calculate_semantic_similarity(
        self,
        item1: Any,
        item2: Any
    ) -> float:
        """의미적 유사도 계산"""
        
        # 텍스트 추출
        text1 = getattr(item1, 'text', getattr(item1, 'name', ''))
        text2 = getattr(item2, 'text', getattr(item2, 'name', ''))
        
        if not text1 or not text2:
            return 0.0
        
        # 간단한 문자 기반 유사도 (실제로는 임베딩 사용 권장)
        similarity = self._calculate_jaccard_similarity(text1, text2)
        
        # 도메인 유사도 보정
        domain1 = getattr(item1, 'domain', None)
        domain2 = getattr(item2, 'domain', None)
        
        if domain1 and domain2 and domain1 == domain2:
            similarity *= 1.2  # 같은 도메인이면 유사도 증가
        
        # 타입 유사도 보정
        type1 = getattr(item1, 'entity_type', getattr(item1, 'concept_type', ''))
        type2 = getattr(item2, 'entity_type', getattr(item2, 'concept_type', ''))
        
        if type1 and type2 and type1 == type2:
            similarity *= 1.1  # 같은 타입이면 유사도 증가
        
        return min(1.0, similarity)
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard 유사도 계산"""
        
        # 문자 단위 집합
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = set1 & set2
        union = set1 | set2
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _determine_implicit_relation_type(
        self,
        item1: Any,
        item2: Any,
        similarity: float
    ) -> str:
        """암시적 관계 유형 결정"""
        
        # 도메인 기반 관계 결정
        domain1 = getattr(item1, 'domain', None)
        domain2 = getattr(item2, 'domain', None)
        
        if domain1 and domain2:
            if domain1 == domain2:
                return "RELATED_TO"  # 같은 도메인
            else:
                return "CROSS_DOMAIN"  # 다른 도메인
        
        # 타입 기반 관계 결정
        type1 = getattr(item1, 'entity_type', getattr(item1, 'concept_type', ''))
        type2 = getattr(item2, 'entity_type', getattr(item2, 'concept_type', ''))
        
        if type1 == "MATERIAL" and type2 == "PROCESS":
            return "USES_IN"
        elif type1 == "EQUIPMENT" and type2 == "PROCESS":
            return "REQUIRED_BY"
        elif similarity > 0.8:
            return "SIMILAR_TO"
        else:
            return "RELATED_TO"
    
    def _calculate_co_occurrence_score(
        self,
        item1: Any,
        item2: Any,
        context: ConversationContext
    ) -> float:
        """동시 출현 점수 계산"""
        
        # 기본 동시 출현 점수 (같은 메시지에 나타나면 1.0)
        base_score = 0.8
        
        # 도메인 관련성 보정
        domain1 = getattr(item1, 'domain', None)
        domain2 = getattr(item2, 'domain', None)
        
        if context.domain:
            if domain1 == context.domain:
                base_score += 0.1
            if domain2 == context.domain:
                base_score += 0.1
        
        # 메시지 순서 보정 (최근 메시지일수록 높은 점수)
        if context.message_sequence:
            recency_factor = 1.0 / (1.0 + context.message_sequence * 0.1)
            base_score *= recency_factor
        
        return min(1.0, base_score)
    
    async def _filter_and_score_relationships(
        self,
        candidates: List[RelationshipCandidate]
    ) -> List[RelationshipCandidate]:
        """관계 후보 필터링 및 점수 계산"""
        
        # 중복 제거
        unique_candidates = self._deduplicate_candidates(candidates)
        
        # 점수 재계산
        scored_candidates = []
        for candidate in unique_candidates:
            # 종합 점수 계산
            final_score = await self._calculate_final_relationship_score(candidate)
            candidate.score = final_score
            
            # 임계값 이상인 경우만 포함
            if final_score.strength > 0.3 and final_score.confidence > 0.2:
                scored_candidates.append(candidate)
        
        # 점수 순으로 정렬
        scored_candidates.sort(
            key=lambda c: c.score.strength * c.score.confidence,
            reverse=True
        )
        
        return scored_candidates
    
    def _deduplicate_candidates(
        self,
        candidates: List[RelationshipCandidate]
    ) -> List[RelationshipCandidate]:
        """관계 후보 중복 제거"""
        
        # 동일한 source-target-type 조합 찾기
        seen_relations = {}
        unique_candidates = []
        
        for candidate in candidates:
            relation_key = f"{candidate.source_id}:{candidate.target_id}:{candidate.relation_type}"
            
            if relation_key in seen_relations:
                # 기존 후보와 점수 비교하여 더 좋은 것 선택
                existing = seen_relations[relation_key]
                if candidate.score.strength > existing.score.strength:
                    # 증거 합치기
                    candidate.evidence.extend(existing.evidence)
                    candidate.score.evidence_count = len(candidate.evidence)
                    seen_relations[relation_key] = candidate
            else:
                seen_relations[relation_key] = candidate
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    async def _calculate_final_relationship_score(
        self,
        candidate: RelationshipCandidate
    ) -> RelationshipScore:
        """최종 관계 점수 계산"""
        
        # 기본 점수
        base_strength = candidate.score.strength
        base_confidence = candidate.score.confidence
        
        # 증거 개수 보정
        evidence_factor = min(1.0, candidate.score.evidence_count / 3.0)
        
        # 관계 타입별 보정
        type_factor = self._get_relation_type_factor(candidate.relation_type)
        
        # 맥락 관련성 보정
        context_factor = 1.0
        if candidate.context:
            context_factor = self._calculate_context_relevance(candidate)
        
        # 최종 점수 계산
        final_strength = base_strength * evidence_factor * type_factor * context_factor
        final_confidence = base_confidence * evidence_factor * type_factor
        
        return RelationshipScore(
            strength=min(1.0, final_strength),
            confidence=min(1.0, final_confidence),
            evidence_count=candidate.score.evidence_count
        )
    
    def _get_relation_type_factor(self, relation_type: str) -> float:
        """관계 타입별 가중치 팩터"""
        
        # 뿌리산업 특화 관계는 높은 가중치
        industry_relations = ["USES_IN", "AFFECTS", "REQUIRES", "CAUSES", "IMPROVES", "MEASURES"]
        
        if relation_type in industry_relations:
            return 1.2
        elif relation_type in ["RELATED_TO", "SIMILAR_TO"]:
            return 0.8
        else:
            return 1.0
    
    def _calculate_context_relevance(self, candidate: RelationshipCandidate) -> float:
        """맥락 관련성 계산"""
        
        if not candidate.context:
            return 1.0
        
        relevance = 1.0
        
        # 도메인 관련성
        if candidate.context.domain:
            # 도메인 정보를 활용한 관련성 계산 (구현 필요)
            pass
        
        # 시간적 관련성 (최근일수록 높은 관련성)
        time_diff = datetime.now() - candidate.context.timestamp
        if time_diff.total_seconds() < 3600:  # 1시간 이내
            relevance *= 1.1
        elif time_diff.total_seconds() > 86400:  # 1일 이후
            relevance *= 0.9
        
        return relevance
    
    async def persist_relationships(
        self,
        candidates: List[RelationshipCandidate]
    ) -> List[GraphRelationship]:
        """관계를 그래프에 저장"""
        
        persisted_relationships = []
        
        for candidate in candidates:
            try:
                # 관계 속성 구성
                properties = {
                    "strength": candidate.score.strength,
                    "confidence": candidate.score.confidence,
                    "evidence_count": candidate.score.evidence_count,
                    "evidence": candidate.evidence[:5],  # 최대 5개 증거만
                    "created_at": datetime.now().isoformat()
                }
                
                if candidate.context:
                    properties.update({
                        "session_id": candidate.context.session_id,
                        "message_sequence": candidate.context.message_sequence,
                        "domain": candidate.context.domain
                    })
                
                # 관계 생성
                relationship = await self.graph_manager.create_relationship(
                    start_node_id=candidate.source_id,
                    end_node_id=candidate.target_id,
                    relationship_type=candidate.relation_type,
                    properties=properties
                )
                
                persisted_relationships.append(relationship)
                
            except Exception as e:
                logger.error(f"관계 저장 실패 ({candidate.relation_type}): {e}")
        
        logger.info(f"관계 저장 완료: {len(persisted_relationships)}개")
        return persisted_relationships
    
    def _update_building_stats(
        self,
        candidates: List[RelationshipCandidate],
        processing_time: float
    ):
        """관계 구축 통계 업데이트"""
        
        self.building_stats["total_relationships_built"] += len(candidates)
        
        if candidates:
            # 평균 관계 강도 업데이트
            avg_strength = sum(c.score.strength for c in candidates) / len(candidates)
            total_built = self.building_stats["total_relationships_built"]
            current_avg = self.building_stats["avg_relationship_strength"]
            
            self.building_stats["avg_relationship_strength"] = (
                (current_avg * (total_built - len(candidates)) + avg_strength * len(candidates)) / total_built
            )
            
            # 관계 타입별 카운트 업데이트
            for candidate in candidates:
                self.building_stats["relationship_types_count"][candidate.relation_type] += 1
        
        # 평균 처리 시간 업데이트
        if self.building_stats["total_relationships_built"] > 0:
            current_avg_time = self.building_stats["avg_processing_time"]
            total = self.building_stats["total_relationships_built"] // len(candidates) if candidates else 1
            
            self.building_stats["avg_processing_time"] = (
                (current_avg_time * (total - 1) + processing_time) / total
            )
    
    def get_building_statistics(self) -> Dict[str, Any]:
        """관계 구축 통계 조회"""
        
        return {
            **self.building_stats,
            "relationship_types": list(self.relationship_types.keys()),
            "similarity_thresholds": self.similarity_thresholds,
            "strength_weights": self.strength_weights,
            "last_updated": datetime.now().isoformat()
        }
    
    async def strengthen_existing_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        new_evidence: str,
        strength_boost: float = 0.1
    ) -> bool:
        """기존 관계 강화"""
        
        try:
            # 기존 관계 찾기
            relationships = await self.graph_manager.find_relationships(
                start_node_id=source_id,
                end_node_id=target_id,
                relationship_type=relation_type
            )
            
            if not relationships:
                return False
            
            relationship = relationships[0]
            
            # 강도 증가 및 증거 추가
            current_strength = relationship.properties.get("strength", 0.5)
            new_strength = min(1.0, current_strength + strength_boost)
            
            current_evidence = relationship.properties.get("evidence", [])
            updated_evidence = current_evidence + [new_evidence]
            
            # 관계 업데이트
            update_query = """
            MATCH ()-[r {id: $rel_id}]->()
            SET r.strength = $new_strength,
                r.evidence = $new_evidence,
                r.evidence_count = $evidence_count,
                r.last_updated = $timestamp
            """
            
            await self.graph_manager.execute_query(update_query, {
                "rel_id": relationship.id,
                "new_strength": new_strength,
                "new_evidence": updated_evidence[:10],  # 최대 10개 증거
                "evidence_count": len(updated_evidence),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.debug(f"관계 강화 완료: {relation_type} ({new_strength:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"관계 강화 실패: {e}")
            return False