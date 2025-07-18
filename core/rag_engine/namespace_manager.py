"""
Namespace Manager - 지능형 네임스페이스 관리자

문서와 지식을 네임스페이스별로 조직화하고 관리하는 실제 시스템.
뿌리산업 도메인 특화 자동 분류와 격리를 제공.

Features:
- 네임스페이스 생성 및 관리
- 자동 도메인 분류
- 네임스페이스 간 문서 이동
- 접근 권한 관리
- 통계 및 모니터링
- 네임스페이스 최적화
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class NamespaceType(Enum):
    """네임스페이스 타입"""
    DEFAULT = "default"
    INDUSTRY = "industry"  # 뿌리산업 특화
    PROJECT = "project"    # 프로젝트별
    USER = "user"         # 사용자별
    SYSTEM = "system"     # 시스템용
    TEMP = "temporary"    # 임시


class AccessLevel(Enum):
    """접근 권한 레벨"""
    PUBLIC = "public"     # 누구나 접근
    RESTRICTED = "restricted"  # 제한적 접근
    PRIVATE = "private"   # 개인 전용
    SYSTEM = "system"     # 시스템 전용


@dataclass
class NamespaceConfig:
    """네임스페이스 설정"""
    name: str
    display_name: str
    namespace_type: NamespaceType
    access_level: AccessLevel
    description: str = ""
    max_documents: int = 10000
    auto_cleanup: bool = False
    cleanup_days: int = 30
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NamespaceStats:
    """네임스페이스 통계"""
    document_count: int = 0
    chunk_count: int = 0
    total_size_bytes: int = 0
    last_updated: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    search_count: int = 0
    avg_document_size: float = 0.0
    top_domains: List[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class DocumentLocation:
    """문서 위치 정보"""
    namespace: str
    document_id: str
    chunk_ids: List[str] = field(default_factory=list)
    indexed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NamespaceMapping:
    """네임스페이스 매핑"""
    source_namespace: str
    target_namespace: str
    mapping_rules: Dict[str, Any]
    confidence: float
    auto_generated: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class NamespaceManager:
    """
    지능형 네임스페이스 관리자
    
    문서와 지식을 도메인별로 조직화하고 효율적으로 관리하는
    실제 동작하는 네임스페이스 시스템.
    """
    
    def __init__(
        self,
        config_manager,
        korean_optimizer=None,
        data_path: Optional[str] = None
    ):
        self.config_manager = config_manager
        self.korean_optimizer = korean_optimizer
        self.data_path = Path(data_path) if data_path else Path("./namespace_data")
        self.data_path.mkdir(exist_ok=True)
        
        # 네임스페이스 레지스트리
        self.namespaces: Dict[str, NamespaceConfig] = {}
        self.namespace_stats: Dict[str, NamespaceStats] = {}
        self.document_locations: Dict[str, DocumentLocation] = {}
        
        # 자동 매핑 규칙
        self.auto_mapping_rules: List[NamespaceMapping] = []
        
        # 뿌리산업 도메인 분류기
        self.domain_classifier = self._init_domain_classifier()
        
        # 접근 통계
        self.access_history: Dict[str, List[datetime]] = {}
        
        # 기본 네임스페이스들
        self.default_namespaces = [
            "default", "주조", "금형", "소성가공", "용접", "표면처리", "열처리"
        ]
        
        logger.info("Namespace Manager 초기화 완료")
    
    def _init_domain_classifier(self) -> Dict[str, Dict[str, Any]]:
        """도메인 분류기 초기화"""
        
        return {
            "주조": {
                "keywords": [
                    "주조", "캐스팅", "용탕", "응고", "주형", "모래주형", "금속주형",
                    "다이캐스팅", "원심주조", "정밀주조", "로스트왁스", "셸몰드",
                    "용해", "탕온", "수축", "기공", "편석"
                ],
                "patterns": [
                    r"(?:주조|캐스팅)\s*(?:공정|기술|방법)",
                    r"(?:용탕|응고)\s*(?:특성|온도|시간)",
                    r"(?:주형|몰드)\s*(?:설계|제작|재료)"
                ],
                "confidence_weight": 1.0
            },
            "금형": {
                "keywords": [
                    "금형", "다이", "몰드", "프레스", "성형", "사출", "압출",
                    "블로우", "스탬핑", "드로잉", "벤딩", "펀칭", "CAD", "CAM",
                    "표면조도", "치수정밀도", "이형제"
                ],
                "patterns": [
                    r"(?:금형|다이)\s*(?:설계|제작|가공)",
                    r"(?:프레스|성형)\s*(?:공정|압력|속도)",
                    r"(?:표면|치수)\s*(?:정밀도|공차|품질)"
                ],
                "confidence_weight": 1.0
            },
            "소성가공": {
                "keywords": [
                    "소성가공", "단조", "압연", "인발", "전조", "압출", "성형",
                    "냉간가공", "열간가공", "자유단조", "형단조", "정밀단조",
                    "변형저항", "소성변형", "가공경화", "재결정"
                ],
                "patterns": [
                    r"(?:소성가공|단조|압연)\s*(?:공정|조건|온도)",
                    r"(?:변형|응력)\s*(?:해석|분석|측정)",
                    r"(?:재료|금속)\s*(?:유동|변형|성형성)"
                ],
                "confidence_weight": 1.0
            },
            "용접": {
                "keywords": [
                    "용접", "접합", "아크용접", "가스용접", "저항용접", "레이저용접",
                    "전자빔용접", "마찰용접", "브레이징", "솔더링", "용접봉",
                    "열영향부", "HAZ", "용접부", "잔류응력", "변형"
                ],
                "patterns": [
                    r"(?:용접|접합)\s*(?:공정|방법|조건)",
                    r"(?:열영향부|HAZ)\s*(?:특성|제어|분석)",
                    r"(?:용접부|접합부)\s*(?:품질|검사|강도)"
                ],
                "confidence_weight": 1.0
            },
            "표면처리": {
                "keywords": [
                    "표면처리", "도금", "코팅", "양극산화", "침탄", "질화",
                    "쇼트피닝", "연마", "화성처리", "PVD", "CVD", "전해연마",
                    "내식성", "내마모성", "표면경도", "피막", "전처리"
                ],
                "patterns": [
                    r"(?:표면처리|코팅|도금)\s*(?:공정|방법|조건)",
                    r"(?:표면|피막)\s*(?:특성|품질|두께)",
                    r"(?:내식성|내마모성)\s*(?:평가|개선|시험)"
                ],
                "confidence_weight": 1.0
            },
            "열처리": {
                "keywords": [
                    "열처리", "소입", "소성", "담금질", "풀림", "노멀라이징",
                    "템퍼링", "침탄", "질화", "고주파", "진공열처리",
                    "경도", "강도", "인성", "조직", "미세구조", "상변태"
                ],
                "patterns": [
                    r"(?:열처리|소입|소성)\s*(?:공정|조건|온도)",
                    r"(?:조직|미세구조)\s*(?:변화|제어|관찰)",
                    r"(?:경도|강도)\s*(?:측정|평가|개선)"
                ],
                "confidence_weight": 1.0
            }
        }
    
    async def initialize(self):
        """네임스페이스 시스템 초기화"""
        
        logger.info("네임스페이스 시스템 초기화 중...")
        
        # 1. 기본 네임스페이스 생성
        await self._create_default_namespaces()
        
        # 2. 저장된 설정 로드
        await self._load_namespace_configs()
        
        # 3. 자동 매핑 규칙 로드
        await self._load_auto_mapping_rules()
        
        # 4. 통계 초기화
        await self._initialize_statistics()
        
        logger.info(f"✅ 네임스페이스 시스템 초기화 완료: {len(self.namespaces)}개 네임스페이스")
    
    async def _create_default_namespaces(self):
        """기본 네임스페이스 생성"""
        
        # 기본 네임스페이스
        default_config = NamespaceConfig(
            name="default",
            display_name="기본",
            namespace_type=NamespaceType.DEFAULT,
            access_level=AccessLevel.PUBLIC,
            description="기본 문서 저장소",
            max_documents=50000
        )
        await self.create_namespace(default_config)
        
        # 뿌리산업 도메인별 네임스페이스
        for domain in ["주조", "금형", "소성가공", "용접", "표면처리", "열처리"]:
            domain_config = NamespaceConfig(
                name=domain,
                display_name=domain,
                namespace_type=NamespaceType.INDUSTRY,
                access_level=AccessLevel.PUBLIC,
                description=f"{domain} 분야 전문 문서",
                tags=[domain, "뿌리산업"],
                metadata={"industry_domain": domain}
            )
            await self.create_namespace(domain_config)
    
    async def create_namespace(
        self,
        config: NamespaceConfig,
        force: bool = False
    ) -> bool:
        """
        네임스페이스 생성
        
        Args:
            config: 네임스페이스 설정
            force: 기존 네임스페이스 덮어쓰기 여부
            
        Returns:
            bool: 생성 성공 여부
        """
        
        if config.name in self.namespaces and not force:
            logger.warning(f"네임스페이스 이미 존재: {config.name}")
            return False
        
        try:
            # 네임스페이스 이름 검증
            if not self._validate_namespace_name(config.name):
                logger.error(f"유효하지 않은 네임스페이스 이름: {config.name}")
                return False
            
            # 네임스페이스 등록
            self.namespaces[config.name] = config
            
            # 통계 초기화
            self.namespace_stats[config.name] = NamespaceStats()
            
            # 네임스페이스 디렉토리 생성
            namespace_dir = self.data_path / config.name
            namespace_dir.mkdir(exist_ok=True)
            
            # 설정 파일 저장
            config_file = namespace_dir / "config.json"
            await self._save_namespace_config(config, config_file)
            
            logger.info(f"네임스페이스 생성 완료: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"네임스페이스 생성 실패 ({config.name}): {e}")
            return False
    
    def _validate_namespace_name(self, name: str) -> bool:
        """네임스페이스 이름 검증"""
        
        # 길이 확인
        if not name or len(name) > 50:
            return False
        
        # 특수 문자 확인 (한글, 영문, 숫자, 언더스코어, 하이픈만 허용)
        pattern = r'^[가-힣a-zA-Z0-9_-]+$'
        if not re.match(pattern, name):
            return False
        
        # 예약어 확인
        reserved_names = ["system", "admin", "root", "config", "temp", "cache"]
        if name.lower() in reserved_names:
            return False
        
        return True
    
    async def detect_namespace(
        self,
        content: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """
        문서 내용 기반 네임스페이스 자동 감지
        
        Args:
            content: 문서 내용
            title: 문서 제목
            metadata: 추가 메타데이터
            
        Returns:
            Tuple[str, float]: (네임스페이스명, 신뢰도)
        """
        
        try:
            # 1. 한국어 최적화로 전처리
            processed_text = content
            if self.korean_optimizer:
                result = await self.korean_optimizer.process_korean_text(content)
                processed_text = result.normalized_text
            
            # 2. 각 도메인별 점수 계산
            domain_scores = {}
            
            for domain, classifier in self.domain_classifier.items():
                score = self._calculate_domain_score(
                    processed_text, title, classifier, metadata
                )
                if score > 0:
                    domain_scores[domain] = score
            
            # 3. 최고 점수 도메인 선택
            if domain_scores:
                best_domain = max(domain_scores, key=domain_scores.get)
                confidence = domain_scores[best_domain]
                
                # 신뢰도 임계값 확인
                if confidence >= 0.6:
                    return best_domain, confidence
            
            # 4. 기본 네임스페이스 반환
            return "default", 0.5
            
        except Exception as e:
            logger.error(f"네임스페이스 감지 실패: {e}")
            return "default", 0.0
    
    def _calculate_domain_score(
        self,
        content: str,
        title: str,
        classifier: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> float:
        """도메인 점수 계산"""
        
        total_score = 0.0
        content_lower = content.lower()
        title_lower = title.lower()
        
        # 1. 키워드 매칭 점수
        keyword_matches = 0
        for keyword in classifier["keywords"]:
            keyword_lower = keyword.lower()
            
            # 제목에서 키워드 매칭 (가중치 2배)
            if keyword_lower in title_lower:
                keyword_matches += 2
            
            # 내용에서 키워드 매칭
            if keyword_lower in content_lower:
                keyword_matches += content_lower.count(keyword_lower)
        
        # 키워드 점수 정규화 (최대 0.6)
        keyword_score = min(0.6, keyword_matches * 0.05)
        total_score += keyword_score
        
        # 2. 패턴 매칭 점수
        pattern_matches = 0
        combined_text = f"{title} {content}"
        
        for pattern in classifier["patterns"]:
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            pattern_matches += matches
        
        # 패턴 점수 정규화 (최대 0.3)
        pattern_score = min(0.3, pattern_matches * 0.1)
        total_score += pattern_score
        
        # 3. 메타데이터 점수
        metadata_score = 0.0
        if metadata:
            # 도메인 정보가 메타데이터에 있는 경우
            if "domain" in metadata:
                meta_domain = metadata["domain"].lower()
                if any(kw.lower() in meta_domain for kw in classifier["keywords"]):
                    metadata_score += 0.1
            
            # 산업 용어가 메타데이터에 있는 경우
            if "industry_terms" in metadata:
                industry_terms = metadata["industry_terms"]
                if isinstance(industry_terms, list):
                    common_terms = set(classifier["keywords"]) & set(industry_terms)
                    metadata_score += len(common_terms) * 0.02
        
        total_score += metadata_score
        
        # 4. 신뢰도 가중치 적용
        final_score = total_score * classifier["confidence_weight"]
        
        return min(1.0, final_score)
    
    async def add_document(
        self,
        document_id: str,
        namespace: Optional[str] = None,
        auto_detect: bool = True,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        문서를 네임스페이스에 추가
        
        Args:
            document_id: 문서 ID
            namespace: 지정할 네임스페이스 (None시 자동 감지)
            auto_detect: 자동 네임스페이스 감지 여부
            content: 문서 내용 (감지용)
            metadata: 문서 메타데이터
            
        Returns:
            str: 할당된 네임스페이스
        """
        
        try:
            # 1. 네임스페이스 결정
            target_namespace = namespace
            
            if not target_namespace and auto_detect and content:
                detected_ns, confidence = await self.detect_namespace(
                    content, metadata=metadata
                )
                if confidence >= 0.6:
                    target_namespace = detected_ns
                    logger.debug(
                        f"자동 네임스페이스 감지: {document_id} -> {detected_ns} "
                        f"(신뢰도: {confidence:.2f})"
                    )
            
            # 기본값 설정
            if not target_namespace:
                target_namespace = "default"
            
            # 2. 네임스페이스 존재 확인
            if target_namespace not in self.namespaces:
                logger.warning(f"네임스페이스 없음: {target_namespace}, default 사용")
                target_namespace = "default"
            
            # 3. 문서 위치 정보 저장
            doc_location = DocumentLocation(
                namespace=target_namespace,
                document_id=document_id,
                metadata=metadata or {}
            )
            
            self.document_locations[document_id] = doc_location
            
            # 4. 네임스페이스 통계 업데이트
            await self._update_namespace_stats(target_namespace, "add_document")
            
            # 5. 접근 기록
            self._record_access(target_namespace)
            
            logger.debug(f"문서 추가: {document_id} -> {target_namespace}")
            return target_namespace
            
        except Exception as e:
            logger.error(f"문서 추가 실패 ({document_id}): {e}")
            return "default"
    
    async def move_document(
        self,
        document_id: str,
        target_namespace: str,
        reason: str = ""
    ) -> bool:
        """
        문서를 다른 네임스페이스로 이동
        
        Args:
            document_id: 문서 ID
            target_namespace: 목표 네임스페이스
            reason: 이동 사유
            
        Returns:
            bool: 이동 성공 여부
        """
        
        if document_id not in self.document_locations:
            logger.error(f"문서를 찾을 수 없음: {document_id}")
            return False
        
        if target_namespace not in self.namespaces:
            logger.error(f"대상 네임스페이스 없음: {target_namespace}")
            return False
        
        try:
            doc_location = self.document_locations[document_id]
            source_namespace = doc_location.namespace
            
            # 이미 같은 네임스페이스에 있는 경우
            if source_namespace == target_namespace:
                return True
            
            # 문서 위치 업데이트
            doc_location.namespace = target_namespace
            doc_location.metadata["moved_from"] = source_namespace
            doc_location.metadata["move_reason"] = reason
            doc_location.metadata["moved_at"] = datetime.now().isoformat()
            
            # 통계 업데이트
            await self._update_namespace_stats(source_namespace, "remove_document")
            await self._update_namespace_stats(target_namespace, "add_document")
            
            logger.info(
                f"문서 이동: {document_id} ({source_namespace} -> {target_namespace})"
            )
            return True
            
        except Exception as e:
            logger.error(f"문서 이동 실패 ({document_id}): {e}")
            return False
    
    async def remove_document(self, document_id: str) -> bool:
        """문서를 네임스페이스에서 제거"""
        
        if document_id not in self.document_locations:
            return False
        
        try:
            doc_location = self.document_locations[document_id]
            namespace = doc_location.namespace
            
            # 문서 위치 정보 삭제
            del self.document_locations[document_id]
            
            # 통계 업데이트
            await self._update_namespace_stats(namespace, "remove_document")
            
            logger.debug(f"문서 제거: {document_id} from {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"문서 제거 실패 ({document_id}): {e}")
            return False
    
    def get_document_namespace(self, document_id: str) -> Optional[str]:
        """문서의 네임스페이스 조회"""
        
        if document_id in self.document_locations:
            return self.document_locations[document_id].namespace
        return None
    
    def list_namespaces(
        self,
        namespace_type: Optional[NamespaceType] = None,
        access_level: Optional[AccessLevel] = None
    ) -> List[NamespaceConfig]:
        """네임스페이스 목록 조회"""
        
        namespaces = list(self.namespaces.values())
        
        # 타입 필터링
        if namespace_type:
            namespaces = [ns for ns in namespaces if ns.namespace_type == namespace_type]
        
        # 접근 레벨 필터링
        if access_level:
            namespaces = [ns for ns in namespaces if ns.access_level == access_level]
        
        return namespaces
    
    def get_namespace_info(self, namespace: str) -> Optional[Dict[str, Any]]:
        """네임스페이스 상세 정보 조회"""
        
        if namespace not in self.namespaces:
            return None
        
        config = self.namespaces[namespace]
        stats = self.namespace_stats[namespace]
        
        # 해당 네임스페이스의 문서들 조회
        documents = [
            doc_id for doc_id, location in self.document_locations.items()
            if location.namespace == namespace
        ]
        
        return {
            "config": {
                "name": config.name,
                "display_name": config.display_name,
                "type": config.namespace_type.value,
                "access_level": config.access_level.value,
                "description": config.description,
                "max_documents": config.max_documents,
                "tags": config.tags,
                "metadata": config.metadata
            },
            "statistics": {
                "document_count": len(documents),
                "chunk_count": stats.chunk_count,
                "total_size_bytes": stats.total_size_bytes,
                "last_updated": stats.last_updated.isoformat() if stats.last_updated else None,
                "last_accessed": stats.last_accessed.isoformat() if stats.last_accessed else None,
                "access_count": stats.access_count,
                "search_count": stats.search_count,
                "quality_score": stats.quality_score
            },
            "documents": documents[:10],  # 최근 10개 문서만
            "access_history": len(self.access_history.get(namespace, []))
        }
    
    async def get_namespace_statistics(self) -> Dict[str, Any]:
        """전체 네임스페이스 통계"""
        
        total_namespaces = len(self.namespaces)
        total_documents = len(self.document_locations)
        
        # 타입별 분포
        type_distribution = {}
        for config in self.namespaces.values():
            ns_type = config.namespace_type.value
            type_distribution[ns_type] = type_distribution.get(ns_type, 0) + 1
        
        # 접근 레벨별 분포
        access_distribution = {}
        for config in self.namespaces.values():
            access_level = config.access_level.value
            access_distribution[access_level] = access_distribution.get(access_level, 0) + 1
        
        # 문서 분포
        document_distribution = {}
        for location in self.document_locations.values():
            ns = location.namespace
            document_distribution[ns] = document_distribution.get(ns, 0) + 1
        
        # 가장 활성화된 네임스페이스
        most_active = max(
            document_distribution.items(),
            key=lambda x: x[1],
            default=("default", 0)
        )
        
        return {
            "summary": {
                "total_namespaces": total_namespaces,
                "total_documents": total_documents,
                "most_active_namespace": most_active[0],
                "most_active_document_count": most_active[1]
            },
            "distributions": {
                "by_type": type_distribution,
                "by_access_level": access_distribution,
                "by_document_count": document_distribution
            },
            "auto_classification": {
                "total_rules": len(self.auto_mapping_rules),
                "industry_domains": len(self.domain_classifier)
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def optimize_namespaces(self) -> Dict[str, Any]:
        """네임스페이스 최적화"""
        
        optimization_results = {
            "moved_documents": [],
            "merged_namespaces": [],
            "created_namespaces": [],
            "recommendations": []
        }
        
        try:
            # 1. 비어있는 네임스페이스 확인
            empty_namespaces = []
            for ns_name, config in self.namespaces.items():
                doc_count = sum(
                    1 for location in self.document_locations.values()
                    if location.namespace == ns_name
                )
                if doc_count == 0 and config.namespace_type != NamespaceType.SYSTEM:
                    empty_namespaces.append(ns_name)
            
            if empty_namespaces:
                optimization_results["recommendations"].append(
                    f"비어있는 네임스페이스 정리 고려: {', '.join(empty_namespaces)}"
                )
            
            # 2. 잘못 분류된 문서 감지 및 재분류
            misclassified_docs = await self._detect_misclassified_documents()
            
            for doc_id, suggested_ns in misclassified_docs:
                success = await self.move_document(
                    doc_id, suggested_ns, "자동 최적화"
                )
                if success:
                    optimization_results["moved_documents"].append({
                        "document_id": doc_id,
                        "new_namespace": suggested_ns
                    })
            
            # 3. 네임스페이스 사용량 분석
            usage_analysis = self._analyze_namespace_usage()
            optimization_results["recommendations"].extend(usage_analysis)
            
            logger.info(f"네임스페이스 최적화 완료: {len(optimization_results['moved_documents'])}개 문서 재분류")
            
        except Exception as e:
            logger.error(f"네임스페이스 최적화 실패: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def _detect_misclassified_documents(self) -> List[Tuple[str, str]]:
        """잘못 분류된 문서 감지"""
        
        misclassified = []
        
        for doc_id, location in self.document_locations.items():
            current_ns = location.namespace
            
            # 시스템 네임스페이스는 제외
            if current_ns == "system":
                continue
            
            # 문서 내용이 있는 경우에만 검사
            if "content" in location.metadata:
                content = location.metadata["content"]
                
                # 자동 감지 실행
                suggested_ns, confidence = await self.detect_namespace(
                    content, metadata=location.metadata
                )
                
                # 현재 네임스페이스와 다르고 신뢰도가 높은 경우
                if (suggested_ns != current_ns and 
                    confidence >= 0.8 and 
                    suggested_ns in self.namespaces):
                    
                    misclassified.append((doc_id, suggested_ns))
        
        return misclassified
    
    def _analyze_namespace_usage(self) -> List[str]:
        """네임스페이스 사용량 분석"""
        
        recommendations = []
        
        # 문서 분포 계산
        doc_counts = {}
        for location in self.document_locations.values():
            ns = location.namespace
            doc_counts[ns] = doc_counts.get(ns, 0) + 1
        
        # 과도하게 많은 문서가 있는 네임스페이스
        for ns_name, count in doc_counts.items():
            config = self.namespaces.get(ns_name)
            if config and count > config.max_documents * 0.8:
                recommendations.append(
                    f"네임스페이스 '{ns_name}' 용량 한계 접근 ({count}/{config.max_documents})"
                )
        
        # 너무 적은 문서가 있는 네임스페이스
        for ns_name, config in self.namespaces.items():
            count = doc_counts.get(ns_name, 0)
            if count < 5 and config.namespace_type == NamespaceType.INDUSTRY:
                recommendations.append(
                    f"네임스페이스 '{ns_name}' 활용도 낮음 ({count}개 문서)"
                )
        
        return recommendations
    
    async def _update_namespace_stats(self, namespace: str, operation: str):
        """네임스페이스 통계 업데이트"""
        
        if namespace not in self.namespace_stats:
            self.namespace_stats[namespace] = NamespaceStats()
        
        stats = self.namespace_stats[namespace]
        
        if operation == "add_document":
            stats.document_count += 1
            stats.last_updated = datetime.now()
        elif operation == "remove_document":
            stats.document_count = max(0, stats.document_count - 1)
            stats.last_updated = datetime.now()
        elif operation == "search":
            stats.search_count += 1
            stats.last_accessed = datetime.now()
    
    def _record_access(self, namespace: str):
        """네임스페이스 접근 기록"""
        
        if namespace not in self.access_history:
            self.access_history[namespace] = []
        
        self.access_history[namespace].append(datetime.now())
        
        # 통계 업데이트
        if namespace in self.namespace_stats:
            self.namespace_stats[namespace].access_count += 1
            self.namespace_stats[namespace].last_accessed = datetime.now()
        
        # 접근 기록 제한 (최근 1000개만 유지)
        if len(self.access_history[namespace]) > 1000:
            self.access_history[namespace] = self.access_history[namespace][-1000:]
    
    async def _save_namespace_config(self, config: NamespaceConfig, file_path: Path):
        """네임스페이스 설정 저장"""
        
        config_data = {
            "name": config.name,
            "display_name": config.display_name,
            "namespace_type": config.namespace_type.value,
            "access_level": config.access_level.value,
            "description": config.description,
            "max_documents": config.max_documents,
            "auto_cleanup": config.auto_cleanup,
            "cleanup_days": config.cleanup_days,
            "tags": config.tags,
            "metadata": config.metadata,
            "created_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    async def _load_namespace_configs(self):
        """저장된 네임스페이스 설정 로드"""
        
        try:
            for namespace_dir in self.data_path.iterdir():
                if not namespace_dir.is_dir():
                    continue
                
                config_file = namespace_dir / "config.json"
                if not config_file.exists():
                    continue
                
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 객체 복원
                config = NamespaceConfig(
                    name=config_data["name"],
                    display_name=config_data["display_name"],
                    namespace_type=NamespaceType(config_data["namespace_type"]),
                    access_level=AccessLevel(config_data["access_level"]),
                    description=config_data.get("description", ""),
                    max_documents=config_data.get("max_documents", 10000),
                    auto_cleanup=config_data.get("auto_cleanup", False),
                    cleanup_days=config_data.get("cleanup_days", 30),
                    tags=config_data.get("tags", []),
                    metadata=config_data.get("metadata", {})
                )
                
                self.namespaces[config.name] = config
                
        except Exception as e:
            logger.error(f"네임스페이스 설정 로드 실패: {e}")
    
    async def _load_auto_mapping_rules(self):
        """자동 매핑 규칙 로드"""
        
        # 실제로는 파일이나 데이터베이스에서 로드
        # 여기서는 기본 규칙만 설정
        pass
    
    async def _initialize_statistics(self):
        """통계 초기화"""
        
        for namespace in self.namespaces:
            if namespace not in self.namespace_stats:
                self.namespace_stats[namespace] = NamespaceStats()
    
    async def cleanup_expired_namespaces(self):
        """만료된 네임스페이스 정리"""
        
        cleaned_count = 0
        
        for ns_name, config in list(self.namespaces.items()):
            if not config.auto_cleanup:
                continue
            
            # 최근 접근 시간 확인
            if ns_name in self.namespace_stats:
                stats = self.namespace_stats[ns_name]
                if stats.last_accessed:
                    days_since_access = (datetime.now() - stats.last_accessed).days
                    if days_since_access > config.cleanup_days:
                        # 문서가 없는 경우에만 정리
                        doc_count = sum(
                            1 for location in self.document_locations.values()
                            if location.namespace == ns_name
                        )
                        if doc_count == 0:
                            await self._delete_namespace(ns_name)
                            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"만료된 네임스페이스 정리: {cleaned_count}개")
        
        return cleaned_count
    
    async def _delete_namespace(self, namespace: str):
        """네임스페이스 삭제"""
        
        try:
            # 레지스트리에서 제거
            if namespace in self.namespaces:
                del self.namespaces[namespace]
            
            if namespace in self.namespace_stats:
                del self.namespace_stats[namespace]
            
            if namespace in self.access_history:
                del self.access_history[namespace]
            
            # 디렉토리 정리
            namespace_dir = self.data_path / namespace
            if namespace_dir.exists():
                import shutil
                shutil.rmtree(namespace_dir)
            
            logger.info(f"네임스페이스 삭제 완료: {namespace}")
            
        except Exception as e:
            logger.error(f"네임스페이스 삭제 실패 ({namespace}): {e}")
    
    async def cleanup(self):
        """Namespace Manager 정리"""
        
        # 만료된 네임스페이스 정리
        await self.cleanup_expired_namespaces()
        
        # 통계 저장
        await self._save_all_statistics()
        
        logger.info("Namespace Manager 정리 완료")
    
    async def _save_all_statistics(self):
        """모든 통계 저장"""
        
        stats_file = self.data_path / "statistics.json"
        
        stats_data = {
            "namespace_stats": {
                ns: {
                    "document_count": stats.document_count,
                    "chunk_count": stats.chunk_count,
                    "total_size_bytes": stats.total_size_bytes,
                    "access_count": stats.access_count,
                    "search_count": stats.search_count,
                    "last_accessed": stats.last_accessed.isoformat() if stats.last_accessed else None,
                    "last_updated": stats.last_updated.isoformat() if stats.last_updated else None,
                    "quality_score": stats.quality_score
                }
                for ns, stats in self.namespace_stats.items()
            },
            "document_locations": {
                doc_id: {
                    "namespace": location.namespace,
                    "indexed_at": location.indexed_at.isoformat(),
                    "metadata": location.metadata
                }
                for doc_id, location in self.document_locations.items()
            },
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"통계 저장 실패: {e}")