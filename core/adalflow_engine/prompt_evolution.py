"""
Prompt Evolution Engine - 프롬프트 자동 진화 시스템

시간이 지남에 따라 사용자 패턴과 성공 사례를 학습하여
프롬프트를 자동으로 진화시키는 시스템.

Features:
- 유전자 알고리즘 기반 프롬프트 진화
- 사용자 피드백 기반 적응형 학습
- A/B 테스트 자동 실행
- 성능 기반 자연 선택
"""

import asyncio
import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """진화 전략"""
    GENETIC_ALGORITHM = "genetic"      # 유전자 알고리즘
    REINFORCEMENT = "reinforcement"    # 강화학습 기반
    GRADIENT_ASCENT = "gradient"       # 기울기 상승법
    ENSEMBLE = "ensemble"              # 앙상블 방법


@dataclass
class PromptGene:
    """프롬프트 유전자 (진화 단위)"""
    content: str
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def gene_id(self) -> str:
        """유전자 고유 ID"""
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


@dataclass
class EvolutionGeneration:
    """진화 세대"""
    generation_number: int
    genes: List[PromptGene]
    avg_fitness: float
    best_fitness: float
    diversity_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class PromptEvolutionEngine:
    """
    프롬프트 자동 진화 엔진
    
    유전자 알고리즘과 강화학습을 결합하여 프롬프트를 
    지속적으로 진화시키는 시스템.
    """
    
    def __init__(
        self,
        ollama_client,
        population_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        max_generations: int = 50
    ):
        self.ollama_client = ollama_client
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.max_generations = max_generations
        
        # 진화 상태
        self.current_generation = 0
        self.population: List[PromptGene] = []
        self.evolution_history: List[EvolutionGeneration] = []
        self.fitness_cache: Dict[str, float] = {}
        
        # 뿌리산업 특화 진화 패턴
        self.industry_keywords = {
            "주조": ["용탕", "주형", "응고", "사형", "정밀주조"],
            "금형": ["프레스", "사출", "성형", "금형설계", "표면조도"],
            "소성가공": ["단조", "압연", "압출", "인발", "소재"],
            "용접": ["아크용접", "TIG", "MIG", "레이저용접", "용접부"],
            "표면처리": ["도금", "코팅", "열처리", "표면경화", "방식"],
            "열처리": ["담금질", "뜨임", "소둔", "경화", "조직"]
        }
    
    async def evolve_prompt_population(
        self,
        base_prompt: str,
        target_domain: str,
        evaluation_context: Dict[str, Any],
        strategy: EvolutionStrategy = EvolutionStrategy.GENETIC_ALGORITHM
    ) -> PromptGene:
        """
        프롬프트 집단 진화
        
        Args:
            base_prompt: 기본 프롬프트
            target_domain: 타겟 도메인
            evaluation_context: 평가 컨텍스트
            strategy: 진화 전략
            
        Returns:
            PromptGene: 최적화된 프롬프트 유전자
        """
        logger.info(f"프롬프트 집단 진화 시작 - 도메인: {target_domain}")
        
        try:
            # 1. 초기 집단 생성
            if not self.population:
                await self._initialize_population(base_prompt, target_domain)
            
            # 2. 진화 루프
            for generation in range(self.max_generations):
                self.current_generation = generation
                logger.info(f"진화 세대 {generation + 1}/{self.max_generations}")
                
                # 적합도 평가
                await self._evaluate_population(evaluation_context, target_domain)
                
                # 세대 통계 기록
                generation_stats = self._record_generation_stats()
                
                # 수렴 조건 확인
                if self._check_convergence(generation_stats):
                    logger.info(f"수렴 달성 - 세대 {generation + 1}에서 진화 종료")
                    break
                
                # 다음 세대 생성
                if strategy == EvolutionStrategy.GENETIC_ALGORITHM:
                    await self._genetic_evolution()
                elif strategy == EvolutionStrategy.REINFORCEMENT:
                    await self._reinforcement_evolution()
                elif strategy == EvolutionStrategy.ENSEMBLE:
                    await self._ensemble_evolution()
                
                # 다양성 유지
                await self._maintain_diversity()
            
            # 3. 최적 유전자 반환
            best_gene = max(self.population, key=lambda g: g.fitness_score)
            logger.info(
                f"진화 완료! 최적 적합도: {best_gene.fitness_score:.3f} "
                f"(세대 {best_gene.generation})"
            )
            
            return best_gene
            
        except Exception as e:
            logger.error(f"진화 실패: {e}")
            # 실패 시 기본 프롬프트로 유전자 생성
            return PromptGene(content=base_prompt, fitness_score=0.5)
    
    async def _initialize_population(self, base_prompt: str, target_domain: str):
        """초기 집단 생성"""
        logger.info("초기 프롬프트 집단 생성 중...")
        
        self.population = []
        
        # 기본 프롬프트 추가
        base_gene = PromptGene(content=base_prompt, generation=0)
        self.population.append(base_gene)
        
        # 변형 프롬프트들 생성
        for i in range(self.population_size - 1):
            variant = await self._create_prompt_variant(
                base_prompt, target_domain, f"variant_{i}"
            )
            gene = PromptGene(
                content=variant,
                generation=0,
                parent_ids=[base_gene.gene_id],
                mutations=[f"initial_variant_{i}"]
            )
            self.population.append(gene)
        
        logger.info(f"초기 집단 생성 완료: {len(self.population)}개 개체")
    
    async def _create_prompt_variant(
        self,
        base_prompt: str,
        target_domain: str,
        variant_type: str
    ) -> str:
        """프롬프트 변형 생성"""
        
        # 도메인별 키워드 선택
        domain_keywords = self.industry_keywords.get(target_domain, [])
        selected_keywords = random.sample(
            domain_keywords, min(3, len(domain_keywords))
        ) if domain_keywords else []
        
        variant_prompt = f"""다음 기본 프롬프트를 {target_domain} 도메인에 특화하여 변형하세요:

**기본 프롬프트**: {base_prompt}

**변형 요구사항**:
- 도메인: {target_domain}
- 강조할 키워드: {', '.join(selected_keywords)}
- 변형 타입: {variant_type}

**변형 방향**:
1. 더 구체적이고 전문적으로
2. 사용자 친화적 표현으로
3. 실행 가능한 지시로
4. 한국어 자연스럽게

변형된 프롬프트만 출력하세요:"""

        try:
            response = await self.ollama_client.generate({
                "model": "qwen3:30b-a3b",
                "prompt": variant_prompt,
                "options": {
                    "temperature": 0.8,  # 창의적 변형을 위해 높은 온도
                    "top_p": 0.9,
                    "num_predict": 800
                }
            })
            
            return response.get('response', base_prompt).strip()
            
        except Exception as e:
            logger.error(f"변형 생성 실패: {e}")
            return base_prompt
    
    async def _evaluate_population(
        self,
        context: Dict[str, Any],
        target_domain: str
    ):
        """집단 적합도 평가"""
        logger.info("집단 적합도 평가 중...")
        
        # 병렬 평가를 위한 세마포어
        semaphore = asyncio.Semaphore(5)  # 동시 평가 제한
        
        async def evaluate_gene(gene: PromptGene):
            async with semaphore:
                # 캐시 확인
                gene_hash = gene.gene_id
                if gene_hash in self.fitness_cache:
                    gene.fitness_score = self.fitness_cache[gene_hash]
                    return
                
                # 적합도 계산
                fitness = await self._calculate_fitness(
                    gene.content, context, target_domain
                )
                gene.fitness_score = fitness
                self.fitness_cache[gene_hash] = fitness
        
        # 모든 유전자 병렬 평가
        tasks = [evaluate_gene(gene) for gene in self.population]
        await asyncio.gather(*tasks)
        
        # 적합도 순으로 정렬
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        avg_fitness = sum(g.fitness_score for g in self.population) / len(self.population)
        best_fitness = self.population[0].fitness_score
        
        logger.info(
            f"적합도 평가 완료 - 평균: {avg_fitness:.3f}, "
            f"최고: {best_fitness:.3f}"
        )
    
    async def _calculate_fitness(
        self,
        prompt: str,
        context: Dict[str, Any],
        target_domain: str
    ) -> float:
        """
        개별 프롬프트 적합도 계산
        
        적합도 구성 요소:
        1. 응답 품질 (40%)
        2. 도메인 특화성 (30%)
        3. 사용자 만족도 (20%)
        4. 실행 효율성 (10%)
        """
        
        fitness_prompt = f"""다음 프롬프트의 적합도를 종합 평가하세요:

**평가 프롬프트**: {prompt}
**타겟 도메인**: {target_domain}
**사용자 컨텍스트**: {json.dumps(context.get('user_profile', {}), ensure_ascii=False)}

**평가 기준**:
1. 응답 품질 (0-10): 명확하고 유용한 응답을 유도하는가?
2. 도메인 특화성 (0-10): {target_domain} 전문성이 반영되었는가?
3. 사용자 친화성 (0-10): 사용자가 이해하기 쉬운가?
4. 실행 효율성 (0-10): 간결하고 효율적인가?

**종합 점수** (JSON):
{{
    "응답품질": 점수,
    "도메인특화성": 점수,
    "사용자친화성": 점수,
    "실행효율성": 점수,
    "가중평균": 가중평균점수,
    "근거": "평가 근거"
}}"""

        try:
            response = await self.ollama_client.generate({
                "model": "qwen3:30b-a3b",
                "prompt": fitness_prompt,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 400
                }
            })
            
            fitness_data = json.loads(response.get('response', '{}'))
            
            # 가중 평균 계산
            weights = {
                "응답품질": 0.4,
                "도메인특화성": 0.3,
                "사용자친화성": 0.2,
                "실행효율성": 0.1
            }
            
            weighted_score = sum(
                fitness_data.get(key, 5.0) * weight
                for key, weight in weights.items()
            )
            
            # 0-1 범위로 정규화
            return max(0.0, min(1.0, weighted_score / 10.0))
            
        except Exception as e:
            logger.error(f"적합도 계산 실패: {e}")
            return 0.5  # 기본값
    
    async def _genetic_evolution(self):
        """유전자 알고리즘 기반 진화"""
        new_population = []
        
        # 엘리트 선택 (상위 몇 개체 보존)
        elite_count = max(1, int(self.population_size * self.elite_ratio))
        elites = self.population[:elite_count]
        new_population.extend(elites)
        
        # 나머지 개체 생성 (교배 + 돌연변이)
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # 교배
                parent1, parent2 = self._select_parents()
                child = await self._crossover(parent1, parent2)
            else:
                # 복제
                parent = self._select_parents()[0]
                child = PromptGene(
                    content=parent.content,
                    generation=self.current_generation + 1,
                    parent_ids=[parent.gene_id]
                )
            
            # 돌연변이
            if random.random() < self.mutation_rate:
                child = await self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        logger.info(f"새로운 세대 생성: {len(self.population)}개 개체")
    
    def _select_parents(self) -> Tuple[PromptGene, PromptGene]:
        """부모 선택 (토너먼트 선택)"""
        tournament_size = 3
        
        def tournament_select():
            candidates = random.sample(self.population, tournament_size)
            return max(candidates, key=lambda g: g.fitness_score)
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2
    
    async def _crossover(self, parent1: PromptGene, parent2: PromptGene) -> PromptGene:
        """교배 (두 프롬프트 결합)"""
        
        crossover_prompt = f"""두 프롬프트의 장점을 결합하여 새로운 프롬프트를 생성하세요:

**부모 1 (적합도: {parent1.fitness_score:.3f})**:
{parent1.content}

**부모 2 (적합도: {parent2.fitness_score:.3f})**:
{parent2.content}

**교배 요구사항**:
1. 두 프롬프트의 강점을 모두 포함
2. 자연스럽고 일관된 구조
3. 뿌리산업 전문성 유지
4. 간결하고 명확한 표현

결합된 프롬프트만 출력하세요:"""

        try:
            response = await self.ollama_client.generate({
                "model": "qwen3:30b-a3b",
                "prompt": crossover_prompt,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            })
            
            child_content = response.get('response', parent1.content).strip()
            
            return PromptGene(
                content=child_content,
                generation=self.current_generation + 1,
                parent_ids=[parent1.gene_id, parent2.gene_id],
                mutations=["crossover"]
            )
            
        except Exception as e:
            logger.error(f"교배 실패: {e}")
            # 실패 시 더 적합한 부모 반환
            better_parent = parent1 if parent1.fitness_score > parent2.fitness_score else parent2
            return PromptGene(
                content=better_parent.content,
                generation=self.current_generation + 1,
                parent_ids=[better_parent.gene_id]
            )
    
    async def _mutate(self, gene: PromptGene) -> PromptGene:
        """돌연변이"""
        
        mutation_types = [
            "키워드 추가", "문체 변경", "구조 개선", 
            "예시 추가", "표현 간소화", "전문성 강화"
        ]
        mutation_type = random.choice(mutation_types)
        
        mutation_prompt = f"""다음 프롬프트에 '{mutation_type}' 돌연변이를 적용하세요:

**원본 프롬프트**:
{gene.content}

**돌연변이 타입**: {mutation_type}

**돌연변이 요구사항**:
1. 기본 의도와 구조는 유지
2. 작은 개선 사항 적용
3. 창의적이고 혁신적 요소 추가
4. 뿌리산업 특화 강화

돌연변이된 프롬프트만 출력하세요:"""

        try:
            response = await self.ollama_client.generate({
                "model": "qwen3:30b-a3b",
                "prompt": mutation_prompt,
                "options": {
                    "temperature": 0.7,  # 창의적 돌연변이
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            })
            
            mutated_content = response.get('response', gene.content).strip()
            
            # 돌연변이 정보 업데이트
            new_mutations = gene.mutations.copy()
            new_mutations.append(mutation_type)
            
            return PromptGene(
                content=mutated_content,
                generation=gene.generation,
                parent_ids=gene.parent_ids,
                mutations=new_mutations
            )
            
        except Exception as e:
            logger.error(f"돌연변이 실패: {e}")
            return gene  # 실패 시 원본 반환
    
    def _record_generation_stats(self) -> EvolutionGeneration:
        """세대 통계 기록"""
        if not self.population:
            return None
        
        fitness_scores = [g.fitness_score for g in self.population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        best_fitness = max(fitness_scores)
        
        # 다양성 점수 계산 (유전자 내용의 유사성 기반)
        diversity_score = self._calculate_diversity()
        
        generation = EvolutionGeneration(
            generation_number=self.current_generation,
            genes=self.population.copy(),
            avg_fitness=avg_fitness,
            best_fitness=best_fitness,
            diversity_score=diversity_score
        )
        
        self.evolution_history.append(generation)
        return generation
    
    def _calculate_diversity(self) -> float:
        """집단 다양성 계산"""
        if len(self.population) < 2:
            return 1.0
        
        # 간단한 다양성 메트릭: 유니크한 프롬프트 비율
        unique_contents = set(g.content for g in self.population)
        diversity = len(unique_contents) / len(self.population)
        
        return diversity
    
    def _check_convergence(self, generation: EvolutionGeneration) -> bool:
        """수렴 조건 확인"""
        if len(self.evolution_history) < 5:
            return False
        
        # 최근 5세대의 성능 향상이 미미한 경우
        recent_best = [g.best_fitness for g in self.evolution_history[-5:]]
        improvement = recent_best[-1] - recent_best[0]
        
        return improvement < 0.01  # 1% 미만 개선 시 수렴 판정
    
    async def _maintain_diversity(self):
        """다양성 유지"""
        current_diversity = self._calculate_diversity()
        
        if current_diversity < 0.5:  # 다양성이 50% 미만인 경우
            logger.info("다양성 부족 감지 - 새로운 변형 추가")
            
            # 무작위 새로운 변형 추가
            base_prompt = self.population[0].content  # 최고 개체 기준
            new_variant = await self._create_prompt_variant(
                base_prompt, "뿌리산업", "diversity_boost"
            )
            
            # 가장 낮은 적합도 개체 교체
            worst_idx = min(
                range(len(self.population)),
                key=lambda i: self.population[i].fitness_score
            )
            
            self.population[worst_idx] = PromptGene(
                content=new_variant,
                generation=self.current_generation,
                mutations=["diversity_injection"]
            )
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """진화 통계 반환"""
        if not self.evolution_history:
            return {"generations": 0}
        
        latest = self.evolution_history[-1]
        all_fitness = [g.best_fitness for g in self.evolution_history]
        
        return {
            "total_generations": len(self.evolution_history),
            "current_best_fitness": latest.best_fitness,
            "current_avg_fitness": latest.avg_fitness,
            "current_diversity": latest.diversity_score,
            "fitness_improvement": all_fitness[-1] - all_fitness[0] if len(all_fitness) > 1 else 0,
            "population_size": len(self.population),
            "convergence_rate": self._calculate_convergence_rate()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """수렴 속도 계산"""
        if len(self.evolution_history) < 2:
            return 0.0
        
        fitness_gains = []
        for i in range(1, len(self.evolution_history)):
            gain = (
                self.evolution_history[i].best_fitness - 
                self.evolution_history[i-1].best_fitness
            )
            fitness_gains.append(gain)
        
        return sum(fitness_gains) / len(fitness_gains) if fitness_gains else 0.0