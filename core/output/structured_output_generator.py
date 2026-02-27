"""
PPuRI-AI Ultimate - 구조화된 출력 생성기
NotebookLM 스타일의 다양한 출력 형식 지원

Features:
- Mind Map 생성 (Mermaid, D3.js)
- Data Table 추출 및 구조화
- Study Guide 생성
- Briefing Document 생성
- FAQ 자동 생성
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """출력 형식"""
    MIND_MAP = "mind_map"
    DATA_TABLE = "data_table"
    STUDY_GUIDE = "study_guide"
    BRIEFING_DOC = "briefing_doc"
    FAQ = "faq"
    TIMELINE = "timeline"
    COMPARISON_TABLE = "comparison_table"


@dataclass
class MindMapNode:
    """마인드맵 노드"""
    id: str
    label: str
    level: int
    parent_id: Optional[str] = None
    children: List['MindMapNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataTableRow:
    """데이터 테이블 행"""
    values: Dict[str, Any]
    source_citation: Optional[str] = None


@dataclass
class DataTable:
    """데이터 테이블"""
    title: str
    columns: List[str]
    rows: List[DataTableRow]
    description: str = ""


@dataclass
class FAQItem:
    """FAQ 항목"""
    question: str
    answer: str
    category: str = ""
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)


@dataclass
class StudyGuideSection:
    """학습 가이드 섹션"""
    title: str
    content: str
    key_points: List[str]
    review_questions: List[str]


class StructuredOutputGenerator:
    """
    NotebookLM 스타일 구조화된 출력 생성기

    지원 형식:
    - Mind Map: 개념 관계 시각화
    - Data Table: 정보 구조화
    - Study Guide: 학습 자료
    - Briefing Doc: 브리핑 문서
    - FAQ: 자주 묻는 질문
    """

    def __init__(self, llm_client, graph_manager=None):
        self.llm = llm_client
        self.graph = graph_manager
        self._initialized = False

    async def initialize(self) -> bool:
        """초기화"""
        self._initialized = True
        return True

    async def generate(
        self,
        documents: List[Dict[str, Any]],
        output_format: OutputFormat,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """구조화된 출력 생성"""

        generators = {
            OutputFormat.MIND_MAP: self._generate_mind_map,
            OutputFormat.DATA_TABLE: self._generate_data_table,
            OutputFormat.STUDY_GUIDE: self._generate_study_guide,
            OutputFormat.BRIEFING_DOC: self._generate_briefing_doc,
            OutputFormat.FAQ: self._generate_faq,
            OutputFormat.TIMELINE: self._generate_timeline,
            OutputFormat.COMPARISON_TABLE: self._generate_comparison_table,
        }

        generator = generators.get(output_format)
        if not generator:
            raise ValueError(f"Unsupported output format: {output_format}")

        return await generator(documents, options or {})

    async def _generate_mind_map(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """마인드맵 생성"""

        combined_content = self._combine_documents(documents)

        prompt = f"""다음 문서 내용을 분석하여 마인드맵 구조를 생성해주세요.
뿌리산업(주조, 금형, 용접, 소성가공, 표면처리, 열처리) 관점에서 핵심 개념을 정리해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "central_topic": "중심 주제",
  "branches": [
    {{
      "id": "1",
      "label": "주요 가지 1",
      "children": [
        {{"id": "1-1", "label": "하위 개념 1"}},
        {{"id": "1-2", "label": "하위 개념 2"}}
      ]
    }},
    {{
      "id": "2",
      "label": "주요 가지 2",
      "children": [...]
    }}
  ]
}}

규칙:
1. 최대 깊이 3단계
2. 각 가지는 최대 5개의 하위 항목
3. 간결하고 명확한 레이블 사용
4. 기술 용어는 한글 우선
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            mind_map_data = json.loads(response.get("content", "{}"))

            # Mermaid 형식으로 변환
            mermaid_code = self._to_mermaid_mindmap(mind_map_data)

            return {
                "format": "mind_map",
                "data": mind_map_data,
                "mermaid_code": mermaid_code,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse mind map JSON")
            return {"error": "Failed to generate mind map"}

    def _to_mermaid_mindmap(self, data: Dict[str, Any]) -> str:
        """Mermaid 마인드맵 코드 생성"""
        lines = ["mindmap"]
        lines.append(f"  root(({data.get('central_topic', '주제')}))")

        for branch in data.get("branches", []):
            lines.append(f"    {branch.get('label', '')}")
            for child in branch.get("children", []):
                lines.append(f"      {child.get('label', '')}")

        return "\n".join(lines)

    async def _generate_data_table(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """데이터 테이블 추출 및 구조화"""

        combined_content = self._combine_documents(documents)
        table_topic = options.get("topic", "핵심 정보")

        prompt = f"""다음 문서에서 '{table_topic}'에 관한 정보를 구조화된 테이블로 정리해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "title": "테이블 제목",
  "description": "테이블 설명",
  "columns": ["컬럼1", "컬럼2", "컬럼3"],
  "rows": [
    {{"컬럼1": "값1", "컬럼2": "값2", "컬럼3": "값3"}},
    {{"컬럼1": "값4", "컬럼2": "값5", "컬럼3": "값6"}}
  ]
}}

규칙:
1. 관련 있는 정보만 포함
2. 일관된 데이터 형식 사용
3. 누락된 값은 "N/A" 표시
4. 숫자 데이터는 단위 포함
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            table_data = json.loads(response.get("content", "{}"))

            # CSV 형식으로도 제공
            csv_content = self._to_csv(table_data)

            return {
                "format": "data_table",
                "data": table_data,
                "csv": csv_content,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse data table JSON")
            return {"error": "Failed to generate data table"}

    def _to_csv(self, table_data: Dict[str, Any]) -> str:
        """CSV 형식 변환"""
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])

        lines = [",".join(columns)]
        for row in rows:
            values = [str(row.get(col, "")).replace(",", ";") for col in columns]
            lines.append(",".join(values))

        return "\n".join(lines)

    async def _generate_study_guide(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """학습 가이드 생성"""

        combined_content = self._combine_documents(documents)

        prompt = f"""다음 기술 문서를 바탕으로 학습 가이드를 생성해주세요.
KITECH 연구원들이 뿌리산업 기술을 학습하는 데 도움이 되도록 작성해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "title": "학습 가이드 제목",
  "overview": "전체 개요 (2-3문장)",
  "learning_objectives": ["학습 목표 1", "학습 목표 2", "학습 목표 3"],
  "sections": [
    {{
      "title": "섹션 1: 기초 개념",
      "content": "섹션 내용 설명",
      "key_points": ["핵심 포인트 1", "핵심 포인트 2"],
      "review_questions": ["복습 질문 1?", "복습 질문 2?"]
    }}
  ],
  "summary": "전체 요약",
  "further_reading": ["추가 학습 자료 1", "추가 학습 자료 2"]
}}
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            guide_data = json.loads(response.get("content", "{}"))

            # Markdown 형식으로도 제공
            markdown_content = self._to_markdown_study_guide(guide_data)

            return {
                "format": "study_guide",
                "data": guide_data,
                "markdown": markdown_content,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse study guide JSON")
            return {"error": "Failed to generate study guide"}

    def _to_markdown_study_guide(self, data: Dict[str, Any]) -> str:
        """Markdown 학습 가이드 생성"""
        lines = [
            f"# {data.get('title', '학습 가이드')}",
            "",
            "## 개요",
            data.get("overview", ""),
            "",
            "## 학습 목표",
        ]

        for obj in data.get("learning_objectives", []):
            lines.append(f"- {obj}")

        lines.append("")

        for section in data.get("sections", []):
            lines.append(f"## {section.get('title', '')}")
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")
            lines.append("### 핵심 포인트")
            for point in section.get("key_points", []):
                lines.append(f"- {point}")
            lines.append("")
            lines.append("### 복습 질문")
            for q in section.get("review_questions", []):
                lines.append(f"1. {q}")
            lines.append("")

        lines.append("## 요약")
        lines.append(data.get("summary", ""))
        lines.append("")
        lines.append("## 추가 학습 자료")
        for resource in data.get("further_reading", []):
            lines.append(f"- {resource}")

        return "\n".join(lines)

    async def _generate_briefing_doc(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """브리핑 문서 생성"""

        combined_content = self._combine_documents(documents)
        audience = options.get("audience", "기술 관리자")

        prompt = f"""다음 문서를 바탕으로 {audience}를 위한 브리핑 문서를 작성해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "title": "브리핑 제목",
  "executive_summary": "핵심 요약 (3-5문장)",
  "key_findings": [
    {{"finding": "주요 발견 1", "implication": "시사점"}},
    {{"finding": "주요 발견 2", "implication": "시사점"}}
  ],
  "recommendations": ["권장사항 1", "권장사항 2"],
  "risks_and_challenges": ["리스크/과제 1", "리스크/과제 2"],
  "next_steps": ["다음 단계 1", "다음 단계 2"],
  "appendix": "추가 참고 정보"
}}
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            briefing_data = json.loads(response.get("content", "{}"))

            return {
                "format": "briefing_doc",
                "data": briefing_data,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse briefing doc JSON")
            return {"error": "Failed to generate briefing document"}

    async def _generate_faq(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """FAQ 자동 생성"""

        combined_content = self._combine_documents(documents)
        faq_count = options.get("count", 10)

        prompt = f"""다음 문서를 분석하여 자주 묻는 질문(FAQ) {faq_count}개를 생성해주세요.
KITECH 연구원이나 뿌리산업 종사자가 궁금해할 만한 실용적인 질문 위주로 작성해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "faqs": [
    {{
      "category": "카테고리 (예: 기초, 응용, 문제해결)",
      "question": "질문 내용?",
      "answer": "상세한 답변",
      "related_topics": ["관련 주제 1", "관련 주제 2"]
    }}
  ]
}}

규칙:
1. 다양한 난이도의 질문 포함 (기초, 중급, 고급)
2. 실무에서 유용한 질문 우선
3. 답변은 간결하면서도 충분히 설명적으로
4. 뿌리산업 맥락에 맞는 질문
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            faq_data = json.loads(response.get("content", "{}"))

            return {
                "format": "faq",
                "data": faq_data,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse FAQ JSON")
            return {"error": "Failed to generate FAQ"}

    async def _generate_timeline(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """타임라인 생성"""

        combined_content = self._combine_documents(documents)

        prompt = f"""다음 문서에서 시간 순서대로 정리할 수 있는 이벤트/발전 과정을 추출하여 타임라인을 생성해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "title": "타임라인 제목",
  "events": [
    {{
      "date": "날짜/시기 (예: 2020년, 1990년대 초)",
      "title": "이벤트 제목",
      "description": "이벤트 설명",
      "significance": "중요성/영향"
    }}
  ]
}}
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            timeline_data = json.loads(response.get("content", "{}"))

            # Mermaid 타임라인으로 변환
            mermaid_code = self._to_mermaid_timeline(timeline_data)

            return {
                "format": "timeline",
                "data": timeline_data,
                "mermaid_code": mermaid_code,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse timeline JSON")
            return {"error": "Failed to generate timeline"}

    def _to_mermaid_timeline(self, data: Dict[str, Any]) -> str:
        """Mermaid 타임라인 코드 생성"""
        lines = ["timeline"]
        lines.append(f"    title {data.get('title', '타임라인')}")

        for event in data.get("events", []):
            lines.append(f"    {event.get('date', '')} : {event.get('title', '')}")

        return "\n".join(lines)

    async def _generate_comparison_table(
        self,
        documents: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """비교 테이블 생성"""

        combined_content = self._combine_documents(documents)
        comparison_topic = options.get("topic", "기술 비교")

        prompt = f"""다음 문서에서 '{comparison_topic}'에 관한 비교 테이블을 생성해주세요.

문서 내용:
{combined_content[:5000]}

출력 형식 (JSON):
{{
  "title": "비교 테이블 제목",
  "items": ["비교 대상 A", "비교 대상 B", "비교 대상 C"],
  "criteria": [
    {{
      "name": "비교 기준 1",
      "values": {{"비교 대상 A": "값", "비교 대상 B": "값", "비교 대상 C": "값"}}
    }},
    {{
      "name": "비교 기준 2",
      "values": {{"비교 대상 A": "값", "비교 대상 B": "값", "비교 대상 C": "값"}}
    }}
  ],
  "conclusion": "비교 결론"
}}
"""

        response = await self.llm.generate(prompt, response_format="json")

        try:
            comparison_data = json.loads(response.get("content", "{}"))

            return {
                "format": "comparison_table",
                "data": comparison_data,
                "generated_at": datetime.now().isoformat()
            }

        except json.JSONDecodeError:
            logger.error("Failed to parse comparison table JSON")
            return {"error": "Failed to generate comparison table"}

    def _combine_documents(self, documents: List[Dict[str, Any]]) -> str:
        """문서 내용 결합"""
        return "\n\n---\n\n".join([
            f"제목: {doc.get('title', 'N/A')}\n내용: {doc.get('content', '')}"
            for doc in documents
        ])
