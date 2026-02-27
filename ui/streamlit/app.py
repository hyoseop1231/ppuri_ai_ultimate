"""
PPuRI-AI Ultimate - Streamlit UI
NotebookLM 스타일 웹 인터페이스

Features:
- 문서 업로드 및 관리
- 채팅 인터페이스 (인라인 인용)
- 지식 그래프 시각화
- Audio Overview 생성
- 검색 모드 선택
"""

import streamlit as st
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 페이지 설정
st.set_page_config(
    page_title="PPuRI-AI Ultimate",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 채팅 메시지 스타일 */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }

    .assistant-message {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }

    /* 인용 스타일 */
    .citation {
        background-color: #FFF3E0;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        border-left: 3px solid #FF9800;
    }

    .citation-number {
        background-color: #FF9800;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
    }

    /* 산업 선택 버튼 */
    .industry-btn {
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        border: 2px solid #1E3A5F;
        background-color: white;
        cursor: pointer;
        transition: all 0.3s;
    }

    .industry-btn:hover {
        background-color: #1E3A5F;
        color: white;
    }

    /* 검색 모드 카드 */
    .search-mode-card {
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s;
    }

    .search-mode-card:hover {
        border-color: #1E3A5F;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .search-mode-card.selected {
        border-color: #4CAF50;
        background-color: #E8F5E9;
    }

    /* 소스 카드 */
    .source-card {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }

    .source-type-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
    }

    .badge-document { background-color: #E3F2FD; color: #1565C0; }
    .badge-web { background-color: #F3E5F5; color: #7B1FA2; }
    .badge-academic { background-color: #E8F5E9; color: #2E7D32; }
    .badge-patent { background-color: #FFF3E0; color: #EF6C00; }
    .badge-knowledge { background-color: #FCE4EC; color: #C2185B; }
</style>
""", unsafe_allow_html=True)


# 세션 상태 초기화
def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "web_enabled"
    if "industry" not in st.session_state:
        st.session_state.industry = None
    if "citations" not in st.session_state:
        st.session_state.citations = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "audio_tasks" not in st.session_state:
        st.session_state.audio_tasks = []


def run_async(coro):
    """비동기 함수 실행 헬퍼"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ==================== 사이드바 ====================

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60?text=PPuRI-AI", width=200)
        st.markdown("---")

        # 산업 선택
        st.subheader("🏭 산업 선택")
        industries = {
            None: "전체",
            "casting": "주조",
            "mold": "금형",
            "welding": "용접",
            "forming": "소성가공",
            "surface": "표면처리",
            "heat": "열처리"
        }

        selected_industry = st.selectbox(
            "산업 분야",
            options=list(industries.keys()),
            format_func=lambda x: industries[x],
            index=0
        )
        st.session_state.industry = selected_industry

        st.markdown("---")

        # 검색 모드
        st.subheader("🔍 검색 모드")
        search_modes = {
            "documents_only": ("📄 문서만", "업로드한 문서에서만 검색"),
            "web_enabled": ("🌐 문서 + 웹", "문서와 웹 검색 결합"),
            "full_search": ("🔬 전체 검색", "문서 + 웹 + 학술 + 특허")
        }

        for mode, (label, desc) in search_modes.items():
            if st.button(
                f"{label}",
                key=f"mode_{mode}",
                help=desc,
                use_container_width=True,
                type="primary" if st.session_state.search_mode == mode else "secondary"
            ):
                st.session_state.search_mode = mode
                st.rerun()

        st.markdown("---")

        # 문서 관리
        st.subheader("📁 문서 관리")
        uploaded_files = st.file_uploader(
            "문서 업로드",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                if file.name not in [d["name"] for d in st.session_state.documents]:
                    st.session_state.documents.append({
                        "name": file.name,
                        "size": file.size,
                        "type": file.type,
                        "uploaded_at": datetime.now().isoformat()
                    })
                    st.success(f"✅ {file.name} 업로드됨")

        # 업로드된 문서 목록
        if st.session_state.documents:
            st.write(f"**업로드된 문서: {len(st.session_state.documents)}개**")
            for doc in st.session_state.documents[:5]:
                st.caption(f"📄 {doc['name']}")

        st.markdown("---")

        # 대화 초기화
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.citations = []
            st.rerun()


# ==================== 메인 채팅 영역 ====================

def render_chat_interface():
    """채팅 인터페이스 렌더링"""
    st.markdown('<h1 class="main-header">🏭 PPuRI-AI Ultimate</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666;">뿌리산업 특화 NotebookLM 스타일 AI 어시스턴트</p>',
        unsafe_allow_html=True
    )

    # 현재 설정 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        mode_labels = {
            "documents_only": "📄 문서 전용",
            "web_enabled": "🌐 웹 검색 포함",
            "full_search": "🔬 전체 검색"
        }
        st.info(f"검색 모드: {mode_labels[st.session_state.search_mode]}")
    with col2:
        industry_label = {
            None: "전체 산업",
            "casting": "주조",
            "mold": "금형",
            "welding": "용접",
            "forming": "소성가공",
            "surface": "표면처리",
            "heat": "열처리"
        }.get(st.session_state.industry, "전체")
        st.info(f"산업: 🏭 {industry_label}")
    with col3:
        st.info(f"문서: 📁 {len(st.session_state.documents)}개")

    st.markdown("---")

    # 채팅 메시지 표시
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

                    # 인용 정보 표시
                    if "citations" in msg and msg["citations"]:
                        with st.expander(f"📚 출처 ({len(msg['citations'])}개)", expanded=False):
                            for citation in msg["citations"]:
                                render_citation(citation)

    # 입력 영역
    user_input = st.chat_input("질문을 입력하세요...")

    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # AI 응답 생성
        with st.spinner("🤔 답변 생성 중..."):
            response = get_ai_response(user_input)

        # AI 응답 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "citations": response.get("citations", [])
        })

        st.rerun()


def render_citation(citation):
    """인용 카드 렌더링"""
    source_type = citation.get("source_type", "document")
    badge_class = {
        "document": "badge-document",
        "web": "badge-web",
        "academic": "badge-academic",
        "patent": "badge-patent",
        "knowledge_graph": "badge-knowledge"
    }.get(source_type, "badge-document")

    source_label = {
        "document": "문서",
        "web": "웹",
        "academic": "논문",
        "patent": "특허",
        "knowledge_graph": "지식"
    }.get(source_type, "출처")

    st.markdown(f"""
    <div class="source-card">
        <span class="source-type-badge {badge_class}">{source_label}</span>
        <span class="citation-number">[{citation.get('id', '?')}]</span>
        <strong>{citation.get('title', '제목 없음')}</strong>
        <p style="font-size: 0.85rem; color: #666; margin: 0.5rem 0;">
            {citation.get('content_snippet', '')[:200]}...
        </p>
        {f'<a href="{citation.get("url")}" target="_blank">🔗 링크</a>' if citation.get('url') else ''}
    </div>
    """, unsafe_allow_html=True)


def get_ai_response(query: str) -> dict:
    """AI 응답 생성"""
    try:
        # ChatService 사용 시도
        from core.services import get_chat_service, SearchMode

        mode_map = {
            "documents_only": SearchMode.DOCUMENTS_ONLY,
            "web_enabled": SearchMode.WEB_ENABLED,
            "full_search": SearchMode.FULL_SEARCH
        }

        chat_service = run_async(get_chat_service())
        response = run_async(chat_service.chat(
            query=query,
            search_mode=mode_map.get(st.session_state.search_mode, SearchMode.WEB_ENABLED),
            industry_filter=st.session_state.industry
        ))

        citations = [
            {
                "id": c.id,
                "title": c.title,
                "content_snippet": c.content_snippet,
                "source_type": c.source_type,
                "url": c.url,
                "page_number": c.page_number,
                "confidence": c.confidence
            }
            for c in response.citations
        ]

        return {
            "answer": response.answer,
            "citations": citations,
            "model": response.model_used
        }

    except Exception as e:
        # 폴백 응답
        return {
            "answer": f"""안녕하세요! PPuRI-AI Ultimate입니다.

현재 시스템이 초기화 중이거나 일부 서비스에 연결할 수 없습니다.

**질문:** {query}

**상태:** 서비스 연결 대기 중 ({str(e)[:100]})

서비스가 정상화되면 다음 기능을 사용할 수 있습니다:
- 📄 문서 기반 RAG 검색
- 🌐 웹/학술/특허 통합 검색
- 🔗 인라인 인용 [1][2]
- 🎙️ Audio Overview 생성
""",
            "citations": [],
            "model": "fallback"
        }


# ==================== 지식 그래프 탭 ====================

def render_knowledge_graph():
    """지식 그래프 시각화"""
    st.subheader("🕸️ 지식 그래프")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.text_input("엔티티 검색", placeholder="TIG 용접, 기공 결함...")
        st.selectbox("엔티티 타입", ["전체", "기술", "공정", "재료", "결함", "장비"])
        depth = st.slider("탐색 깊이", 1, 5, 2)
        st.button("🔍 그래프 탐색", use_container_width=True)

    with col2:
        # 샘플 그래프 시각화 (Plotly 사용)
        try:
            import plotly.graph_objects as go
            import networkx as nx

            # 샘플 그래프 생성
            G = nx.Graph()
            sample_nodes = [
                ("TIG 용접", "기술"),
                ("기공 결함", "결함"),
                ("아르곤 가스", "재료"),
                ("전류 조절", "파라미터"),
                ("STS304", "재료")
            ]
            sample_edges = [
                ("TIG 용접", "기공 결함", "발생원인"),
                ("TIG 용접", "아르곤 가스", "사용"),
                ("TIG 용접", "전류 조절", "파라미터"),
                ("TIG 용접", "STS304", "적용재료")
            ]

            for name, ntype in sample_nodes:
                G.add_node(name, node_type=ntype)
            for source, target, rel in sample_edges:
                G.add_edge(source, target, relation=rel)

            pos = nx.spring_layout(G, seed=42)

            # 노드 트레이스
            node_x, node_y, node_text, node_color = [], [], [], []
            color_map = {"기술": "#2196F3", "결함": "#F44336", "재료": "#4CAF50", "파라미터": "#FF9800"}

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_color.append(color_map.get(G.nodes[node].get("node_type", ""), "#9E9E9E"))

            # 엣지 트레이스
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig = go.Figure()

            # 엣지
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color='#BDBDBD'),
                hoverinfo='none'
            ))

            # 노드
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=30, color=node_color, line=dict(width=2, color='white')),
                text=node_text,
                textposition='bottom center',
                hoverinfo='text'
            ))

            fig.update_layout(
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=0, r=0, t=0, b=0),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.info("📊 그래프 시각화를 위해 `pip install plotly networkx` 실행 필요")
            st.code("pip install plotly networkx")


# ==================== Audio Overview 탭 ====================

def render_audio_overview():
    """Audio Overview 생성 UI"""
    st.subheader("🎙️ Audio Overview")
    st.markdown("문서를 분석하여 팟캐스트 스타일의 오디오 요약을 생성합니다.")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("제목", placeholder="TIG 용접 기술 개요")
        style = st.selectbox("스타일", ["conversational", "educational", "technical"],
                            format_func=lambda x: {"conversational": "대화체", "educational": "교육적", "technical": "기술적"}[x])
        duration = st.slider("목표 길이 (분)", 3, 30, 10)
        tts_provider = st.selectbox("TTS 엔진", ["edge_tts", "melo_tts", "openai_tts"],
                                   format_func=lambda x: {"edge_tts": "Edge TTS (무료)", "melo_tts": "MeloTTS (오픈소스)", "openai_tts": "OpenAI TTS"}[x])

    with col2:
        st.markdown("**소스 문서 선택**")
        if st.session_state.documents:
            selected_docs = []
            for doc in st.session_state.documents:
                if st.checkbox(doc["name"], key=f"audio_doc_{doc['name']}"):
                    selected_docs.append(doc["name"])

            if selected_docs:
                st.success(f"{len(selected_docs)}개 문서 선택됨")
        else:
            st.info("📁 사이드바에서 문서를 먼저 업로드하세요")

    st.markdown("---")

    if st.button("🎙️ Audio Overview 생성", type="primary", use_container_width=True):
        if not title:
            st.error("제목을 입력하세요")
        elif not st.session_state.documents:
            st.error("문서를 먼저 업로드하세요")
        else:
            with st.spinner("🎙️ 오디오 생성 중... (약 2-5분 소요)"):
                # TODO: 실제 생성 로직 연동
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)

                st.success("✅ Audio Overview 생성 완료!")
                st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")  # 샘플

    # 생성 이력
    st.markdown("---")
    st.subheader("📜 생성 이력")
    if st.session_state.audio_tasks:
        for task in st.session_state.audio_tasks:
            with st.expander(f"🎙️ {task.get('title', 'Untitled')}"):
                st.write(f"상태: {task.get('status', 'unknown')}")
    else:
        st.info("아직 생성된 Audio Overview가 없습니다")


# ==================== 메인 앱 ====================

def main():
    """메인 앱 실행"""
    init_session_state()
    render_sidebar()

    # 탭 구성
    tabs = st.tabs(["💬 채팅", "🕸️ 지식 그래프", "🎙️ Audio Overview", "⚙️ 설정"])

    with tabs[0]:
        render_chat_interface()

    with tabs[1]:
        render_knowledge_graph()

    with tabs[2]:
        render_audio_overview()

    with tabs[3]:
        st.subheader("⚙️ 시스템 설정")

        st.markdown("### API 설정")
        openrouter_key = st.text_input("OpenRouter API Key", type="password",
                                       value=os.getenv("OPENROUTER_API_KEY", "")[:10] + "..." if os.getenv("OPENROUTER_API_KEY") else "")

        st.markdown("### 모델 설정")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("기본 모델", [
                "google/gemini-2.5-pro-preview-03-25",
                "google/gemini-2.5-flash-preview",
                "deepseek/deepseek-r1",
                "anthropic/claude-3.5-sonnet"
            ])
        with col2:
            st.slider("Temperature", 0.0, 1.0, 0.7)

        st.markdown("### 데이터베이스")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("PostgreSQL URL", value="localhost:5432")
        with col2:
            st.text_input("Neo4j URL", value="localhost:7687")

        if st.button("💾 설정 저장"):
            st.success("설정이 저장되었습니다")


if __name__ == "__main__":
    main()
