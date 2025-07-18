# 🤝 기여 가이드

PPuRI-AI Ultimate 프로젝트에 기여해 주셔서 감사합니다!

## 🚀 빠른 시작

### 개발 환경 설정
```bash
# 프로젝트 포크 및 클론
git clone https://github.com/your-username/ppuri-ai-ultimate.git
cd ppuri-ai-ultimate

# 개발 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 환경 설정
cp .env.example .env
cp config/config.yaml.example config/config.yaml
```

## 📝 기여 방법

### 1. Issue 생성
- 버그 리포트, 기능 요청, 질문 등
- 명확한 제목과 상세한 설명
- 관련 라벨 추가

### 2. Pull Request 프로세스
1. **Fork** 프로젝트
2. **Feature 브랜치** 생성
   ```bash
   git checkout -b feature/amazing-new-feature
   ```
3. **변경사항 구현**
4. **테스트 작성 및 실행**
   ```bash
   pytest tests/ -v
   black . --line-length 120
   ruff check .
   ```
5. **커밋 및 푸시**
   ```bash
   git commit -m "✨ Add amazing new feature"
   git push origin feature/amazing-new-feature
   ```
6. **Pull Request 생성**

## 📋 코딩 스타일

### Python 코드 스타일
- **Black** 포매터 사용 (line-length: 120)
- **Ruff** 린터 사용
- **Type hints** 필수
- **Docstring** 모든 public 함수/클래스

### 커밋 메시지 컨벤션
```
<타입>(<스코프>): <제목>

<본문>

<푸터>
```

**타입:**
- ✨ `feat`: 새로운 기능
- 🐛 `fix`: 버그 수정
- 📚 `docs`: 문서 변경
- 🎨 `style`: 코드 스타일 변경
- ♻️ `refactor`: 리팩토링
- ✅ `test`: 테스트 추가/수정
- 🔧 `chore`: 기타 변경사항

## 🧪 테스트

### 테스트 실행
```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=core --cov=ui --cov-report=html

# 특정 모듈 테스트
pytest tests/test_rag_engine.py -v
```

### 테스트 작성 가이드
- 모든 새로운 기능에 대한 테스트 작성
- pytest fixture 활용
- 단위 테스트 + 통합 테스트
- Mock 객체 적절히 활용

## 🏗️ 아키텍처 가이드

### 새 컴포넌트 추가
```python
# core/new_component/new_feature.py
class NewFeature:
    """새로운 기능 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def process(self, input_data: Any) -> Any:
        """메인 처리 로직"""
        pass
```

### MCP 도구 추가
```python
# mcp_tools/your_tool.py
from core.mcp.base_tool import BaseMCPTool

class YourTool(BaseMCPTool):
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """도구 실행 로직"""
        return {"result": "success"}
```

## 📊 성능 가이드

### 성능 최적화 원칙
- 비동기 처리 우선 사용
- 메모리 효율적 스트리밍
- 지연 로딩 (Lazy Loading)
- 적절한 캐싱 전략

### 프로파일링
```bash
# 성능 프로파일링
python -m cProfile -o profile.stats main.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumtime').print_stats(20)"
```

## 🌐 국제화

### 한국어 지원 강화
- 뿌리산업 전문 용어 추가
- KoNLPy 형태소 분석 개선
- 산업별 도메인 분류 정확도 향상

### 다국어 지원 추가
- i18n 시스템 구축
- 번역 리소스 관리
- 로케일별 설정

## 🔒 보안 가이드

### 보안 체크리스트
- [ ] 입력 검증 및 살균
- [ ] SQL 인젝션 방지
- [ ] XSS 방지
- [ ] CSRF 토큰 사용
- [ ] 민감 정보 로깅 금지
- [ ] 안전한 암호화 사용

### 의존성 보안
```bash
# 보안 취약점 스캔
pip-audit
bandit -r core/ ui/ api/
```

## 📖 문서화

### 코드 문서화
```python
def process_document(
    self,
    content: str,
    namespace: str = "default",
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessedDocument:
    """
    문서를 처리하여 구조화된 형태로 변환
    
    Args:
        content: 처리할 문서 내용
        namespace: 문서가 속할 네임스페이스
        metadata: 추가 메타데이터
        
    Returns:
        ProcessedDocument: 처리된 문서 객체
        
    Raises:
        ProcessingError: 문서 처리 중 오류 발생 시
        
    Example:
        >>> processor = DocumentProcessor()
        >>> doc = processor.process_document("Hello World", "test")
        >>> print(doc.content)
        "Hello World"
    """
```

## 🎯 개발 로드맵

### 우선순위 높음
- [ ] API 엔드포인트 실제 구현체 연결
- [ ] 프론트엔드 React/Vue 컴포넌트
- [ ] 실제 Ollama 모델 연동 테스트
- [ ] ChromaDB 성능 최적화

### 우선순위 중간
- [ ] 다국어 지원 확장
- [ ] 모바일 반응형 UI
- [ ] 고급 분석 대시보드
- [ ] API 캐싱 최적화

### 우선순위 낮음
- [ ] 플러그인 시스템
- [ ] 클러스터 배포 지원
- [ ] 고급 보안 기능
- [ ] 엔터프라이즈 기능

## 🙋‍♀️ 도움 요청

### 커뮤니티 지원
- **GitHub Issues**: 버그 리포트, 기능 요청
- **GitHub Discussions**: 질문, 아이디어 공유
- **Discord**: 실시간 개발자 채팅 (링크 추가 예정)

### 메인테이너 연락
- 복잡한 아키텍처 변경 전 사전 논의
- 대규모 기능 추가 시 RFC 작성
- 보안 이슈는 private 보고

---

**함께 만들어가는 뿌리산업 AI 혁신! 🏭✨**