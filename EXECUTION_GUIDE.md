# 🚀 PPuRI-AI Ultimate 실행 가이드

Shell 환경 문제로 직접 실행이 어려운 상황을 위한 **5가지 대안 실행 방법**을 제공합니다.

## 🎯 핵심 목표

**단 하나의 명령어로 PPuRI-AI Ultimate 서버를 실행하고 브라우저에서 `http://localhost:8002` 접속**

---

## 🔥 즉시 실행 방법들

### 1️⃣ **원클릭 실행** (가장 쉬움)

macOS Finder에서 더블클릭으로 실행:

```bash
# 파일을 실행 가능하게 만들기
chmod +x start_server.command

# 더블클릭으로 실행하거나 터미널에서:
./start_server.command
```

**결과**: 터미널 창이 열리고 서버가 자동으로 시작됩니다.

### 2️⃣ **백그라운드 데몬 실행** (안정적)

서버를 백그라운드에서 실행하고 관리:

```bash
# 서버 시작
python3 daemon_server.py start

# 상태 확인
python3 daemon_server.py status

# 서버 중지
python3 daemon_server.py stop

# 서버 재시작
python3 daemon_server.py restart
```

**장점**: 터미널을 닫아도 서버가 계속 실행됩니다.

### 3️⃣ **Docker 컨테이너 실행** (격리된 환경)

Docker를 사용한 컨테이너 실행:

```bash
# 실행 권한 부여
chmod +x docker_run.sh

# Docker로 실행
./docker_run.sh
```

또는 개별 명령어:

```bash
# 이미지 빌드
docker build -f Dockerfile.simple -t ppuri-ai-ultimate .

# 컨테이너 실행
docker run -d --name ppuri-ai-ultimate -p 8002:8002 ppuri-ai-ultimate
```

**장점**: 의존성 충돌 없이 깨끗한 환경에서 실행됩니다.

### 4️⃣ **Jupyter 노트북 제어판** (시각적)

Jupyter에서 서버를 시각적으로 관리:

```bash
# Jupyter 설치 (필요시)
pip install jupyter

# 노트북 실행
jupyter notebook server_control.ipynb
```

**장점**: 웹 인터페이스에서 서버를 제어하고 모니터링할 수 있습니다.

### 5️⃣ **VS Code 통합 실행** (개발자용)

VS Code에서 직접 실행:

1. **Command Palette** 열기: `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows/Linux)
2. **"Tasks: Run Task"** 선택
3. **"🚀 Start PPuRI-AI Server"** 선택

또는 **F5**를 눌러 디버그 모드로 실행

**장점**: 코드 편집과 서버 실행을 동시에 할 수 있습니다.

---

## 🌐 브라우저 접속 주소

서버가 성공적으로 시작되면 다음 주소로 접속:

- **메인 페이지**: http://localhost:8002
- **API 문서**: http://localhost:8002/docs
- **헬스 체크**: http://localhost:8002/api/health
- **상태 정보**: http://localhost:8002/api/status

---

## 🧪 테스트 계정

API 테스트를 위한 기본 계정:

- **사용자명**: `admin_001`
- **비밀번호**: `admin_pass_001`

---

## 🎯 성공 확인 방법

### 1. **터미널 메시지 확인**
```
🚀 PPuRI-AI Ultimate Simple Server 시작
✅ 서버 시작 완료 - http://localhost:8002
🌐 브라우저에서 http://localhost:8002 접속 가능
```

### 2. **브라우저 접속 확인**
- 아름다운 웹 인터페이스가 표시됨
- 서버 상태가 "실행 중"으로 표시됨
- Agno 에이전트와 LlamaIndex 워크플로우 정보 확인

### 3. **API 테스트**
```bash
curl http://localhost:8002/api/health
```

예상 응답:
```json
{
  "status": "healthy",
  "server": "PPuRI-AI Ultimate Simple Server",
  "version": "3.0.0"
}
```

---

## 🛠️ 의존성 설치

필요한 패키지가 없는 경우:

```bash
pip install fastapi uvicorn requests
```

Docker 사용 시:
```bash
# Docker 설치 확인
docker --version

# Docker가 없다면 https://docs.docker.com/get-docker/ 에서 설치
```

---

## 🔧 문제 해결

### **포트 충돌**
다른 프로그램이 8002 포트를 사용 중인 경우:

```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8002

# 프로세스 종료
kill -9 [PID]
```

### **권한 문제**
macOS에서 실행 권한 오류 시:

```bash
chmod +x start_server.command
chmod +x docker_run.sh
```

### **Python 버전**
Python 3.8 이상 필요:

```bash
python3 --version
```

---

## 🎉 최종 목표 달성

**어떤 방법을 사용하든 최종 결과는 동일합니다:**

1. ✅ 서버가 `localhost:8002`에서 실행
2. ✅ 브라우저에서 아름다운 웹 인터페이스 확인
3. ✅ Agno 초경량 에이전트 (3μs, 6.5KB) 동작
4. ✅ LlamaIndex 워크플로우 통합 확인
5. ✅ 주조 전문 AI 분석 시스템 활성화
6. ✅ 완전한 RESTful API 엔드포인트 제공

---

## 💡 추천 방법

### **처음 사용자**: 1️⃣ 원클릭 실행 (`start_server.command`)
### **개발자**: 5️⃣ VS Code 통합 실행
### **서버 관리자**: 2️⃣ 백그라운드 데몬 실행
### **Docker 사용자**: 3️⃣ Docker 컨테이너 실행
### **연구자/분석가**: 4️⃣ Jupyter 노트북 제어판

---

**이제 Shell 환경 제약 없이 어떤 방법으로든 PPuRI-AI Ultimate를 실행할 수 있습니다!** 🚀