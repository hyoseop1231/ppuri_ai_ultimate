# 🚀 지금 당장 실행하기

## 💥 1초 만에 서버 실행!

### 🔥 **가장 빠른 방법**

터미널에서 한 줄만 실행:

```bash
python3 simple_stable_server.py
```

### 🖱️ **가장 쉬운 방법**

Finder에서 `start_server.command` 파일을 **더블클릭**

---

## 🌐 접속 주소

**http://localhost:8002**

---

## ✅ 성공 확인

브라우저에서 아래와 같은 화면이 보이면 성공:

```
🏭 PPuRI-AI Ultimate
✅ 서버가 정상적으로 실행 중입니다!
```

---

## 🔧 문제 시 해결책

### **Python 없음**
```bash
# macOS
brew install python3

# Ubuntu/Debian
sudo apt install python3
```

### **의존성 없음**
```bash
pip install fastapi uvicorn
```

### **포트 충돌**
```bash
# 다른 프로그램이 8002 포트 사용 중
lsof -i :8002
kill -9 [PID]
```

---

## 🎯 5가지 대안 방법

| 방법 | 명령어 | 장점 |
|------|--------|------|
| **1. 직접 실행** | `python3 simple_stable_server.py` | 가장 빠름 |
| **2. 원클릭** | `더블클릭 start_server.command` | 가장 쉬움 |
| **3. 데몬** | `python3 daemon_server.py start` | 백그라운드 실행 |
| **4. Docker** | `./docker_run.sh` | 격리된 환경 |
| **5. Jupyter** | `jupyter notebook server_control.ipynb` | 시각적 제어 |

---

## 🎉 최종 목표

**브라우저에서 `localhost:8002` 접속하여 PPuRI-AI Ultimate 시스템 확인!**

- ✅ Agno 초경량 에이전트 (3μs)
- ✅ LlamaIndex 워크플로우  
- ✅ 주조 전문 AI 분석
- ✅ 완전한 웹 인터페이스

**지금 바로 실행하세요!** 🚀