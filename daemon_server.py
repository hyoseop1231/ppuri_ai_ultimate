#!/usr/bin/env python3
"""
Daemon Server - 백그라운드 자동 실행 서버
"""

import os
import sys
import subprocess
import signal
import time
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ServerDaemon:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.pid_file = self.project_dir / "server.pid"
        self.process = None
        
    def start_daemon(self):
        """데몬으로 서버 시작"""
        logger.info("🚀 PPuRI-AI Ultimate 데몬 서버 시작")
        
        # 기존 프로세스 확인
        if self.is_running():
            logger.info("⚠️ 서버가 이미 실행 중입니다.")
            return
        
        try:
            # 백그라운드에서 서버 실행
            cmd = [sys.executable, "simple_stable_server.py"]
            self.process = subprocess.Popen(
                cmd,
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 새로운 세션 그룹 생성
            )
            
            # PID 파일에 저장
            with open(self.pid_file, 'w') as f:
                f.write(str(self.process.pid))
            
            logger.info(f"✅ 서버 시작 완료 (PID: {self.process.pid})")
            logger.info("🌐 브라우저에서 http://localhost:8002 접속 가능")
            
            # 시그널 핸들러 등록
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)
            
            # 프로세스 모니터링
            self.monitor_process()
            
        except Exception as e:
            logger.error(f"❌ 서버 시작 실패: {e}")
    
    def stop_daemon(self):
        """데몬 서버 종료"""
        logger.info("🛑 서버 종료 중...")
        
        if not self.is_running():
            logger.info("⚠️ 실행 중인 서버가 없습니다.")
            return
        
        try:
            # PID 파일에서 PID 읽기
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 프로세스 종료
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            
            # PID 파일 삭제
            self.pid_file.unlink()
            
            logger.info("✅ 서버가 정상적으로 종료되었습니다.")
            
        except Exception as e:
            logger.error(f"❌ 서버 종료 실패: {e}")
    
    def restart_daemon(self):
        """데몬 서버 재시작"""
        logger.info("🔄 서버 재시작 중...")
        self.stop_daemon()
        time.sleep(2)
        self.start_daemon()
    
    def status_daemon(self):
        """데몬 서버 상태 확인"""
        if self.is_running():
            with open(self.pid_file, 'r') as f:
                pid = f.read().strip()
            logger.info(f"✅ 서버 실행 중 (PID: {pid})")
            logger.info("🌐 http://localhost:8002")
        else:
            logger.info("❌ 서버가 실행되지 않았습니다.")
    
    def is_running(self):
        """서버 실행 상태 확인"""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 프로세스 존재 확인
            os.kill(pid, 0)
            return True
            
        except (OSError, ValueError):
            # PID 파일 삭제
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
    
    def monitor_process(self):
        """프로세스 모니터링"""
        try:
            while True:
                if self.process and self.process.poll() is not None:
                    logger.error("❌ 서버 프로세스가 예상치 못하게 종료됨")
                    # 자동 재시작
                    logger.info("🔄 5초 후 자동 재시작...")
                    time.sleep(5)
                    self.start_daemon()
                    break
                
                time.sleep(10)  # 10초마다 확인
                
        except KeyboardInterrupt:
            logger.info("🛑 모니터링 중단")
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"📡 시그널 {signum} 수신")
        self.stop_daemon()
        sys.exit(0)

def main():
    """메인 실행"""
    daemon = ServerDaemon()
    
    if len(sys.argv) < 2:
        print("🚀 PPuRI-AI Ultimate 데몬 서버")
        print("=" * 40)
        print("사용법:")
        print("  python3 daemon_server.py start   - 서버 시작")
        print("  python3 daemon_server.py stop    - 서버 종료") 
        print("  python3 daemon_server.py restart - 서버 재시작")
        print("  python3 daemon_server.py status  - 상태 확인")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        daemon.start_daemon()
    elif command == "stop":
        daemon.stop_daemon()
    elif command == "restart":
        daemon.restart_daemon()
    elif command == "status":
        daemon.status_daemon()
    else:
        print(f"❌ 알 수 없는 명령어: {command}")

if __name__ == "__main__":
    main()