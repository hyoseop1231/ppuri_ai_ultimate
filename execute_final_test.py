#!/usr/bin/env python3
"""
Execute Final Test - 최종 테스트 직접 실행
"""

# 최종 테스트 코드 실행
import sys
import os
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 최종 테스트 실행
exec(open(project_root / "final_test_execution.py").read())