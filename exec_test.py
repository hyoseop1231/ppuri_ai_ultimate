#!/usr/bin/env python3
"""
Execute Test - 직접 실행 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """기본 import 테스트"""
    print("기본 import 테스트:")
    
    # 1. 기본 라이브러리
    try:
        import asyncio
        print("✅ asyncio")
    except Exception as e:
        print(f"❌ asyncio: {e}")
    
    # 2. 타입 힌트
    try:
        from typing import Dict, Any, List, Optional
        print("✅ typing")
    except Exception as e:
        print(f"❌ typing: {e}")
    
    # 3. 프로젝트 모듈들
    try:
        from core.agents.base_agent import BaseIndustrialAgent
        print("✅ BaseIndustrialAgent")
    except Exception as e:
        print(f"❌ BaseIndustrialAgent: {e}")
        return False
    
    try:
        from core.agents.casting_agent import CastingExpertAgent
        print("✅ CastingExpertAgent")
    except Exception as e:
        print(f"❌ CastingExpertAgent: {e}")
        return False
    
    return True

def test_class_creation():
    """클래스 생성 테스트"""
    print("\n클래스 생성 테스트:")
    
    try:
        from core.agents.casting_agent import CastingExpertAgent
        agent = CastingExpertAgent()
        print("✅ CastingExpertAgent 생성 성공")
        print(f"   도메인: {agent.domain}")
        print(f"   도구 수: {len(agent.tools)}")
        return agent
    except Exception as e:
        print(f"❌ CastingExpertAgent 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_sync_methods(agent):
    """동기 메서드 테스트"""
    print("\n동기 메서드 테스트:")
    
    try:
        metrics = agent.get_metrics()
        print("✅ get_metrics() 성공")
        print(f"   총 요청 수: {metrics['total_requests']}")
        print(f"   평균 응답 시간: {metrics['average_response_time']}")
        print(f"   메모리 사용량: {metrics['memory_usage']}")
        return True
    except Exception as e:
        print(f"❌ get_metrics() 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 PPuRI-AI Ultimate 직접 실행 테스트")
    print("=" * 40)
    
    # 1. 기본 import 테스트
    if not test_basic_imports():
        return
    
    # 2. 클래스 생성 테스트
    agent = test_class_creation()
    if not agent:
        return
    
    # 3. 동기 메서드 테스트
    if not test_sync_methods(agent):
        return
    
    print("\n✅ 모든 동기 테스트 성공!")
    print("⚠️  비동기 테스트는 별도 실행 필요")

if __name__ == "__main__":
    main()