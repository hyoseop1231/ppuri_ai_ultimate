#!/usr/bin/env python3
"""
Import Verification - 모든 모듈의 import 검증
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name, import_statement):
    """모듈 import 테스트"""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: 성공")
        return True
    except Exception as e:
        print(f"❌ {module_name}: 실패 - {e}")
        return False

def main():
    """메인 검증"""
    print("🔍 PPuRI-AI Ultimate 모듈 import 검증")
    print("=" * 50)
    
    # 테스트할 모듈들
    modules = [
        ("BaseIndustrialAgent", "from core.agents.base_agent import BaseIndustrialAgent"),
        ("CastingExpertAgent", "from core.agents.casting_agent import CastingExpertAgent"),
        ("BaseIndustrialWorkflow", "from core.workflows.base_workflow import BaseIndustrialWorkflow"),
        ("IndustrialAnalysisWorkflow", "from core.workflows.analysis_workflow import IndustrialAnalysisWorkflow"),
        ("Industrial AI Router", "from api.routes.industrial_ai import router"),
        ("Constants", "from api.constants import HTTPStatus"),
        ("Response Models", "from api.models.responses import SuccessResponse, ErrorResponse"),
        ("Auth Middleware", "from api.middleware.auth import JWTManager"),
        ("Security Middleware", "from api.middleware.security import SecurityHeadersMiddleware"),
        ("Connection Pool", "from api.database.connection_pool import connection_pool_manager"),
    ]
    
    # 각 모듈 테스트
    success_count = 0
    for module_name, import_statement in modules:
        if test_import(module_name, import_statement):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"결과: {success_count}/{len(modules)} 모듈 import 성공")
    
    if success_count == len(modules):
        print("✅ 모든 모듈이 정상적으로 import됩니다!")
    else:
        print("⚠️  일부 모듈에서 import 오류가 발생했습니다.")
    
    # 추가 검증: 클래스 인스턴스 생성 테스트
    print("\n🧪 클래스 인스턴스 생성 테스트")
    print("-" * 30)
    
    try:
        from core.agents.casting_agent import CastingExpertAgent
        agent = CastingExpertAgent()
        print("✅ CastingExpertAgent 인스턴스 생성 성공")
        print(f"   - 도메인: {agent.domain}")
        print(f"   - 도구 수: {len(agent.tools)}")
    except Exception as e:
        print(f"❌ CastingExpertAgent 인스턴스 생성 실패: {e}")
    
    try:
        from core.workflows.analysis_workflow import IndustrialAnalysisWorkflow
        workflow = IndustrialAnalysisWorkflow()
        print("✅ IndustrialAnalysisWorkflow 인스턴스 생성 성공")
        print(f"   - 워크플로우 이름: {workflow.workflow_name}")
        print(f"   - 에이전트 수: {len(workflow.agents)}")
    except Exception as e:
        print(f"❌ IndustrialAnalysisWorkflow 인스턴스 생성 실패: {e}")

if __name__ == "__main__":
    main()