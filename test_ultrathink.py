#!/usr/bin/env python3
"""
ULTRATHINK 기능 직접 테스트

뿌리산업 특화 AI의 3단계 사고 시스템을 테스트합니다.
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from core.kitech_base.think_ui import ThinkBlockManager, ThinkLevel
from core.config.config_manager import ConfigManager


async def demo_ultrathink():
    """ULTRATHINK 데모 실행"""
    
    print("""
    🏭 PPuRI-AI Ultimate - ULTRATHINK 데모
    ════════════════════════════════════════
    뿌리산업 특화 3단계 사고 시스템 테스트
    
    🧠 THINK: 기본 분석
    🚀 MEGATHINK: 복합 관계 분석  
    ⚡ ULTRATHINK: 최종 통합 결론
    ════════════════════════════════════════
    """)
    
    # 설정 매니저 초기화
    config_manager = ConfigManager()
    config_manager.initialize()
    
    # THINK 블록 매니저 생성
    think_manager = ThinkBlockManager(config_manager)
    
    # 세션 시작
    session_id = "test-ultrathink-demo"
    session = await think_manager.start_think_session(session_id)
    
    if not session:
        print("❌ THINK 블록이 비활성화되어 있습니다.")
        return
    
    # 테스트 주제
    topic = "주조 공정에서 용탕 온도 최적화"
    industry_domain = "주조"
    
    print(f"📋 분석 주제: {topic}")
    print(f"🏭 산업 도메인: {industry_domain}")
    print("\n" + "="*50)
    
    # 점진적 사고 과정 실행
    async for think_block in think_manager.generate_progressive_think(
        session_id, topic, industry_domain
    ):
        # 사고 블록 표시
        formatted = think_manager.format_think_block_for_display(think_block)
        print(f"\n{formatted}")
        
        # 실제 사고 과정처럼 약간의 지연
        await asyncio.sleep(0.5)
    
    print("\n" + "="*50)
    
    # 세션 요약 출력
    summary = think_manager.format_session_summary(session_id)
    print(f"\n{summary}")
    
    # 세션 종료 및 통계
    stats = think_manager.end_think_session(session_id)
    if stats:
        print(f"\n📊 **세션 통계**:")
        print(f"- 세션 ID: {stats['session_id']}")
        print(f"- 총 사고 블록: {stats['total_blocks']}개")
        print(f"- 총 처리 시간: {stats['total_time']:.2f}초")
        print(f"- 세션 지속 시간: {stats['duration']:.2f}초")
        print(f"- 발견된 산업 용어: {stats['industry_terms_found']}개")
        print(f"- 평균 신뢰도: {stats['avg_confidence']:.2f}")
    
    # 분석 결과
    analytics = think_manager.get_think_analytics()
    print(f"\n🔍 **시스템 분석**:")
    print(f"- THINK 블록 활성화: {analytics['enabled']}")
    print(f"- 지원 레벨: {', '.join(analytics['supported_levels'])}")
    print(f"- 산업 템플릿: {', '.join(analytics['industry_templates'])}")
    print(f"- 총 사용 블록: {analytics['usage_stats']['total_blocks']}개")
    print(f"- 평균 처리 시간: {analytics['usage_stats']['avg_processing_time']:.3f}초")
    
    print("\n✅ ULTRATHINK 데모 완료!")


async def interactive_ultrathink():
    """대화형 ULTRATHINK 테스트"""
    
    print("\n🎯 대화형 ULTRATHINK 모드")
    print("주제를 입력하면 3단계 사고 과정을 보여드립니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    # 설정 및 매니저 초기화
    config_manager = ConfigManager()
    config_manager.initialize()
    think_manager = ThinkBlockManager(config_manager)
    
    session_counter = 0
    
    while True:
        try:
            # 사용자 입력
            topic = input("💭 분석할 주제를 입력하세요: ").strip()
            
            if topic.lower() in ['quit', 'exit', '종료']:
                print("👋 ULTRATHINK 세션을 종료합니다.")
                break
            
            if not topic:
                continue
            
            # 산업 도메인 선택
            domains = ["주조", "금형", "소성가공", "용접", "표면처리", "열처리"]
            print(f"\n🏭 산업 도메인을 선택하세요:")
            for i, domain in enumerate(domains, 1):
                print(f"  {i}. {domain}")
            
            try:
                domain_choice = input("선택 (1-6, 기본값=1): ").strip()
                domain_idx = int(domain_choice) - 1 if domain_choice else 0
                domain = domains[domain_idx] if 0 <= domain_idx < len(domains) else domains[0]
            except:
                domain = domains[0]
            
            print(f"\n🚀 선택된 도메인: {domain}")
            print("="*60)
            
            # 세션 시작
            session_counter += 1
            session_id = f"interactive-{session_counter}"
            session = await think_manager.start_think_session(session_id)
            
            # 사고 과정 실행
            async for think_block in think_manager.generate_progressive_think(
                session_id, topic, domain
            ):
                formatted = think_manager.format_think_block_for_display(think_block)
                print(f"\n{formatted}")
                await asyncio.sleep(0.3)
            
            # 간단한 세션 요약
            stats = think_manager.end_think_session(session_id)
            if stats:
                print(f"\n📈 처리 완료: {stats['total_blocks']}개 블록, {stats['total_time']:.2f}초")
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 사용자 중단으로 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    print("ULTRATHINK 테스트 모드를 선택하세요:")
    print("1. 데모 모드 (자동 테스트)")
    print("2. 대화형 모드 (직접 입력)")
    
    try:
        choice = input("선택 (1 또는 2): ").strip()
        
        if choice == "2":
            asyncio.run(interactive_ultrathink())
        else:
            asyncio.run(demo_ultrathink())
            
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")