#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
네이버 쇼핑 트렌드 머신러닝 분석 시스템 (모듈화 버전)

모듈화된 구조로 개선된 트렌드 분석 시스템입니다.
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.core.data_manager import DataManager

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🛍️  네이버 쇼핑 트렌드 머신러닝 분석 시스템")
    print("=" * 60)
    print("모듈화 버전 v2.1")
    print("CPU 및 AMD GPU 지원")
    print("-" * 60)
    
    try:
        # 설정 초기화
        Config.initialize()
        
        # 데이터 관리자 초기화
        data_manager = DataManager()
        
        # 전체 프로세스 실행
        data_manager.run_full_process()
        
        print("\n" + "=" * 60)
        print("🎉 모든 분석이 완료되었습니다!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다:")
        print(f"오류 내용: {str(e)}")
        print("\n상세 오류 정보:")
        import traceback
        traceback.print_exc()

def run_web_dashboard():
    """웹 대시보드 실행"""
    try:
        from web.app import main as web_main
        web_main()
    except ImportError as e:
        print(f"웹 대시보드 모듈을 찾을 수 없습니다: {e}")
        print("streamlit run web/app.py 명령을 직접 실행해보세요.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--web", "--dashboard"]:
            run_web_dashboard()
        elif sys.argv[1] in ["--help", "-h"]:
            print("\n사용법:")
            print("  python main.py           # 기본 전체 실행")
            print("  python main.py --web     # 웹 대시보드 실행")
            print("  python main.py --help    # 도움말")
        else:
            print(f"알 수 없는 옵션: {sys.argv[1]}")
            print("python main.py --help 로 사용법을 확인하세요.")
    else:
        main() 