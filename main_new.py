#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
네이버 쇼핑 트렌드 머신러닝 분석 프로그램 (로컬 버전)

기존 1500+ 줄의 코드를 모듈화하여 관리하기 쉽게 개선했습니다.
- 구글 코랩 의존성 제거
- AMD GPU 및 CPU 환경에서 동작
- 모듈화된 구조로 유지보수성 향상
"""

import sys
import traceback
import pandas as pd
from config import Config
from data_collector import NaverDataCollector
from data_analyzer import DataAnalyzer
from ml_models import MLModelTrainer
from utils import get_user_input

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🛍️  네이버 쇼핑 트렌드 머신러닝 분석 시스템")
    print("=" * 60)
    print("로컬 환경 최적화 버전 v2.0")
    print("CPU 및 AMD GPU 지원")
    print("-" * 60)
    
    try:
        # 1. 설정 초기화
        print("\n1️⃣  시스템 설정 초기화...")
        Config.initialize()
        
        # 2. 데이터 수집
        print("\n2️⃣  데이터 수집 모듈")
        
        # 플랫폼 선택
        print("\n📊 데이터 플랫폼 선택:")
        print("1. 네이버 쇼핑 트렌드만")
        print("2. 구글 트렌드만") 
        print("3. 네이버 + 구글 트렌드 (통합 분석)")
        
        platform_choice = get_user_input("플랫폼을 선택하세요 (1-3)", "1")
        
        # 키워드 입력 (공통)
        collector = NaverDataCollector()  # 키워드 입력용
        keywords = collector.get_keywords_from_user()
        start_year, end_year = collector.get_date_range_from_user()
        
        if platform_choice == "1":
            # 네이버만
            print("\n📊 네이버 쇼핑 트렌드 데이터 수집...")
            collector.setup_credentials()
            existing_df, keywords_to_collect = collector.check_saved_data(keywords)
            
            if keywords_to_collect:
                print(f"\n새로운 데이터 수집: {', '.join(keywords_to_collect)}")
                result_df = collector.collect_data(keywords_to_collect, start_year, end_year, existing_df)
            else:
                result_df = existing_df
                print("\n기존 데이터를 사용합니다.")
            
        elif platform_choice == "2":
            # 구글 트렌드만
            print("\n🌐 구글 트렌드 데이터 수집...")
            from data_collector import GoogleTrendsCollector
            google_collector = GoogleTrendsCollector()
            result_df = google_collector.collect_trends_data(keywords, start_year, end_year)
            
        elif platform_choice == "3":
            # 통합 수집
            print("\n🔄 다중 플랫폼 데이터 수집...")
            from data_collector import MultiPlatformCollector
            
            # 네이버 API 인증
            collector.setup_credentials()
            
            # 다중 플랫폼 수집기 초기화
            multi_collector = MultiPlatformCollector(collector.client_id, collector.client_secret)
            
            # 플랫폼 선택
            platforms = ['naver', 'google']
            use_existing = get_user_input("기존 네이버 데이터를 활용하시겠습니까?", "y", bool)
            
            if use_existing:
                existing_df, keywords_to_collect = collector.check_saved_data(keywords)
                if not existing_df.empty:
                    print("기존 네이버 데이터를 로드했습니다.")
                    # 구글 데이터만 추가 수집
                    from data_collector import GoogleTrendsCollector
                    google_collector = GoogleTrendsCollector()
                    google_data = google_collector.collect_trends_data(keywords, start_year, end_year)
                    
                    if not google_data.empty:
                        existing_df['source'] = 'naver'  # 기존 데이터에 소스 추가
                        result_df = pd.concat([existing_df, google_data], ignore_index=True)
                        print(f"✅ 다중 플랫폼 데이터 통합 완료: {len(result_df)}개 포인트")
                    else:
                        result_df = existing_df
                        result_df['source'] = 'naver'
                else:
                    result_df = multi_collector.collect_multi_platform_data(keywords, start_year, end_year, platforms)
            else:
                result_df = multi_collector.collect_multi_platform_data(keywords, start_year, end_year, platforms)
        else:
            print("❌ 잘못된 선택입니다. 네이버 데이터를 사용합니다.")
            collector.setup_credentials()
            existing_df, keywords_to_collect = collector.check_saved_data(keywords)
            result_df = collector.collect_data(keywords_to_collect, start_year, end_year, existing_df)
        
        if result_df.empty:
            print("❌ 수집된 데이터가 없습니다. 프로그램을 종료합니다.")
            return
        
        print(f"✅ 데이터 준비 완료: {len(result_df)}개 데이터 포인트")
        
        # 플랫폼 정보 출력
        if 'source' in result_df.columns:
            platform_counts = result_df['source'].value_counts()
            print("플랫폼별 데이터 분포:")
            for platform, count in platform_counts.items():
                print(f"  {platform}: {count}개")
        else:
            print("플랫폼: 네이버 쇼핑 트렌드")
        
        # 3. 데이터 분석
        print("\n3️⃣  데이터 분석 모듈")
        run_analysis = get_user_input("데이터 분석을 수행하시겠습니까?", "y", bool)
        
        if run_analysis:
            analyzer = DataAnalyzer(result_df)
            analyzer.run_full_analysis()
        
        # 4. 머신러닝 예측
        print("\n4️⃣  머신러닝 예측 모듈")
        run_prediction = get_user_input("머신러닝 예측을 수행하시겠습니까?", "y", bool)
        
        if run_prediction:
            trainer = MLModelTrainer(result_df)
            
            # 예측할 키워드 개수 설정
            available_keywords = result_df['keyword'].unique()
            max_keywords = min(3, len(available_keywords))
            
            print(f"사용 가능한 키워드: {', '.join(available_keywords)}")
            
            # 예측 방법 선택 (다중 플랫폼 자동 감지)
            if 'source' in result_df.columns and len(result_df['source'].unique()) > 1:
                print("다중 플랫폼 데이터가 감지되었습니다.")
                use_multi_platform = get_user_input("다중 플랫폼 통합 예측을 사용하시겠습니까?", "y", bool)
                
                if use_multi_platform:
                    prediction_results = trainer.run_multi_platform_predictions(
                        keywords=available_keywords, 
                        max_keywords=max_keywords
                    )
                else:
                    # 단일 플랫폼별 예측
                    platform_choice = get_user_input(f"예측에 사용할 플랫폼을 선택하세요 ({'/'.join(result_df['source'].unique())})", 
                                                   result_df['source'].unique()[0])
                    
                    platform_data = result_df[result_df['source'] == platform_choice].drop(columns=['source'])
                    trainer_single = MLModelTrainer(platform_data)
                    
                    use_enhanced = get_user_input("개선된 예측 방법을 사용하시겠습니까?", "y", bool)
                    if use_enhanced:
                        prediction_results = trainer_single.run_predictions_enhanced(
                            keywords=available_keywords, 
                            max_keywords=max_keywords
                        )
                    else:
                        prediction_results = trainer_single.run_predictions(
                            keywords=available_keywords, 
                            max_keywords=max_keywords
                        )
            else:
                # 단일 플랫폼 데이터
                use_enhanced = get_user_input("개선된 예측 방법을 사용하시겠습니까? (범위 제약, 정규화 등 적용)", "y", bool)
                
                if use_enhanced:
                    prediction_results = trainer.run_predictions_enhanced(
                        keywords=available_keywords, 
                        max_keywords=max_keywords
                    )
                else:
                    prediction_results = trainer.run_predictions(
                        keywords=available_keywords, 
                        max_keywords=max_keywords
                    )
            
            if prediction_results:
                print(f"\n✅ {len(prediction_results)}개 키워드 예측 완료!")
                for keyword, result in prediction_results.items():
                    print(f"\n📈 {keyword} 예측 결과:")
                    print(result.to_string(index=False))
            else:
                print("\n❌ 예측 결과가 없습니다.")
        
        # 5. 완료 메시지
        print("\n" + "=" * 60)
        print("🎉 모든 분석이 완료되었습니다!")
        print("=" * 60)
        print(f"📁 결과 파일 위치: {Config.SAVE_DIR}")
        print("📊 생성된 파일들:")
        print("   - CSV 데이터 파일들")
        print("   - PNG 시각화 이미지들")
        print("   - 훈련된 모델 파일들")
        print("-" * 60)
        print("💡 팁: 다음 실행 시 저장된 데이터와 모델을 재사용할 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다:")
        print(f"오류 내용: {str(e)}")
        print("\n상세 오류 정보:")
        traceback.print_exc()
        print("\n문제가 지속되면 각 모듈을 개별적으로 실행해보세요.")

def show_menu():
    """고급 사용자를 위한 메뉴 표시"""
    print("\n" + "=" * 60)
    print("🔧 고급 옵션 메뉴")
    print("=" * 60)
    print("1. 데이터 수집만 실행")
    print("2. 데이터 분석만 실행")
    print("3. 머신러닝 예측만 실행")
    print("4. 전체 실행 (권장)")
    print("5. 웹 대시보드 실행 (NEW!)")
    print("6. 설정 확인")
    print("7. 종료")
    print("-" * 60)
    
    choice = get_user_input("선택하세요 (1-7)", "4")
    return choice

def run_web_dashboard():
    """웹 대시보드 실행"""
    print("\n" + "=" * 60)
    print("🌐 웹 대시보드 시작")
    print("=" * 60)
    
    try:
        import streamlit
        print("✅ Streamlit 패키지 확인됨")
    except ImportError:
        print("❌ Streamlit가 설치되지 않았습니다.")
        install = get_user_input("지금 설치하시겠습니까?", "y", bool)
        if install:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit 설치 완료")
        else:
            print("웹 대시보드 실행을 취소합니다.")
            return
    
    print("\n🚀 웹 대시보드를 시작합니다...")
    print("📱 브라우저에서 http://localhost:8501 을 열어주세요.")
    print("⏹️  중단하려면 터미널에서 Ctrl+C를 누르세요.")
    print("-" * 60)
    
    try:
        import subprocess
        import os
        
        # Streamlit 앱 실행
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        process = subprocess.Popen(cmd)
        
        print("웹 대시보드가 실행 중입니다...")
        print("메뉴로 돌아가려면 브라우저를 닫고 여기서 Ctrl+C를 누르세요.")
        
        # 프로세스 대기
        process.wait()
        
    except KeyboardInterrupt:
        print("\n웹 대시보드를 종료합니다.")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"웹 대시보드 실행 중 오류: {e}")
        print("수동으로 실행하려면: streamlit run streamlit_app.py")

def run_individual_modules():
    """개별 모듈 실행"""
    Config.initialize()
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            # 데이터 수집만
            collector = NaverDataCollector()
            collector.setup_credentials()
            keywords = collector.get_keywords_from_user()
            start_year, end_year = collector.get_date_range_from_user()
            existing_df, keywords_to_collect = collector.check_saved_data(keywords)
            
            if keywords_to_collect:
                result_df = collector.collect_data(keywords_to_collect, start_year, end_year, existing_df)
                print(f"✅ 데이터 수집 완료: {len(result_df)}개 데이터 포인트")
            else:
                print("✅ 기존 데이터를 사용합니다.")
                
        elif choice == "2":
            # 데이터 분석만
            try:
                from utils import load_file
                result_df = load_file(Config.DATA_FILE)
                if result_df is not None:
                    analyzer = DataAnalyzer(result_df)
                    analyzer.run_full_analysis()
                else:
                    print("❌ 분석할 데이터가 없습니다. 먼저 데이터를 수집하세요.")
            except Exception as e:
                print(f"❌ 데이터 로드 실패: {e}")
                
        elif choice == "3":
            # 머신러닝 예측만
            try:
                from utils import load_file
                result_df = load_file(Config.DATA_FILE)
                if result_df is not None:
                    trainer = MLModelTrainer(result_df)
                    results = trainer.run_predictions()
                    print(f"✅ 예측 완료: {len(results)}개 키워드")
                else:
                    print("❌ 예측할 데이터가 없습니다. 먼저 데이터를 수집하세요.")
            except Exception as e:
                print(f"❌ 예측 실행 실패: {e}")
                
        elif choice == "4":
            # 전체 실행
            main()
            break
            
        elif choice == "5":
            # 웹 대시보드 실행
            run_web_dashboard()
            
        elif choice == "6":
            # 설정 확인
            print(f"\n📋 현재 설정:")
            print(f"   저장 디렉토리: {Config.SAVE_DIR}")
            print(f"   API URL: {Config.API_URL}")
            print(f"   시각화 DPI: {Config.DPI}")
            print(f"   LSTM 에포크: {Config.LSTM_EPOCHS}")
            
        elif choice == "7":
            print("👋 프로그램을 종료합니다.")
            break
            
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "--menu":
                run_individual_modules()
            elif sys.argv[1] == "--web" or sys.argv[1] == "--dashboard":
                run_web_dashboard()
            elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
                print("\n사용법:")
                print("  python main_new.py           # 기본 전체 실행")
                print("  python main_new.py --menu    # 고급 메뉴")
                print("  python main_new.py --web     # 웹 대시보드 실행")
                print("  python main_new.py --help    # 도움말")
            else:
                print(f"알 수 없는 옵션: {sys.argv[1]}")
                print("python main_new.py --help 로 사용법을 확인하세요.")
        else:
            main()
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        print("python main_new.py --menu 로 개별 모듈을 실행해보세요.") 