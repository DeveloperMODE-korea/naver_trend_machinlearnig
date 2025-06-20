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
        collector = NaverDataCollector()
        
        # API 인증 정보 설정
        collector.setup_credentials()
        
        # 키워드 입력
        keywords = collector.get_keywords_from_user()
        
        # 날짜 범위 입력
        start_year, end_year = collector.get_date_range_from_user()
        
        # 저장된 데이터 확인
        existing_df, keywords_to_collect = collector.check_saved_data(keywords)
        
        # 데이터 수집 실행
        if keywords_to_collect:
            print(f"\n새로운 데이터 수집: {', '.join(keywords_to_collect)}")
            result_df = collector.collect_data(keywords_to_collect, start_year, end_year, existing_df)
        else:
            result_df = existing_df
            print("\n기존 데이터를 사용합니다.")
        
        if result_df.empty:
            print("❌ 수집된 데이터가 없습니다. 프로그램을 종료합니다.")
            return
        
        print(f"✅ 데이터 준비 완료: {len(result_df)}개 데이터 포인트")
        
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
    print("5. 설정 확인")
    print("6. 종료")
    print("-" * 60)
    
    choice = get_user_input("선택하세요 (1-6)", "4")
    return choice

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
            # 설정 확인
            print(f"\n📋 현재 설정:")
            print(f"   저장 디렉토리: {Config.SAVE_DIR}")
            print(f"   API URL: {Config.API_URL}")
            print(f"   시각화 DPI: {Config.DPI}")
            print(f"   LSTM 에포크: {Config.LSTM_EPOCHS}")
            
        elif choice == "6":
            print("👋 프로그램을 종료합니다.")
            break
            
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--menu":
            run_individual_modules()
        else:
            main()
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        print("python main_new.py --menu 로 개별 모듈을 실행해보세요.") 