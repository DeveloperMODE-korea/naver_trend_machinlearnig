"""
데이터 관리자 모듈

전체 데이터 수집, 분석, 예측 프로세스를 통합 관리합니다.
"""

from ..collectors import NaverDataCollector, GoogleTrendsCollector, MultiPlatformCollector
from ..analysis.statistics import StatisticsAnalyzer
from ..models.ensemble_model import EnsembleModel
from ..utils.common_utils import CommonUtils

class DataManager:
    """데이터 수집, 분석, 예측을 통합 관리하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.data_df = None
        self.platforms = []
        self.keywords = []
        self.start_year = 2020
        self.end_year = 2024
    
    def get_user_preferences(self):
        """사용자 설정을 받습니다"""
        print("\n📋 분석 설정")
        
        # 플랫폼 선택
        print("\n📊 데이터 플랫폼 선택:")
        print("1. 네이버 쇼핑 트렌드만")
        print("2. 구글 트렌드만") 
        print("3. 네이버 + 구글 트렌드 (통합 분석)")
        
        platform_choice = CommonUtils.get_user_input("플랫폼을 선택하세요 (1-3)", "1")
        
        if platform_choice == "1":
            self.platforms = ['naver']
        elif platform_choice == "2":
            self.platforms = ['google']
        elif platform_choice == "3":
            self.platforms = ['naver', 'google']
        else:
            print("잘못된 선택입니다. 네이버만 사용합니다.")
            self.platforms = ['naver']
        
        # 키워드 입력
        collector = NaverDataCollector()  # 키워드 입력용
        self.keywords = collector.get_keywords_from_user()
        self.start_year, self.end_year = collector.get_date_range_from_user()
    
    def collect_data(self):
        """데이터를 수집합니다"""
        print(f"\n2️⃣  데이터 수집 ({', '.join(self.platforms)})")
        
        if len(self.platforms) == 1:
            if 'naver' in self.platforms:
                collector = NaverDataCollector()
                collector.setup_credentials()
                existing_df, keywords_to_collect = collector.check_saved_data(self.keywords)
                
                if keywords_to_collect:
                    self.data_df = collector.collect_data(keywords_to_collect, self.start_year, self.end_year, existing_df)
                else:
                    self.data_df = existing_df
                    
            elif 'google' in self.platforms:
                collector = GoogleTrendsCollector()
                self.data_df = collector.collect_trends_data(self.keywords, self.start_year, self.end_year)
        else:
            # 다중 플랫폼
            naver_collector = NaverDataCollector()
            naver_collector.setup_credentials()
            
            multi_collector = MultiPlatformCollector(naver_collector.client_id, naver_collector.client_secret)
            self.data_df = multi_collector.collect_multi_platform_data(
                self.keywords, self.start_year, self.end_year, self.platforms
            )
        
        if self.data_df is None or self.data_df.empty:
            raise ValueError("데이터 수집에 실패했습니다.")
        
        print(f"✅ 데이터 준비 완료: {len(self.data_df)}개 데이터 포인트")
    
    def analyze_data(self):
        """데이터를 분석합니다"""
        print("\n3️⃣  데이터 분석")
        
        run_analysis = CommonUtils.get_user_input("데이터 분석을 수행하시겠습니까?", "y", bool)
        
        if run_analysis:
            analyzer = StatisticsAnalyzer(self.data_df)
            analyzer.run_full_analysis()
    
    def predict_trends(self):
        """트렌드를 예측합니다"""
        print("\n4️⃣  머신러닝 예측")
        
        run_prediction = CommonUtils.get_user_input("머신러닝 예측을 수행하시겠습니까?", "y", bool)
        
        if run_prediction:
            model_trainer = EnsembleModel(self.data_df)
            
            # 예측할 키워드 개수 설정
            available_keywords = self.data_df['keyword'].unique()
            max_keywords = min(3, len(available_keywords))
            
            print(f"사용 가능한 키워드: {', '.join(available_keywords)}")
            
            # 다중 플랫폼 예측 자동 감지
            if 'source' in self.data_df.columns and len(self.data_df['source'].unique()) > 1:
                print("다중 플랫폼 데이터가 감지되었습니다.")
                use_multi_platform = CommonUtils.get_user_input("다중 플랫폼 통합 예측을 사용하시겠습니까?", "y", bool)
                
                if use_multi_platform:
                    prediction_results = model_trainer.run_multi_platform_predictions(
                        keywords=available_keywords, 
                        max_keywords=max_keywords
                    )
                else:
                    prediction_results = model_trainer.run_predictions_enhanced(
                        keywords=available_keywords, 
                        max_keywords=max_keywords
                    )
            else:
                prediction_results = model_trainer.run_predictions_enhanced(
                    keywords=available_keywords, 
                    max_keywords=max_keywords
                )
            
            if prediction_results:
                print(f"\n✅ {len(prediction_results)}개 키워드 예측 완료!")
                for keyword, result in prediction_results.items():
                    print(f"\n📈 {keyword} 예측 결과:")
                    print(result.to_string(index=False))
    
    def run_full_process(self):
        """전체 프로세스를 실행합니다"""
        print("🚀 모듈화된 트렌드 분석 시스템을 시작합니다...")
        
        # 1. 사용자 설정
        self.get_user_preferences()
        
        # 2. 데이터 수집
        self.collect_data()
        
        # 3. 데이터 분석
        self.analyze_data()
        
        # 4. 트렌드 예측
        self.predict_trends()
        
        print("\n🎯 전체 분석 프로세스가 완료되었습니다!") 