import pandas as pd
from .naver_collector import NaverDataCollector
from .google_collector import GoogleTrendsCollector
from ..utils.file_utils import FileManager

class MultiPlatformCollector:
    """네이버 + 구글 트렌드 통합 수집 클래스"""
    
    def __init__(self, naver_client_id=None, naver_client_secret=None):
        """초기화"""
        self.naver_collector = NaverDataCollector(naver_client_id, naver_client_secret)
        self.google_collector = GoogleTrendsCollector()
    
    def collect_multi_platform_data(self, keywords, start_year, end_year, platforms=['naver', 'google']):
        """다중 플랫폼 데이터를 수집합니다"""
        print(f"\n🔄 다중 플랫폼 데이터 수집 시작...")
        print(f"플랫폼: {', '.join(platforms)}")
        
        all_data = []
        
        # 네이버 데이터 수집
        if 'naver' in platforms:
            print(f"\n📊 네이버 데이터 수집...")
            naver_data = self.naver_collector.collect_data(keywords, start_year, end_year)
            if not naver_data.empty:
                naver_data['source'] = 'naver'
                all_data.append(naver_data)
                print(f"네이버 데이터: {len(naver_data)}개 포인트")
        
        # 구글 트렌드 데이터 수집
        if 'google' in platforms:
            print(f"\n🌐 구글 트렌드 데이터 수집...")
            google_data = self.google_collector.collect_trends_data(keywords, start_year, end_year)
            if not google_data.empty:
                all_data.append(google_data)
                print(f"구글 트렌드 데이터: {len(google_data)}개 포인트")
        
        # 데이터 통합
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['source', 'keyword', 'date'])
            
            # 저장
            FileManager.save_file(combined_df, 'multi_platform_data.csv')
            
            print(f"\n✅ 다중 플랫폼 데이터 수집 완료: {len(combined_df)}개 포인트")
            print(f"플랫폼별 데이터 분포:")
            for source in combined_df['source'].unique():
                count = len(combined_df[combined_df['source'] == source])
                print(f"  {source}: {count}개")
            
            return combined_df
        else:
            print("\n❌ 수집된 데이터가 없습니다.")
            return pd.DataFrame()
    
    def compare_platforms(self, keyword, data_df):
        """플랫폼 간 데이터를 비교합니다"""
        print(f"\n📈 '{keyword}' 플랫폼 간 비교 분석")
        
        keyword_data = data_df[data_df['keyword'] == keyword]
        
        if keyword_data.empty:
            print(f"'{keyword}' 데이터가 없습니다.")
            return None
        
        # 플랫폼별 통계
        platform_stats = keyword_data.groupby('source')['ratio'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("플랫폼별 통계:")
        print(platform_stats)
        
        # 상관관계 분석 (같은 기간 데이터가 있는 경우)
        pivot_data = keyword_data.pivot_table(
            index=['year', 'month'], 
            columns='source', 
            values='ratio'
        ).dropna()
        
        if len(pivot_data) > 1 and len(pivot_data.columns) > 1:
            correlation = pivot_data.corr()
            print(f"\n플랫폼 간 상관관계:")
            print(correlation)
            
            return platform_stats, correlation
        else:
            print("상관관계 분석을 위한 충분한 겹치는 데이터가 없습니다.")
            return platform_stats, None
