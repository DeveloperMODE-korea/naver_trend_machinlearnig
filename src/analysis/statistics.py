"""
통계 분석 모듈

기본 통계, 상관관계, 성장률 등의 분석을 담당합니다.
"""

import pandas as pd
import numpy as np
from ..utils.file_utils import FileManager
from .visualization import VisualizationEngine

class StatisticsAnalyzer:
    """통계 분석 클래스"""
    
    def __init__(self, data_df):
        """초기화"""
        self.data_df = data_df.copy()
        self.viz = VisualizationEngine(data_df)
        self.prepare_data()
    
    def prepare_data(self):
        """데이터 전처리"""
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return
        
        self.data_df = FileManager.safe_parse_date_column(self.data_df, 'date')
        
        if not self.data_df.empty:
            self.data_df['yearmonth'] = self.data_df['date'].dt.strftime('%Y-%m')
            print(f"분석 준비 완료: {len(self.data_df)}개 데이터 포인트")
    
    def basic_statistics(self):
        """기본 통계 분석"""
        print("\n=== 기본 통계 분석 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        # 키워드별 통계
        stats_df = self.data_df.groupby('keyword')['ratio'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        stats_df.columns = ['키워드', '데이터 개수', '평균', '표준편차', '최소값', '최대값']
        
        print("\n키워드별 기본 통계:")
        print(stats_df)
        
        # 연도별 평균
        yearly_avg = self.data_df.groupby(['year', 'keyword'])['ratio'].mean().reset_index()
        yearly_avg.columns = ['연도', '키워드', '평균 검색 비율']
        
        print("\n연도별 키워드 평균 검색 비율:")
        print(yearly_avg.pivot(index='연도', columns='키워드', values='평균 검색 비율'))
        
        # 결과 저장
        FileManager.save_file(stats_df, 'basic_statistics.csv')
        FileManager.save_file(yearly_avg, 'yearly_statistics.csv')
        
        return stats_df, yearly_avg
    
    def correlation_analysis(self):
        """상관관계 분석"""
        print("\n=== 상관관계 분석 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        # 피벗 테이블 생성
        pivot_df = self.data_df.pivot_table(
            index='date', 
            columns='keyword', 
            values='ratio'
        ).fillna(0)
        
        if pivot_df.empty:
            print("상관관계 분석을 위한 데이터가 부족합니다.")
            return None
        
        # 상관관계 계산
        correlation = pivot_df.corr()
        print("\n키워드 간 상관관계:")
        print(correlation)
        
        # 시각화
        self.viz.plot_correlation_heatmap(correlation)
        
        # 결과 저장
        FileManager.save_file(correlation, 'correlation_matrix.csv')
        
        return correlation
    
    def growth_rate_analysis(self):
        """성장률 분석"""
        print("\n=== 성장률 분석 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        keywords = self.data_df['keyword'].unique()
        growth_rates = []
        
        for keyword in keywords:
            keyword_data = self.data_df[self.data_df['keyword'] == keyword].sort_values('date')
            
            if len(keyword_data) >= 12:
                first_year = keyword_data.iloc[:12]['ratio'].mean()
                last_year = keyword_data.iloc[-12:]['ratio'].mean()
                growth_rate = ((last_year - first_year) / first_year) * 100 if first_year > 0 else 0
                
                growth_rates.append({
                    'keyword': keyword,
                    'first_year_avg': first_year,
                    'last_year_avg': last_year,
                    'growth_rate': growth_rate
                })
        
        if growth_rates:
            growth_df = pd.DataFrame(growth_rates)
            growth_df.columns = ['키워드', '첫해 평균', '마지막해 평균', '성장률(%)']
            
            print("\n키워드별 성장률:")
            print(growth_df)
            
            # 시각화
            self.viz.plot_growth_rate(growth_df)
            
            # 결과 저장
            FileManager.save_file(growth_df, 'growth_rate_analysis.csv')
            
            return growth_df
        else:
            print("성장률 분석을 위한 충분한 데이터가 없습니다.")
            return None
    
    def monthly_analysis(self):
        """월별 분석"""
        print("\n=== 월별 분석 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        # 월별 평균
        monthly_avg = self.data_df.groupby(['month', 'keyword'])['ratio'].mean().reset_index()
        FileManager.save_file(monthly_avg, 'monthly_averages.csv')
        
        # 월별 최고 인기 키워드 찾기
        try:
            monthly_max = self.data_df.groupby('yearmonth')['ratio'].max().reset_index()
            monthly_top = pd.merge(
                monthly_max,
                self.data_df[['yearmonth', 'keyword', 'ratio']],
                on=['yearmonth', 'ratio'],
                how='left'
            ).drop_duplicates(['yearmonth', 'ratio'])
            
            if not monthly_top.empty:
                print("월별 최고 인기 키워드:")
                print(monthly_top[['yearmonth', 'keyword', 'ratio']])
                
                # 시각화
                self.viz.plot_monthly_trends(monthly_avg, monthly_top)
                
                FileManager.save_file(monthly_top, 'monthly_top_keywords.csv')
                return monthly_top
            
        except Exception as e:
            print(f"월별 분석 중 오류: {e}")
            return None
    
    def multi_platform_analysis(self):
        """다중 플랫폼 분석"""
        print("\n=== 다중 플랫폼 분석 ===")
        
        # source 컬럼이 있는지 확인
        if 'source' not in self.data_df.columns:
            print("단일 플랫폼 데이터입니다. 다중 플랫폼 분석을 건너뜁니다.")
            return
        
        platforms = self.data_df['source'].unique()
        print(f"발견된 플랫폼: {', '.join(platforms)}")
        
        # 플랫폼별 기본 통계
        platform_stats = self.data_df.groupby(['source', 'keyword'])['ratio'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("\n플랫폼별 키워드 통계:")
        print(platform_stats)
        FileManager.save_file(platform_stats.reset_index(), 'multi_platform_statistics.csv')
        
        # 시각화
        self.viz.plot_platform_comparison()
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("\n🔍 전체 데이터 분석을 시작합니다...")
        
        # 기본 통계
        self.basic_statistics()
        
        # 상관관계 분석
        self.correlation_analysis()
        
        # 트렌드 시각화
        self.viz.plot_trend_charts()
        
        # 월별 분석
        self.monthly_analysis()
        
        # 성장률 분석
        self.growth_rate_analysis()
        
        # 다중 플랫폼 분석 (해당되는 경우)
        self.multi_platform_analysis()
        
        print("\n✅ 전체 분석이 완료되었습니다!")
        print(f"결과는 {self.data_df} 폴더에 저장되었습니다.") 