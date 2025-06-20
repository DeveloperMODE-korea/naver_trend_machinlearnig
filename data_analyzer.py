import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from config import Config
from utils import safe_parse_date_column, save_file
import os

class DataAnalyzer:
    """데이터 분석 및 시각화 클래스"""
    
    def __init__(self, data_df):
        """초기화"""
        self.data_df = data_df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """데이터 전처리"""
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return
        
        # 날짜 파싱
        self.data_df = safe_parse_date_column(self.data_df, 'date')
        
        if not self.data_df.empty:
            self.data_df['yearmonth'] = self.data_df['date'].dt.strftime('%Y-%m')
            print(f"분석 준비 완료: {len(self.data_df)}개 데이터 포인트")
        else:
            print("유효한 데이터가 없습니다.")
    
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
        save_file(stats_df, 'basic_statistics.csv')
        save_file(yearly_avg, 'yearly_statistics.csv')
        
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
        
        # 상관관계 히트맵
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                   center=0, linewidths=0.5, square=True)
        plt.title('키워드 간 상관관계', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'correlation_heatmap.png')
        plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
        plt.show()
        
        # 결과 저장
        save_file(correlation, 'correlation_matrix.csv')
        
        return correlation
    
    def trend_visualization(self):
        """트렌드 시각화"""
        print("\n=== 트렌드 시각화 ===")
        
        if self.data_df.empty:
            print("시각화할 데이터가 없습니다.")
            return
        
        keywords = self.data_df['keyword'].unique()
        
        # 1. 전체 트렌드
        plt.figure(figsize=Config.FIGURE_SIZE)
        for keyword in keywords:
            keyword_data = self.data_df[self.data_df['keyword'] == keyword]
            plt.plot(keyword_data['date'], keyword_data['ratio'], 
                    marker='o', linewidth=2, label=keyword)
        
        plt.title('네이버 쇼핑 검색 트렌드 (전체 기간)', fontsize=16)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('검색 비율', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'overall_trend.png')
        plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
        plt.show()
        
        # 2. 연도별 박스플롯
        plt.figure(figsize=Config.FIGURE_SIZE)
        sns.boxplot(x='year', y='ratio', hue='keyword', data=self.data_df)
        plt.title('연도별 키워드 검색 비율 분포', fontsize=16)
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('검색 비율', fontsize=12)
        plt.legend(title='키워드', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'yearly_boxplot.png')
        plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
        plt.show()
        
        # 3. 월별 계절성 분석
        monthly_avg = self.data_df.groupby(['month', 'keyword'])['ratio'].mean().reset_index()
        
        plt.figure(figsize=Config.FIGURE_SIZE)
        for keyword in keywords:
            keyword_data = monthly_avg[monthly_avg['keyword'] == keyword]
            plt.plot(keyword_data['month'], keyword_data['ratio'], 
                    marker='o', linewidth=2, label=keyword)
        
        plt.title('월별 평균 검색 비율 (계절성)', fontsize=16)
        plt.xlabel('월', fontsize=12)
        plt.ylabel('평균 검색 비율', fontsize=12)
        plt.xticks(range(1, 13))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'monthly_seasonality.png')
        plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
        plt.show()
        
        # 월별 데이터 저장
        save_file(monthly_avg, 'monthly_averages.csv')
    
    def heatmap_analysis(self):
        """히트맵 분석"""
        print("\n=== 히트맵 분석 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return
        
        # 월별 키워드 인기도 히트맵
        try:
            heatmap_data = self.data_df.pivot_table(
                index='yearmonth',
                columns='keyword',
                values='ratio',
                aggfunc='mean'
            ).fillna(0)
            
            if heatmap_data.size > 0 and not heatmap_data.empty:
                plt.figure(figsize=(15, 8))
                sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, 
                           fmt='.1f', linewidths=0.5)
                plt.title('월별 키워드 인기도 히트맵', fontsize=16)
                plt.xlabel('키워드', fontsize=12)
                plt.ylabel('년-월', fontsize=12)
                plt.tight_layout()
                
                plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'monthly_heatmap.png')
                plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
                plt.show()
                
                # 히트맵 데이터 저장
                save_file(heatmap_data, 'heatmap_data.csv')
            else:
                print("히트맵에 표시할 데이터가 충분하지 않습니다.")
        
        except Exception as e:
            print(f"히트맵 생성 중 오류: {e}")
    
    def monthly_top_keywords(self):
        """월별 최고 인기 키워드 분석"""
        print("\n=== 월별 최고 인기 키워드 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return None
        
        try:
            # 월별 최대값 찾기
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
                keywords = self.data_df['keyword'].unique()
                keyword_colors = {kw: plt.cm.Set3(i) for i, kw in enumerate(keywords)}
                
                plt.figure(figsize=(15, 8))
                for i, (ym, kw, ratio) in enumerate(zip(
                    monthly_top['yearmonth'], 
                    monthly_top['keyword'], 
                    monthly_top['ratio']
                )):
                    if pd.notna(kw) and pd.notna(ratio):
                        color = keyword_colors.get(kw, plt.cm.Set3(0))
                        plt.bar(i, ratio, color=color, 
                               label=kw if kw not in [l.get_label() for l in plt.gca().get_legend_handles_labels()[0]] else "")
                        plt.text(i, ratio/2, kw, ha='center', rotation=90, 
                               color='black', fontweight='bold')
                
                plt.xticks(range(len(monthly_top)), monthly_top['yearmonth'], rotation=45)
                plt.title('월별 최고 인기 키워드', fontsize=16)
                plt.xlabel('년-월', fontsize=12)
                plt.ylabel('검색 비율', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 중복 제거된 범례
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper right')
                
                plt.tight_layout()
                
                plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'monthly_top_keywords.png')
                plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
                plt.show()
                
                # 결과 저장
                save_file(monthly_top, 'monthly_top_keywords.csv')
                
                return monthly_top
            else:
                print("월별 최고 인기 키워드를 찾을 수 없습니다.")
                return None
                
        except Exception as e:
            print(f"월별 최고 인기 키워드 분석 중 오류: {e}")
            return None
    
    def time_series_decomposition(self, keywords=None):
        """시계열 분해 분석"""
        print("\n=== 시계열 분해 분석 ===")
        
        if self.data_df.empty:
            print("분석할 데이터가 없습니다.")
            return
        
        if keywords is None:
            keywords = self.data_df['keyword'].unique()
        
        for keyword in keywords:
            keyword_data = self.data_df[self.data_df['keyword'] == keyword].sort_values('date')
            
            if len(keyword_data) >= 12:  # 최소 12개월 데이터 필요
                try:
                    # 시계열 인덱스로 변환
                    ts_data = keyword_data.set_index('date')['ratio']
                    
                    # 시계열 분해
                    decomposition = seasonal_decompose(ts_data, model='additive', period=12)
                    
                    # 결과 시각화
                    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
                    
                    decomposition.observed.plot(ax=axes[0], title=f'{keyword} - 원본 데이터')
                    decomposition.trend.plot(ax=axes[1], title='추세')
                    decomposition.seasonal.plot(ax=axes[2], title='계절성')
                    decomposition.resid.plot(ax=axes[3], title='잔차')
                    
                    plt.suptitle(f'{keyword} 시계열 분해', fontsize=16)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(Config.SAVE_DIR, 'plots', f'{keyword}_decomposition.png')
                    plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
                    plt.show()
                    
                    # 정상성 검정 (ADF 테스트)
                    adf_result = adfuller(ts_data.dropna())
                    print(f'\n{keyword} ADF 검정 결과:')
                    print(f'ADF 통계량: {adf_result[0]:.4f}')
                    print(f'p-value: {adf_result[1]:.4f}')
                    print(f'정상성: {"정상" if adf_result[1] < 0.05 else "비정상"}')
                    
                except Exception as e:
                    print(f"\n{keyword} 시계열 분해 중 오류: {e}")
            else:
                print(f"\n{keyword}는 시계열 분해를 위한 데이터가 부족합니다 (최소 12개월 필요).")
    
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
            
            # 성장률 시각화
            plt.figure(figsize=(12, 6))
            colors = ['green' if rate > 0 else 'red' for rate in growth_df['성장률(%)']]
            plt.bar(growth_df['키워드'], growth_df['성장률(%)'], color=colors, alpha=0.7)
            plt.title('키워드별 검색 비율 성장률', fontsize=16)
            plt.xlabel('키워드', fontsize=12)
            plt.ylabel('성장률 (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'growth_rate.png')
            plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
            plt.show()
            
            # 결과 저장
            save_file(growth_df, 'growth_rate_analysis.csv')
            
            return growth_df
        else:
            print("성장률 분석을 위한 충분한 데이터가 없습니다.")
            return None
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("\n🔍 전체 데이터 분석을 시작합니다...")
        
        # 기본 통계
        self.basic_statistics()
        
        # 상관관계 분석
        self.correlation_analysis()
        
        # 트렌드 시각화
        self.trend_visualization()
        
        # 히트맵 분석
        self.heatmap_analysis()
        
        # 월별 최고 인기 키워드
        self.monthly_top_keywords()
        
        # 성장률 분석
        self.growth_rate_analysis()
        
        # 시계열 분해 (첫 3개 키워드만)
        keywords = self.data_df['keyword'].unique()[:3]
        self.time_series_decomposition(keywords)
        
        print("\n✅ 전체 분석이 완료되었습니다!")
        print(f"결과는 {Config.SAVE_DIR} 폴더에 저장되었습니다.") 