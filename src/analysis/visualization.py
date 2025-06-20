"""
시각화 엔진 모듈

각종 차트와 그래프 생성을 담당합니다.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from ..config import Config

class VisualizationEngine:
    """시각화 엔진 클래스"""
    
    def __init__(self, data_df):
        """초기화"""
        self.data_df = data_df.copy()
        self.keywords = self.data_df['keyword'].unique()
    
    def plot_correlation_heatmap(self, correlation):
        """상관관계 히트맵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                   center=0, linewidths=0.5, square=True)
        plt.title('키워드 간 상관관계', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'correlation_heatmap.png')
        plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_trend_charts(self):
        """트렌드 차트들"""
        # 전체 트렌드
        plt.figure(figsize=Config.FIGURE_SIZE)
        for keyword in self.keywords:
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
        
        # 연도별 박스플롯
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
    
    def plot_monthly_trends(self, monthly_avg, monthly_top):
        """월별 트렌드"""
        # 월별 계절성 분석
        plt.figure(figsize=Config.FIGURE_SIZE)
        for keyword in self.keywords:
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
        
        # 히트맵
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
        except Exception as e:
            print(f"히트맵 생성 중 오류: {e}")
    
    def plot_growth_rate(self, growth_df):
        """성장률 차트"""
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
    
    def plot_platform_comparison(self):
        """플랫폼 비교 차트"""
        if 'source' not in self.data_df.columns:
            return
        
        platforms = self.data_df['source'].unique()
        keywords = self.data_df['keyword'].unique()
        
        # 플랫폼별 키워드 트렌드 비교
        fig, axes = plt.subplots(len(keywords), 1, figsize=(15, 5 * len(keywords)))
        if len(keywords) == 1:
            axes = [axes]
        
        for idx, keyword in enumerate(keywords):
            keyword_data = self.data_df[self.data_df['keyword'] == keyword]
            
            for platform in platforms:
                platform_data = keyword_data[keyword_data['source'] == platform]
                if not platform_data.empty:
                    axes[idx].plot(platform_data['date'], platform_data['ratio'], 
                                 marker='o', linewidth=2, label=f'{platform}', alpha=0.8)
            
            axes[idx].set_title(f'{keyword} - 플랫폼별 검색 트렌드 비교', fontsize=14)
            axes[idx].set_xlabel('날짜')
            axes[idx].set_ylabel('검색 비율')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(Config.SAVE_DIR, 'plots', 'all_keywords_platform_comparison.png')
        plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
        plt.show() 