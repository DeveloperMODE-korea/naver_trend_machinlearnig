"""
이상치 탐지 모듈

통계적 방법을 사용하여 데이터의 이상치를 탐지하고 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
from ..config import Config

class OutlierDetector:
    """이상치 탐지 클래스"""
    
    def __init__(self, methods: List[str] = ['iqr', 'zscore', 'isolation']):
        """초기화
        
        Args:
            methods: 사용할 이상치 탐지 방법들
                    - 'iqr': 사분위수 범위 방법
                    - 'zscore': Z-점수 방법  
                    - 'isolation': 격리 포레스트 (추후 구현)
        """
        self.methods = methods
        self.outlier_results = {}
    
    def detect_outliers_iqr(self, data: pd.Series, factor: float = 1.5) -> Tuple[np.ndarray, Dict]:
        """IQR 방법으로 이상치를 탐지합니다
        
        Args:
            data: 검사할 데이터 시리즈
            factor: IQR 배수 (기본: 1.5)
            
        Returns:
            이상치 마스크와 통계 정보
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        stats_info = {
            'method': 'IQR',
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(data)) * 100
        }
        
        return outlier_mask, stats_info
    
    def detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """Z-점수 방법으로 이상치를 탐지합니다
        
        Args:
            data: 검사할 데이터 시리즈
            threshold: Z-점수 임계값 (기본: 3.0)
            
        Returns:
            이상치 마스크와 통계 정보
        """
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > threshold
        
        stats_info = {
            'method': 'Z-Score',
            'mean': data.mean(),
            'std': data.std(),
            'threshold': threshold,
            'max_zscore': z_scores.max(),
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(data)) * 100
        }
        
        return outlier_mask, stats_info
    
    def detect_outliers_modified_zscore(self, data: pd.Series, threshold: float = 3.5) -> Tuple[np.ndarray, Dict]:
        """수정된 Z-점수 방법으로 이상치를 탐지합니다 (중앙값 기반)
        
        Args:
            data: 검사할 데이터 시리즈
            threshold: 수정된 Z-점수 임계값 (기본: 3.5)
            
        Returns:
            이상치 마스크와 통계 정보
        """
        median = data.median()
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation
        
        if mad == 0:
            # MAD가 0인 경우 표준편차 사용
            modified_z_scores = np.abs(data - median) / data.std()
        else:
            modified_z_scores = 0.6745 * (data - median) / mad
        
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        stats_info = {
            'method': 'Modified Z-Score',
            'median': median,
            'mad': mad,
            'threshold': threshold,
            'max_modified_zscore': np.abs(modified_z_scores).max(),
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(data)) * 100
        }
        
        return outlier_mask, stats_info
    
    def detect_keyword_outliers(self, data_df: pd.DataFrame, keyword: str) -> Dict:
        """특정 키워드의 이상치를 탐지합니다
        
        Args:
            data_df: 데이터프레임
            keyword: 검사할 키워드
            
        Returns:
            이상치 탐지 결과
        """
        keyword_data = data_df[data_df['keyword'] == keyword]['ratio']
        
        # 데이터 타입 확인 및 변환
        keyword_data = pd.to_numeric(keyword_data, errors='coerce').dropna()
        
        if len(keyword_data) < 10:
            return {
                'keyword': keyword,
                'status': 'insufficient_data',
                'data_count': len(keyword_data),
                'message': '이상치 탐지를 위한 데이터가 부족합니다 (최소 10개 필요)'
            }
        
        results = {
            'keyword': keyword,
            'data_count': len(keyword_data),
            'methods': {}
        }
        
        # IQR 방법
        if 'iqr' in self.methods:
            outlier_mask, stats_info = self.detect_outliers_iqr(keyword_data)
            results['methods']['iqr'] = {
                'outlier_indices': keyword_data.index[outlier_mask].tolist(),
                'outlier_values': keyword_data[outlier_mask].tolist(),
                'stats': stats_info
            }
        
        # Z-Score 방법
        if 'zscore' in self.methods:
            outlier_mask, stats_info = self.detect_outliers_zscore(keyword_data)
            results['methods']['zscore'] = {
                'outlier_indices': keyword_data.index[outlier_mask].tolist(),
                'outlier_values': keyword_data[outlier_mask].tolist(),
                'stats': stats_info
            }
        
        # Modified Z-Score 방법
        if 'modified_zscore' in self.methods:
            outlier_mask, stats_info = self.detect_outliers_modified_zscore(keyword_data)
            results['methods']['modified_zscore'] = {
                'outlier_indices': keyword_data.index[outlier_mask].tolist(),
                'outlier_values': keyword_data[outlier_mask].tolist(),
                'stats': stats_info
            }
        
        # 합의 기반 이상치 (2개 이상 방법에서 탐지된 경우)
        all_outlier_indices = set()
        for method_result in results['methods'].values():
            all_outlier_indices.update(method_result['outlier_indices'])
        
        consensus_outliers = []
        for idx in all_outlier_indices:
            detection_count = sum(1 for method_result in results['methods'].values() 
                                if idx in method_result['outlier_indices'])
            if detection_count >= 2:  # 2개 이상 방법에서 탐지
                consensus_outliers.append(idx)
        
        results['consensus_outliers'] = {
            'indices': consensus_outliers,
            'values': keyword_data.loc[consensus_outliers].tolist() if consensus_outliers else [],
            'count': len(consensus_outliers)
        }
        
        return results
    
    def detect_all_outliers(self, data_df: pd.DataFrame) -> Dict:
        """모든 키워드의 이상치를 탐지합니다
        
        Args:
            data_df: 데이터프레임
            
        Returns:
            전체 이상치 탐지 결과
        """
        keywords = data_df['keyword'].unique()
        all_results = {}
        
        print(f"\n🔍 이상치 탐지 시작 ({len(keywords)}개 키워드)...")
        
        for keyword in keywords:
            print(f"  📊 {keyword} 분석 중...")
            result = self.detect_keyword_outliers(data_df, keyword)
            all_results[keyword] = result
            
            if result.get('status') != 'insufficient_data':
                consensus_count = result['consensus_outliers']['count']
                if consensus_count > 0:
                    print(f"    ⚠️  합의 기반 이상치: {consensus_count}개")
                else:
                    print(f"    ✅ 이상치 없음")
        
        # 전체 요약
        total_consensus_outliers = sum(r['consensus_outliers']['count'] 
                                     for r in all_results.values() 
                                     if r.get('status') != 'insufficient_data')
        
        summary = {
            'total_keywords': len(keywords),
            'analyzed_keywords': len([r for r in all_results.values() 
                                    if r.get('status') != 'insufficient_data']),
            'total_consensus_outliers': total_consensus_outliers,
            'outlier_percentage': (total_consensus_outliers / len(data_df)) * 100 if len(data_df) > 0 else 0
        }
        
        print(f"\n📋 이상치 탐지 완료:")
        print(f"  분석된 키워드: {summary['analyzed_keywords']}/{summary['total_keywords']}")
        print(f"  발견된 이상치: {summary['total_consensus_outliers']}개")
        print(f"  이상치 비율: {summary['outlier_percentage']:.2f}%")
        
        return {
            'summary': summary,
            'results': all_results
        }
    
    def clean_outliers(self, data_df: pd.DataFrame, method: str = 'remove') -> pd.DataFrame:
        """이상치를 처리합니다
        
        Args:
            data_df: 원본 데이터프레임
            method: 처리 방법 ('remove', 'winsorize', 'interpolate')
            
        Returns:
            처리된 데이터프레임
        """
        # 먼저 이상치 탐지
        outlier_results = self.detect_all_outliers(data_df)
        
        cleaned_df = data_df.copy()
        total_removed = 0
        
        for keyword, result in outlier_results['results'].items():
            if result.get('status') == 'insufficient_data':
                continue
            
            outlier_indices = result['consensus_outliers']['indices']
            
            if not outlier_indices:
                continue
            
            keyword_mask = cleaned_df['keyword'] == keyword
            outlier_mask = cleaned_df.index.isin(outlier_indices)
            target_mask = keyword_mask & outlier_mask
            
            if method == 'remove':
                # 이상치 제거
                cleaned_df = cleaned_df[~target_mask]
                total_removed += len(outlier_indices)
                
            elif method == 'winsorize':
                # 이상치를 극값으로 대체 (winsorization)
                keyword_data = cleaned_df[keyword_mask]['ratio']
                p5 = keyword_data.quantile(0.05)
                p95 = keyword_data.quantile(0.95)
                
                cleaned_df.loc[target_mask & (cleaned_df['ratio'] < p5), 'ratio'] = p5
                cleaned_df.loc[target_mask & (cleaned_df['ratio'] > p95), 'ratio'] = p95
                
            elif method == 'interpolate':
                # 이상치를 보간법으로 대체
                keyword_data = cleaned_df[keyword_mask].sort_values('date')
                ratio_series = keyword_data['ratio'].copy()
                
                for idx in outlier_indices:
                    if idx in ratio_series.index:
                        ratio_series.loc[idx] = np.nan
                
                # 선형 보간
                ratio_series = ratio_series.interpolate(method='linear')
                
                # 결과 적용
                for idx in keyword_data.index:
                    if idx in ratio_series.index:
                        cleaned_df.loc[idx, 'ratio'] = ratio_series.loc[idx]
        
        print(f"\n✅ 이상치 처리 완료 ({method} 방법):")
        print(f"  원본 데이터: {len(data_df)}개")
        print(f"  처리된 데이터: {len(cleaned_df)}개")
        if method == 'remove':
            print(f"  제거된 데이터: {total_removed}개")
        
        return cleaned_df
    
    def plot_outlier_analysis(self, data_df: pd.DataFrame, keyword: str, save_plot: bool = True):
        """이상치 분석 결과를 시각화합니다
        
        Args:
            data_df: 데이터프레임
            keyword: 시각화할 키워드
            save_plot: 플롯 저장 여부
        """
        result = self.detect_keyword_outliers(data_df, keyword)
        
        if result.get('status') == 'insufficient_data':
            print(f"{keyword}: 시각화할 데이터가 부족합니다.")
            return
        
        keyword_data = data_df[data_df['keyword'] == keyword]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{keyword} 이상치 분석', fontsize=16)
        
        # 1. 시계열 플롯
        axes[0, 0].plot(keyword_data['date'], keyword_data['ratio'], 'b-', alpha=0.7, label='데이터')
        
        # 합의 기반 이상치 표시
        consensus_indices = result['consensus_outliers']['indices']
        if consensus_indices:
            outlier_data = keyword_data.loc[consensus_indices]
            axes[0, 0].scatter(outlier_data['date'], outlier_data['ratio'], 
                             color='red', s=50, label='이상치', zorder=5)
        
        axes[0, 0].set_title('시계열 데이터')
        axes[0, 0].set_ylabel('검색 비율')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 박스플롯
        box_data = [keyword_data['ratio']]
        axes[0, 1].boxplot(box_data, labels=[keyword])
        axes[0, 1].set_title('박스플롯')
        axes[0, 1].set_ylabel('검색 비율')
        
        # 3. 히스토그램
        axes[1, 0].hist(keyword_data['ratio'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(keyword_data['ratio'].mean(), color='red', linestyle='--', label='평균')
        axes[1, 0].axvline(keyword_data['ratio'].median(), color='green', linestyle='--', label='중앙값')
        axes[1, 0].set_title('분포 히스토그램')
        axes[1, 0].set_xlabel('검색 비율')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].legend()
        
        # 4. Z-score 플롯
        z_scores = np.abs(stats.zscore(keyword_data['ratio']))
        axes[1, 1].plot(range(len(z_scores)), z_scores, 'o-', alpha=0.7)
        axes[1, 1].axhline(y=3, color='red', linestyle='--', label='임계값 (3.0)')
        axes[1, 1].set_title('Z-Score 분석')
        axes[1, 1].set_xlabel('데이터 포인트')
        axes[1, 1].set_ylabel('|Z-Score|')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = f"{Config.SAVE_DIR}/plots/{keyword}_outlier_analysis.png"
            plt.savefig(plot_path, dpi=Config.DPI, bbox_inches='tight')
            print(f"이상치 분석 플롯 저장: {plot_path}")
        
        plt.show()
    
    def get_outlier_report(self, outlier_results: Dict) -> str:
        """이상치 탐지 리포트를 생성합니다
        
        Args:
            outlier_results: detect_all_outliers의 결과
            
        Returns:
            리포트 문자열
        """
        report = "\n" + "="*50 + "\n"
        report += "🔍 이상치 탐지 리포트\n"
        report += "="*50 + "\n"
        
        summary = outlier_results['summary']
        report += f"분석된 키워드: {summary['analyzed_keywords']}/{summary['total_keywords']}\n"
        report += f"총 이상치: {summary['total_consensus_outliers']}개\n"
        report += f"이상치 비율: {summary['outlier_percentage']:.2f}%\n\n"
        
        # 키워드별 상세 결과
        for keyword, result in outlier_results['results'].items():
            if result.get('status') == 'insufficient_data':
                report += f"🔍 {keyword}: 데이터 부족 ({result['data_count']}개)\n"
                continue
            
            consensus_count = result['consensus_outliers']['count']
            report += f"🔍 {keyword}:\n"
            report += f"  데이터 개수: {result['data_count']}개\n"
            report += f"  합의 기반 이상치: {consensus_count}개\n"
            
            for method, method_result in result['methods'].items():
                method_count = len(method_result['outlier_indices'])
                percentage = method_result['stats']['outlier_percentage']
                report += f"  {method_result['stats']['method']}: {method_count}개 ({percentage:.1f}%)\n"
            
            if consensus_count > 0:
                outlier_values = result['consensus_outliers']['values']
                report += f"  이상치 값: {outlier_values}\n"
            
            report += "\n"
        
        return report 