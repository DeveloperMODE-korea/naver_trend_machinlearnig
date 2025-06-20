"""
품질 모니터링 모듈

실시간 데이터 품질 추적, 경고 시스템, 품질 메트릭을 담당합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import os
from ..config import Config
from ..utils.file_utils import FileManager

class QualityMonitor:
    """데이터 품질 모니터링 클래스"""
    
    def __init__(self, quality_thresholds: Dict[str, float] = None):
        """초기화
        
        Args:
            quality_thresholds: 품질 임계값 딕셔너리
        """
        self.quality_thresholds = quality_thresholds or {
            'completeness_min': 85.0,      # 완성도 최소 85%
            'validity_min': 90.0,          # 유효성 최소 90%
            'consistency_min': 95.0,       # 일관성 최소 95%
            'sufficiency_min': 70.0,       # 충분성 최소 70%
            'outlier_max': 5.0,            # 이상치 최대 5%
            'prediction_error_max': 30.0,  # 예측 오차 최대 30%
        }
        
        self.quality_history = []
        self.alerts = []
        self.monitor_log = []
    
    def assess_data_quality(self, data_df: pd.DataFrame, 
                          validator_results: Dict = None,
                          outlier_results: Dict = None) -> Dict[str, Any]:
        """종합적인 데이터 품질을 평가합니다
        
        Args:
            data_df: 평가할 데이터프레임
            validator_results: DataValidator의 결과
            outlier_results: OutlierDetector의 결과
            
        Returns:
            품질 평가 결과
        """
        assessment_time = datetime.now()
        
        # 1. 기본 품질 메트릭
        quality_metrics = self._calculate_basic_metrics(data_df)
        
        # 2. 이상치 관련 메트릭
        if outlier_results:
            quality_metrics.update(self._calculate_outlier_metrics(outlier_results))
        
        # 3. 예측 품질 메트릭 (있는 경우)
        if validator_results:
            quality_metrics.update(self._calculate_prediction_metrics(validator_results))
        
        # 4. 전체 품질 점수 계산
        overall_score = self._calculate_overall_score(quality_metrics)
        
        # 5. 품질 등급 결정
        quality_grade = self._determine_quality_grade(overall_score)
        
        # 6. 경고 생성
        alerts = self._generate_alerts(quality_metrics)
        
        # 7. 개선 권장사항
        recommendations = self._generate_recommendations(quality_metrics)
        
        assessment_result = {
            'timestamp': assessment_time,
            'data_count': len(data_df),
            'keyword_count': len(data_df['keyword'].unique()),
            'quality_metrics': quality_metrics,
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'alerts': alerts,
            'recommendations': recommendations,
            'passed_thresholds': self._check_thresholds(quality_metrics)
        }
        
        # 이력 저장
        self.quality_history.append(assessment_result)
        self.alerts.extend(alerts)
        
        self._log_assessment(assessment_result)
        
        return assessment_result
    
    def _calculate_basic_metrics(self, data_df: pd.DataFrame) -> Dict[str, float]:
        """기본 품질 메트릭을 계산합니다"""
        metrics = {}
        
        # 완성도 (결측값 비율)
        total_cells = len(data_df) * len(data_df.select_dtypes(include=[np.number]).columns)
        missing_cells = data_df.select_dtypes(include=[np.number]).isnull().sum().sum()
        metrics['completeness'] = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 100
        
        # 유효성 (범위 내 값 비율)
        valid_ratio_count = ((data_df['ratio'] >= 0) & (data_df['ratio'] <= 100)).sum()
        metrics['validity'] = (valid_ratio_count / len(data_df)) * 100
        
        # 일관성 (데이터 타입 일관성)
        consistency_issues = 0
        if not pd.api.types.is_datetime64_any_dtype(data_df['date']):
            consistency_issues += 1
        if not pd.api.types.is_numeric_dtype(data_df['ratio']):
            consistency_issues += 1
        
        metrics['consistency'] = (1 - consistency_issues / 2) * 100
        
        # 충분성 (키워드별 최소 데이터 개수)
        keyword_counts = data_df['keyword'].value_counts()
        min_required = 12
        sufficient_keywords = (keyword_counts >= min_required).sum()
        metrics['sufficiency'] = (sufficient_keywords / len(keyword_counts)) * 100
        
        # 신선도 (최신 데이터 비율)
        if 'date' in data_df.columns and pd.api.types.is_datetime64_any_dtype(data_df['date']):
            latest_date = data_df['date'].max()
            six_months_ago = latest_date - timedelta(days=180)
            recent_data_count = (data_df['date'] >= six_months_ago).sum()
            metrics['freshness'] = (recent_data_count / len(data_df)) * 100
        else:
            metrics['freshness'] = 0
        
        # 균형성 (키워드별 데이터 분포의 균등성)
        keyword_counts = data_df['keyword'].value_counts()
        if len(keyword_counts) > 1:
            cv = keyword_counts.std() / keyword_counts.mean()  # 변동계수
            metrics['balance'] = max(0, (1 - cv) * 100)
        else:
            metrics['balance'] = 100
        
        return metrics
    
    def _calculate_outlier_metrics(self, outlier_results: Dict) -> Dict[str, float]:
        """이상치 관련 메트릭을 계산합니다"""
        if not outlier_results or 'summary' not in outlier_results:
            return {'outlier_percentage': 0.0}
        
        return {
            'outlier_percentage': outlier_results['summary']['outlier_percentage'],
            'outlier_severity': min(100, outlier_results['summary']['outlier_percentage'] * 2)
        }
    
    def _calculate_prediction_metrics(self, validator_results: Dict) -> Dict[str, float]:
        """예측 관련 메트릭을 계산합니다"""
        if not hasattr(self, 'validation_results') or not validator_results:
            return {}
        
        # 예측값 보정 비율 계산
        total_predictions = len(validator_results)
        corrected_predictions = sum(1 for r in validator_results.values() if r.get('corrections_made', False))
        
        prediction_reliability = (1 - corrected_predictions / total_predictions) * 100 if total_predictions > 0 else 100
        
        return {
            'prediction_reliability': prediction_reliability,
            'prediction_corrections': corrected_predictions
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """전체 품질 점수를 계산합니다"""
        # 가중치 설정
        weights = {
            'completeness': 0.2,
            'validity': 0.25,
            'consistency': 0.15,
            'sufficiency': 0.15,
            'freshness': 0.1,
            'balance': 0.1,
            'outlier_percentage': -0.05  # 음의 가중치 (이상치가 많을수록 점수 감소)
        }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                if metric == 'outlier_percentage':
                    # 이상치는 낮을수록 좋음 (5% 이하를 100점으로 설정)
                    normalized_value = max(0, 100 - metrics[metric] * 20)
                else:
                    normalized_value = metrics[metric]
                
                score += normalized_value * abs(weight)
                total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0
    
    def _determine_quality_grade(self, score: float) -> str:
        """품질 점수를 기반으로 등급을 결정합니다"""
        if score >= 90:
            return "A (우수)"
        elif score >= 80:
            return "B (양호)"
        elif score >= 70:
            return "C (보통)"
        elif score >= 60:
            return "D (미흡)"
        else:
            return "F (불량)"
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """임계값 통과 여부를 확인합니다"""
        passed = {}
        
        threshold_mapping = {
            'completeness': 'completeness_min',
            'validity': 'validity_min',
            'consistency': 'consistency_min',
            'sufficiency': 'sufficiency_min'
        }
        
        for metric, threshold_key in threshold_mapping.items():
            if metric in metrics:
                if threshold_key in self.quality_thresholds:
                    passed[metric] = metrics[metric] >= self.quality_thresholds[threshold_key]
                else:
                    # 임계값이 설정되지 않은 경우 기본값 사용
                    default_thresholds = {
                        'completeness_min': 85.0,
                        'validity_min': 90.0,
                        'consistency_min': 95.0,
                        'sufficiency_min': 70.0
                    }
                    default_value = default_thresholds.get(threshold_key, 0)
                    passed[metric] = metrics[metric] >= default_value
        
        # 이상치는 반대 (낮을수록 좋음)
        if 'outlier_percentage' in metrics:
            outlier_threshold = self.quality_thresholds.get('outlier_max', 5.0)
            passed['outlier_percentage'] = metrics['outlier_percentage'] <= outlier_threshold
        
        return passed
    
    def _generate_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """품질 메트릭을 기반으로 경고를 생성합니다"""
        alerts = []
        current_time = datetime.now()
        
        # 임계값 기반 경고
        completeness_threshold = self.quality_thresholds.get('completeness_min', 85.0)
        if metrics.get('completeness', 100) < completeness_threshold:
            alerts.append({
                'type': 'warning',
                'category': 'completeness',
                'message': f"데이터 완성도가 낮습니다 ({metrics['completeness']:.1f}% < {completeness_threshold}%)",
                'severity': 'medium',
                'timestamp': current_time
            })
        
        validity_threshold = self.quality_thresholds.get('validity_min', 90.0)
        if metrics.get('validity', 100) < validity_threshold:
            alerts.append({
                'type': 'error',
                'category': 'validity',
                'message': f"데이터 유효성이 낮습니다 ({metrics['validity']:.1f}% < {validity_threshold}%)",
                'severity': 'high',
                'timestamp': current_time
            })
        
        outlier_threshold = self.quality_thresholds.get('outlier_max', 5.0)
        if metrics.get('outlier_percentage', 0) > outlier_threshold:
            alerts.append({
                'type': 'warning',
                'category': 'outliers',
                'message': f"이상치 비율이 높습니다 ({metrics['outlier_percentage']:.1f}% > {outlier_threshold}%)",
                'severity': 'medium',
                'timestamp': current_time
            })
        
        sufficiency_threshold = self.quality_thresholds.get('sufficiency_min', 70.0)
        if metrics.get('sufficiency', 100) < sufficiency_threshold:
            alerts.append({
                'type': 'info',
                'category': 'sufficiency',
                'message': f"데이터 충분성이 부족합니다 ({metrics['sufficiency']:.1f}% < {sufficiency_threshold}%)",
                'severity': 'low',
                'timestamp': current_time
            })
        
        return alerts
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """품질 개선을 위한 권장사항을 생성합니다"""
        recommendations = []
        
        if metrics.get('completeness', 100) < 90:
            recommendations.append("• 결측값 처리 로직을 개선하세요")
            recommendations.append("• 데이터 수집 과정에서 누락을 방지하세요")
        
        if metrics.get('validity', 100) < 95:
            recommendations.append("• 입력 데이터의 범위 검증을 강화하세요")
            recommendations.append("• 데이터 변환 과정에서의 오류를 점검하세요")
        
        if metrics.get('outlier_percentage', 0) > 3:
            recommendations.append("• 이상치 탐지 및 처리 규칙을 재검토하세요")
            recommendations.append("• 데이터 수집 소스의 신뢰성을 확인하세요")
        
        if metrics.get('sufficiency', 100) < 80:
            recommendations.append("• 각 키워드별로 더 많은 데이터를 수집하세요")
            recommendations.append("• 분석 기간을 늘려 충분한 샘플을 확보하세요")
        
        if metrics.get('balance', 100) < 70:
            recommendations.append("• 키워드별 데이터 수집량의 균형을 맞추세요")
            recommendations.append("• 편향된 키워드에 대한 추가 수집을 고려하세요")
        
        if metrics.get('freshness', 100) < 60:
            recommendations.append("• 더 최신의 데이터를 수집하세요")
            recommendations.append("• 정기적인 데이터 업데이트 스케줄을 설정하세요")
        
        return recommendations
    
    def _log_assessment(self, assessment_result: Dict[str, Any]):
        """평가 결과를 로그에 기록합니다"""
        log_entry = {
            'timestamp': assessment_result['timestamp'].isoformat(),
            'overall_score': assessment_result['overall_score'],
            'quality_grade': assessment_result['quality_grade'],
            'alert_count': len(assessment_result['alerts']),
            'data_count': assessment_result['data_count']
        }
        
        self.monitor_log.append(log_entry)
        
        # 로그 파일에도 저장
        log_file = os.path.join(Config.SAVE_DIR, 'quality_monitor.log')
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"로그 저장 실패: {e}")
    
    def get_quality_dashboard(self) -> Dict[str, Any]:
        """품질 대시보드 정보를 생성합니다"""
        if not self.quality_history:
            return {"message": "품질 평가 이력이 없습니다."}
        
        latest_assessment = self.quality_history[-1]
        
        # 최근 경고들 (24시간 이내)
        recent_alerts = [
            alert for alert in self.alerts 
            if alert['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        # 품질 트렌드 (최근 5개 평가)
        recent_assessments = self.quality_history[-5:]
        quality_trend = [a['overall_score'] for a in recent_assessments]
        
        # 경고 통계
        alert_stats = {
            'total': len(recent_alerts),
            'high': len([a for a in recent_alerts if a['severity'] == 'high']),
            'medium': len([a for a in recent_alerts if a['severity'] == 'medium']),
            'low': len([a for a in recent_alerts if a['severity'] == 'low'])
        }
        
        dashboard = {
            'last_updated': latest_assessment['timestamp'],
            'current_score': latest_assessment['overall_score'],
            'current_grade': latest_assessment['quality_grade'],
            'data_summary': {
                'total_records': latest_assessment['data_count'],
                'keywords': latest_assessment['keyword_count']
            },
            'quality_metrics': latest_assessment['quality_metrics'],
            'quality_trend': quality_trend,
            'recent_alerts': recent_alerts,
            'alert_statistics': alert_stats,
            'threshold_status': latest_assessment['passed_thresholds'],
            'recommendations': latest_assessment['recommendations']
        }
        
        return dashboard
    
    def export_quality_report(self, filename: str = None) -> str:
        """품질 리포트를 파일로 내보냅니다"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'quality_thresholds': self.quality_thresholds,
            'assessment_history': [
                {
                    **assessment,
                    'timestamp': assessment['timestamp'].isoformat()
                }
                for assessment in self.quality_history
            ],
            'dashboard': self.get_quality_dashboard()
        }
        
        # timestamp를 다시 isoformat으로 변환
        if 'dashboard' in report_data and 'last_updated' in report_data['dashboard']:
            report_data['dashboard']['last_updated'] = report_data['dashboard']['last_updated'].isoformat()
        
        for alert in report_data['dashboard'].get('recent_alerts', []):
            alert['timestamp'] = alert['timestamp'].isoformat()
        
        filepath = os.path.join(Config.SAVE_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            print(f"품질 리포트 저장: {filepath}")
            return filepath
        except Exception as e:
            print(f"리포트 저장 실패: {e}")
            return ""
    
    def print_quality_summary(self, assessment_result: Dict[str, Any] = None):
        """품질 평가 요약을 출력합니다"""
        if not assessment_result and self.quality_history:
            assessment_result = self.quality_history[-1]
        
        if not assessment_result:
            print("품질 평가 결과가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📊 데이터 품질 평가 요약")
        print("="*60)
        
        print(f"평가 시간: {assessment_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"데이터 규모: {assessment_result['data_count']:,}개 레코드, {assessment_result['keyword_count']}개 키워드")
        print(f"전체 품질 점수: {assessment_result['overall_score']:.1f}점")
        print(f"품질 등급: {assessment_result['quality_grade']}")
        
        print(f"\n📈 품질 메트릭:")
        metrics = assessment_result['quality_metrics']
        for metric, value in metrics.items():
            status = "✅" if assessment_result['passed_thresholds'].get(metric, True) else "❌"
            print(f"  {status} {metric}: {value:.1f}%")
        
        if assessment_result['alerts']:
            print(f"\n⚠️  경고 사항 ({len(assessment_result['alerts'])}개):")
            for alert in assessment_result['alerts']:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(alert['severity'], "ℹ️")
                print(f"  {severity_icon} {alert['message']}")
        
        if assessment_result['recommendations']:
            print(f"\n💡 개선 권장사항:")
            for rec in assessment_result['recommendations']:
                print(f"  {rec}")
        
        print("="*60)
    
    def clear_history(self, days_to_keep: int = 30):
        """오래된 이력을 정리합니다"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 품질 이력 정리
        self.quality_history = [
            assessment for assessment in self.quality_history
            if assessment['timestamp'] > cutoff_date
        ]
        
        # 경고 이력 정리
        self.alerts = [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff_date
        ]
        
        print(f"✅ {days_to_keep}일 이전의 품질 이력이 정리되었습니다.") 