"""
데이터 검증 모듈

예측값 검증, 범위 체크, 데이터 무결성을 담당합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

class DataValidator:
    """데이터 검증 클래스"""
    
    def __init__(self, valid_range: Tuple[float, float] = (0, 100)):
        """초기화
        
        Args:
            valid_range: 유효한 값 범위 (기본: 0-100)
        """
        self.valid_range = valid_range
        self.validation_results = {}
    
    def validate_predictions(self, predictions: np.ndarray, keyword: str = "unknown") -> np.ndarray:
        """예측값을 검증하고 보정합니다
        
        Args:
            predictions: 예측값 배열
            keyword: 키워드명 (로깅용)
            
        Returns:
            보정된 예측값
        """
        original_predictions = predictions.copy()
        validated_predictions = predictions.copy()
        issues = []
        
        # 1. NaN/Inf 값 검사
        nan_mask = np.isnan(predictions) | np.isinf(predictions)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            issues.append(f"NaN/Inf 값 {nan_count}개 발견")
            
            # 이전/이후 값의 평균으로 대체
            for i in range(len(validated_predictions)):
                if nan_mask[i]:
                    # 이전 유효값 찾기
                    prev_val = None
                    for j in range(i-1, -1, -1):
                        if not nan_mask[j]:
                            prev_val = validated_predictions[j]
                            break
                    
                    # 이후 유효값 찾기
                    next_val = None
                    for j in range(i+1, len(validated_predictions)):
                        if not nan_mask[j]:
                            next_val = validated_predictions[j]
                            break
                    
                    # 대체값 계산
                    if prev_val is not None and next_val is not None:
                        validated_predictions[i] = (prev_val + next_val) / 2
                    elif prev_val is not None:
                        validated_predictions[i] = prev_val
                    elif next_val is not None:
                        validated_predictions[i] = next_val
                    else:
                        validated_predictions[i] = np.mean(self.valid_range)
        
        # 2. 범위 벗어난 값 검사
        min_val, max_val = self.valid_range
        out_of_range_mask = (validated_predictions < min_val) | (validated_predictions > max_val)
        if np.any(out_of_range_mask):
            out_count = np.sum(out_of_range_mask)
            issues.append(f"범위 벗어난 값 {out_count}개 발견 (유효범위: {min_val}-{max_val})")
            
            # 범위 내로 클리핑
            validated_predictions = np.clip(validated_predictions, min_val, max_val)
        
        # 3. 급격한 변화 검사 (이상 변동률 > 50%)
        if len(validated_predictions) > 1:
            changes = np.abs(np.diff(validated_predictions))
            mean_change = np.mean(changes)
            threshold = mean_change + 2 * np.std(changes)
            
            extreme_changes = changes > threshold
            if np.any(extreme_changes):
                extreme_count = np.sum(extreme_changes)
                issues.append(f"급격한 변화 {extreme_count}개 발견")
                
                # 급격한 변화 완화
                for i in range(len(extreme_changes)):
                    if extreme_changes[i]:
                        # 이전값과 다음값의 선형 보간
                        if i == 0:
                            validated_predictions[i+1] = (validated_predictions[i] + validated_predictions[i+2]) / 2
                        else:
                            validated_predictions[i+1] = (validated_predictions[i] + validated_predictions[i+1]) / 2
        
        # 4. 검증 결과 저장
        self.validation_results[keyword] = {
            'original_min': np.min(original_predictions),
            'original_max': np.max(original_predictions),
            'validated_min': np.min(validated_predictions),
            'validated_max': np.max(validated_predictions),
            'corrections_made': len(issues) > 0,
            'issues': issues,
            'correction_count': np.sum(original_predictions != validated_predictions)
        }
        
        if issues:
            print(f"⚠️  [{keyword}] 예측값 검증 결과:")
            for issue in issues:
                print(f"  - {issue}")
            print(f"  ✅ 총 {self.validation_results[keyword]['correction_count']}개 값 보정됨")
        
        return validated_predictions
    
    def validate_raw_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """원시 데이터를 검증합니다
        
        Args:
            data_df: 검증할 데이터프레임
            
        Returns:
            검증된 데이터프레임
        """
        validated_df = data_df.copy()
        total_issues = []
        
        # 1. 필수 컬럼 검사
        required_columns = ['date', 'keyword', 'ratio']
        missing_columns = [col for col in required_columns if col not in validated_df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        # 2. 날짜 형식 검사
        if not pd.api.types.is_datetime64_any_dtype(validated_df['date']):
            try:
                validated_df['date'] = pd.to_datetime(validated_df['date'])
                total_issues.append("날짜 형식 자동 변환됨")
            except:
                total_issues.append("날짜 형식 변환 실패")
        
        # 3. ratio 값 검증
        ratio_issues = []
        original_count = len(validated_df)
        
        # NaN 제거
        nan_count = validated_df['ratio'].isna().sum()
        if nan_count > 0:
            validated_df = validated_df.dropna(subset=['ratio'])
            ratio_issues.append(f"NaN 값 {nan_count}개 제거")
        
        # 음수 값 처리
        negative_count = (validated_df['ratio'] < 0).sum()
        if negative_count > 0:
            validated_df.loc[validated_df['ratio'] < 0, 'ratio'] = 0
            ratio_issues.append(f"음수 값 {negative_count}개를 0으로 변환")
        
        # 과도하게 큰 값 처리 (평균의 10배 이상)
        mean_ratio = validated_df['ratio'].mean()
        threshold = mean_ratio * 10
        extreme_count = (validated_df['ratio'] > threshold).sum()
        if extreme_count > 0:
            validated_df.loc[validated_df['ratio'] > threshold, 'ratio'] = threshold
            ratio_issues.append(f"극값 {extreme_count}개를 {threshold:.1f}로 제한")
        
        # ratio 컬럼을 float 타입으로 확실히 변환
        validated_df['ratio'] = pd.to_numeric(validated_df['ratio'], errors='coerce')
        
        # 4. 중복 데이터 검사
        duplicates = validated_df.duplicated(subset=['date', 'keyword']).sum()
        if duplicates > 0:
            validated_df = validated_df.drop_duplicates(subset=['date', 'keyword'])
            total_issues.append(f"중복 데이터 {duplicates}개 제거")
        
        # 5. 검증 결과 요약
        final_count = len(validated_df)
        removed_count = original_count - final_count
        
        print(f"\n📋 데이터 검증 완료:")
        print(f"  원본 데이터: {original_count}개")
        print(f"  최종 데이터: {final_count}개")
        if removed_count > 0:
            print(f"  제거된 데이터: {removed_count}개")
        
        if ratio_issues:
            print(f"  비율 값 보정:")
            for issue in ratio_issues:
                print(f"    - {issue}")
        
        if total_issues:
            print(f"  기타 이슈:")
            for issue in total_issues:
                print(f"    - {issue}")
        
        return validated_df
    
    def get_data_quality_score(self, data_df: pd.DataFrame) -> Dict[str, float]:
        """데이터 품질 점수를 계산합니다
        
        Args:
            data_df: 평가할 데이터프레임
            
        Returns:
            품질 점수 딕셔너리
        """
        scores = {}
        
        # 1. 완성도 점수 (결측값 비율)
        total_cells = len(data_df) * len(data_df.columns)
        missing_cells = data_df.isnull().sum().sum()
        completeness_score = (1 - missing_cells / total_cells) * 100
        scores['완성도'] = completeness_score
        
        # 2. 일관성 점수 (데이터 타입 및 형식)
        consistency_issues = 0
        # 날짜 형식 확인
        if not pd.api.types.is_datetime64_any_dtype(data_df['date']):
            consistency_issues += 1
        # 숫자 형식 확인
        if not pd.api.types.is_numeric_dtype(data_df['ratio']):
            consistency_issues += 1
        
        consistency_score = max(0, (1 - consistency_issues / 2) * 100)
        scores['일관성'] = consistency_score
        
        # 3. 유효성 점수 (범위 내 값 비율)
        valid_ratio_count = ((data_df['ratio'] >= 0) & (data_df['ratio'] <= 100)).sum()
        validity_score = (valid_ratio_count / len(data_df)) * 100
        scores['유효성'] = validity_score
        
        # 4. 충분성 점수 (키워드별 데이터 개수)
        keyword_counts = data_df['keyword'].value_counts()
        min_required = 12  # 최소 12개월 데이터
        sufficient_keywords = (keyword_counts >= min_required).sum()
        sufficiency_score = (sufficient_keywords / len(keyword_counts)) * 100
        scores['충분성'] = sufficiency_score
        
        # 5. 전체 품질 점수
        overall_score = np.mean(list(scores.values()))
        scores['전체점수'] = overall_score
        
        return scores
    
    def get_validation_report(self) -> str:
        """검증 결과 리포트를 생성합니다"""
        if not self.validation_results:
            return "검증 결과가 없습니다."
        
        report = "\n" + "="*50 + "\n"
        report += "📊 데이터 검증 리포트\n"
        report += "="*50 + "\n"
        
        total_keywords = len(self.validation_results)
        corrected_keywords = sum(1 for r in self.validation_results.values() if r['corrections_made'])
        
        report += f"검증된 키워드: {total_keywords}개\n"
        report += f"보정이 필요한 키워드: {corrected_keywords}개\n\n"
        
        for keyword, result in self.validation_results.items():
            report += f"🔍 {keyword}:\n"
            report += f"  원본 범위: {result['original_min']:.2f} ~ {result['original_max']:.2f}\n"
            report += f"  보정 범위: {result['validated_min']:.2f} ~ {result['validated_max']:.2f}\n"
            
            if result['corrections_made']:
                report += f"  보정된 값: {result['correction_count']}개\n"
                for issue in result['issues']:
                    report += f"    - {issue}\n"
            else:
                report += f"  ✅ 검증 통과\n"
            report += "\n"
        
        return report 