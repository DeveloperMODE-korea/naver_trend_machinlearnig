import os
import pandas as pd
import numpy as np
from ..config import Config

class FileManager:
    """파일 관리 클래스"""
    
    @staticmethod
    def save_file(df, filename, file_format='csv'):
        """데이터프레임을 지정된 형식으로 저장합니다"""
        filepath = os.path.join(Config.SAVE_DIR, filename)
        try:
            if file_format == 'csv':
                df.to_csv(filepath, index=False, encoding='utf-8')
            elif file_format == 'excel':
                df.to_excel(filepath, index=False)
            elif file_format == 'pickle':
                df.to_pickle(filepath)
            print(f"파일 저장 완료: {filepath}")
            return True
        except Exception as e:
            print(f"파일 저장 실패: {e}")
            return False

    @staticmethod
    def load_file(filename, file_format='csv'):
        """저장된 파일을 불러옵니다"""
        filepath = os.path.join(Config.SAVE_DIR, filename)
        try:
            if file_format == 'csv':
                df = pd.read_csv(filepath)
            elif file_format == 'excel':
                df = pd.read_excel(filepath)
            elif file_format == 'pickle':
                df = pd.read_pickle(filepath)
            
            # 날짜 컬럼이 있다면 datetime으로 변환
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            print(f"파일 로드 완료: {filepath}")
            return df
        except Exception as e:
            print(f"파일 로드 실패: {e}")
            return None

    @staticmethod
    def clean_for_export(df):
        """NaN, inf 값 처리하여 내보내기 가능하게 변환합니다"""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna('')
        return df
    
    @staticmethod
    def safe_parse_date_column(df, date_col='date'):
        """날짜 컬럼을 안전하게 파싱합니다"""
        df_copy = df.copy()
        df_copy[date_col] = df_copy[date_col].astype(str)
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce', infer_datetime_format=True)
        
        nat_count = df_copy[date_col].isna().sum()
        if nat_count > 0:
            print(f"날짜 파싱 실패값: {nat_count}개 → 제거합니다.")
            df_copy = df_copy.dropna(subset=[date_col])
        
        return df_copy 