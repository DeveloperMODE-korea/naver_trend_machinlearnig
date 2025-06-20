import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import Config
import joblib
import prophet
from prophet.serialize import model_to_json, model_from_json
from packaging import version
import tensorflow as tf

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

def check_prophet_version():
    """Prophet 버전이 model_to_json을 지원하는지 확인합니다"""
    required_version = "1.1"
    current_version = prophet.__version__
    return version.parse(current_version) >= version.parse(required_version)

def save_model(model, model_type: str, keyword: str, save_dir: str = None) -> bool:
    """모델을 저장합니다"""
    if save_dir is None:
        save_dir = os.path.join(Config.SAVE_DIR, 'models')
    
    keyword_safe = keyword.replace(" ", "_").lower()
    
    try:
        if model_type == "prophet":
            if check_prophet_version():
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.json")
                with open(model_path, "w") as fout:
                    fout.write(model_to_json(model))
            else:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                joblib.dump(model, model_path)
            print(f"{keyword}의 Prophet 모델이 저장되었습니다: {model_path}")
            
        elif model_type == "lstm":
            model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.keras")
            model.save(model_path, save_format='keras')
            print(f"{keyword}의 LSTM 모델이 저장되었습니다: {model_path}")
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        return True
    except Exception as e:
        print(f"{keyword}의 {model_type} 모델 저장 중 오류: {e}")
        return False

def load_model(model_type: str, keyword: str, save_dir: str = None):
    """저장된 모델을 로드합니다"""
    if save_dir is None:
        save_dir = os.path.join(Config.SAVE_DIR, 'models')
    
    keyword_safe = keyword.replace(" ", "_").lower()
    
    try:
        if model_type == "prophet":
            if check_prophet_version():
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.json")
                if not os.path.exists(model_path):
                    return None
                with open(model_path, "r") as fin:
                    model = model_from_json(fin.read())
            else:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                if not os.path.exists(model_path):
                    return None
                model = joblib.load(model_path)
            print(f"{keyword}의 Prophet 모델을 로드했습니다")
            
        elif model_type == "lstm":
            keras_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.keras")
            h5_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.h5")
            
            if os.path.exists(keras_path):
                model = tf.keras.models.load_model(keras_path)
            elif os.path.exists(h5_path):
                model = tf.keras.models.load_model(h5_path)
            else:
                return None
            print(f"{keyword}의 LSTM 모델을 로드했습니다")
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        return model
    except Exception as e:
        print(f"{keyword}의 {model_type} 모델 로드 중 오류: {e}")
        return None

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

def clean_for_export(df):
    """NaN, inf 값 처리하여 내보내기 가능하게 변환합니다"""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna('')
    return df

def create_sequences(data, seq_length):
    """LSTM을 위한 시퀀스 데이터를 생성합니다"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_user_input(prompt, default=None, input_type=str):
    """사용자 입력을 받는 헬퍼 함수"""
    try:
        if default is not None:
            user_input = input(f"{prompt} (기본값: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if input_type == int:
            return int(user_input)
        elif input_type == float:
            return float(user_input)
        elif input_type == bool:
            return user_input.lower() in ['y', 'yes', 'true', '1']
        else:
            return user_input
    except ValueError:
        if default is not None:
            print(f"유효하지 않은 입력입니다. 기본값 {default}를 사용합니다.")
            return default
        else:
            raise ValueError("유효하지 않은 입력입니다.")

def ensemble_weighted(pred_list, weights):
    """가중평균 앙상블을 수행합니다"""
    arrs = [np.array(a) for a in pred_list if a is not None]
    min_len = min([len(a) for a in arrs])
    arrs = [a[:min_len] for a in arrs]
    weights = np.array(weights)[:len(arrs)]
    weights = weights / weights.sum()
    arrs = np.stack(arrs, axis=0)
    return np.average(arrs, axis=0, weights=weights) 