import os
import joblib
import prophet
from prophet.serialize import model_to_json, model_from_json
from packaging import version
import tensorflow as tf
from ..config import Config

class ModelManager:
    """모델 관리 클래스"""
    
    @staticmethod
    def check_prophet_version():
        """Prophet 버전이 model_to_json을 지원하는지 확인합니다"""
        required_version = "1.1"
        current_version = prophet.__version__
        return version.parse(current_version) >= version.parse(required_version)

    @staticmethod
    def save_model(model, model_type: str, keyword: str, save_dir: str = None) -> bool:
        """모델을 저장합니다"""
        if save_dir is None:
            save_dir = os.path.join(Config.SAVE_DIR, 'models')
        
        keyword_safe = keyword.replace(" ", "_").lower()
        
        try:
            if model_type == "prophet":
                if ModelManager.check_prophet_version():
                    model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.json")
                    with open(model_path, "w") as fout:
                        fout.write(model_to_json(model))
                else:
                    model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                    joblib.dump(model, model_path)
                print(f"{keyword}의 Prophet 모델이 저장되었습니다: {model_path}")
                
            elif model_type in ["lstm", "lstm_constrained"]:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.keras")
                model.save(model_path, save_format='keras')
                print(f"{keyword}의 {model_type.upper()} 모델이 저장되었습니다: {model_path}")
                
            elif model_type == "transformer":
                import torch
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pth")
                torch.save(model.state_dict(), model_path)
                print(f"{keyword}의 Transformer 모델이 저장되었습니다: {model_path}")
                
            elif model_type in ["xgboost", "catboost"]:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                joblib.dump(model, model_path)
                print(f"{keyword}의 {model_type.upper()} 모델이 저장되었습니다: {model_path}")
                
            else:
                raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
            return True
        except Exception as e:
            print(f"{keyword}의 {model_type} 모델 저장 중 오류: {e}")
            return False

    @staticmethod
    def load_model(model_type: str, keyword: str, save_dir: str = None):
        """저장된 모델을 로드합니다"""
        if save_dir is None:
            save_dir = os.path.join(Config.SAVE_DIR, 'models')
        
        keyword_safe = keyword.replace(" ", "_").lower()
        
        try:
            if model_type == "prophet":
                if ModelManager.check_prophet_version():
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
                
            elif model_type in ["lstm", "lstm_constrained"]:
                keras_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.keras")
                h5_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.h5")
                
                if os.path.exists(keras_path):
                    model = tf.keras.models.load_model(keras_path)
                elif os.path.exists(h5_path):
                    model = tf.keras.models.load_model(h5_path)
                else:
                    return None
                print(f"{keyword}의 {model_type.upper()} 모델을 로드했습니다")
                
            elif model_type == "transformer":
                import torch
                from ..models.transformer_model import TimeSeriesTransformer
                
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pth")
                if not os.path.exists(model_path):
                    return None
                
                # 기본 파라미터로 모델 생성 후 가중치 로드
                d_model = 64
                nhead = 8
                if d_model % nhead != 0:
                    d_model = ((d_model // nhead) + 1) * nhead
                    
                model = TimeSeriesTransformer(
                    input_dim=1,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=4,
                    seq_len=12,
                    pred_len=6
                )
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"{keyword}의 Transformer 모델을 로드했습니다")
                
            elif model_type in ["xgboost", "catboost"]:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                if not os.path.exists(model_path):
                    return None
                model = joblib.load(model_path)
                print(f"{keyword}의 {model_type.upper()} 모델을 로드했습니다")
                
            else:
                # 지원되지 않는 모델 타입은 None 반환 (오류 발생하지 않도록)
                return None
                
            return model
        except Exception as e:
            print(f"{keyword}의 {model_type} 모델 로드 중 오류: {e}")
            return None 