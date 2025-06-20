import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from config import Config
from utils import save_model, load_model, create_sequences, ensemble_weighted, get_user_input
import os
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """머신러닝 모델 훈련 클래스"""
    
    def __init__(self, data_df):
        """초기화"""
        self.data_df = data_df.copy()
        self.models = {}
        
    def prepare_keyword_data(self, keyword):
        """특정 키워드의 데이터를 준비합니다"""
        keyword_data = self.data_df[self.data_df['keyword'] == keyword].sort_values('date')
        
        if len(keyword_data) < 12:
            print(f"{keyword}는 예측을 위한 충분한 데이터가 없습니다.")
            return None
        
        return keyword_data
    
    def train_prophet_model(self, keyword, use_saved=True):
        """Prophet 모델 훈련"""
        print(f"\n🔮 {keyword}에 대한 Prophet 모델 훈련 중...")
        
        if use_saved:
            saved_model = load_model('prophet', keyword)
            if saved_model is not None:
                return saved_model
        
        keyword_data = self.prepare_keyword_data(keyword)
        if keyword_data is None:
            return None
        
        # Prophet 데이터 형식으로 변환
        prophet_data = pd.DataFrame({
            'ds': keyword_data['date'],
            'y': keyword_data['ratio']
        }).dropna()
        
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(prophet_data)
            
            save_model(model, 'prophet', keyword)
            return model
            
        except Exception as e:
            print(f"{keyword} Prophet 모델 훈련 중 오류: {e}")
            return None
    
    def train_lstm_model(self, keyword, use_saved=True):
        """LSTM 모델 훈련"""
        print(f"\n🧠 {keyword}에 대한 LSTM 모델 훈련 중...")
        
        if use_saved:
            saved_model = load_model('lstm', keyword)
            if saved_model is not None:
                return saved_model, None
        
        keyword_data = self.prepare_keyword_data(keyword)
        if keyword_data is None:
            return None, None
        
        # 데이터 준비
        values = keyword_data['ratio'].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
        
        seq_length = min(12, len(keyword_data) // 2)
        X, y = create_sequences(scaled_values, seq_length)
        
        if len(X) < 10:
            return None, None
        
        try:
            # 모델 구축
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # 훈련
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=8,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(patience=10)],
                verbose=0
            )
            
            save_model(model, 'lstm', keyword)
            return model, scaler
            
        except Exception as e:
            print(f"{keyword} LSTM 모델 훈련 중 오류: {e}")
            return None, None
    
    def ensemble_prediction(self, keyword, periods=6):
        """앙상블 예측"""
        print(f"\n🎯 {keyword}에 대한 앙상블 예측...")
        
        keyword_data = self.prepare_keyword_data(keyword)
        if keyword_data is None:
            return None
        
        predictions = []
        model_names = []
        
        # Prophet 예측
        prophet_model = self.train_prophet_model(keyword)
        if prophet_model:
            try:
                future = prophet_model.make_future_dataframe(periods=periods, freq='M')
                forecast = prophet_model.predict(future)
                prophet_future = forecast['yhat'].values[-periods:]
                predictions.append(prophet_future)
                model_names.append('Prophet')
            except:
                pass
        
        # LSTM 예측
        lstm_model, scaler = self.train_lstm_model(keyword)
        if lstm_model and scaler:
            try:
                values = keyword_data['ratio'].values.reshape(-1, 1)
                scaled_values = scaler.transform(values)
                seq_length = min(12, len(keyword_data) // 2)
                
                last_sequence = scaled_values[-seq_length:].reshape(1, seq_length, 1)
                lstm_future = []
                
                current_sequence = last_sequence.copy()
                for _ in range(periods):
                    next_pred = lstm_model.predict(current_sequence)[0][0]
                    lstm_future.append(next_pred)
                    current_sequence = np.append(
                        current_sequence[:, 1:, :], 
                        np.array([[[next_pred]]]), 
                        axis=1
                    )
                
                lstm_future = scaler.inverse_transform(
                    np.array(lstm_future).reshape(-1, 1)
                ).flatten()
                predictions.append(lstm_future)
                model_names.append('LSTM')
            except:
                pass
        
        if predictions:
            # 단순 평균 앙상블
            ensemble_pred = np.mean(predictions, axis=0)
            ensemble_std = np.std(predictions, axis=0)
            
            # 미래 날짜 생성
            last_date = keyword_data['date'].iloc[-1]
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
            
            # 시각화
            plt.figure(figsize=(14, 7))
            plt.plot(keyword_data['date'], keyword_data['ratio'], 'b-', 
                    marker='o', label='실제 데이터')
            plt.plot(future_dates, ensemble_pred, 'r-', linewidth=2, 
                    marker='s', label='앙상블 예측')
            plt.fill_between(future_dates, 
                           ensemble_pred - ensemble_std,
                           ensemble_pred + ensemble_std,
                           color='red', alpha=0.2, label='예측 불확실성')
            
            plt.title(f'{keyword} 앙상블 예측 결과', fontsize=16)
            plt.xlabel('날짜', fontsize=12)
            plt.ylabel('검색 비율', fontsize=12)
            plt.grid(True, alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(Config.SAVE_DIR, 'plots', f'{keyword}_prediction.png')
            plt.savefig(plot_path, dpi=Config.DPI)
            plt.show()
            
            # 결과 반환
            result_df = pd.DataFrame({
                '날짜': [d.strftime('%Y-%m') for d in future_dates],
                '앙상블 예측': ensemble_pred,
                '하한': ensemble_pred - ensemble_std,
                '상한': ensemble_pred + ensemble_std
            })
            
            print(f"\n{keyword} 미래 6개월 예측:")
            print(result_df)
            
            return result_df
        else:
            print(f"{keyword}: 예측 실패")
            return None
    
    def run_predictions(self, keywords=None, max_keywords=3):
        """예측 실행"""
        if keywords is None:
            keywords = self.data_df['keyword'].unique()
        
        if len(keywords) > max_keywords:
            keywords = keywords[:max_keywords]
        
        print(f"\n🚀 {len(keywords)}개 키워드 예측 시작...")
        
        results = {}
        for keyword in keywords:
            result = self.ensemble_prediction(keyword)
            if result is not None:
                results[keyword] = result
        
        return results 