# 필요한 라이브러리 설치
!pip install pandas matplotlib seaborn requests scikit-learn statsmodels prophet tensorflow

# Google 드라이브와 스프레드시트 연동을 위한 라이브러리 설치
!pip install gspread oauth2client google-api-python-client

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
from matplotlib import rc
import time
import os
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Google API 관련 라이브러리
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# KerasTuner 설치 (Colab 환경에서만 필요)
try:
    import kerastuner as kt
except ImportError:
    import os
    os.system('pip install -q -U keras-tuner')
    import kerastuner as kt

# Google Colab에서 실행 중인지 확인
try:
    from google.colab import drive
    IN_COLAB = True
    print("Google Colab에서 실행 중입니다. Google Drive 설정 중...")
    # Google Drive 연결
    drive.mount('/content/drive')
    # 저장할 디렉토리 경로 설정
    SAVE_DIR = '/content/drive/MyDrive/naver_shopping_analysis'
    # 디렉토리가 없으면 생성
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"파일이 저장될 경로: {SAVE_DIR}")

    # Google API 인증 (Google Colab에서)
    auth.authenticate_user()

    # 구글 스프레드시트 연동 설정
    def setup_google_sheets():
        try:
            from google.auth import default
            creds, _ = default()
            gc = gspread.authorize(creds)

            # 스프레드시트 생성 또는 열기
            try:
                sheet = gc.open("Naver Shopping Keyword Analysis")
                print("기존 스프레드시트를 사용합니다: Naver Shopping Keyword Analysis")
            except:
                sheet = gc.create("Naver Shopping Keyword Analysis")
                print("새 스프레드시트를 생성했습니다: Naver Shopping Keyword Analysis")

                # 스프레드시트 공유 설정 (선택적)
                # sheet.share('your-email@example.com', perm_type='user', role='writer')

            return sheet
        except Exception as e:
            print(f"구글 스프레드시트 설정 오류: {e}")
            return None

    # 구글 스프레드시트 객체
    google_sheet = setup_google_sheets()

except ImportError:
    IN_COLAB = False
    SAVE_DIR = '.'  # 로컬 경로
    print("로컬 환경에서 실행 중입니다.")
    google_sheet = None

# 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] =False

# 네이버 API 인증 정보 입력 받기
client_id = input("네이버 API client_id를 입력하세요: ").strip()
client_secret = input("네이버 API client_secret을 입력하세요: ").strip()

# API URL
url = "https://openapi.naver.com/v1/datalab/search"

# API 헤더 정의 추가
headers = {
    "X-Naver-Client-Id": client_id,
    "X-Naver-Client-Secret": client_secret,
    "Content-Type": "application/json"
}

# 사용자로부터 키워드 입력 받기
print("분석할 키워드를 입력해주세요. 쉼표(,)로 구분하여 입력하세요.")
print("예시: 물티슈, 기저귀, 커피, 마스크, 생수")

# 콘솔 입력 또는 Colab input() 사용
user_keywords_input = input("키워드 입력: ")
keywords_ko = [keyword.strip() for keyword in user_keywords_input.split(',')]

# 최소 1개, 최대 10개의 키워드로 제한
if len(keywords_ko) == 0:
    print("키워드가 입력되지 않았습니다. 기본 키워드를 사용합니다.")
    keywords_ko = ['물티슈', '기저귀', '커피', '마스크', '생수']
elif len(keywords_ko) > 10:
    print("입력된 키워드가 너무 많습니다. 처음 10개의 키워드만 사용합니다.")
    keywords_ko = keywords_ko[:10]

print(f"분석할 키워드 ({len(keywords_ko)}개): {', '.join(keywords_ko)}")

# 설정
start_year = int(input("분석 시작 연도를 입력하세요 (기본값: 2022): ") or "2022")
end_year = int(input("분석 종료 연도를 입력하세요 (기본값: 2024): ") or "2024")

# 이전 데이터 불러오기 여부 확인
use_saved_data = False
if IN_COLAB:
    saved_data_path = os.path.join(SAVE_DIR, 'naver_shopping_data_extended.csv')
    if os.path.exists(saved_data_path):
        load_saved = input(f"\n이전에 저장된 데이터가 발견되었습니다 ({saved_data_path}).\n이 데이터를 사용하시겠습니까? (y/n, 기본값: y): ").lower().strip()
        use_saved_data = load_saved != 'n'  # 기본값은 'y'

        if use_saved_data:
            try:
                # 저장된 데이터 로드
                result_df = pd.read_csv(saved_data_path)

                # 날짜 열을 datetime 형식으로 변환
                result_df['date'] = pd.to_datetime(result_df['date'])

                print(f"저장된 데이터를 성공적으로 불러왔습니다. 총 {len(result_df)}개의 데이터 포인트.")

                # 불러온 데이터의 키워드 목록 확인
                saved_keywords = result_df['keyword'].unique()
                print(f"불러온 데이터의 키워드: {', '.join(saved_keywords)}")

                # 현재 입력한 키워드와 비교
                new_keywords = [k for k in keywords_ko if k not in saved_keywords]
                if new_keywords:
                    print(f"새로운 키워드 발견: {', '.join(new_keywords)}")
                    add_new_keywords = input("새 키워드에 대한 데이터를 수집하시겠습니까? (y/n, 기본값: y): ").lower().strip() != 'n'

                    if add_new_keywords:
                        # 새 키워드만 수집
                        keywords_to_collect = new_keywords
                    else:
                        # 새 키워드 무시
                        keywords_ko = [k for k in keywords_ko if k in saved_keywords]
                        keywords_to_collect = []
                else:
                    keywords_to_collect = []

                # 날짜 범위 확인
                saved_min_date = result_df['date'].min()
                saved_max_date = result_df['date'].max()
                current_date = datetime.now()

                print(f"저장된 데이터의 날짜 범위: {saved_min_date.strftime('%Y-%m-%d')} ~ {saved_max_date.strftime('%Y-%m-%d')}")

                # 추가 데이터 수집 여부 확인
                collect_additional = False

                # 지정한 시작 연도가 저장된 데이터의 최소 연도보다 이전인 경우
                if start_year < saved_min_date.year:
                    collect_additional = True
                    print(f"시작 연도({start_year})가 저장된 데이터 시작 연도({saved_min_date.year})보다 이전입니다.")

                # 지정한 종료 연도가 저장된 데이터의 최대 연도보다 이후인 경우 (현재 연도 이하일 때만)
                if end_year > saved_max_date.year and end_year <= current_date.year:
                    collect_additional = True
                    print(f"종료 연도({end_year})가 저장된 데이터 종료 연도({saved_max_date.year})보다 이후입니다.")

                if collect_additional:
                    collect_decision = input("추가 데이터를 수집하시겠습니까? (y/n, 기본값: y): ").lower().strip() != 'n'
                    if not collect_decision:
                        collect_additional = False
                        # 시작 연도와 종료 연도를 저장된 데이터 범위로 제한
                        start_year = saved_min_date.year
                        end_year = saved_max_date.year

                # 저장된 모델 체크
                model_paths = {}
                model_types = ['prophet', 'lstm']

                for keyword in saved_keywords:
                    model_paths[keyword] = {}
                    keyword_safe = keyword.replace(' ', '_').lower()

                    for model_type in model_types:
                        if model_type == 'linear':
                            # 선형 모델은 파일로 저장되지 않으므로 패스
                            continue

                        model_path = os.path.join(SAVE_DIR, f"{keyword_safe}_{model_type}_model")
                        if os.path.exists(model_path) or (model_type == 'prophet' and os.path.exists(f"{model_path}.json")):
                            model_paths[keyword][model_type] = model_path

                if any(model_paths[keyword] for keyword in model_paths):
                    print("\n저장된 모델 파일이 발견되었습니다:")
                    for keyword in model_paths:
                        if model_paths[keyword]:
                            print(f"  {keyword}: {', '.join(model_paths[keyword].keys())}")

                # 전체 데이터를 수집할 필요가 없는 경우
                if not collect_additional and not keywords_to_collect:
                    print("\n추가 데이터 수집이 필요하지 않습니다. 저장된 데이터로 분석을 진행합니다.")
            except Exception as e:
                print(f"저장된 데이터를 불러오는 중 오류가 발생했습니다: {e}")
                print("새로운 데이터를 수집합니다.")
                use_saved_data = False

# 파일 저장 함수
def save_file(df, filename, file_format='csv'):
    filepath = os.path.join(SAVE_DIR, filename)
    if file_format == 'csv':
        df.to_csv(filepath, index=False, encoding='utf-8')
    elif file_format == 'excel':
        df.to_excel(filepath, index=False)
    elif file_format == 'pickle':
        df.to_pickle(filepath)
    print(f"파일 저장 완료: {filepath}")

# 모델 저장 함수
import os
import prophet
from prophet.serialize import model_to_json, model_from_json
import joblib
from packaging import version
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product

def check_prophet_version():
    """Prophet 버전이 model_to_json을 지원하는지 확인합니다."""
    required_version = "1.1"
    current_version = prophet.__version__
    return version.parse(current_version) >= version.parse(required_version)

def save_model(model, model_type: str, keyword: str, save_dir: str = SAVE_DIR) -> bool:
    """
    Prophet 모델을 저장합니다.

    Args:
        model: 저장할 Prophet 모델.
        model_type: 모델 유형 ("prophet").
        keyword: 키워드 (파일명에 사용).
        save_dir: 저장 디렉토리 경로.

    Returns:
        bool: 저장 성공 여부.
    """
    keyword_safe = keyword.replace(" ", "_").lower()
    try:
        if model_type == "prophet":
            if check_prophet_version():
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.json")
                with open(model_path, "w") as fout:
                    fout.write(model_to_json(model))
                print(f"{keyword}의 Prophet 모델이 저장되었습니다: {model_path}")
            else:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                joblib.dump(model, model_path)
                print(f"{keyword}의 Prophet 모델이 joblib으로 저장되었습니다: {model_path}")
        elif model_type == "lstm":
            model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.keras")
            model.save(model_path, save_format='keras')
            print(f"{keyword}의 {model_type} 모델이 저장되었습니다: {model_path}")
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        return True
    except Exception as e:
        print(f"{keyword}의 {model_type} 모델 저장 중 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False

def load_model(model_type: str, keyword: str, save_dir: str = SAVE_DIR):
    """
    Prophet/LSTM 모델을 로드합니다.

    Args:
        model_type: 모델 유형 ("prophet" 또는 "lstm").
        keyword: 키워드 (파일명에 사용).
        save_dir: 저장 디렉토리 경로.

    Returns:
        모델 객체 또는 None (로드 실패 시).
    """
    keyword_safe = keyword.replace(" ", "_").lower()
    try:
        if model_type == "prophet":
            if check_prophet_version():
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.json")
                if not os.path.exists(model_path):
                    print(f"{keyword}의 Prophet 모델 파일이 없습니다: {model_path}")
                    return None
                with open(model_path, "r") as fin:
                    model = model_from_json(fin.read())
                print(f"{keyword}의 Prophet 모델을 로드했습니다: {model_path}")
            else:
                model_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.pkl")
                if not os.path.exists(model_path):
                    print(f"{keyword}의 Prophet 모델 파일이 없습니다: {model_path}")
                    return None
                model = joblib.load(model_path)
                print(f"{keyword}의 Prophet 모델을 joblib으로 로드했습니다: {model_path}")
        elif model_type == "lstm":
            keras_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.keras")
            h5_path = os.path.join(save_dir, f"{keyword_safe}_{model_type}_model.h5")
            import tensorflow as tf
            if os.path.exists(keras_path):
                model = tf.keras.models.load_model(keras_path)
            elif os.path.exists(h5_path):
                model = tf.keras.models.load_model(h5_path)
            else:
                print(f"{keyword}의 LSTM 모델 파일을 찾을 수 없습니다.")
                return None
            print(f"{keyword}의 LSTM 모델을 로드했습니다.")
        else:
            raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
        return model
    except Exception as e:
        print(f"{keyword}의 {model_type} 모델 로드 중 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return None

# 구글 스프레드시트 업데이트 함수
def update_google_sheet(sheet_name, df, start_cell='A1'):
    if not IN_COLAB or google_sheet is None:
        print("구글 스프레드시트 업데이트를 건너뜁니다 (Colab이 아니거나 설정 실패)")
        return False

    try:
        # 전처리: 구글 시트 업로드 가능한 형태로 변환
        sheet_df = clean_for_sheets(df)

        # 워크시트 가져오기 또는 생성
        try:
            worksheet = google_sheet.worksheet(sheet_name)
        except:
            worksheet = google_sheet.add_worksheet(title=sheet_name, rows="1000", cols="26")

        # DataFrame을 리스트로 변환
        values = [sheet_df.columns.tolist()] + sheet_df.values.tolist()

        # 데이터 업데이트
        worksheet.clear()
        worksheet.update(start_cell, values)

        print(f"구글 스프레드시트 업데이트 완료: {sheet_name}")
        return True
    except Exception as e:
        print(f"구글 스프레드시트 업데이트 오류: {e}")
        return False

# 결과를 저장할 데이터프레임
if not 'result_df' in locals() or result_df.empty:
    result_df = pd.DataFrame(columns=['date', 'year', 'month', 'keyword', 'ratio'])

# 데이터 수집이 필요한지 확인
collect_data = not use_saved_data or collect_additional or keywords_to_collect

if collect_data:
    print("\n데이터 수집을 시작합니다...")

    # 기존에 불러온 데이터가 있고, 일부 키워드만 수집하는 경우
    if use_saved_data and keywords_to_collect:
        # 새 키워드만 수집
        collection_keywords = keywords_to_collect
        print(f"새 키워드 {', '.join(collection_keywords)}에 대한 데이터만 수집합니다.")

        # 임시 데이터프레임에 수집
        temp_df = pd.DataFrame(columns=['date', 'year', 'month', 'keyword', 'ratio'])
    else:
        # 모든 키워드 수집 또는 새로 시작
        collection_keywords = keywords_ko
        temp_df = result_df if use_saved_data else pd.DataFrame(columns=['date', 'year', 'month', 'keyword', 'ratio'])

    # 각 연도, 월별로 데이터 수집
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 현재 날짜 이후 데이터는 건너뜀
            current_date = datetime.now()
            if year > current_date.year or (year == current_date.year and month > current_date.month):
                print(f"{year}-{month}에 대한 데이터는 사용할 수 없습니다.")
                continue

            # 이미 저장된 데이터에 해당 연도-월 데이터가 있는지 확인
            if use_saved_data:
                date_obj = datetime(year, month, 1)
                existing_data = result_df[(result_df['year'] == year) & (result_df['month'] == month)]

                # 기존 키워드에 대한 데이터가 있고, 추가 수집이 필요 없는 경우 건너뜀
                if not keywords_to_collect and len(existing_data) == len(keywords_ko):
                    print(f"{year}-{month}의 데이터는 이미 수집되어 있습니다.")
                    continue

            # 시작일과 종료일 설정
            if month == 12:
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year+1}-01-01"
            else:
                start_date = f"{year}-{month:02d}-01"
                end_date = f"{year}-{month+1:02d}-01"

            print(f"데이터 수집 중: {start_date} ~ {end_date}")

            # 각 키워드에 대한 결과 저장
            for keyword in keywords_ko:
                # 요청 본문 설정 (한글 키워드 사용)
                body = {
                    "startDate": start_date,
                    "endDate": end_date,
                    "timeUnit": "month",
                    "keywordGroups": [
                        {
                            "groupName": keyword,
                            "keywords": [keyword]
                        }
                    ],
                    "device": "pc",
                    "ages": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
                    "gender": "f"
                }

                # API 요청
                try:
                    response = requests.post(url, headers=headers, data=json.dumps(body))
                    response_data = response.json()

                    # 응답 확인 및 데이터 추출
                    if 'results' in response_data and len(response_data['results']) > 0:
                        for item in response_data['results'][0]['data']:
                            period = item['period']
                            ratio = item['ratio']

                            # 날짜 파싱 방식 수정 - 'YYYY-MM-DD' 형식이 아닌 'YYYY-MM' 형식임
                            try:
                                # 올바른 년월 형식으로 변환
                                year_month = period.split('-')
                                year_val = int(year_month[0])
                                month_val = int(year_month[1])
                                date_obj = datetime(year_val, month_val, 1)

                                # 결과를 데이터프레임에 추가
                                new_row = pd.DataFrame({
                                    'date': [date_obj],
                                    'year': [date_obj.year],
                                    'month': [date_obj.month],
                                    'keyword': [keyword],  # 한글 키워드 사용
                                    'ratio': [ratio]
                                })
                                result_df = pd.concat([result_df, new_row], ignore_index=True)
                            except Exception as e:
                                print(f"날짜 변환 오류: {period}, 오류: {e}")
                    else:
                        print(f"키워드 '{keyword}'에 대한 데이터를 찾을 수 없습니다")

                    # API 호출 간격 (초)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"API 요청 오류: {e}")
                    print(f"응답 내용: {response.text if 'response' in locals() else '요청 실패'}")

    # 임시 데이터프레임에 수집한 경우, 기존 데이터와 병합
    if use_saved_data and keywords_to_collect and not temp_df.empty:
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        print(f"새로운 데이터 {len(temp_df)}개 행이 기존 데이터에 추가되었습니다.")

# 결과 정렬
if not result_df.empty:
    # 데이터 타입 변환 - 'ratio' 컬럼을 float 타입으로 명시적으로 변환
    result_df['ratio'] = pd.to_numeric(result_df['ratio'], errors='coerce')

    # NaN 값이 있는지 확인하고 처리
    if result_df['ratio'].isna().any():
        print(f"경고: ratio 컬럼에서 {result_df['ratio'].isna().sum()}개의 NaN 값이 발견되어 제거되었습니다")
        result_df = result_df.dropna(subset=['ratio'])

    result_df = result_df.sort_values(by=['keyword', 'date'])

    # Fix [1]: Timestamp 직렬화 오류 해결
    result_df["date"] = result_df["date"].astype(str)

    # Fix [2]: 특정 키워드('수영복') 처리 확인
    if "수영복" in keywords_ko and "수영복" not in result_df["keyword"].values:
        print("Warning: '수영복' keyword is missing in the dataset. Check data collection or fill missing values.")

    # 안전한 데이터 파싱을 위한 유틸리티 함수
    def safe_parse_date_column(df, date_col='date'):
        """날짜 컬럼을 안전하게 파싱하고 에러 처리"""
        df_copy = df.copy()

        # 문자열로 강제 변환 (숫자/None 섞여있는 경우 방지)
        df_copy[date_col] = df_copy[date_col].astype(str)

        # 자동 형식 파서로 변환 시도
        df_copy[date_col] = pd.to_datetime(
            df_copy[date_col],
            errors='coerce',
            infer_datetime_format=True
        )

        nat_count = df_copy[date_col].isna().sum()

        if nat_count == 0:
            print(f"날짜 파싱 성공! 자동 형식 감지 사용.")
            return df_copy
        else:
            print(f"자동 파싱 실패값: {nat_count}개 → 제거합니다.")
            df_copy = df_copy.dropna(subset=[date_col])
            return df_copy

    # 구글 시트 업로드용 데이터프레임 전처리
    def clean_for_sheets(df):
        """NaN, inf 값 처리하여 구글 시트 업로드 가능하게 변환"""
        # 복사본 생성
        df = df.copy()
        # inf, -inf 값을 NaN으로 변환
        df = df.replace([np.inf, -np.inf], np.nan)
        # NaN 값을 빈 문자열로 대체
        df = df.fillna('')
        return df

    # 구글 스프레드시트 업데이트 함수 개선
    def update_google_sheet(sheet_name, df, start_cell='A1'):
        if not IN_COLAB or google_sheet is None:
            print("구글 스프레드시트 업데이트를 건너뜁니다 (Colab이 아니거나 설정 실패)")
            return False

        try:
            # 전처리: 구글 시트 업로드 가능한 형태로 변환
            sheet_df = clean_for_sheets(df)

            # 워크시트 가져오기 또는 생성
            try:
                worksheet = google_sheet.worksheet(sheet_name)
            except:
                worksheet = google_sheet.add_worksheet(title=sheet_name, rows="1000", cols="26")

            # DataFrame을 리스트로 변환
            values = [sheet_df.columns.tolist()] + sheet_df.values.tolist()

            # 데이터 업데이트
            worksheet.clear()
            worksheet.update(start_cell, values)

            print(f"구글 스프레드시트 업데이트 완료: {sheet_name}")
            return True
        except Exception as e:
            print(f"구글 스프레드시트 업데이트 오류: {e}")
            return False

    print("\n수집된 데이터:")
    print(result_df.head())
    print(f"총 데이터 수: {len(result_df)}")
    print(f"데이터 유형: {result_df.dtypes}")

    # 데이터 저장
    save_file(result_df, 'naver_shopping_data.csv')

    # 구글 스프레드시트에 원본 데이터 업로드 - 개선된 함수 사용
    update_google_sheet('Raw_Data', result_df)

    # 기본 통계 분석
    print("\n기본 통계 분석:")
    stats_df = result_df.groupby('keyword')['ratio'].agg(['mean', 'std', 'min', 'max']).reset_index()
    stats_df.columns = ['키워드', '평균', '표준편차', '최소값', '최대값']
    print(stats_df)

    # 구글 스프레드시트에 통계 요약 업로드 - 개선된 함수 사용
    update_google_sheet('Statistics', stats_df)

    # 연도별, 키워드별 평균 검색 비율
    yearly_avg = result_df.groupby(['year', 'keyword'])['ratio'].mean().reset_index()
    yearly_avg.columns = ['연도', '키워드', '평균 검색 비율']
    print("\n연도별 키워드 평균 검색 비율:")
    print(yearly_avg)

    # 상관관계 분석을 위한 피벗 테이블 생성
    pivot_df = result_df.pivot_table(index='date', columns='keyword', values='ratio').reset_index()
    print("\n키워드 상관관계:")
    correlation = pivot_df.drop(columns=['date']).corr()
    print(correlation)

    # 1. 월별 인기 키워드 시각화
    # 각 달마다 가장 인기 있는 키워드 찾기
    print("\n월별 인기 키워드 분석:")

    # '년월' 컬럼 추가 (YYYY-MM 형식) - 안전한 날짜 파싱 사용
    result_df = safe_parse_date_column(result_df, 'date')
    if len(result_df) == 0:
        print("날짜 파싱 후 유효한 데이터가 없습니다. 분석을 건너뜁니다.")
    else:
        result_df['yearmonth'] = result_df['date'].dt.strftime('%Y-%m')

        # 월별 인기 키워드 찾기 - 개선된 방식
        try:
            # [5] 빈 DataFrame 문제 해결 - 데이터 존재 여부 확인
            if result_df.empty:
                print("분석할 데이터가 없습니다.")
            else:
                # 먼저 각 yearmonth 그룹 내에서 최대 ratio 값 찾기
                max_per_month = result_df.groupby('yearmonth')['ratio'].max().reset_index()

                # 이 최대값과 원본 데이터를 병합하여 해당 키워드 찾기
                monthly_top = pd.merge(
                    max_per_month,
                    result_df[['yearmonth', 'keyword', 'ratio']],
                    on=['yearmonth', 'ratio'],
                    how='left'
                )

                # 중복 제거 (같은 값이 있을 경우)
                monthly_top = monthly_top.drop_duplicates(['yearmonth', 'ratio'])

                if monthly_top.empty:
                    print("월별 최고 인기 키워드를 찾을 수 없습니다. 대체 방법을 시도합니다.")
                    raise ValueError("Empty DataFrame")

                print("월별 최고 인기 키워드:")
                print(monthly_top)
        except Exception as e:
            print(f"월별 인기 키워드 분석 중 오류 발생: {e}")
            print("대체 방법으로 분석을 시도합니다...")

            try:
                # 대체 방법: 각 날짜별로 키워드 중 최대값 찾기
                temp_pivot = result_df.pivot_table(
                    index='yearmonth',
                    columns='keyword',
                    values='ratio',
                    aggfunc='mean'
                ).reset_index()

                # 각 행에서 최댓값을 가진 컬럼 찾기
                temp_pivot['max_keyword'] = temp_pivot.iloc[:, 1:].idxmax(axis=1)
                temp_pivot['max_value'] = temp_pivot.iloc[:, 1:].max(axis=1)

                if pd.isna(temp_pivot['max_keyword']).all():
                    print("모든 데이터에서 최고 인기 키워드를 찾을 수 없습니다.")
                    monthly_top = pd.DataFrame(columns=['yearmonth', 'keyword', 'ratio'])
                else:
                    # 오타 수정: drop나() -> dropna()
                    monthly_top = temp_pivot[['yearmonth', 'max_keyword', 'max_value']].dropna()
                    monthly_top.columns = ['yearmonth', 'keyword', 'ratio']
                    print("월별 최고 인기 키워드 (대체 방법):")
                    print(monthly_top)
            except Exception as inner_e:
                print(f"대체 방법도 실패했습니다: {inner_e}")
                monthly_top = pd.DataFrame(columns=['yearmonth', 'keyword', 'ratio'])

    # 월별 인기 키워드 시각화
    plt.figure(figsize=(15, 8))

    # 히트맵으로 월별 키워드 인기도 표시
    try:
        # [6] 히트맵 오류 해결 - 데이터 충분성 검증
        if result_df.empty:
            print("히트맵을 생성할 데이터가 없습니다.")
        else:
            heatmap_data = result_df.pivot_table(
                index='yearmonth',
                columns='keyword',
                values='ratio',
                aggfunc='mean'
            ).fillna(0)

            # 시각적 효과를 위한 정렬
            heatmap_data = heatmap_data.sort_index()

            # 데이터가 충분한지 확인
            if heatmap_data.size > 0 and not heatmap_data.empty:
                sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=0.5)
                plt.title('Monthly Keyword Popularity Heatmap', fontsize=16)
                plt.xlabel('Keyword', fontsize=12)
                plt.ylabel('Year-Month', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_DIR, 'monthly_keyword_heatmap.png'), dpi=300)
                plt.show()
            else:
                print("히트맵에 표시할 데이터가 충분하지 않습니다.")
    except Exception as e:
        print(f"히트맵 시각화 중 오류 발생: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

    # 월별 최고 인기 키워드 막대 그래프
    try:
        plt.figure(figsize=(15, 8))

        # [7] 키워드 누락 데이터 처리
        # 누락된 키워드 체크
        available_keywords = result_df['keyword'].unique()
        missing_keywords = [kw for kw in keywords_ko if kw not in available_keywords]
        if missing_keywords:
            print(f"Warning: 다음 키워드에 대한 데이터가 누락되었습니다: {', '.join(missing_keywords)}")

        # 각 키워드별로 다른 색상 지정
        keyword_colors = {}
        for i, kw in enumerate(keywords_ko):
            keyword_colors[kw] = plt.cm.Set3(i)

        # 월별 최고 인기 키워드 데이터가 있는지 확인
        if monthly_top.empty:
            print("월별 최고 인기 키워드 데이터가 없어 막대 그래프를 그릴 수 없습니다.")
        else:
            # 키워드가 누락된 행 건너뛰기
            for i, (ym, kw, ratio) in enumerate(zip(monthly_top['yearmonth'], monthly_top['keyword'], monthly_top['ratio'])):
                if pd.isna(kw) or pd.isna(ratio):  # NaN 값 확인
                    print(f"Warning: NaN 값 발견 (yearmonth: {ym}, keyword: {kw}, ratio: {ratio})")
                    continue
                # 누락된 키워드의 색상 자동 할당
                if kw not in keyword_colors:
                    keyword_colors[kw] = plt.cm.Set3(len(keyword_colors))
                    print(f"Warning: 키워드 '{kw}'가 원래 키워드 목록에 없습니다. 색상이 자동 할당됩니다.")

                # 막대 그래프 그리기
                plt.bar(i, ratio, color=keyword_colors[kw],
                        label=kw if kw not in [l.get_label() for l in plt.gca().get_legend_handles_labels()[0]] else "")
                plt.text(i, ratio/2, kw, ha='center', color='black', fontweight='bold')

            plt.xticks(range(len(monthly_top)), monthly_top['yearmonth'], rotation=45)
            plt.title('Monthly Top Keywords', fontsize=16)
            plt.xlabel('Year-Month', fontsize=12)
            plt.ylabel('Search Ratio', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)

            # 중복 없는 범례
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, 'monthly_top_keywords.png'), dpi=300)
            plt.show()
    except Exception as e:
        print(f"월별 최고 인기 키워드 막대 그래프 생성 중 오류: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")

    # 데이터 시각화 1: 전체 트렌드
    plt.figure(figsize=(14, 7))
    for keyword in keywords_ko:
        keyword_data = result_df[result_df['keyword'] == keyword]
        plt.plot(keyword_data['date'], keyword_data['ratio'], marker='o', linewidth=2, label=keyword)

    plt.title('Naver Shopping Search Trend (All Period)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Search Ratio', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('naver_shopping_trend_all.png', dpi=300)
    plt.show()

    # 데이터 시각화 2: 히트맵으로 상관관계 표시
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Keyword Correlation', fontsize=16)
    plt.tight_layout()
    plt.savefig('keyword_correlation_heatmap.png', dpi=300)
    plt.show()

    # 데이터 시각화 3: 연도별 키워드 변화 (박스플롯)
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='year', y='ratio', hue='keyword', data=result_df)
    plt.title('Yearly Keyword Search Ratio Distribution', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Search Ratio', fontsize=12)
    plt.legend(title='Keyword')
    plt.tight_layout()
    plt.savefig('yearly_keyword_boxplot.png', dpi=300)
    plt.show()

    # 데이터 시각화 4: 월별 평균 검색량 (계절성 파악)
    monthly_avg = result_df.groupby(['month', 'keyword'])['ratio'].mean().reset_index()
    plt.figure(figsize=(14, 7))
    for keyword in keywords_ko:
        keyword_data = monthly_avg[monthly_avg['keyword'] == keyword]
        plt.plot(keyword_data['month'], keyword_data['ratio'], marker='o', linewidth=2, label=keyword)

    plt.title('Monthly Average Search Ratio (Seasonality)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Search Ratio', fontsize=12)
    plt.xticks(range(1, 13))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('monthly_seasonality.png', dpi=300)
    plt.show()

    # 시계열 분해 (계절성, 추세, 잔차)
    print("\nTime Series Decomposition Analysis:")
    for keyword in keywords_ko:
        keyword_data = result_df[result_df['keyword'] == keyword].sort_values(by='date')

        if len(keyword_data) >= 12:  # 최소 12개월 데이터 필요
            # 시계열 인덱스로 변환
            ts_data = keyword_data.set_index('date')['ratio']

            try:
                # 시계열 분해
                decomposition = seasonal_decompose(ts_data, model='additive', period=12)

                # 결과 시각화
                plt.figure(figsize=(14, 10))
                plt.subplot(411)
                plt.plot(ts_data, label='Original Data')
                plt.legend(loc='best')
                plt.title(f'{keyword} Time Series Decomposition')

                plt.subplot(412)
                plt.plot(decomposition.trend, label='Trend')
                plt.legend(loc='best')

                plt.subplot(413)
                plt.plot(decomposition.seasonal, label='Seasonality')
                plt.legend(loc='best')

                plt.subplot(414)
                plt.plot(decomposition.resid, label='Residual')
                plt.legend(loc='best')

                plt.tight_layout()
                plt.savefig(f'{keyword}_time_series_decomposition.png', dpi=300)
                plt.show()

                # 정상성 검정 (ADF 테스트)
                adf_result = adfuller(ts_data.dropna())
                print(f'\n{keyword} ADF Test Results:')
                print(f'ADF Statistic: {adf_result[0]}')
                print(f'p-value: {adf_result[1]}')
                print(f'Stationarity: {"Stationary" if adf_result[1] < 0.05 else "Non-stationary"}')

            except Exception as e:
                print(f"\nError during {keyword} time series decomposition: {e}")
        else:
            print(f"\n{keyword} doesn't have enough data for time series decomposition.")

# Prophet 예측 코드보다 위에 함수 정의 추가
def tune_prophet_hyperparams(df, param_grid, initial='365 days', period='180 days', horizon='180 days'):
    """
    Prophet 하이퍼파라미터 튜닝(Grid Search)
    df: Prophet 입력 데이터프레임(ds, y)
    param_grid: {'changepoint_prior_scale': [...], 'seasonality_prior_scale': [...], ...}
    """
    best_score = float('inf')
    best_params = None
    best_model = None
    all_results = []
    # 파라미터 조합 생성
    keys, values = zip(*param_grid.items())
    for v in product(*values):
        params = dict(zip(keys, v))
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0)
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df)
        # cross-validation
        df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmse = df_p['rmse'].values[0]
        all_results.append((params, rmse))
        if rmse < best_score:
            best_score = rmse
            best_params = params
            best_model = model
    print(f"Prophet 최적 파라미터: {best_params}, RMSE: {best_score:.4f}")
    return best_model, best_params, all_results

    # Prophet 모델 사용 여부 확인
    use_prophet = input("\nProphet 모델을 사용하여 예측하시겠습니까? (y/n, 기본값: n): ").lower().strip() == 'y'

    if use_prophet:
        # Prophet 모델 개선
        print("\nProphet 모델로 예측을 시작합니다.")
        for keyword in keywords_ko:
            keyword_data = result_df[result_df['keyword'] == keyword].sort_values(by='date')

            # 저장된 모델 확인
            saved_model = None
            if use_saved_data and keyword in model_paths and 'prophet' in model_paths[keyword]:
                use_saved_model = input(f"\n{keyword}에 대한 저장된 Prophet 모델이 있습니다. 이 모델을 사용하시겠습니까? (y/n, 기본값: y): ").lower().strip() != 'n'
                if use_saved_model:
                    saved_model = load_model('prophet', keyword)

            if len(keyword_data) >= 12:  # 최소 12개월 데이터 필요
                # Prophet 데이터 형식으로 변환 (ds, y)
                prophet_data = pd.DataFrame({
                    'ds': keyword_data['date'],
                    'y': keyword_data['ratio']
                })

                # NaN 값 확인 및 제거
                nan_count = prophet_data['ds'].isna().sum() + prophet_data['y'].isna().sum()
                if nan_count > 0:
                    print(f"Warning: Prophet 데이터에서 {nan_count}개의 NaN 값 발견. 제거합니다.")
                    prophet_data = prophet_data.dropna(subset=['ds', 'y'])

                    if len(prophet_data) < 12:
                        print(f"{keyword}의 유효한 데이터가 12개 미만으로 Prophet 예측을 건너뜁니다.")
                        continue

                # 하이퍼파라미터 튜닝 적용
                if saved_model is None:
                    print(f"\n{keyword} Prophet 하이퍼파라미터 튜닝 중...")
                    param_grid = {
                        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
                        'seasonality_prior_scale': [1.0, 5.0, 10.0, 20.0]
                    }
                    model, best_params, all_results = tune_prophet_hyperparams(
                        prophet_data,
                        param_grid,
                        initial='365 days',
                        period='180 days',
                        horizon='180 days'
                    )
                    print(f"{keyword} Prophet 최적 파라미터: {best_params}")
                    save_model(model, 'prophet', keyword)
                else:
                    model = saved_model
                    print(f"{keyword}에 대해 저장된 Prophet 모델을 사용합니다.")

                # 미래 12개월 예측
                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                # 예측 시각화 (개선된 버전)
                plt.figure(figsize=(14, 7))

                # 실제 데이터 플로팅
                plt.plot(prophet_data['ds'], prophet_data['y'], 'ko', markersize=4, label='실제 데이터')

                # 예측값 및 신뢰구간 플로팅
                plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='예측값')
                plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                color='blue', alpha=0.2, label='95% 신뢰구간')

                # 미래 예측 부분 강조
                future_forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
                plt.plot(future_forecast['ds'], future_forecast['yhat'], 'r-', linewidth=2, label='미래 예측')

                plt.title(f'{keyword} 검색 비율 예측 (Prophet)', fontsize=16)
                plt.xlabel('날짜', fontsize=12)
                plt.ylabel('검색 비율', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{keyword}_prophet_prediction.png', dpi=300)
                plt.show()

                # 성분 시각화 (개선된 버전)
                fig2 = model.plot_components(forecast)
                plt.tight_layout()
                plt.savefig(f'{keyword}_prophet_components.png', dpi=300)
                plt.show()

                # 예측 결과 출력
                print(f"\n{keyword} 향후 6개월 예측 (Prophet):")
                future_forecast = forecast[forecast['ds'] > prophet_data['ds'].max()].head(6)
                future_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                future_df.columns = ['날짜', '예측값', '하한', '상한']
                future_df['날짜'] = future_df['날짜'].dt.strftime('%Y-%m')
                print(future_df)

            else:
                print(f"\n{keyword}는 Prophet 예측을 위한 충분한 데이터가 없습니다.")

    # LSTM 모델 사용 여부 확인
    use_lstm = input("\nLSTM 모델을 사용하여 예측하시겠습니까? (y/n, 기본값: n): ").lower().strip() == 'y'

    if use_lstm:
        # 몇 개의 키워드에 대해 LSTM을 적용할지 물어보기
        try:
            lstm_count = int(input("몇 개의 키워드에 대해 LSTM을 적용할까요? (기본값: 2): ") or "2")
            lstm_count = min(lstm_count, len(keywords_ko))  # 입력된 키워드 수를 넘지 않도록
        except ValueError:
            lstm_count = 2
            print("유효하지 않은 입력입니다. 기본값 2를 사용합니다.")

        # LSTM 모델 개선
        print(f"\n머신러닝 예측 모델 3: LSTM (처음 {lstm_count}개 키워드에 적용)")
        for keyword in keywords_ko[:lstm_count]:
            keyword_data = result_df[result_df['keyword'] == keyword].sort_values(by='date')

            # 저장된 모델 확인
            saved_model = None
            if use_saved_data and keyword in model_paths and 'lstm' in model_paths[keyword]:
                use_saved_model = input(f"\n{keyword}에 대한 저장된 LSTM 모델이 있습니다. 이 모델을 사용하시겠습니까? (y/n, 기본값: y): ").lower().strip() != 'n'
                if use_saved_model:
                    saved_model = load_model('lstm', keyword)

            if len(keyword_data) >= 12:  # 최소 12개월 데이터 필요
                # 데이터 정규화
                values = keyword_data['ratio'].values.reshape(-1, 1)
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(values)

                # 시퀀스 데이터 생성 (시간 지연 특성)
                def create_sequences(data, seq_length):
                    xs, ys = [], []
                    for i in range(len(data) - seq_length):
                        x = data[i:i+seq_length]
                        y = data[i+seq_length]
                        xs.append(x)
                        ys.append(y)
                    return np.array(xs), np.array(ys)

                # 시퀀스 길이 (데이터에 따라 동적으로 조정)
                seq_length = min(12, len(keyword_data) // 2)  # 기존 6에서 12로 변경
                if seq_length < 3:  # 최소 3개월은 필요
                    seq_length = 3

                print(f"{keyword}의 시퀀스 길이: {seq_length}")

                # 데이터 준비
                X, y = create_sequences(scaled_values, seq_length)

                # 문제 5: X와 y의 차원 확인 및 조정
                if len(X) != len(y):
                    print(f"Warning: X와 y의 차원이 일치하지 않습니다 (X: {len(X)}, y: {len(y)})")
                    # 더 작은 길이에 맞춤
                    min_len = min(len(X), len(y))
                    X = X[:min_len]
                    y = y[:min_len]
                    print(f"차원 맞춤 완료: X와 y 모두 {min_len} 길이로 조정됨")

                if saved_model is None and len(X) >= 10:  # 충분한 시퀀스 샘플이 있는지 확인
                    # 훈련/테스트 분할 (80/20)
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    # KerasTuner로 하이퍼파라미터 튜닝
                    tuner = kt.BayesianOptimization(
                        lambda hp: build_lstm_model(hp, seq_length),
                        objective='val_loss',
                        max_trials=10,
                        directory=SAVE_DIR,
                        project_name=f'lstm_tuning_{keyword}'
                    )
                    print(f"\n{keyword} LSTM 하이퍼파라미터 튜닝 중...")
                    tuner.search(
                        X_train, y_train,
                        epochs=50,
                        batch_size=max(8, len(X_train) // 10),
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)],
                        verbose=2
                    )
                    best_hp = tuner.get_best_hyperparameters(1)[0]
                    print(f"{keyword} 최적 하이퍼파라미터: units1={best_hp.get('units1')}, units2={best_hp.get('units2')}, dropout={best_hp.get('dropout')}, learning_rate={best_hp.get('learning_rate')}")
                    model = tuner.hypermodel.build(best_hp)

                    # 최적 모델로 추가 학습 (조기종료 포함)
                    history = model.fit(
                        X_train, y_train,
                        epochs=150,
                        batch_size=max(8, len(X_train) // 10),
                        validation_data=(X_test, y_test),
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                        ],
                        verbose=1
                    )

                    # 모델 저장
                    save_model(model, 'lstm', keyword)

                    # 먼저 예측 수행 - all_predictions 정의
                    all_sequences = create_sequences(scaled_values, seq_length)[0]
                    all_predictions = model.predict(all_sequences)
                    all_predictions = scaler.inverse_transform(all_predictions)

                    # 손실 시각화
                    plt.figure(figsize=(12,5))
                    plt.subplot(1,2,1)
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Val Loss')
                    plt.title('LSTM Loss')
                    plt.legend()

                    # 예측 시각화 (이제 all_predictions가 정의되어 있음)
                    plt.subplot(1,2,2)
                    plt.subplot(1,2,1)
                    plt.plot(history.history['loss'], label='Train Loss')
                    plt.plot(history.history['val_loss'], label='Val Loss')
                    plt.title('LSTM Loss')
                    plt.legend()

                    # 예측 시각화 (이제 all_predictions가 정의되어 있음)
                    plt.subplot(1,2,2)
                    # 날짜와 예측 배열 길이 일치 확인
                    date_range = keyword_data['date'][seq_length:].values
                    if len(date_range) != len(all_predictions):
                        print(f"경고: 날짜({len(date_range)})와 예측({len(all_predictions)})의 길이가 다릅니다. 최소 길이로 조정합니다.")
                        min_len = min(len(date_range), len(all_predictions))
                        date_range = date_range[:min_len]
                        all_predictions = all_predictions[:min_len]

                    plt.plot(date_range, all_predictions, label='Predicted')
                    plt.plot(keyword_data['date'], keyword_data['ratio'], label='Actual')
                    plt.title('Actual vs Predicted')
                    plt.legend()

                    plt.tight_layout()
                    plt.show()
                elif saved_model is not None:
                    # 저장된 모델 사용
                    model = saved_model
                    print(f"{keyword}에 대해 저장된 LSTM 모델을 사용합니다.")
                else:
                    print(f"\n{keyword}는 LSTM 모델링을 위한 충분한 데이터가 없습니다.")
                    continue

                # 예측 및 시각화
                try:
                    # 모든 시퀀스 데이터에 대한 예측
                    all_sequences = create_sequences(scaled_values, seq_length)[0]
                    all_predictions = model.predict(all_sequences)
                    all_predictions = scaler.inverse_transform(all_predictions)

                    # 예측 날짜 생성
                    pred_dates = keyword_data['date'][seq_length:].values

                    # 미래 6개월 예측
                    last_sequence = scaled_values[-seq_length:].reshape(1, seq_length, 1)
                    future_predictions = []

                    current_sequence = last_sequence.copy()
                    for _ in range(6):  # 6개월 예측
                        # 다음 값 예측
                        next_pred = model.predict(current_sequence)[0]
                        future_predictions.append(next_pred)

                        # 시퀀스 업데이트
                        current_sequence = np.append(current_sequence[:, 1:, :], np.array([next_pred]).reshape(1, 1, 1), axis=1)

                    # 역정규화
                    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                    # 날짜 생성
                    last_date = keyword_data['date'].iloc[-1]
                    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(len(future_predictions))]

                    # 개선된 시각화
                    plt.figure(figsize=(14, 7))

                    # 실제 데이터 (전체)
                    plt.plot(keyword_data['date'], keyword_data['ratio'], 'b-', marker='o', markersize=3, label='실제 데이터')

                    # 예측 데이터 (과거)
                    plt.plot(pred_dates, all_predictions, 'g--', alpha=0.7, label='과거 예측')

                    # 미래 예측
                    plt.plot(future_dates, future_predictions, 'r-', linewidth=2, marker='s', label='미래 예측')

                    # 미래 예측 불확실성 범위 (단순 추정)
                    uncertainty = np.std(keyword_data['ratio'].values) * 0.5
                    plt.fill_between(
                        future_dates,
                        future_predictions.flatten() - uncertainty,
                        future_predictions.flatten() + uncertainty,
                        color='red', alpha=0.2, label='예측 불확실성'
                    )

                    plt.title(f'{keyword} 검색 비율 예측 (LSTM)', fontsize=16)
                    plt.xlabel('날짜', fontsize=12)
                    plt.ylabel('검색 비율', fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend(fontsize=12)
                    plt.tight_layout()
                    plt.savefig(f'{keyword}_lstm_prediction.png', dpi=300)
                    plt.show()

                    # 미래 예측 결과 출력
                    print(f"\n{keyword} 향후 6개월 예측 (LSTM):")
                    future_df = pd.DataFrame({
                        '날짜': [d.strftime('%Y-%m') for d in future_dates],
                        '예측 검색 비율': future_predictions.flatten(),
                        '하한 예측': future_predictions.flatten() - uncertainty,
                        '상한 예측': future_predictions.flatten() + uncertainty
                    })
                    print(future_df)

                    # 모델 평가: 과거 데이터와의 RMSE 계산
                    actual = keyword_data['ratio'].values[seq_length:]
                    predicted = all_predictions.flatten()
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    r2 = r2_score(actual, predicted)
                    print(f"\n{keyword} LSTM 모델 평가:")
                    print(f"RMSE: {rmse:.4f}")
                    print(f"R² Score: {r2:.4f}")

                except Exception as e:
                    print(f"\n{keyword} LSTM 예측 중 오류: {e}")
            else:
                print(f"\n{keyword}는 LSTM 예측을 위한 충분한 데이터가 없습니다 (최소 12개월 필요).")

    # 키워드 간 인사이트 분석
    print("\n키워드 간 인사이트 분석:")

    # 1. 키워드 간 성장률 비교
    growth_rates = []
    for keyword in keywords_ko:
        keyword_data = result_df[result_df['keyword'] == keyword].sort_values(by='date')
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
        plt.bar(growth_df['키워드'], growth_df['성장률(%)'])
        plt.title('키워드별 검색 비율 성장률', fontsize=16)
        plt.xlabel('키워드', fontsize=12)
        plt.ylabel('성장률 (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig('keyword_growth_rate.png', dpi=300)
        plt.show()

    # 2. 계절성 패턴 비교
    print("\n키워드별 계절성 패턴:")
    for keyword in keywords_ko:
        keyword_data = result_df[result_df['keyword'] == keyword]
        if not keyword_data.empty:
            monthly_pattern = keyword_data.groupby('month')['ratio'].mean()
            print(f"\n{keyword} 월별 평균 검색 비율:")
            print(monthly_pattern)
        else:
            print(f"\n{keyword}에 대한 데이터가 없습니다.")

    # 최종 결과 저장
    if not result_df.empty:
        result_df.to_csv('naver_shopping_data_extended.csv', index=False)
        print("\n모든 데이터가 'naver_shopping_data_extended.csv'에 저장되었습니다.")
        print("모든 분석 및 예측이 완료되었습니다.")
    else:
        print("\n저장할 데이터가 없습니다.")

# 최상위 else: (데이터가 수집되지 않았을 때)
if 'result_df' not in locals() or result_df.empty:
    print("\n데이터가 수집되지 않았습니다. API 응답을 확인해주세요.")
    print("프로그램을 종료합니다.")
    exit()

# (이미 model_to_json 방식을 사용 중이므로, Prophet 저장 오류는 해결됨)
# Fix [3]: Prophet 모델 저장 오류 시 older 버전일 경우 joblib 사용
import joblib
# joblib.dump(model, 'prophet_model.pkl')

# LSTM 모델 빌더 함수 (KerasTuner용)
def build_lstm_model(hp, seq_length):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    # LSTM 유닛 수 튜닝
    units1 = hp.Int('units1', min_value=32, max_value=256, step=32)
    units2 = hp.Int('units2', min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

    model.add(LSTM(units1, return_sequences=True, activation='tanh', input_shape=(seq_length, 1), recurrent_dropout=0.2))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(units2, return_sequences=False, activation='tanh', recurrent_dropout=0.2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# 추가 모델 및 앙상블 함수
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
try:
    import xgboost as xgb
except ImportError:
    import os
    os.system('pip install xgboost')
    import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    import os
    os.system('pip install lightgbm')
    import lightgbm as lgb

def predict_random_forest(X, y, future_steps=6):
    # X: (n_samples, n_features), y: (n_samples,)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    # 미래 예측 (마지막 시퀀스 기준)
    last_seq = X[-1].reshape(1, -1)
    future_preds = []
    for _ in range(future_steps):
        next_pred = model.predict(last_seq)[0]
        future_preds.append(next_pred)
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1] = next_pred
    return preds, np.array(future_preds)

def predict_xgboost(X, y, future_steps=6):
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    last_seq = X[-1].reshape(1, -1)
    future_preds = []
    for _ in range(future_steps):
        next_pred = model.predict(last_seq)[0]
        future_preds.append(next_pred)
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1] = next_pred
    return preds, np.array(future_preds)

def predict_lightgbm(X, y, future_steps=6):
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    last_seq = X[-1].reshape(1, -1)
    future_preds = []
    for _ in range(future_steps):
        next_pred = model.predict(last_seq)[0]
        future_preds.append(next_pred)
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1] = next_pred
    return preds, np.array(future_preds)

def predict_arima(y, order=(1,1,1), future_steps=6):
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    preds = model_fit.fittedvalues
    future_preds = model_fit.forecast(steps=future_steps)
    return preds, future_preds  # .values 제거

def ensemble_average(*args):
    # 여러 예측값을 받아 평균 앙상블
    arrs = [np.array(a) for a in args if a is not None]
    min_len = min([len(a) for a in arrs])
    arrs = [a[:min_len] for a in arrs]
    return np.mean(arrs, axis=0)

# 1. 가중평균 앙상블 함수
from typing import List

def ensemble_weighted(pred_list: List[np.ndarray], weights: List[float]):
    arrs = [np.array(a) for a in pred_list if a is not None]
    min_len = min([len(a) for a in arrs])
    arrs = [a[:min_len] for a in arrs]
    weights = np.array(weights)[:len(arrs)]
    weights = weights / weights.sum()
    arrs = np.stack(arrs, axis=0)
    return np.average(arrs, axis=0, weights=weights)

# --- 다양한 모델 예측 및 앙상블 (가중평균, 성능비교, 신뢰구간) ---
for keyword in keywords_ko:
    keyword_data = result_df[result_df['keyword'] == keyword].sort_values(by='date')
    if len(keyword_data) < 12:
        print(f"{keyword}는 예측을 위한 충분한 데이터가 없습니다.")
        continue
    print(f"\n[{keyword}] 다양한 모델 예측 및 앙상블 결과:")
    # 데이터 준비
    values = keyword_data['ratio'].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)
    seq_length = min(12, len(keyword_data) // 2)
    if seq_length < 3:
        seq_length = 3
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    X, y = create_sequences(scaled_values, seq_length)
    if len(X) < 10:
        print(f"{keyword}는 회귀/시계열 모델 예측을 위한 충분한 데이터가 없습니다.")
        continue
    # 2D 변환 (회귀모델용)
    X_rf = X.reshape(X.shape[0], -1)
    # Prophet 예측값(과거)
    prophet_preds = None
    try:
        prophet_data = pd.DataFrame({'ds': keyword_data['date'], 'y': keyword_data['ratio']})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(prophet_data)
        forecast = model.predict(model.make_future_dataframe(periods=6, freq='M'))
        prophet_preds = forecast['yhat'].values[:len(keyword_data)]
        prophet_future = forecast['yhat'].values[-6:]
    except Exception as e:
        print(f"Prophet 예측 오류: {e}")
        prophet_future = None
    # LSTM 예측값(과거)
    lstm_preds = None
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        model = Sequential([
            LSTM(64, return_sequences=True, activation='tanh', input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation='tanh'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X, y, epochs=30, batch_size=8, verbose=0)
        lstm_preds = scaler.inverse_transform(model.predict(X)).flatten()
        # 미래 예측
        last_seq = scaled_values[-seq_length:].reshape(1, seq_length, 1)
        lstm_future = []
        for _ in range(6):
            next_pred = model.predict(last_seq)[0][0]
            lstm_future.append(next_pred)
            last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)
        lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"LSTM 예측 오류: {e}")
        lstm_future = None
    # RandomForest
    try:
        rf_preds, rf_future = predict_random_forest(X_rf, y, future_steps=6)
        rf_preds = scaler.inverse_transform(rf_preds.reshape(-1, 1)).flatten()
        rf_future = scaler.inverse_transform(rf_future.reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"RandomForest 예측 오류: {e}")
        rf_preds, rf_future = None, None
    # XGBoost
    try:
        xgb_preds, xgb_future = predict_xgboost(X_rf, y, future_steps=6)
        xgb_preds = scaler.inverse_transform(xgb_preds.reshape(-1, 1)).flatten()
        xgb_future = scaler.inverse_transform(xgb_future.reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"XGBoost 예측 오류: {e}")
        xgb_preds, xgb_future = None, None
    # LightGBM
    try:
        lgb_preds, lgb_future = predict_lightgbm(X_rf, y, future_steps=6)
        lgb_preds = scaler.inverse_transform(lgb_preds.reshape(-1, 1)).flatten()
        lgb_future = scaler.inverse_transform(lgb_future.reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"LightGBM 예측 오류: {e}")
        lgb_preds, lgb_future = None, None
    # ARIMA
    try:
        arima_preds, arima_future = predict_arima(keyword_data['ratio'].values, order=(1,1,1), future_steps=6)
        arima_preds = arima_preds.flatten()
        arima_future = arima_future.flatten()
    except Exception as e:
        print(f"ARIMA 예측 오류: {e}")
        arima_preds, arima_future = None, None
    # 모델별 성능(RMSE, R2)
    from sklearn.metrics import mean_squared_error, r2_score
    y_true = keyword_data['ratio'].values[seq_length:]
    model_names = ['Prophet', 'LSTM', 'RandomForest', 'XGBoost', 'LightGBM', 'ARIMA']
    model_preds = [prophet_preds, lstm_preds, rf_preds, xgb_preds, lgb_preds, arima_preds]
    perf_table = []
    for name, preds in zip(model_names, model_preds):
        if preds is not None and len(preds) == len(y_true):
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            r2 = r2_score(y_true, preds)
            perf_table.append({'모델':name, 'RMSE':rmse, 'R2':r2})
    perf_df = pd.DataFrame(perf_table)
    print("\n모델별 성능 비교:")
    print(perf_df)
    # 가중평균 앙상블 (성능 역수 기반 가중치)
    weights = []
    for row in perf_table:
        weights.append(1/(row['RMSE']+1e-8))
    # 예측값과 가중치 개수 일치 (zip으로 동시 필터링)
    valid_pairs = [(p, w) for p, w in zip(model_preds, weights) if p is not None and len(p) == len(y_true)]
    if valid_pairs:
        valid_preds, valid_weights = zip(*valid_pairs)
        ensemble_weighted_past = ensemble_weighted(valid_preds, valid_weights)
        # 미래 예측도 동일 가중치로
        future_preds_list = [prophet_future, lstm_future, rf_future, xgb_future, lgb_future, arima_future]
        valid_future_pairs = [(p, w) for p, w in zip(future_preds_list, valid_weights) if p is not None and len(p) == 6]
        if valid_future_pairs:
            valid_future_preds, valid_future_weights = zip(*valid_future_pairs)
            ensemble_weighted_future = ensemble_weighted(valid_future_preds, valid_future_weights)
        else:
            ensemble_weighted_future = None
    else:
        ensemble_weighted_past = None
        ensemble_weighted_future = None
    # 신뢰구간(표준편차 기반)
    all_future_preds = np.stack([p for p in [prophet_future, lstm_future, rf_future, xgb_future, lgb_future, arima_future] if p is not None and len(p) == 6], axis=0)
    ensemble_std = np.std(all_future_preds, axis=0)
    # 시각화
    plt.figure(figsize=(16, 7))
    plt.plot(keyword_data['date'][seq_length:], y_true, 'k-', label='실제')
    for name, preds in zip(model_names, model_preds):
        if preds is not None and len(preds) == len(keyword_data['date'][seq_length:]):
            plt.plot(keyword_data['date'][seq_length:], preds, label=name)
        elif preds is not None:
            # 길이가 다르면 최소 길이로 맞춰서 그리기 (경고 출력)
            min_len = min(len(preds), len(keyword_data['date'][seq_length:]))
            print(f"경고: {name} 예측값 길이({len(preds)})와 날짜({len(keyword_data['date'][seq_length:])})가 달라 최소 길이({min_len})로 맞춥니다.")
            plt.plot(keyword_data['date'][seq_length:][:min_len], preds[:min_len], label=name)
    plt.plot(keyword_data['date'][seq_length:], ensemble_weighted_past, 'r--', label='앙상블(가중평균)')
    # 미래 예측
    future_dates = [keyword_data['date'].iloc[-1] + pd.DateOffset(months=i+1) for i in range(6)]
    plt.plot(future_dates, ensemble_weighted_future, 'ro-', label='앙상블 미래(가중)')
    # 신뢰구간
    plt.fill_between(future_dates, ensemble_weighted_future-ensemble_std, ensemble_weighted_future+ensemble_std, color='red', alpha=0.2, label='앙상블 신뢰구간')
    plt.title(f'{keyword} 다양한 모델 예측 및 앙상블(가중평균, 신뢰구간)', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('검색 비율', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 결과 출력
    print(f"앙상블 미래 6개월 예측(가중평균):")
    print(pd.DataFrame({'날짜':[d.strftime('%Y-%m') for d in future_dates], '앙상블 예측(가중)':ensemble_weighted_future, '신뢰구간 하한':ensemble_weighted_future-ensemble_std, '신뢰구간 상한':ensemble_weighted_future+ensemble_std}))