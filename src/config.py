import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class Config:
    """애플리케이션 설정 관리 클래스"""
    
    # 파일 경로 설정
    SAVE_DIR = './results'  # 로컬 저장 디렉토리
    DATA_FILE = 'naver_shopping_data_extended.csv'
    
    # API 설정
    API_URL = "https://openapi.naver.com/v1/datalab/search"
    API_TIMEOUT = 30
    API_DELAY = 0.5  # API 호출 간격 (초)
    
    # 시각화 설정
    FIGURE_SIZE = (14, 7)
    DPI = 300
    
    # 모델 설정
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 8
    PROPHET_PERIODS = 12  # 예측할 개월 수
    
    @classmethod
    def setup_directories(cls):
        """필요한 디렉토리를 생성합니다"""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, 'models'), exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, 'plots'), exist_ok=True)
    
    @classmethod
    def setup_font(cls):
        """운영체제에 맞는 한글 폰트를 설정합니다"""
        try:
            system = platform.system()
            if system == 'Windows':
                # Windows에서 사용 가능한 한글 폰트 설정
                font_candidates = ['Malgun Gothic', 'NanumGothic', 'Gulim']
                for font in font_candidates:
                    try:
                        plt.rcParams['font.family'] = font
                        plt.rcParams['axes.unicode_minus'] = False
                        print(f"한글 폰트 설정 완료: {font}")
                        break
                    except:
                        continue
                else:
                    print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            elif system == 'Darwin':  # macOS
                plt.rcParams['font.family'] = 'AppleGothic'
                plt.rcParams['axes.unicode_minus'] = False
            else:  # Linux
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"폰트 설정 중 오류 발생: {e}")
            print("기본 폰트를 사용합니다.")
    
    @classmethod
    def initialize(cls):
        """전체 설정을 초기화합니다"""
        cls.setup_directories()
        cls.setup_font()
        print(f"설정 초기화 완료!")
        print(f"저장 디렉토리: {cls.SAVE_DIR}") 