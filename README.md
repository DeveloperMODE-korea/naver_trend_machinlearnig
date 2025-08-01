# 네이버 쇼핑 트렌드 머신러닝 분석 시스템 🛍️

## 개요
네이버 쇼핑 검색 트렌드 데이터를 수집하고 머신러닝을 활용하여 미래 트렌드를 예측하는 시스템입니다.

## 이미지
![image](https://github.com/user-attachments/assets/5e5849e7-cd18-4647-ad43-a2648d3db034)


### 주요 개선사항
- ✅ **구글 코랩 의존성 제거**: 로컬 환경에서 완전 독립 실행
- ✅ **AMD GPU 지원**: CPU 및 AMD GPU 환경 최적화
- ✅ **모듈화 구조**: 1500+ 줄 코드를 6개 모듈로 분리
- ✅ **향상된 사용성**: 직관적인 메뉴와 에러 처리

## 시스템 요구사항

### 환경
- **Python 3.8+** (권장: 3.11)
- **Windows 10/11** (현재 AMD GPU 최적화)
- **RAM**: 최소 8GB (권장: 16GB)

### GPU 지원
- **CPU 모드**: 모든 환경에서 동작
- **AMD GPU**: ROCm 지원 (선택사항)
- **NVIDIA GPU**: CUDA 지원 (추후 업데이트 예정)

## 설치 방법

### 1. 저장소 복제 및 이동
```bash
git clone https://github.com/DeveloperMODE-korea/naver_trend_machinlearnig.git
cd naver_trend_machinlearnig
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**새로 추가된 패키지:**
- `pytrends>=4.9.0`: 구글 트렌드 데이터 수집

## 사용 방법

### 🆕 새로운 실행 방법 (모듈화 v2.1)
```bash
# 간소화된 메인 실행
python main.py

# 웹 대시보드 실행 (권장)
python main.py --web

# 도움말
python main.py --help
```

### 기존 방법 (호환성 유지)
```bash
# 기본 실행 (전체 프로세스)
python main_new.py

# 고급 사용자 메뉴
python main_new.py --menu
```

### 메뉴 옵션
1. **데이터 수집만 실행** - 네이버 API를 통한 데이터 수집
2. **데이터 분석만 실행** - 기존 데이터의 통계 분석 및 시각화
3. **머신러닝 예측만 실행** - Prophet, LSTM 등을 활용한 예측
4. **전체 실행** - 모든 단계를 순차적으로 실행
5. **설정 확인** - 현재 시스템 설정 확인

## 프로젝트 구조 (모듈화 v2.1)

```
naver_trend_machinlearnig/
├── main.py                     # 🆕 새로운 메인 실행 파일 (간소화)
├── main_new.py                 # 기존 메인 파일 (호환성)
├── requirements.txt            # 패키지 목록
├── src/                        # 🆕 모든 소스 코드 (모듈화)
│   ├── __init__.py
│   ├── config.py               # 설정 관리
│   ├── core/                   # 🆕 핵심 기능
│   │   ├── __init__.py
│   │   └── data_manager.py     # 데이터 프로세스 통합 관리
│   ├── collectors/             # 🆕 데이터 수집기들
│   │   ├── __init__.py
│   │   ├── naver_collector.py  # 네이버 API 수집
│   │   ├── google_collector.py # 구글 트렌드 수집
│   │   └── multi_platform_collector.py # 다중 플랫폼
│   ├── models/                 # 🆕 머신러닝 모델들
│   │   ├── __init__.py
│   │   ├── prophet_model.py    # Prophet 시계열 예측
│   │   ├── lstm_model.py       # LSTM 신경망
│   │   └── ensemble_model.py   # 앙상블 모델
│   ├── analysis/               # 🆕 데이터 분석
│   │   ├── __init__.py
│   │   ├── statistics.py       # 통계 분석
│   │   └── visualization.py    # 시각화 엔진
│   └── utils/                  # 🆕 유틸리티들
│       ├── __init__.py
│       ├── file_utils.py       # 파일 관리
│       ├── model_utils.py      # 모델 관리
│       └── common_utils.py     # 공통 기능
├── web/                        # 🆕 웹 인터페이스
│   ├── __init__.py
│   ├── app.py                  # Streamlit 대시보드 (개선)
│   ├── pages/                  # 웹 페이지들
│   └── components/             # 웹 컴포넌트들
├── run_dashboard.bat           # Windows 웹 실행
├── run_dashboard.sh            # Unix/Linux 웹 실행
├── venv/                       # 가상환경
└── results/                    # 결과 파일 저장소
    ├── models/                 # 훈련된 모델 파일
    ├── plots/                  # 생성된 그래프 이미지
    └── *.csv                   # 분석 결과 데이터
```

### 🆕 모듈화 개선사항
- **📦 체계적인 구조**: 기능별 모듈 분리로 가독성 향상
- **🔧 향상된 유지보수성**: 모듈 간 독립성으로 수정 용이
- **⚡ 성능 최적화**: 필요한 모듈만 로드하여 빠른 실행
- **🛡️ 안정성 향상**: 에러 격리 및 디버깅 용이
- **💡 확장성**: 새로운 기능 추가 시 기존 코드 영향 최소화

## 주요 기능

### 📊 데이터 수집
- **네이버 쇼핑 트렌드**: API 연동을 통한 키워드별 월간 검색 비율
- **구글 트렌드**: 글로벌 검색 트렌드 데이터 (pytrends 활용)
- **다중 플랫폼 통합**: 네이버 + 구글 트렌드 동시 분석
- **스마트 데이터 관리**: 기존 데이터 재사용 및 증분 업데이트

### 📈 데이터 분석
- **기본 통계 분석**: 키워드별 평균, 표준편차, 성장률
- **상관관계 분석**: 키워드 간 상관관계 히트맵
- **시계열 분석**: 추세, 계절성, 잔차 분해
- **시각화**: 트렌드 그래프, 박스플롯, 히트맵

### 🔍 데이터 품질 검증 (NEW!)
- **DataValidator**: 예측값 범위 제한, NaN/Inf 보정, 급격한 변화 완화
- **OutlierDetector**: IQR과 Z-Score 기반 다중 방법 이상치 탐지
- **QualityMonitor**: 완성도/유효성/일관성/충분성 등 종합 품질 평가
- **자동 검증**: 모든 예측 과정에서 실시간 품질 체크

### 🤖 머신러닝 예측 (대폭 확장!)
- **기본 모델**: Prophet, LSTM (범위 제약 적용)
- **🆕 Transformer**: BERT/GPT 스타일 어텐션 메커니즘 시계열 예측
- **🆕 XGBoost**: 고급 특성 엔지니어링 + 그래디언트 부스팅
- **🆕 CatBoost**: 카테고리 특성 자동 처리 부스팅
- **🆕 AutoML**: Optuna 기반 자동 모델 선택 및 하이퍼파라미터 튜닝
- **🎯 동적 앙상블**: 실시간 성능 평가 기반 가중치 자동 조정
- **⚖️ 수학적 최적화**: scipy.optimize 기반 가중치 최적화
- **다중 플랫폼 예측**: 플랫폼별 독립 예측 후 통합
- **신뢰구간**: 모델 간 일치도 기반 신뢰도 계산
- **검증 강화**: 예측값 자동 보정 및 신뢰도 평가

### 🧠 AI 리포트 (NEW!)
- **Claude AI 기반**: 전문가 수준의 데이터 해석
- **4가지 리포트 유형**: 데이터 분석, 예측 분석, 품질 분석, 비즈니스 인사이트
- **한국 시장 특화**: 국내 소비 트렌드에 맞춘 전문 분석
- **실행 가능한 권장사항**: 구체적인 비즈니스 액션 아이템 제공

## API 설정

### 네이버 개발자 센터
1. [네이버 개발자 센터](https://developers.naver.com/) 접속
2. 애플리케이션 등록 및 데이터랩 API 신청
3. Client ID와 Client Secret 확보
4. 프로그램 실행 시 입력

### 구글 트렌드 (자동 설정)
- **pytrends** 라이브러리 자동 사용
- 별도 API 키 불필요 (웹 스크래핑 방식)
- 지역 설정: 한국(KR) 기본값
- 일일 요청 제한 고려 필요

### Claude AI API (NEW!)
1. [Anthropic Console](https://console.anthropic.com/) 접속
2. 계정 생성 및 로그인
3. API 키 생성 (Dashboard → API Keys)
4. 웹 대시보드의 🧠 AI 리포트 페이지에서 입력
5. **주의**: API 사용량에 따라 과금됩니다

## 결과 파일

### CSV 파일
- `naver_shopping_data_extended.csv`: 네이버 단일 플랫폼 데이터
- `multi_platform_data.csv`: 네이버 + 구글 통합 데이터
- `basic_statistics.csv`: 기본 통계 결과
- `multi_platform_statistics.csv`: 플랫폼별 통계
- `correlation_matrix.csv`: 키워드 간 상관관계 행렬
- `platform_correlation_summary.csv`: 플랫폼 간 상관관계 요약
- `growth_rate_analysis.csv`: 성장률 분석 결과

### 이미지 파일
- `overall_trend.png`: 전체 키워드 트렌드
- `correlation_heatmap.png`: 키워드 간 상관관계 히트맵
- `monthly_heatmap.png`: 월별 인기도 히트맵
- `*_prediction.png`: 키워드별 단일 플랫폼 예측 결과
- `*_multi_platform_prediction.png`: 키워드별 다중 플랫폼 통합 예측
- `*_platform_comparison.png`: 플랫폼별 트렌드 비교
- `platform_correlation_distribution.png`: 플랫폼 간 상관관계 분포

### 모델 파일
- `*_prophet_model.json`: Prophet 모델
- `*_lstm_model.keras`: LSTM 모델

## 성능 최적화

### CPU 모드 (권장)
```python
# config.py에서 설정
LSTM_EPOCHS = 50      # 빠른 실행
LSTM_BATCH_SIZE = 8   # 메모리 효율성
```

### GPU 가속 (선택사항)
AMD GPU 사용 시 ROCm 설치:
```bash
pip install tensorflow-rocm
```

## 문제 해결

### 일반적인 오류

#### 1. 모듈 import 오류
```bash
# 가상환경 활성화 확인
venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. 한글 폰트 오류
- Windows: 자동으로 'Malgun Gothic' 설정
- 문제 지속 시 시스템 폰트 확인

#### 3. 메모리 부족
```python
# config.py에서 배치 사이즈 조정
LSTM_BATCH_SIZE = 4  # 기본값 8에서 감소
```

#### 4. API 오류
- 네이버 API 키 재확인
- 요청 한도 확인 (일일 1,000회)
- 네트워크 연결 상태 확인

### 로그 및 디버깅
```bash
# 상세 오류 정보 확인
python main_new.py --menu
# 개별 모듈 테스트
```

## 기여 방법

### 버그 리포트
1. 오류 상황 및 환경 정보 제공
2. 재현 가능한 최소 코드 예제
3. 예상 결과 vs 실제 결과

### 기능 요청
1. 구체적인 사용 사례 설명
2. 예상되는 구현 방법
3. 기존 기능과의 호환성 고려

## 라이선스
MIT License - 자유롭게 사용, 수정, 배포 가능

## 지원 및 문의
- **이슈 제기**: GitHub Issues 활용
- **업데이트**: 정기적으로 새 기능 및 최적화 제공

---

### 변경 이력
- **v2.0** (2025): 로컬 환경 최적화, 모듈화 구조
- **v1.0** (2025): 초기 구글 코랩 버전

## 🌐 웹 대시보드 (NEW!)

### 빠른 실행
```bash
# Windows
run_dashboard.bat

# macOS/Linux
chmod +x run_dashboard.sh
./run_dashboard.sh
```

브라우저에서 `http://localhost:8501` 접속!

### 웹 대시보드 기능
- **🏠 홈**: 시스템 개요 및 상태 확인
- **📊 데이터 수집**: 직관적인 UI로 데이터 수집
- **📈 데이터 분석**: 인터랙티브 차트 및 시각화
- **🔍 품질 검증**: 실시간 데이터 품질 모니터링
- **🤖 ML 예측**: 3가지 예측 모드 지원 (NEW!)
  - 🔧 **기본 앙상블**: Prophet + LSTM
  - ⚡ **고급 앙상블**: 5개 모델 + 동적 가중치
  - 🤖 **AutoML 강화**: 자동 최적화 + 앙상블 결합
- **🧠 AI 리포트**: Claude AI 기반 전문가 수준 분석 리포트
- **🌐 다중 플랫폼**: 플랫폼 간 비교 분석

### 웹 UI 장점
- ✅ **사용자 친화적**: 코딩 지식 없이도 분석 가능
- ✅ **실시간 시각화**: Plotly 기반 인터랙티브 차트
- ✅ **진행률 표시**: 데이터 수집/예측 진행 상황 실시간 확인
- ✅ **결과 다운로드**: CSV 형태로 결과 즉시 다운로드
- ✅ **반응형 디자인**: 데스크톱, 태블릿, 모바일 지원

## 🎯 사용 권장사항

### 데이터 수집
- 최소 12개월 이상의 데이터 수집 권장
- 주요 키워드는 3-5개 정도로 제한
- 정기적인 데이터 업데이트 (월 1회 권장)

### 예측 성능 최적화
- **🔧 기본 앙상블**: 빠른 예측이 필요한 경우
- **⚡ 고급 앙상블**: 높은 정확도가 중요한 경우 (권장)
- **🤖 AutoML 강화**: 최고 성능을 원하는 경우 (시간 소요)
- 예측 기간은 6-12개월 권장
- 복잡한 패턴: Transformer 모델 효과적
- 계절성 강한 데이터: Prophet + XGBoost 조합 우수
- 카테고리 특성 많은 경우: CatBoost 활용

### 모델 선택 가이드
| 상황 | 추천 모드 | 특징 | 예상 시간 |
|------|-----------|------|-----------|
| 빠른 분석 | 기본 앙상블 | 2개 모델, 빠른 실행 | 2-3분 |
| 일반 업무 | 고급 앙상블 | 5개 모델, 동적 가중치 | 5-10분 |
| 중요 의사결정 | AutoML 강화 | 자동 최적화, 최고 성능 | 10-20분 |

### 새로운 모델별 특징
- **Transformer**: 복잡한 시퀀스 패턴, 장기 의존성 학습 우수
- **XGBoost**: 다양한 특성 엔지니어링, 안정적 성능
- **CatBoost**: 카테고리 변수 자동 처리, 오버피팅 방지
- **AutoML**: 최적 모델 자동 선택, 하이퍼파라미터 자동 튜닝

### AI 리포트 활용
- API 키는 안전한 곳에 보관
- 리포트는 비즈니스 의사결정 참고자료로 활용
- 정기적인 리포트 생성으로 트렌드 모니터링

### 다음 업데이트 예정
- 🔄 NVIDIA GPU 지원 확대
- 🔍 더 많은 ML 모델 추가 (ARIMA, SARIMA, Prophet+)
- 📊 고급 대시보드 위젯
- 🔄 자동 스케줄링 기능
- 📧 이메일 알림 시스템
- 🎯 A/B 테스트 기능

**즐거운 데이터 분석 되세요! 🚀**


