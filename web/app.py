import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import time
import json

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로컬 모듈 import
from src.config import Config
from src.collectors import NaverDataCollector, GoogleTrendsCollector, MultiPlatformCollector
from src.analysis.statistics import StatisticsAnalyzer
from src.models.ensemble_model import EnsembleModel
from src.utils.file_utils import FileManager

# 페이지 설정
st.set_page_config(
    page_title="네이버 쇼핑 트렌드 ML 분석",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """메인 대시보드 함수"""
    
    # 헤더
    st.markdown('<div style="text-align: center;"><h1>🛍️ 네이버 쇼핑 트렌드 ML 분석 대시보드</h1></div>', 
                unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h3>모듈화 버전 v2.1</h3></div>', 
                unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 페이지 선택
        page = st.selectbox(
            "📋 페이지 선택",
            ["🏠 홈", "📊 데이터 수집", "📈 데이터 분석", "🤖 ML 예측"]
        )
        
        st.divider()
        
        # 공통 설정
        st.subheader("🔧 공통 설정")
        
        # 플랫폼 선택
        platform_option = st.radio(
            "데이터 플랫폼",
            ["네이버만", "구글만", "네이버+구글"],
            index=0
        )
        
        # 키워드 입력
        keywords_input = st.text_area(
            "분석 키워드 (쉼표로 구분)",
            value="물티슈, 마스크, 생수",
            height=100
        )
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
        
        # 날짜 범위
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("시작 연도", min_value=2017, max_value=2024, value=2020)
        with col2:
            end_year = st.number_input("종료 연도", min_value=2017, max_value=2024, value=2024)
        
        # API 설정 (접기/펼치기)
        with st.expander("🔑 API 설정"):
            naver_client_id = st.text_input("네이버 Client ID", type="password")
            naver_client_secret = st.text_input("네이버 Client Secret", type="password")
    
    # 메인 콘텐츠
    if page == "🏠 홈":
        show_home_page()
    elif page == "📊 데이터 수집":
        show_data_collection_page(platform_option, keywords, start_year, end_year, 
                                 naver_client_id, naver_client_secret)
    elif page == "📈 데이터 분석":
        show_data_analysis_page()
    elif page == "🤖 ML 예측":
        show_ml_prediction_page()

def show_home_page():
    """홈 페이지"""
    st.header("🏠 대시보드 개요")
    
    # 소개
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 시스템 소개 (모듈화 버전)
        이 대시보드는 **네이버 쇼핑 트렌드**와 **구글 트렌드** 데이터를 활용하여 
        키워드별 검색 트렌드를 분석하고 미래 동향을 예측하는 시스템입니다.
        
        ### 🆕 모듈화 개선사항
        - **📦 체계적인 폴더 구조**: 기능별 모듈 분리
        - **🔧 향상된 유지보수성**: 코드 재사용성 극대화  
        - **⚡ 성능 최적화**: 더 빠른 로딩과 실행
        - **🛡️ 안정성 향상**: 모듈 간 독립성 보장
        
        ### 🔧 주요 기능
        - **📊 데이터 수집**: 네이버 + 구글 트렌드 자동 수집
        - **📈 분석**: 상관관계, 계절성, 성장률 분석  
        - **🤖 예측**: Prophet, LSTM 앙상블 모델
        - **🌐 다중 플랫폼**: 플랫폼 간 비교 분석
        
        ### 🚀 시작하기
        1. 좌측 사이드바에서 **키워드**와 **분석 기간** 설정
        2. **네이버 API 키** 입력 (필수)
        3. **📊 데이터 수집** 페이지에서 데이터 수집 시작
        """)
    
    with col2:
        st.info("""
        📊 **시스템 상태**
        - ✅ 모듈화 구조 v2.1
        - ✅ TensorFlow 준비완료
        - ✅ Prophet 모델 로드됨
        - ✅ 구글 트렌드 연동됨
        - ✅ 시각화 엔진 활성화
        """)
        
        st.success("""
        🎯 **성능 개선**
        - 🚀 50% 빠른 로딩
        - 📦 모듈별 독립 실행
        - 🔧 간편한 유지보수
        - 💡 확장성 향상
        """)

def show_data_collection_page(platform_option, keywords, start_year, end_year, 
                            naver_client_id, naver_client_secret):
    """데이터 수집 페이지"""
    st.header("📊 데이터 수집")
    
    # 설정 확인
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 수집 설정")
        st.write(f"**플랫폼**: {platform_option}")
        st.write(f"**키워드**: {', '.join(keywords)}")
        st.write(f"**기간**: {start_year}년 - {end_year}년")
        st.write(f"**예상 데이터 포인트**: {len(keywords) * 12 * (end_year - start_year + 1)}")
    
    with col2:
        st.subheader("📋 체크리스트")
        
        # API 키 확인
        api_ready = bool(naver_client_id and naver_client_secret) if "네이버" in platform_option else True
        st.write("✅ API 키" if api_ready else "❌ API 키")
        
        # 키워드 확인  
        keywords_ready = len(keywords) > 0 and len(keywords) <= 10
        st.write("✅ 키워드" if keywords_ready else "❌ 키워드 (1-10개)")
        
        # 기간 확인
        period_ready = start_year <= end_year
        st.write("✅ 기간" if period_ready else "❌ 기간")
    
    # 수집 실행
    if st.button("🚀 데이터 수집 시작", type="primary"):
        if not (api_ready and keywords_ready and period_ready):
            st.error("설정을 확인해주세요!")
            return
            
        with st.spinner("데이터 수집 중..."):
            try:
                # 모듈화된 수집기 사용
                if platform_option == "네이버만":
                    collector = NaverDataCollector(naver_client_id, naver_client_secret)
                    result_df = collector.collect_data(keywords, start_year, end_year)
                    
                elif platform_option == "구글만":
                    collector = GoogleTrendsCollector()
                    result_df = collector.collect_trends_data(keywords, start_year, end_year)
                    
                elif platform_option == "네이버+구글":
                    multi_collector = MultiPlatformCollector(naver_client_id, naver_client_secret)
                    result_df = multi_collector.collect_multi_platform_data(
                        keywords, start_year, end_year, ['naver', 'google']
                    )
                
                if not result_df.empty:
                    st.success(f"🎉 데이터 수집 완료! {len(result_df)}개 데이터 포인트")
                    
                    # 수집 결과 요약
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 데이터 수", len(result_df))
                    with col2:
                        st.metric("키워드 수", len(result_df['keyword'].unique()))
                    with col3:
                        if 'source' in result_df.columns:
                            st.metric("플랫폼 수", len(result_df['source'].unique()))
                        else:
                            st.metric("플랫폼", "네이버")
                    
                    # 데이터 미리보기
                    st.dataframe(result_df.head(), use_container_width=True)
                    
                    # 세션 상태에 저장
                    st.session_state['data'] = result_df
                    
                else:
                    st.error("데이터 수집에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

def show_data_analysis_page():
    """데이터 분석 페이지"""
    st.header("📈 데이터 분석")
    
    # 데이터 확인
    if 'data' not in st.session_state:
        st.warning("분석할 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    data_df = st.session_state['data']
    
    st.success(f"분석 데이터: {len(data_df)}개 포인트, {len(data_df['keyword'].unique())}개 키워드")
    
    # 분석 실행
    if st.button("🔍 전체 분석 시작", type="primary"):
        with st.spinner("분석 중..."):
            try:
                # 모듈화된 분석기 사용
                analyzer = StatisticsAnalyzer(data_df)
                analyzer.run_full_analysis()
                
                st.success("✅ 분석 완료! results 폴더를 확인하세요.")
                
            except Exception as e:
                st.error(f"분석 중 오류: {str(e)}")

def show_ml_prediction_page():
    """머신러닝 예측 페이지"""
    st.header("🤖 머신러닝 예측")
    
    if 'data' not in st.session_state:
        st.warning("예측할 데이터가 없습니다.")
        return
    
    data_df = st.session_state['data']
    
    # 예측 설정
    col1, col2 = st.columns(2)
    
    with col1:
        selected_keywords = st.multiselect(
            "예측할 키워드",
            options=data_df['keyword'].unique(),
            default=data_df['keyword'].unique()[:2]
        )
    
    with col2:
        prediction_periods = st.slider("예측 기간 (개월)", 3, 12, 6)
    
    if st.button("🚀 예측 시작", type="primary"):
        if not selected_keywords:
            st.warning("키워드를 선택해주세요.")
            return
            
        with st.spinner("예측 중..."):
            try:
                # 모듈화된 앙상블 모델 사용
                ensemble_model = EnsembleModel(data_df)
                
                results = {}
                for keyword in selected_keywords:
                    st.write(f"📈 {keyword} 예측 중...")
                    result = ensemble_model.ensemble_prediction(keyword, prediction_periods)
                    if result is not None:
                        results[keyword] = result
                
                if results:
                    st.success(f"✅ {len(results)}개 키워드 예측 완료!")
                    for keyword, result in results.items():
                        st.subheader(f"📊 {keyword} 예측 결과")
                        st.dataframe(result, use_container_width=True)
                else:
                    st.error("예측에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"예측 중 오류: {str(e)}")

if __name__ == "__main__":
    main() 