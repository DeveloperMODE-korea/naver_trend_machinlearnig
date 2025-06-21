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
from src.models.ensemble_model import EnsembleModel, AdvancedEnsembleModel
from src.utils.file_utils import FileManager
# 🆕 데이터 검증 모듈들
from src.validation.data_validator import DataValidator
from src.validation.outlier_detector import OutlierDetector
from src.validation.quality_monitor import QualityMonitor
# 🆕 AI 리포트 모듈
from src.ai.ai_reporter import AIReporter

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
            ["🏠 홈", "📊 데이터 수집", "📈 데이터 분석", "🔍 품질 검증", "🤖 ML 예측", "🧠 AI 리포트"]
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
    elif page == "🔍 품질 검증":
        show_quality_validation_page()
    elif page == "🤖 ML 예측":
        show_ml_prediction_page()
    elif page == "🧠 AI 리포트":
        show_ai_report_page()

def show_home_page():
    """홈 페이지"""
    st.header("🏠 대시보드 개요")
    
    # 소개
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 시스템 소개 (고급 모델 확장 버전)
        이 대시보드는 **네이버 쇼핑 트렌드**와 **구글 트렌드** 데이터를 활용하여 
        키워드별 검색 트렌드를 분석하고 **AI 기반 고도화된 예측**을 수행하는 시스템입니다.
        
        ### 🆕 최신 업데이트 - 모델 다양성 확장
        - **🧠 Transformer 모델**: BERT/GPT 스타일 어텐션 메커니즘
        - **📊 XGBoost/CatBoost**: 고성능 그래디언트 부스팅 모델  
        - **🔄 AutoML**: 자동 모델 선택 및 하이퍼파라미터 튜닝
        - **🎯 동적 가중치**: 성능 기반 앙상블 최적화
        - **⚖️ 적응적 예측**: 실시간 모델 성능 평가 및 조정
        
        ### 🔧 핵심 기능
        - **📊 데이터 수집**: 네이버 + 구글 트렌드 자동 수집
        - **📈 분석**: 상관관계, 계절성, 성장률 분석
        - **🔍 품질 검증**: 이상치 탐지, 데이터 품질 모니터링
        - **🤖 AI 예측**: 
          - 🔧 기본 앙상블 (Prophet + LSTM)
          - ⚡ 고급 앙상블 (5개 모델 + 동적 가중치)
          - 🤖 AutoML 강화 예측 (자동 최적화)
        - **🧠 AI 리포트**: Claude AI 기반 전문가 수준 인사이트
        - **🌐 다중 플랫폼**: 플랫폼 간 비교 분석
        
        ### 🚀 새로운 예측 모델들
        1. **🔮 Transformer**: 어텐션 메커니즘 기반 고급 시계열 예측
        2. **🌲 XGBoost**: 특성 엔지니어링 + 그래디언트 부스팅
        3. **🐱 CatBoost**: 카테고리 특성 특화 부스팅
        4. **🎯 적응적 앙상블**: 실시간 성능 평가로 최적 가중치 자동 조정
        5. **🤖 AutoML**: Optuna 기반 하이퍼파라미터 자동 최적화
        
        ### 🚀 시작하기
        1. 좌측 사이드바에서 **키워드**와 **분석 기간** 설정
        2. **네이버 API 키** 입력 (필수)
        3. **📊 데이터 수집** 페이지에서 데이터 수집 시작
        4. **🤖 ML 예측**에서 고급 예측 모드 선택
        """)
    
    with col2:
        st.info("""
        📊 **시스템 상태**
        - ✅ 모듈화 구조 v2.1
        - ✅ TensorFlow + PyTorch 준비됨
        - ✅ Prophet + LSTM 기본 모델
        - 🆕 Transformer 모델 지원
        - 🆕 XGBoost + CatBoost 지원
        - 🆕 AutoML (Optuna) 준비됨
        - ✅ 구글 트렌드 연동됨
        - ✅ 데이터 검증 모듈 활성화
        - 🧠 Claude AI 리포터 활성화
        """)
        
        st.success("""
        🎯 **최신 기능**
        - 🚀 5개 모델 앙상블 지원
        - 🎯 동적 가중치 자동 조정
        - 🤖 AutoML 하이퍼파라미터 최적화
        - 📊 실시간 성능 평가
        - 🔍 고급 특성 엔지니어링
        - 💡 카테고리 특성 자동 처리
        - ⚖️ 수학적 최적화 기반 가중치
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
    
    # 분석 옵션
    col1, col2 = st.columns(2)
    with col1:
        show_charts = st.checkbox("📊 차트 생성 및 표시", value=True)
        show_tables = st.checkbox("📋 분석 결과 테이블", value=True)
    
    with col2:
        show_interactive = st.checkbox("🔄 인터랙티브 차트", value=True)
        show_images = st.checkbox("🖼️ 저장된 이미지 표시", value=True)
    
    # 분석 실행
    if st.button("🔍 전체 분석 시작", type="primary"):
        with st.spinner("분석 중..."):
            try:
                # 모듈화된 분석기 사용
                analyzer = StatisticsAnalyzer(data_df)
                
                # 기본 통계 분석
                stats_df, yearly_avg = analyzer.basic_statistics()
                
                # 상관관계 분석
                correlation = analyzer.correlation_analysis()
                
                # 성장률 분석
                growth_df = analyzer.growth_rate_analysis()
                
                # 월별 분석
                monthly_top = analyzer.monthly_analysis()
                
                # 다중 플랫폼 분석 (해당되는 경우)
                analyzer.multi_platform_analysis()
                
                st.success("✅ 분석 완료!")
                
                # 결과 표시
                if show_tables:
                    st.subheader("📊 분석 결과 테이블")
                    
                    # 기본 통계
                    if stats_df is not None:
                        st.write("**📈 키워드별 기본 통계**")
                        st.dataframe(stats_df, use_container_width=True)
                    
                    # 성장률
                    if growth_df is not None:
                        st.write("**📈 키워드별 성장률**")
                        st.dataframe(growth_df, use_container_width=True)
                    
                    # 상관관계
                    if correlation is not None:
                        st.write("**🔗 키워드 간 상관관계**")
                        st.dataframe(correlation, use_container_width=True)
                
                # 인터랙티브 차트 생성
                if show_interactive:
                    st.subheader("🔄 인터랙티브 차트")
                    
                    # 전체 트렌드 차트
                    if not data_df.empty:
                        fig_trend = px.line(
                            data_df, 
                            x='date', 
                            y='ratio', 
                            color='keyword',
                            title='📈 키워드별 검색 트렌드',
                            labels={'ratio': '검색 비율', 'date': '날짜'}
                        )
                        fig_trend.update_layout(height=500)
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # 상관관계 히트맵
                    if correlation is not None and not correlation.empty:
                        fig_corr = px.imshow(
                            correlation,
                            text_auto=True,
                            aspect="auto",
                            title="🔗 키워드 간 상관관계 히트맵",
                            color_continuous_scale="RdBu_r"
                        )
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # 성장률 바차트
                    if growth_df is not None and not growth_df.empty:
                        fig_growth = px.bar(
                            growth_df,
                            x='키워드',
                            y='성장률(%)',
                            title="📊 키워드별 성장률",
                            color='성장률(%)',
                            color_continuous_scale="RdYlGn"
                        )
                        fig_growth.update_layout(height=400)
                        st.plotly_chart(fig_growth, use_container_width=True)
                    
                    # 월별 패턴 분석
                    if not data_df.empty:
                        monthly_avg = data_df.groupby(['month', 'keyword'])['ratio'].mean().reset_index()
                        fig_monthly = px.line(
                            monthly_avg,
                            x='month',
                            y='ratio',
                            color='keyword',
                            title="📅 월별 평균 검색 비율 (계절성)",
                            labels={'ratio': '평균 검색 비율', 'month': '월'}
                        )
                        fig_monthly.update_layout(height=400)
                        st.plotly_chart(fig_monthly, use_container_width=True)
                
                # 저장된 이미지 표시
                if show_images:
                    st.subheader("🖼️ 분석 결과 이미지")
                    
                    plot_dir = os.path.join(Config.SAVE_DIR, 'plots')
                    
                    if os.path.exists(plot_dir):
                        # 이미지 파일 목록
                        image_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
                        
                        if image_files:
                            # 이미지를 2열로 표시
                            cols = st.columns(2)
                            
                            for idx, image_file in enumerate(image_files):
                                image_path = os.path.join(plot_dir, image_file)
                                
                                with cols[idx % 2]:
                                    # 파일명에서 제목 생성
                                    title = image_file.replace('.png', '').replace('_', ' ').title()
                                    st.write(f"**{title}**")
                                    
                                    try:
                                        st.image(image_path, use_column_width=True)
                                    except Exception as e:
                                        st.error(f"이미지 로드 실패: {image_file}")
                        else:
                            st.info("생성된 이미지가 없습니다.")
                    else:
                        st.info("plots 폴더가 없습니다.")
                
                # 데이터 다운로드 옵션
                st.subheader("💾 결과 다운로드")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if stats_df is not None:
                        csv_stats = stats_df.to_csv(index=False)
                        st.download_button(
                            label="📊 기본 통계 CSV",
                            data=csv_stats,
                            file_name="basic_statistics.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if correlation is not None:
                        csv_corr = correlation.to_csv()
                        st.download_button(
                            label="🔗 상관관계 CSV",
                            data=csv_corr,
                            file_name="correlation_matrix.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    if growth_df is not None:
                        csv_growth = growth_df.to_csv(index=False)
                        st.download_button(
                            label="📈 성장률 CSV",
                            data=csv_growth,
                            file_name="growth_rate.csv",
                            mime="text/csv"
                        )
                
                # 세션에 분석 결과 저장 (AI 리포트에서 사용)
                st.session_state['analysis_results'] = {
                    'basic_statistics': stats_df,
                    'correlation_matrix': correlation,
                    'growth_rates': growth_df,
                    'monthly_analysis': monthly_top
                }
                
                st.info("💡 분석 결과가 저장되었습니다. 🧠 AI 리포트 페이지에서 전문가 분석을 확인하세요!")
                
            except Exception as e:
                st.error(f"분석 중 오류: {str(e)}")
                import traceback
                st.error(f"상세 오류: {traceback.format_exc()}")

def show_quality_validation_page():
    """데이터 품질 검증 페이지"""
    st.header("🔍 데이터 품질 검증")
    
    # 데이터 확인
    if 'data' not in st.session_state:
        st.warning("검증할 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    data_df = st.session_state['data']
    
    st.success(f"검증 데이터: {len(data_df)}개 포인트, {len(data_df['keyword'].unique())}개 키워드")
    
    # 검증 옵션
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 검증 옵션")
        
        validate_outliers = st.checkbox("이상치 탐지", value=True)
        validate_quality = st.checkbox("품질 평가", value=True)
        clean_data = st.checkbox("데이터 정리", value=False)
        
        if clean_data:
            clean_method = st.selectbox(
                "정리 방법",
                ["interpolate", "remove", "winsorize"],
                index=0
            )
    
    with col2:
        st.subheader("📊 품질 임계값")
        
        completeness_threshold = st.slider("완성도 최소값 (%)", 70, 100, 85)
        validity_threshold = st.slider("유효성 최소값 (%)", 80, 100, 90)
        consistency_threshold = st.slider("일관성 최소값 (%)", 80, 100, 95)
        sufficiency_threshold = st.slider("충분성 최소값 (%)", 50, 100, 70)
        outlier_threshold = st.slider("이상치 최대값 (%)", 1, 20, 5)
    
    # 검증 실행
    if st.button("🚀 품질 검증 시작", type="primary"):
        
        # 검증 모듈 초기화
        validator = DataValidator()
        outlier_detector = OutlierDetector()
        
        # 사용자 정의 임계값으로 품질 모니터 초기화
        quality_thresholds = {
            'completeness_min': completeness_threshold,
            'validity_min': validity_threshold,
            'consistency_min': consistency_threshold,
            'sufficiency_min': sufficiency_threshold,
            'outlier_max': outlier_threshold
        }
        quality_monitor = QualityMonitor(quality_thresholds)
        
        with st.spinner("품질 검증 중..."):
            try:
                # 1. 원시 데이터 검증
                st.subheader("📋 원시 데이터 검증")
                validated_df = validator.validate_raw_data(data_df)
                
                # 2. 이상치 탐지
                if validate_outliers:
                    st.subheader("🔍 이상치 탐지")
                    outlier_results = outlier_detector.detect_all_outliers(validated_df)
                    
                    # 이상치 요약 표시
                    summary = outlier_results['summary']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("분석된 키워드", f"{summary['analyzed_keywords']}/{summary['total_keywords']}")
                    with col2:
                        st.metric("발견된 이상치", summary['total_consensus_outliers'])
                    with col3:
                        st.metric("이상치 비율", f"{summary['outlier_percentage']:.2f}%")
                    
                    # 키워드별 이상치 상세 정보
                    if st.expander("키워드별 이상치 상세"):
                        for keyword, result in outlier_results['results'].items():
                            if result.get('status') != 'insufficient_data':
                                consensus_count = result['consensus_outliers']['count']
                                if consensus_count > 0:
                                    st.write(f"**{keyword}**: {consensus_count}개 이상치")
                                    outlier_values = result['consensus_outliers']['values']
                                    st.write(f"  이상치 값: {outlier_values}")
                else:
                    outlier_results = None
                
                # 3. 종합 품질 평가
                if validate_quality:
                    st.subheader("📊 종합 품질 평가")
                    
                    quality_assessment = quality_monitor.assess_data_quality(
                        validated_df, 
                        outlier_results=outlier_results
                    )
                    
                    # 세션에 품질 검증 결과 저장 (AI 리포트에서 사용)
                    st.session_state['quality_assessment'] = quality_assessment
                    st.session_state['outlier_results'] = outlier_results
                    
                    # 품질 점수 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        score = quality_assessment['overall_score']
                        grade = quality_assessment['quality_grade']
                        
                        # 점수에 따른 색상 결정
                        if score >= 90:
                            score_color = "green"
                        elif score >= 80:
                            score_color = "orange"
                        elif score >= 70:
                            score_color = "yellow"
                        else:
                            score_color = "red"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {score_color}20;">
                            <h2 style="color: {score_color};">품질 점수</h2>
                            <h1 style="color: {score_color};">{score:.1f}점</h1>
                            <h3>{grade}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # 품질 메트릭 표시
                        metrics = quality_assessment['quality_metrics']
                        thresholds = quality_assessment['passed_thresholds']
                        
                        st.write("**품질 메트릭:**")
                        
                        # 메트릭 순서 정의 (보기 좋게)
                        metric_names = {
                            'completeness': '완성도',
                            'validity': '유효성',
                            'consistency': '일관성',
                            'sufficiency': '충분성',
                            'freshness': '신선도',
                            'balance': '균형성',
                            'outlier_percentage': '이상치 비율'
                        }
                        
                        for metric, korean_name in metric_names.items():
                            if metric in metrics:
                                value = metrics[metric]
                                status = "✅" if thresholds.get(metric, True) else "❌"
                                
                                # 이상치는 낮을수록 좋음
                                if metric == 'outlier_percentage':
                                    st.write(f"{status} **{korean_name}**: {value:.1f}% (낮을수록 좋음)")
                                else:
                                    st.write(f"{status} **{korean_name}**: {value:.1f}%")
                    
                    # 경고사항
                    if quality_assessment['alerts']:
                        st.subheader("⚠️ 경고사항")
                        for alert in quality_assessment['alerts']:
                            severity_colors = {"high": "error", "medium": "warning", "low": "info"}
                            severity_color = severity_colors.get(alert['severity'], 'info')
                            
                            with st.container():
                                if severity_color == "error":
                                    st.error(alert['message'])
                                elif severity_color == "warning":
                                    st.warning(alert['message'])
                                else:
                                    st.info(alert['message'])
                    
                    # 개선 권장사항
                    if quality_assessment['recommendations']:
                        st.subheader("💡 개선 권장사항")
                        for rec in quality_assessment['recommendations']:
                            st.write(rec)
                
                # 4. 데이터 정리 (선택한 경우)
                if clean_data and outlier_results and outlier_results['summary']['total_consensus_outliers'] > 0:
                    st.subheader("🧹 데이터 정리")
                    
                    cleaned_df = outlier_detector.clean_outliers(validated_df, method=clean_method)
                    
                    # 정리 결과 요약
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("원본 데이터", len(validated_df))
                    with col2:
                        st.metric("정리된 데이터", len(cleaned_df))
                    
                    # 세션에 정리된 데이터 저장
                    if st.button("정리된 데이터 적용"):
                        st.session_state['data'] = cleaned_df
                        st.success("정리된 데이터가 적용되었습니다!")
                
                st.success("✅ 품질 검증 완료!")
                st.info("💡 품질 검증 결과가 저장되었습니다. 🧠 AI 리포트 페이지에서 전문가 분석을 확인하세요!")
                
            except Exception as e:
                st.error(f"품질 검증 중 오류: {str(e)}")

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
        
        prediction_periods = st.slider("예측 기간 (개월)", 3, 12, 6)
    
    with col2:
        st.subheader("🚀 예측 모드 선택")
        
        prediction_mode = st.radio(
            "예측 방법",
            ["🔧 기본 앙상블", "⚡ 고급 앙상블 (동적 가중치)", "🤖 AutoML 강화 예측"],
            index=1
        )
        
        if prediction_mode == "🤖 AutoML 강화 예측":
            st.info("🔄 최적 모델 자동 선택 및 하이퍼파라미터 튜닝")
            automl_trials = st.selectbox("AutoML 시도 횟수", [10, 20, 30, 50], index=1)
        
        if prediction_mode == "⚡ 고급 앙상블 (동적 가중치)":
            st.info("🎯 모델 성능 기반 동적 가중치 조정")
            
    # 고급 옵션
    with st.expander("🔧 고급 옵션"):
        show_individual_models = st.checkbox("개별 모델 예측값 표시", value=True)
        show_confidence_intervals = st.checkbox("신뢰구간 표시", value=True)
        show_model_weights = st.checkbox("모델 가중치 표시", value=False)
    
    if st.button("🚀 예측 시작", type="primary"):
        if not selected_keywords:
            st.warning("키워드를 선택해주세요.")
            return
            
        with st.spinner("예측 중..."):
            try:
                results = {}
                
                for keyword in selected_keywords:
                    st.write(f"📈 {keyword} 예측 중...")
                    
                    # 키워드별 데이터 추출
                    keyword_data = data_df[data_df['keyword'] == keyword].copy()
                    
                    if prediction_mode == "🔧 기본 앙상블":
                        # 기본 앙상블 모델 사용
                        ensemble_model = EnsembleModel(data_df)
                        result = ensemble_model.ensemble_prediction(keyword, prediction_periods)
                        
                    elif prediction_mode == "⚡ 고급 앙상블 (동적 가중치)":
                        # 고급 앙상블 모델 사용
                        advanced_model = AdvancedEnsembleModel(data_df, enable_automl=False)
                        result = advanced_model.adaptive_ensemble_prediction(
                            keyword_data, keyword, prediction_periods
                        )
                        
                        # 앙상블 인사이트 표시
                        if show_model_weights:
                            insights = advanced_model.get_ensemble_insights()
                            st.info("🎯 앙상블 인사이트:\n" + "\n".join(insights))
                    
                    elif prediction_mode == "🤖 AutoML 강화 예측":
                        # AutoML 강화 예측
                        advanced_model = AdvancedEnsembleModel(data_df, enable_automl=True)
                        result = advanced_model.run_automl_enhanced_prediction(
                            keyword_data, keyword, prediction_periods
                        )
                    
                    if result is not None:
                        results[keyword] = result
                
                if results:
                    st.success(f"✅ {len(results)}개 키워드 예측 완료!")
                    
                    for keyword, result in results.items():
                        st.subheader(f"📊 {keyword} 예측 결과")
                        
                        # 기본 결과 표시 (중복 제거)
                        display_columns = ['날짜', '앙상블 예측']
                        
                        if show_confidence_intervals and '신뢰도' in result.columns:
                            display_columns.append('신뢰도')
                        
                        if 'AutoML 강화 예측' in result.columns:
                            display_columns.append('AutoML 강화 예측')
                        
                        # 개별 모델 예측값 포함 (중복 제거)
                        if show_individual_models:
                            model_cols = [col for col in result.columns 
                                        if '예측' in col and col not in display_columns]
                            display_columns.extend(model_cols)
                        
                        # 메타 정보 (중복 제거)
                        meta_cols = ['사용된 모델', '주요 모델', 'AutoML 모델']
                        for col in meta_cols:
                            if col in result.columns and col not in display_columns:
                                display_columns.append(col)
                        
                        # 중복 제거 (최종 안전장치)
                        display_columns = list(dict.fromkeys(display_columns))
                        
                        # 결과 표시
                        st.dataframe(
                            result[display_columns].round(2), 
                            use_container_width=True
                        )
                        
                        # 시각화 (Plotly 차트)
                        try:
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            
                            # 앙상블 예측
                            fig.add_trace(go.Scatter(
                                x=result['날짜'],
                                y=result['앙상블 예측'],
                                mode='lines+markers',
                                name='앙상블 예측',
                                line=dict(color='blue', width=3)
                            ))
                            
                            # AutoML 예측 (있는 경우)
                            if 'AutoML 강화 예측' in result.columns:
                                fig.add_trace(go.Scatter(
                                    x=result['날짜'],
                                    y=result['AutoML 강화 예측'],
                                    mode='lines+markers',
                                    name='AutoML 강화 예측',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                            
                            # 개별 모델 예측 (옵션)
                            if show_individual_models:
                                colors = ['green', 'orange', 'purple', 'brown', 'pink']
                                for i, col in enumerate([c for c in result.columns if '예측' in c and c not in ['앙상블 예측', 'AutoML 강화 예측']]):
                                    fig.add_trace(go.Scatter(
                                        x=result['날짜'],
                                        y=result[col],
                                        mode='lines',
                                        name=col,
                                        line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                                        opacity=0.7
                                    ))
                            
                            fig.update_layout(
                                title=f'{keyword} 예측 결과',
                                xaxis_title='날짜',
                                yaxis_title='검색 비율',
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except ImportError:
                            st.info("시각화를 위해서는 plotly 패키지가 필요합니다.")
                    
                    # 세션에 예측 결과 저장 (AI 리포트에서 사용)
                    st.session_state['prediction_results'] = results
                    st.session_state['prediction_mode'] = prediction_mode
                    
                    # 결과 요약
                    st.subheader("📋 예측 요약")
                    
                    summary_data = []
                    for keyword, result in results.items():
                        avg_prediction = result['앙상블 예측'].mean()
                        trend = "상승" if result['앙상블 예측'].iloc[-1] > result['앙상블 예측'].iloc[0] else "하락"
                        
                        if 'AutoML 강화 예측' in result.columns:
                            automl_avg = result['AutoML 강화 예측'].mean()
                            summary_data.append({
                                '키워드': keyword,
                                '앙상블 평균': f"{avg_prediction:.2f}",
                                'AutoML 평균': f"{automl_avg:.2f}",
                                '트렌드': trend,
                                '평균 신뢰도': f"{result['신뢰도'].mean():.1f}%" if '신뢰도' in result.columns else "N/A"
                            })
                        else:
                            summary_data.append({
                                '키워드': keyword,
                                '평균 예측값': f"{avg_prediction:.2f}",
                                '트렌드': trend,
                                '평균 신뢰도': f"{result['신뢰도'].mean():.1f}%" if '신뢰도' in result.columns else "N/A"
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    st.info("💡 예측 결과가 저장되었습니다. 🧠 AI 리포트 페이지에서 전문가 분석을 확인하세요!")
                else:
                    st.error("예측에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"예측 중 오류: {str(e)}")
                
                # 일반적인 오류 원인과 해결 방법 안내
                with st.expander("🔧 오류 해결 방법"):
                    st.markdown("""
                    **일반적인 오류와 해결 방법:**
                    
                    1. **중복 컬럼명 오류**: 페이지를 새로고침하고 다시 시도하세요.
                    
                    2. **모델 타입 지원 오류**: 
                       - Transformer/XGBoost/CatBoost 모델이 실패해도 Prophet+LSTM은 정상 작동합니다.
                       - 기본 앙상블 모드를 사용해보세요.
                    
                    3. **데이터 부족 오류**: 
                       - 더 많은 키워드 데이터를 수집하거나
                       - 분석 기간을 늘려보세요.
                    
                    4. **텐서 크기 불일치**: 
                       - Transformer 모델의 알려진 이슈입니다.
                       - 고급 앙상블에서 Prophet+LSTM만 사용됩니다.
                    
                    5. **패키지 설치 문제**: 
                       ```bash
                       pip install -r requirements.txt
                       ```
                    
                    **권장사항**: 
                    - 🔧 기본 앙상블 모드는 항상 안정적으로 작동합니다.
                    - ⚡ 고급 앙상블은 Prophet+LSTM 조합으로 작동합니다.
                    """)
                
                # 기본 앙상블로 재시도 제안
                if prediction_mode != "🔧 기본 앙상블":
                    st.info("💡 문제가 지속되면 '🔧 기본 앙상블' 모드를 시도해보세요. 이 모드는 Prophet+LSTM만 사용하여 안정적으로 작동합니다.")

def show_ai_report_page():
    """AI 리포트 페이지"""
    st.header("🧠 Claude AI 전문가 리포트")
    
    # 데이터 확인
    if 'data' not in st.session_state:
        st.warning("리포트를 생성할 데이터가 없습니다. 먼저 데이터를 수집해주세요.")
        return
    
    data_df = st.session_state['data']
    
    # AI 리포터 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.success(f"분석 데이터: {len(data_df)}개 포인트, {len(data_df['keyword'].unique())}개 키워드")
        
        # 산업 분야 선택
        industry_context = st.selectbox(
            "🏭 산업 분야",
            ["일반 소비재", "식품/음료", "생활용품", "화장품/뷰티", "패션/의류", "전자제품", "건강/의료", "기타"],
            index=0
        )
        
        # 리포트 유형 선택
        report_types = st.multiselect(
            "📄 생성할 리포트 유형",
            ["📊 데이터 분석 리포트", "🔮 예측 분석 리포트", "🔍 품질 분석 리포트", "💼 비즈니스 인사이트"],
            default=["📊 데이터 분석 리포트"]
        )
    
    with col2:
        st.info("""
        🧠 **Claude AI 리포터**
        - 전문가 수준의 분석
        - 비즈니스 인사이트 제공
        - 실행 가능한 권장사항
        - 한국 시장 특화 해석
        """)
        
        # API 키 설정
        claude_api_key = st.text_input(
            "🔑 Claude API Key", 
            type="password",
            help="Claude AI API 키를 입력하세요. Anthropic 웹사이트에서 발급받을 수 있습니다."
        )
    
    # AI 리포트 생성
    if st.button("🚀 AI 리포트 생성", type="primary"):
        if not claude_api_key:
            st.error("Claude API 키를 입력해주세요!")
            return
        
        if not report_types:
            st.error("생성할 리포트 유형을 선택해주세요!")
            return
        
        try:
            # AI 리포터 초기화
            ai_reporter = AIReporter(api_key=claude_api_key)
            
            if not ai_reporter.is_available():
                st.error("Claude AI 서비스에 연결할 수 없습니다. API 키를 확인해주세요.")
                return
            
            # 각 리포트 유형별로 생성
            for report_type in report_types:
                with st.expander(f"{report_type}", expanded=True):
                    with st.spinner(f"{report_type} 생성 중..."):
                        
                        if report_type == "📊 데이터 분석 리포트":
                            # 기본 분석 수행
                            analyzer = StatisticsAnalyzer(data_df)
                            
                            # 상관관계 분석
                            pivot_df = data_df.pivot_table(
                                index='date', columns='keyword', values='ratio'
                            ).fillna(0)
                            correlation_matrix = pivot_df.corr() if not pivot_df.empty else None
                            
                            # 성장률 분석
                            keywords = data_df['keyword'].unique()
                            growth_rates = []
                            for keyword in keywords:
                                keyword_data = data_df[data_df['keyword'] == keyword].sort_values('date')
                                if len(keyword_data) >= 12:
                                    first_year = keyword_data.iloc[:12]['ratio'].mean()
                                    last_year = keyword_data.iloc[-12:]['ratio'].mean()
                                    growth_rate = ((last_year - first_year) / first_year) * 100 if first_year > 0 else 0
                                    growth_rates.append({'키워드': keyword, '성장률(%)': growth_rate})
                            
                            growth_df = pd.DataFrame(growth_rates) if growth_rates else None
                            
                            # AI 리포트 생성
                            report = ai_reporter.generate_data_analysis_report(
                                data_df, 
                                correlation_matrix=correlation_matrix,
                                growth_rates=growth_df
                            )
                            
                            st.markdown(report)
                            
                            # 저장 옵션
                            if st.button(f"💾 {report_type} 저장", key=f"save_analysis"):
                                saved_path = ai_reporter.save_report(report, "analysis")
                                if saved_path:
                                    st.success(f"리포트가 저장되었습니다: {saved_path}")
                        
                        elif report_type == "🔮 예측 분석 리포트":
                            # 예측 결과가 세션에 있는지 확인
                            if 'prediction_results' in st.session_state:
                                prediction_results = st.session_state['prediction_results']
                                
                                report = ai_reporter.generate_prediction_report(prediction_results)
                                st.markdown(report)
                                
                                if st.button(f"💾 {report_type} 저장", key=f"save_prediction"):
                                    saved_path = ai_reporter.save_report(report, "prediction")
                                    if saved_path:
                                        st.success(f"리포트가 저장되었습니다: {saved_path}")
                            else:
                                st.warning("예측 결과가 없습니다. 먼저 ML 예측을 실행해주세요.")
                        
                        elif report_type == "🔍 품질 분석 리포트":
                            # 품질 검증 결과가 세션에 있는지 확인
                            if 'quality_assessment' in st.session_state:
                                quality_assessment = st.session_state['quality_assessment']
                                outlier_results = st.session_state.get('outlier_results')
                                
                                report = ai_reporter.generate_quality_report(quality_assessment, outlier_results)
                                st.markdown(report)
                                
                                if st.button(f"💾 {report_type} 저장", key=f"save_quality"):
                                    saved_path = ai_reporter.save_report(report, "quality")
                                    if saved_path:
                                        st.success(f"리포트가 저장되었습니다: {saved_path}")
                            else:
                                st.warning("품질 검증 결과가 없습니다. 먼저 품질 검증을 실행해주세요.")
                        
                        elif report_type == "💼 비즈니스 인사이트":
                            keywords = data_df['keyword'].unique().tolist()
                            
                            report = ai_reporter.generate_business_insights(
                                keywords, data_df, industry_context
                            )
                            st.markdown(report)
                            
                            if st.button(f"💾 {report_type} 저장", key=f"save_business"):
                                saved_path = ai_reporter.save_report(report, "business")
                                if saved_path:
                                    st.success(f"리포트가 저장되었습니다: {saved_path}")
            
            st.success("✅ AI 리포트 생성 완료!")
            
        except Exception as e:
            st.error(f"AI 리포트 생성 중 오류: {str(e)}")
            st.info("API 키가 올바른지, 그리고 Claude API 서비스가 정상인지 확인해주세요.")

if __name__ == "__main__":
    main() 