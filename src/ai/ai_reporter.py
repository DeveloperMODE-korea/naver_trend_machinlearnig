"""
AI 리포트 생성 모듈

Claude AI API를 사용하여 데이터 분석 결과를 해석하고 
비즈니스 인사이트를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import os

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("⚠️  Claude AI 패키지가 설치되지 않았습니다. pip install anthropic으로 설치하세요.")

from ..config import Config

class AIReporter:
    """Claude AI 기반 리포트 생성 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """초기화
        
        Args:
            api_key: Claude API 키 (없으면 환경변수에서 가져옴)
        """
        if not CLAUDE_AVAILABLE:
            self.client = None
            return
        
        # API 키 설정
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("✅ Claude AI 초기화 완료")
            except Exception as e:
                print(f"❌ Claude AI 초기화 실패: {e}")
                self.client = None
        else:
            print("⚠️  Claude API 키가 설정되지 않았습니다.")
            self.client = None
    
    def is_available(self) -> bool:
        """AI 리포트 기능 사용 가능 여부 확인"""
        return self.client is not None
    
    def generate_data_analysis_report(self, data_df: pd.DataFrame, 
                                    statistics_results: Dict = None,
                                    correlation_matrix: pd.DataFrame = None,
                                    growth_rates: pd.DataFrame = None) -> str:
        """데이터 분석 결과를 AI가 해석하는 리포트 생성"""
        
        if not self.is_available():
            return "❌ AI 리포트 기능을 사용할 수 없습니다. Claude API 키를 확인하세요."
        
        # 데이터 요약 정보 수집
        summary_info = self._extract_data_summary(data_df, statistics_results, correlation_matrix, growth_rates)
        
        # AI 프롬프트 구성
        prompt = self._create_analysis_prompt(summary_info)
        
        try:
            # Claude AI에게 분석 요청
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            ai_report = response.content[0].text
            
            # 리포트 헤더 추가
            report_header = f"""
🤖 **Claude AI 분석 리포트**
📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 분석 데이터: {len(data_df)}개 포인트, {len(data_df['keyword'].unique())}개 키워드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            return report_header + ai_report
            
        except Exception as e:
            return f"❌ AI 리포트 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_prediction_report(self, prediction_results: Dict[str, pd.DataFrame]) -> str:
        """예측 결과를 AI가 해석하는 리포트 생성"""
        
        if not self.is_available():
            return "❌ AI 리포트 기능을 사용할 수 없습니다."
        
        # 예측 결과 요약
        prediction_summary = self._extract_prediction_summary(prediction_results)
        
        # AI 프롬프트 구성
        prompt = self._create_prediction_prompt(prediction_summary)
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            ai_report = response.content[0].text
            
            report_header = f"""
🔮 **Claude AI 예측 분석 리포트**
📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 예측 키워드: {', '.join(prediction_results.keys())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            return report_header + ai_report
            
        except Exception as e:
            return f"❌ AI 예측 리포트 생성 중 오류: {str(e)}"
    
    def generate_quality_report(self, quality_assessment: Dict, outlier_results: Dict = None) -> str:
        """데이터 품질 검증 결과를 AI가 해석하는 리포트 생성"""
        
        if not self.is_available():
            return "❌ AI 리포트 기능을 사용할 수 없습니다."
        
        # 품질 정보 요약
        quality_summary = self._extract_quality_summary(quality_assessment, outlier_results)
        
        # AI 프롬프트 구성
        prompt = self._create_quality_prompt(quality_summary)
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            ai_report = response.content[0].text
            
            report_header = f"""
🔍 **Claude AI 품질 분석 리포트**
📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💯 품질 점수: {quality_assessment['overall_score']:.1f}점 ({quality_assessment['quality_grade']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            return report_header + ai_report
            
        except Exception as e:
            return f"❌ AI 품질 리포트 생성 중 오류: {str(e)}"
    
    def generate_business_insights(self, keywords: List[str], 
                                 market_data: pd.DataFrame,
                                 industry_context: str = "일반 소비재") -> str:
        """비즈니스 인사이트 및 전략 제안 리포트 생성"""
        
        if not self.is_available():
            return "❌ AI 리포트 기능을 사용할 수 없습니다."
        
        # 시장 데이터 요약
        market_summary = self._extract_market_summary(keywords, market_data, industry_context)
        
        # AI 프롬프트 구성
        prompt = self._create_business_prompt(market_summary)
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3500,
                temperature=0.4,  # 창의적 제안을 위해 온도 약간 높임
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            ai_report = response.content[0].text
            
            report_header = f"""
💼 **Claude AI 비즈니스 인사이트 리포트**
📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏭 산업 분야: {industry_context}
🎯 분석 키워드: {', '.join(keywords)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            return report_header + ai_report
            
        except Exception as e:
            return f"❌ AI 비즈니스 리포트 생성 중 오류: {str(e)}"
    
    def _extract_data_summary(self, data_df: pd.DataFrame, 
                            statistics_results: Dict = None,
                            correlation_matrix: pd.DataFrame = None,
                            growth_rates: pd.DataFrame = None) -> Dict:
        """데이터 분석 결과 요약 정보 추출"""
        
        keywords = data_df['keyword'].unique().tolist()
        
        # 기본 통계
        basic_stats = {}
        for keyword in keywords:
            keyword_data = data_df[data_df['keyword'] == keyword]['ratio']
            basic_stats[keyword] = {
                'mean': float(keyword_data.mean()),
                'std': float(keyword_data.std()),
                'min': float(keyword_data.min()),
                'max': float(keyword_data.max()),
                'trend': 'increasing' if keyword_data.iloc[-5:].mean() > keyword_data.iloc[:5].mean() else 'decreasing'
            }
        
        # 상관관계 요약
        correlation_summary = {}
        if correlation_matrix is not None:
            for i, keyword1 in enumerate(keywords):
                for j, keyword2 in enumerate(keywords):
                    if i < j:  # 중복 제거
                        corr_value = correlation_matrix.loc[keyword1, keyword2]
                        if abs(corr_value) > 0.5:  # 강한 상관관계만
                            correlation_summary[f"{keyword1}-{keyword2}"] = float(corr_value)
        
        # 성장률 요약
        growth_summary = {}
        if growth_rates is not None:
            for _, row in growth_rates.iterrows():
                growth_summary[row['키워드']] = float(row['성장률(%)'])
        
        return {
            'keywords': keywords,
            'data_period': f"{data_df['date'].min().strftime('%Y-%m')} ~ {data_df['date'].max().strftime('%Y-%m')}",
            'total_data_points': len(data_df),
            'basic_statistics': basic_stats,
            'correlations': correlation_summary,
            'growth_rates': growth_summary
        }
    
    def _extract_prediction_summary(self, prediction_results: Dict[str, pd.DataFrame]) -> Dict:
        """예측 결과 요약 정보 추출"""
        
        summary = {}
        
        for keyword, result_df in prediction_results.items():
            predictions = result_df['앙상블 예측'].values
            confidence = result_df['신뢰도'].mean()
            
            # 트렌드 분석
            trend = 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
            volatility = np.std(predictions)
            
            summary[keyword] = {
                'predictions': predictions.tolist(),
                'average_confidence': float(confidence),
                'trend': trend,
                'volatility': float(volatility),
                'peak_month': int(np.argmax(predictions) + 1),
                'lowest_month': int(np.argmin(predictions) + 1)
            }
        
        return summary
    
    def _extract_quality_summary(self, quality_assessment: Dict, outlier_results: Dict = None) -> Dict:
        """품질 검증 결과 요약 정보 추출"""
        
        return {
            'overall_score': quality_assessment['overall_score'],
            'quality_grade': quality_assessment['quality_grade'],
            'metrics': quality_assessment['quality_metrics'],
            'alerts': [alert['message'] for alert in quality_assessment.get('alerts', [])],
            'recommendations': quality_assessment.get('recommendations', []),
            'outlier_info': outlier_results['summary'] if outlier_results else None
        }
    
    def _extract_market_summary(self, keywords: List[str], market_data: pd.DataFrame, industry_context: str) -> Dict:
        """시장 데이터 요약 정보 추출"""
        
        # 키워드별 시장 포지션 분석
        keyword_positions = {}
        for keyword in keywords:
            keyword_data = market_data[market_data['keyword'] == keyword]['ratio']
            if not keyword_data.empty:
                avg_performance = keyword_data.mean()
                recent_trend = keyword_data.iloc[-6:].mean() - keyword_data.iloc[-12:-6].mean()
                
                keyword_positions[keyword] = {
                    'average_performance': float(avg_performance),
                    'recent_trend': float(recent_trend),
                    'market_share_rank': None  # 상대적 순위는 계산 가능
                }
        
        return {
            'industry_context': industry_context,
            'keywords': keywords,
            'keyword_positions': keyword_positions,
            'analysis_period': f"{market_data['date'].min().strftime('%Y-%m')} ~ {market_data['date'].max().strftime('%Y-%m')}"
        }
    
    def _create_analysis_prompt(self, summary_info: Dict) -> str:
        """데이터 분석을 위한 AI 프롬프트 생성"""
        
        return f"""
당신은 한국 시장 전문가이자 데이터 분석가입니다. 네이버 쇼핑 트렌드 데이터 분석 결과를 바탕으로 전문적이고 실용적인 리포트를 작성해주세요.

## 분석 데이터 정보
- 분석 키워드: {', '.join(summary_info['keywords'])}
- 분석 기간: {summary_info['data_period']}
- 총 데이터 포인트: {summary_info['total_data_points']}개

## 키워드별 기본 통계
{json.dumps(summary_info['basic_statistics'], indent=2, ensure_ascii=False)}

## 주요 상관관계 (0.5 이상)
{json.dumps(summary_info['correlations'], indent=2, ensure_ascii=False)}

## 성장률 분석
{json.dumps(summary_info['growth_rates'], indent=2, ensure_ascii=False)}

다음 구조로 한국어 리포트를 작성해주세요:

## 📊 핵심 발견사항
- 가장 주목할 만한 3가지 트렌드

## 🔍 키워드별 상세 분석
- 각 키워드의 성과와 특징 분석
- 시장에서의 위치와 의미

## 🔗 키워드 간 관계 분석
- 강한 상관관계를 보이는 키워드들의 비즈니스적 의미
- 경쟁 관계 또는 보완 관계 분석

## 📈 시장 트렌드 해석
- 전체적인 시장 흐름과 패턴
- 계절성이나 특별한 이벤트의 영향

## 💡 전략적 시사점
- 각 키워드에 대한 구체적인 비즈니스 제안
- 향후 주목해야 할 키워드나 트렌드

전문적이면서도 실무진이 바로 활용할 수 있는 구체적인 인사이트를 제공해주세요.
"""
    
    def _create_prediction_prompt(self, prediction_summary: Dict) -> str:
        """예측 분석을 위한 AI 프롬프트 생성"""
        
        return f"""
당신은 한국 시장 전문가이자 미래 예측 분석가입니다. 네이버 쇼핑 트렌드 예측 결과를 바탕으로 미래 시장 전망 리포트를 작성해주세요.

## 예측 결과 요약
{json.dumps(prediction_summary, indent=2, ensure_ascii=False)}

다음 구조로 한국어 리포트를 작성해주세요:

## 🔮 예측 결과 요약
- 각 키워드의 향후 6개월 전망 한 문장 요약

## 📊 키워드별 미래 전망
- 예측된 트렌드 방향과 변동성 분석
- 신뢰도 평가 및 주의사항
- 피크 시점과 저점 시점의 비즈니스적 의미

## ⚡ 주요 기회와 위험
- 성장이 예상되는 키워드의 기회 요인
- 하락이 예상되는 키워드의 위험 요인
- 시장 변화에 대비한 전략

## 📅 월별 실행 계획
- 각 월별로 집중해야 할 키워드
- 마케팅 및 재고 관리 타이밍

## 🎯 추천 액션 아이템
- 즉시 실행 가능한 구체적 행동 계획
- 중장기 전략 방향

예측의 불확실성을 인정하면서도 실무진이 의사결정에 활용할 수 있는 명확한 가이드라인을 제공해주세요.
"""
    
    def _create_quality_prompt(self, quality_summary: Dict) -> str:
        """품질 분석을 위한 AI 프롬프트 생성"""
        
        return f"""
당신은 데이터 품질 전문가입니다. 다음 데이터 품질 검증 결과를 분석하여 개선방안을 제시해주세요.

## 품질 평가 결과
- 전체 점수: {quality_summary['overall_score']:.1f}점
- 품질 등급: {quality_summary['quality_grade']}
- 품질 메트릭: {json.dumps(quality_summary['metrics'], indent=2, ensure_ascii=False)}

## 발견된 문제점
{quality_summary['alerts']}

## 기존 권장사항
{quality_summary['recommendations']}

## 이상치 정보
{json.dumps(quality_summary['outlier_info'], indent=2, ensure_ascii=False) if quality_summary['outlier_info'] else "이상치 정보 없음"}

다음 구조로 한국어 리포트를 작성해주세요:

## 🔍 데이터 품질 진단
- 현재 데이터 품질 수준의 전반적 평가
- 비즈니스 의사결정에 미치는 영향도

## ⚠️ 주요 문제점 분석
- 각 품질 이슈의 근본 원인 분석
- 문제의 우선순위와 영향도 평가

## 🛠️ 즉시 개선 방안
- 단기간에 실행 가능한 구체적 해결책
- 각 방안의 예상 효과와 소요 시간

## 📈 장기 품질 개선 전략
- 근본적 품질 향상을 위한 시스템 개선안
- 지속적 모니터링 체계 구축 방안

## ✅ 품질 관리 체크리스트
- 일상적으로 확인해야 할 품질 지표
- 문제 발생 시 대응 절차

실무진이 바로 적용할 수 있는 구체적이고 실행 가능한 개선 방안을 제시해주세요.
"""
    
    def _create_business_prompt(self, market_summary: Dict) -> str:
        """비즈니스 인사이트를 위한 AI 프롬프트 생성"""
        
        return f"""
당신은 한국 시장 전문가이자 경영 컨설턴트입니다. 다음 시장 분석 데이터를 바탕으로 비즈니스 전략 리포트를 작성해주세요.

## 시장 분석 정보
- 산업 분야: {market_summary['industry_context']}
- 분석 키워드: {', '.join(market_summary['keywords'])}
- 분석 기간: {market_summary['analysis_period']}

## 키워드별 시장 포지션
{json.dumps(market_summary['keyword_positions'], indent=2, ensure_ascii=False)}

다음 구조로 한국어 리포트를 작성해주세요:

## 🏪 시장 환경 분석
- {market_summary['industry_context']} 시장의 현재 상황
- 주요 트렌드와 소비자 행동 변화

## 🎯 키워드별 전략 포지셔닝
- 각 키워드의 시장 내 위치와 경쟁력
- 성장 잠재력과 시장 기회 평가

## 💰 수익성 분석
- 높은 수익성이 예상되는 키워드
- 투자 대비 효과가 큰 영역

## 🚀 성장 전략 제안
- 단기 실행 가능한 마케팅 전략
- 중장기 사업 확장 방향

## ⚔️ 경쟁 대응 전략
- 경쟁사 대비 차별화 포인트
- 시장 점유율 확대 방안

## 📊 성과 측정 지표
- 전략 실행 후 모니터링할 KPI
- 성공 기준과 수정 시점

## 🔮 미래 시나리오
- 베스트/워스트 케이스 시나리오
- 각 상황별 대응 계획

{market_summary['industry_context']} 업계의 특성을 고려하여 실제 비즈니스에 바로 적용 가능한 구체적이고 실용적인 전략을 제시해주세요.
"""
    
    def save_report(self, report_content: str, report_type: str, filename: str = None) -> str:
        """AI 리포트를 파일로 저장"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_report_{report_type}_{timestamp}.md"
        
        filepath = os.path.join(Config.SAVE_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ AI 리포트 저장 완료: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ AI 리포트 저장 실패: {e}")
            return ""