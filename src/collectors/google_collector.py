import pandas as pd
import time
from ..config import Config
from ..utils.file_utils import FileManager

class GoogleTrendsCollector:
    """구글 트렌드 데이터 수집 클래스"""
    
    def __init__(self, geo='KR', tz=540):
        """초기화
        
        Args:
            geo: 지역 코드 (KR=한국, US=미국, ''=전세계)
            tz: 시간대 (540=한국시간)
        """
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='ko-KR', tz=tz)
            self.geo = geo
            print(f"구글 트렌드 수집기 초기화 완료 (지역: {geo})")
        except ImportError:
            print("❌ pytrends 패키지가 설치되지 않았습니다.")
            print("pip install pytrends를 실행해주세요.")
            self.pytrends = None
        except Exception as e:
            print(f"구글 트렌드 초기화 오류: {e}")
            self.pytrends = None
    
    def collect_trends_data(self, keywords, start_year, end_year):
        """구글 트렌드 데이터를 수집합니다"""
        if self.pytrends is None:
            print("구글 트렌드 수집기가 초기화되지 않았습니다.")
            return pd.DataFrame()
        
        print(f"\n🌐 구글 트렌드 데이터 수집 시작...")
        print(f"키워드: {', '.join(keywords)}")
        print(f"기간: {start_year}년 - {end_year}년")
        print(f"지역: {self.geo}")
        
        all_data = []
        
        try:
            # 시간 범위 설정
            timeframe = f'{start_year}-01-01 {end_year}-12-31'
            
            # 키워드를 5개씩 묶어서 처리 (구글 트렌드 API 제한)
            keyword_chunks = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
            
            for chunk_idx, keyword_chunk in enumerate(keyword_chunks):
                print(f"\n키워드 그룹 {chunk_idx + 1}: {', '.join(keyword_chunk)}")
                
                try:
                    # 구글 트렌드 데이터 요청
                    self.pytrends.build_payload(
                        keyword_chunk, 
                        cat=0,  # 모든 카테고리
                        timeframe=timeframe,
                        geo=self.geo,
                        gprop=''  # 웹 검색
                    )
                    
                    # 시간별 관심도 데이터 가져오기
                    interest_over_time_df = self.pytrends.interest_over_time()
                    
                    if not interest_over_time_df.empty:
                        # 'isPartial' 컬럼 제거 (있는 경우)
                        if 'isPartial' in interest_over_time_df.columns:
                            interest_over_time_df = interest_over_time_df.drop(columns=['isPartial'])
                        
                        # 데이터 형식 변환 (네이버 형식과 통일)
                        for keyword in keyword_chunk:
                            if keyword in interest_over_time_df.columns:
                                keyword_data = interest_over_time_df[keyword]
                                
                                for date, ratio in keyword_data.items():
                                    if pd.notna(ratio):
                                        all_data.append({
                                            'date': date,
                                            'year': date.year,
                                            'month': date.month,
                                            'keyword': keyword,
                                            'ratio': ratio,
                                            'source': 'google'
                                        })
                        
                        print(f"  ✅ 성공: {len(keyword_chunk)}개 키워드")
                    else:
                        print(f"  ❌ 데이터 없음: {', '.join(keyword_chunk)}")
                    
                    # API 호출 간격
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"  ❌ 오류: {e}")
                    continue
            
            if all_data:
                result_df = pd.DataFrame(all_data)
                result_df = result_df.sort_values(['keyword', 'date'])
                print(f"\n✅ 구글 트렌드 수집 완료: {len(result_df)}개 데이터 포인트")
                return result_df
            else:
                print("\n❌ 수집된 구글 트렌드 데이터가 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"\n❌ 구글 트렌드 수집 중 전체 오류: {e}")
            return pd.DataFrame()
    
    def get_related_keywords(self, keyword, limit=10):
        """관련 키워드를 가져옵니다"""
        if self.pytrends is None:
            return []
        
        try:
            self.pytrends.build_payload([keyword], geo=self.geo)
            related_queries = self.pytrends.related_queries()
            
            if keyword in related_queries and related_queries[keyword]['top'] is not None:
                top_related = related_queries[keyword]['top']
                return top_related['query'].head(limit).tolist()
            else:
                return []
        except Exception as e:
            print(f"관련 키워드 조회 오류: {e}")
            return []
    
    def get_regional_interest(self, keyword):
        """지역별 관심도를 가져옵니다"""
        if self.pytrends is None:
            return pd.DataFrame()
        
        try:
            self.pytrends.build_payload([keyword], geo=self.geo)
            regional_interest = self.pytrends.interest_by_region(
                resolution='REGION', 
                inc_low_vol=True, 
                inc_geo_code=True
            )
            return regional_interest
        except Exception as e:
            print(f"지역별 관심도 조회 오류: {e}")
            return pd.DataFrame() 