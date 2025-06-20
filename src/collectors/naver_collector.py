import requests
import json
import pandas as pd
import time
import os
from datetime import datetime
from ..config import Config
from ..utils.common_utils import CommonUtils
from ..utils.file_utils import FileManager

class NaverDataCollector:
    """네이버 쇼핑 검색 트렌드 데이터 수집 클래스"""
    
    def __init__(self, client_id=None, client_secret=None):
        """초기화"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.headers = None
        
        if client_id and client_secret:
            self.setup_headers()
    
    def setup_credentials(self):
        """API 인증 정보를 설정합니다"""
        print("네이버 API 인증 정보를 입력해주세요.")
        self.client_id = CommonUtils.get_user_input("네이버 API client_id를 입력하세요")
        self.client_secret = CommonUtils.get_user_input("네이버 API client_secret을 입력하세요")
        self.setup_headers()
    
    def setup_headers(self):
        """API 헤더를 설정합니다"""
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json"
        }
    
    def get_keywords_from_user(self):
        """사용자로부터 키워드 입력을 받습니다"""
        print("분석할 키워드를 입력해주세요. 쉼표(,)로 구분하여 입력하세요.")
        print("예시: 물티슈, 기저귀, 커피, 마스크, 생수")
        
        user_input = CommonUtils.get_user_input("키워드 입력")
        keywords = [keyword.strip() for keyword in user_input.split(',')]
        
        # 키워드 개수 제한
        if len(keywords) == 0:
            print("키워드가 입력되지 않았습니다. 기본 키워드를 사용합니다.")
            keywords = ['물티슈', '기저귀', '커피', '마스크', '생수']
        elif len(keywords) > 10:
            print("입력된 키워드가 너무 많습니다. 처음 10개의 키워드만 사용합니다.")
            keywords = keywords[:10]
        
        print(f"분석할 키워드 ({len(keywords)}개): {', '.join(keywords)}")
        return keywords
    
    def get_date_range_from_user(self):
        """사용자로부터 날짜 범위를 입력받습니다"""
        start_year = CommonUtils.get_user_input("분석 시작 연도를 입력하세요", 2017, int)
        end_year = CommonUtils.get_user_input("분석 종료 연도를 입력하세요", 2024, int)
        return start_year, end_year
    
    def check_saved_data(self, keywords):
        """저장된 데이터가 있는지 확인합니다"""
        try:
            result_df = FileManager.load_file(Config.DATA_FILE)
            if result_df is not None and not result_df.empty:
                print(f"이전에 저장된 데이터가 발견되었습니다.")
                
                use_saved = CommonUtils.get_user_input("이 데이터를 사용하시겠습니까?", "y", bool)
                
                if use_saved:
                    saved_keywords = result_df['keyword'].unique()
                    print(f"불러온 데이터의 키워드: {', '.join(saved_keywords)}")
                    
                    new_keywords = [k for k in keywords if k not in saved_keywords]
                    if new_keywords:
                        print(f"새로운 키워드 발견: {', '.join(new_keywords)}")
                        collect_new = CommonUtils.get_user_input("새 키워드에 대한 데이터를 수집하시겠습니까?", "y", bool)
                        return result_df, new_keywords if collect_new else []
                    
                    return result_df, []
        except Exception as e:
            print(f"저장된 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        
        return pd.DataFrame(), keywords
    
    def collect_data_for_period(self, keyword, start_date, end_date):
        """특정 기간의 데이터를 수집합니다"""
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
            "device": "",
            "ages": [],
            "gender": ""
        }
        
        try:
            response = requests.post(
                Config.API_URL, 
                headers=self.headers, 
                data=json.dumps(body),
                timeout=Config.API_TIMEOUT
            )
            response_data = response.json()
            
            if 'results' in response_data and len(response_data['results']) > 0:
                data_points = []
                for item in response_data['results'][0]['data']:
                    period = item['period']
                    ratio = item['ratio']
                    
                    try:
                        year_month = period.split('-')
                        year_val = int(year_month[0])
                        month_val = int(year_month[1])
                        date_obj = datetime(year_val, month_val, 1)
                        
                        data_points.append({
                            'date': date_obj,
                            'year': date_obj.year,
                            'month': date_obj.month,
                            'keyword': keyword,
                            'ratio': ratio
                        })
                    except Exception as e:
                        print(f"날짜 변환 오류: {period}, 오류: {e}")
                
                return data_points
            else:
                print(f"키워드 '{keyword}'에 대한 데이터를 찾을 수 없습니다")
                return []
                
        except Exception as e:
            print(f"API 요청 오류: {e}")
            if 'response' in locals():
                print(f"응답 내용: {response.text}")
            return []
    
    def collect_data(self, keywords, start_year, end_year, existing_df=None):
        """전체 데이터를 수집합니다"""
        if not self.headers:
            print("API 인증 정보가 설정되지 않았습니다.")
            return pd.DataFrame()
        
        if existing_df is None:
            result_df = pd.DataFrame(columns=['date', 'year', 'month', 'keyword', 'ratio'])
        else:
            result_df = existing_df.copy()
        
        print(f"\n데이터 수집을 시작합니다...")
        print(f"키워드: {', '.join(keywords)}")
        print(f"기간: {start_year}년 - {end_year}년")
        
        current_date = datetime.now()
        total_requests = 0
        successful_requests = 0
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 현재 날짜 이후는 건너뛰기
                if year > current_date.year or (year == current_date.year and month > current_date.month):
                    continue
                
                # 시작일과 종료일 설정
                if month == 12:
                    start_date = f"{year}-{month:02d}-01"
                    end_date = f"{year+1}-01-01"
                else:
                    start_date = f"{year}-{month:02d}-01"
                    end_date = f"{year}-{month+1:02d}-01"
                
                print(f"데이터 수집 중: {start_date} ~ {end_date}")
                
                for keyword in keywords:
                    total_requests += 1
                    
                    # 기존 데이터에 이미 있는지 확인
                    if not result_df.empty and 'year' in result_df.columns:
                        existing_data = result_df[
                            (result_df['year'] == year) & 
                            (result_df['month'] == month) & 
                            (result_df['keyword'] == keyword)
                        ]
                        
                        if not existing_data.empty:
                            print(f"  {keyword}: 이미 존재하는 데이터")
                            successful_requests += 1
                            continue
                    
                    data_points = self.collect_data_for_period(keyword, start_date, end_date)
                    
                    if data_points:
                        new_rows = pd.DataFrame(data_points)
                        result_df = pd.concat([result_df, new_rows], ignore_index=True)
                        successful_requests += 1
                        print(f"  {keyword}: 성공 ({len(data_points)}개 데이터 포인트)")
                    else:
                        print(f"  {keyword}: 실패")
                    
                    # API 호출 간격
                    time.sleep(Config.API_DELAY)
        
        print(f"\n데이터 수집 완료!")
        print(f"총 요청: {total_requests}, 성공: {successful_requests}")
        
        # 데이터 정렬 및 정리
        if not result_df.empty:
            result_df['ratio'] = pd.to_numeric(result_df['ratio'], errors='coerce')
            
            # NaN 값 제거
            nan_count = result_df['ratio'].isna().sum()
            if nan_count > 0:
                print(f"경고: {nan_count}개의 NaN 값이 발견되어 제거되었습니다")
                result_df = result_df.dropna(subset=['ratio'])
            
            result_df = result_df.sort_values(by=['keyword', 'date'])
            
            # 데이터 저장
            FileManager.save_file(result_df, Config.DATA_FILE)
            
            print(f"수집된 데이터: {len(result_df)}개 행")
            print(f"키워드별 데이터 개수:")
            for keyword in result_df['keyword'].unique():
                count = len(result_df[result_df['keyword'] == keyword])
                print(f"  {keyword}: {count}개")
        
        return result_df 