"""
웹 인터페이스 모듈

Streamlit 기반 웹 대시보드와 관련 컴포넌트들을 관리합니다.
"""

from .app import main as run_web_dashboard

__all__ = [
    'run_web_dashboard',
] 