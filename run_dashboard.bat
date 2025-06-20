@echo off
echo ===============================================
echo     네이버 쇼핑 트렌드 웹 대시보드 시작
echo ===============================================
echo.

REM 가상환경 활성화 확인
if exist "venv\Scripts\activate.bat" (
    echo 가상환경 활성화 중...
    call venv\Scripts\activate.bat
) else (
    echo 경고: 가상환경이 설정되지 않았습니다.
    echo 'python -m venv venv' 명령으로 가상환경을 먼저 생성하세요.
    pause
    exit /b 1
)

REM 필요 패키지 확인
echo 필수 패키지 확인 중...
python -c "import streamlit; print('✅ Streamlit 설치됨')" 2>nul || (
    echo ❌ Streamlit가 설치되지 않았습니다.
    echo 패키지를 설치하시겠습니까? (Y/N)
    set /p install_choice=
    if /i "%install_choice%"=="Y" (
        echo 패키지 설치 중...
        pip install -r requirements.txt
    ) else (
        echo 패키지 설치를 취소했습니다.
        pause
        exit /b 1
    )
)

echo.
echo 🚀 웹 대시보드를 시작합니다...
echo 📱 브라우저에서 http://localhost:8501 을 열어주세요.
echo ⏹️  중단하려면 Ctrl+C를 누르세요.
echo.

REM Streamlit 앱 실행 (모듈화 버전)
streamlit run web/app.py --server.port 8501 --server.address localhost

echo.
echo 대시보드가 종료되었습니다.
pause 