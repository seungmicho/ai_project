@echo off
REM ==========================================================
REM   AI 패션 비서 - Streamlit 실행기
REM   더블클릭하면 자동으로 옷장 UI가 브라우저에 열림
REM ==========================================================

REM 프로젝트 루트로 이동
cd /d "%~dp0"

REM venv 활성화 + streamlit 실행 (backend/app에서)
echo.
echo [AI 패션 비서] 옷장을 여는 중입니다...
echo 브라우저가 자동으로 열리고, 옷장 UI가 뜨면 성공!
echo (이 검은 창은 끄지 마세요 — 닫으면 앱도 꺼져요)
echo.

cd backend\app
"%~dp0.venv\Scripts\python.exe" -m streamlit run streamlit_app.py

REM 에러 시 창이 바로 닫히지 않도록 pause
echo.
echo Streamlit이 종료되었습니다.
pause
