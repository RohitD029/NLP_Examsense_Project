@echo off
title ExamSense - RAG Study Assistant
echo ===================================================
echo   Starting ExamSense (Streamlit)...
echo ===================================================
call .venv312\Scripts\activate.bat
streamlit run app.py
pause
