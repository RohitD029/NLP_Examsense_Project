@echo off
title ExamSense System Diagnostic
call .venv312\Scripts\activate.bat
python scripts\doctor.py
pause
