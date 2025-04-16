@echo off
set GRADIO_TEMP_DIR=%~dp0TEMP

call conda activate GPTSoVits311
python .\api_fox_ui.py
pause