@echo off
set GRADIO_TEMP_DIR=%~dp0TEMP

call conda activate gptsovits
python .\webui.py
pause