@echo off
set GRADIO_TEMP_DIR=%~dp0TEMP

call conda activate GPTSoVits311

python .\tools\uvr5\webui.py "cuda" True 9872 False

pause