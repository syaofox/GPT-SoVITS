set GRADIO_TEMP_DIR=%~dp0TEMP

call conda activate GPTSoVits311
python .\webui.py
pause