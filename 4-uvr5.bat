set GRADIO_TEMP_DIR=%~dp0TEMP
set PYTHONPATH=%PYTHONPATH%;%~dp0;%~dp0GPT_SoVITS

uv run python .\tools\uvr5\webui.py "cuda" True 9872 False

