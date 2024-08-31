@echo off
set MY_PATH=.\ffmpeg\bin
set PATH=%PATH%;%MY_PATH%
set GRADIO_TEMP_DIR=%~dp0TEMP

call conda activate gptsovits
python .\fox\srt_asr.py