@echo off
cd /d "D:\aisound\GPT-SoVITS"

set MY_PATH=.\ffmpeg\bin
set PATH=%PATH%;%MY_PATH%

call conda activate gptsovits
python webui.py zh_CN