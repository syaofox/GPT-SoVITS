@echo off
set MY_PATH=.\ffmpeg\bin
set PATH=%PATH%;%MY_PATH%

runtime\python.exe ./srt_asr.py
