#! /bin/bash
# 临时设置语言为中文
export LANG="zh_CN.UTF-8"
export LANGUAGE="zh_CN:zh"
export LC_ALL="zh_CN.UTF-8"

if [ ! -d "/tmp/uvr5" ]; then
    mkdir -p /tmp/uvr5
fi
export TEMP="/tmp/uvr5"

.venv/bin/python tools/uvr5/webui.py "cuda" True 9873 False

