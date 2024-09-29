import os
import sys
from pathlib import Path


def setenv():
    # 获取当前脚本的父目录路径
    script_dir = Path(__file__).resolve().parent.parent

    # 更新 PATH 环境变量
    os.environ['PATH'] += os.pathsep + str(script_dir / 'ffmpeg' / 'bin')

    # 设置 GRADIO_TEMP_DIR 环境变量
    os.environ['GRADIO_TEMP_DIR'] = str(script_dir / 'TEMP')

    # 需要添加到 sys.path 的目录列表
    ex_paths = [
        script_dir,
        script_dir / 'tools',
        script_dir / 'tools' / 'asr',
        script_dir / 'GPT_SoVITS',
        script_dir / 'tools' / 'uvr5',
    ]

    # 将目录添加到 sys.path
    sys.path.extend(str(path) for path in ex_paths)
