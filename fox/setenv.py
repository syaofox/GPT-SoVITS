import os
from pathlib import Path


def setenv():
    # 获取当前脚本的绝对路径
    script_dir = Path(__file__).resolve().parent.parent

    # 设置环境变量
    my_path = script_dir / "ffmpeg" / "bin"
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + str(my_path)
    os.environ["GRADIO_TEMP_DIR"] = str(script_dir / "TEMP")

    import sys

    ex_paths = [
        script_dir,
        script_dir / "tools",
        script_dir / "tools" / "asr",
        script_dir / "GPT_SoVITS",
        script_dir / "tools" / "uvr5",
    ]

    for ex_path in ex_paths:
        sys.path.append(str(ex_path))
