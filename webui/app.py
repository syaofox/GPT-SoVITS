import os
import sys
import argparse
import logging

# 使用os.path.expanduser('~')获取用户主目录，避免硬编码用户名
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    print(f"nltk_data 不存在，开始下载到 {nltk_data_path}")
    import nltk
    # 设置NLTK数据目录
    nltk.data.path.append(nltk_data_path)
    nltk.download("averaged_perceptron_tagger_eng")

# 添加项目根目录到Python路径
WEBUI_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(WEBUI_ROOT, ".."))
gpt_sovits_path = os.path.abspath(os.path.join(PROJECT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, PROJECT_ROOT)  # 添加项目根目录
sys.path.insert(0, WEBUI_ROOT)  # 添加webui目录
sys.path.insert(0, gpt_sovits_path)  # 添加gpt_sovits目录

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"


from models.logger import set_level, info, error, debug
from services.prompt_service import PromptService
from services.gptsovits_server import GPTSoVITSServer
from ui.main_ui import MainUI
from ui.event_handlers import EventHandlers
from models.constand import TEMP_DIR, OUTPUT_DIR, PROMPT_DIR


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="IndexTTS WebUI")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="日志级别 (debug, info, warning, error, critical)",
    )
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置日志级别
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    set_level(log_level_map[args.log_level])

    try:
        info("gptsovits WebUI 启动中...")

        debug("初始化提示词服务...")
        prompt_service = PromptService()

        debug("初始化 TTS 服务...")
        tts_service = GPTSoVITSServer()

        debug("创建必要目录...")
        os.makedirs(PROMPT_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        debug("构建用户界面...")
        main_ui = MainUI()
        event_handlers = EventHandlers(tts_service, prompt_service)

        debug("启动 Gradio 界面...")
        demo = main_ui.build(event_handlers)
        demo.queue(20)

        info(
            f"服务器准备就绪，监听地址: {os.environ['GRADIO_SERVER_NAME']}:{os.environ['GRADIO_SERVER_PORT']}"
        )
        demo.launch(favicon_path="webui/favicon.png")

    except Exception as e:
        error(f"程序启动出错: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
