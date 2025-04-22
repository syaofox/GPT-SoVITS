"""
GUI应用主入口

启动GPT-SoVITS GUI应用程序
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTranslator, QLocale

from gui.main_window import MainWindow


def setup_environment():
    """设置环境变量和工作目录"""
    # 获取脚本所在目录
    script_dir = Path(__file__).resolve().parent
    
    # 将项目根目录添加到Python路径
    root_dir = script_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    # 确保输出目录存在
    output_dir = root_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 切换工作目录到项目根目录
    os.chdir(root_dir)


def main():
    """主函数"""
    # 设置环境
    setup_environment()
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("GPT-SoVITS GUI")
    
    # 设置翻译器
    translator = QTranslator()
    if translator.load(QLocale.system(), "qt", "_", 
                       QApplication.instance().applicationDirPath()):
        app.installTranslator(translator)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 