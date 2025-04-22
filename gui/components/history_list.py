"""
历史记录列表组件

显示和管理output目录中的音频文件
"""

import os
import subprocess
import glob
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListView, QMessageBox
from PySide6.QtGui import QStandardItemModel, QStandardItem


class HistoryList(QWidget):
    """历史记录列表组件"""
    
    audio_selected = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("历史记录")
        self.layout.addWidget(title_label)
        
        # 历史列表
        self.history_model = QStandardItemModel()
        self.history_view = QListView()
        self.history_view.setModel(self.history_model)
        self.history_view.clicked.connect(self.on_item_clicked)
        self.history_view.doubleClicked.connect(self.on_item_double_clicked)
        
        # 设置为不可编辑
        self.history_view.setEditTriggers(QListView.NoEditTriggers)
        
        self.layout.addWidget(self.history_view)
        
        # 刷新按钮，替代清空按钮
        self.refresh_button = QPushButton("刷新列表")
        self.refresh_button.clicked.connect(self.load_output_files)
        self.layout.addWidget(self.refresh_button)
        
        # 初始加载output目录下的wav文件
        self.load_output_files()
    
    def get_project_root(self):
        """获取项目根目录"""
        # 获取当前文件所在目录，然后向上两级得到项目根目录
        current_dir = Path(os.path.abspath(__file__))
        # components目录的上级是gui，再上级是项目根目录
        return current_dir.parent.parent.parent
    
    def load_output_files(self):
        """加载output目录下的音频文件"""
        self.history_model.clear()
        
        # 获取项目根目录下的output文件夹
        output_dir = self.get_project_root() / "output"
        
        if not output_dir.exists():
            return
        
        # 查找所有wav文件
        wav_files = glob.glob(str(output_dir / "*.wav"))
        # 按文件修改时间排序（最新的在前面）
        wav_files.sort(key=os.path.getmtime, reverse=True)
        
        for wav_file in wav_files:
            file_path = Path(wav_file)
            # 提取文件名作为显示文本
            file_name = file_path.name
            
            # 尝试从文件名解析时间戳
            try:
                # 假设文件名格式为 YYYYMMDD_HHMMSS_*.wav
                timestamp_part = file_name[:15]  # 取前15个字符 (YYYYMMDD_HHMMSS)
                timestamp = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                display_text = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {file_name}"
            except (ValueError, IndexError):
                # 如果无法解析时间戳，直接使用文件名
                display_text = file_name
            
            item = QStandardItem(display_text)
            item.setData(str(file_path), Qt.UserRole)
            self.history_model.appendRow(item)
    
    def update_history(self, history=None):
        """更新历史记录（保留该方法以兼容现有代码）"""
        # 直接调用加载文件方法
        self.load_output_files()
    
    def on_item_clicked(self, index):
        """历史项点击回调"""
        if not index.isValid():
            return
            
        item = self.history_model.itemFromIndex(index)
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            self.audio_selected.emit(path)
    
    def on_item_double_clicked(self, index):
        """历史项双击回调 - 使用系统播放器打开"""
        if not index.isValid():
            return
            
        item = self.history_model.itemFromIndex(index)
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            try:
                # 使用系统默认程序打开音频文件
                if os.name == 'nt':  # Windows
                    os.startfile(path)
                elif os.name == 'posix':  # macOS/Linux
                    # 对于macOS，用open命令；对于Linux，用xdg-open命令
                    if os.uname().sysname == 'Darwin':  # macOS
                        subprocess.Popen(['open', path])
                    else:  # Linux
                        subprocess.Popen(['xdg-open', path])
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "打开失败",
                    f"无法使用系统播放器打开文件: {str(e)}"
                ) 