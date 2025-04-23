"""
历史记录列表组件

显示和管理output目录中的音频文件
"""

import os
import subprocess
import glob
import re
import winreg
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListView, QMessageBox, QMenu, QInputDialog
from PySide6.QtGui import QStandardItemModel, QStandardItem, QCursor


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
        
        # 设置右键菜单
        self.history_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Adobe Audition 路径
        self.audition_path = self.find_audition_path()
        
        self.layout.addWidget(self.history_view)
        
        # 刷新按钮，替代清空按钮
        self.refresh_button = QPushButton("刷新列表")
        self.refresh_button.clicked.connect(self.load_output_files)
        self.layout.addWidget(self.refresh_button)
        
        # 初始加载output目录下的wav文件
        self.load_output_files()
    
    def find_audition_path(self):
        """查找 Adobe Audition 的安装路径"""
        # 默认路径
        default_paths = [
            r"C:\Program Files\Adobe\Adobe Audition CC 2023\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition 2022\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition 2021\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition 2020\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition CC 2019\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition CC 2018\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition 2023\Adobe Audition.exe",
            r"C:\Program Files\Adobe\Adobe Audition 2024\Adobe Audition.exe",
        ]
        
        # 首先检查默认路径
        for path in default_paths:
            if os.path.exists(path):
                return path
        
        # 尝试从注册表中查找
        try:
            # 检查64位注册表
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Adobe\Adobe Audition") as key:
                version = winreg.EnumKey(key, 0)  # 获取第一个版本
                with winreg.OpenKey(key, version) as version_key:
                    install_path = winreg.QueryValueEx(version_key, "InstallPath")[0]
                    exe_path = os.path.join(install_path, "Adobe Audition.exe")
                    if os.path.exists(exe_path):
                        return exe_path
        except Exception:
            pass  # 如果注册表查找失败，继续尝试其他方法
        
        return None
    
    def set_audition_path(self):
        """手动设置 Adobe Audition 路径"""
        current_path = self.audition_path or ""
        path, ok = QInputDialog.getText(
            self, 
            "设置 Adobe Audition 路径", 
            "请输入 Adobe Audition 执行文件的完整路径:",
            text=current_path
        )
        
        if ok and path:
            if os.path.exists(path) and path.lower().endswith(".exe"):
                self.audition_path = path
                return True
            else:
                QMessageBox.warning(
                    self,
                    "无效路径",
                    "指定的 Adobe Audition 路径无效，请确保文件存在且以 .exe 结尾"
                )
        return False
    
    def show_context_menu(self, position):
        """显示右键菜单"""
        index = self.history_view.indexAt(position)
        if not index.isValid():
            return
            
        # 获取选中的文件路径
        item = self.history_model.itemFromIndex(index)
        path = item.data(Qt.UserRole)
        
        if not path or not os.path.exists(path):
            return
            
        menu = QMenu(self)
        
        # 使用系统默认程序打开
        open_action = menu.addAction("使用默认程序打开")
        
        # 打开文件所在文件夹
        open_folder_action = menu.addAction("打开所在文件夹")
        
        # 发送到 Adobe Audition
        send_to_audition_action = menu.addAction("发送到 Adobe Audition")
        send_to_audition_action.setEnabled(self.audition_path is not None)
        
        # 设置 Adobe Audition 路径
        set_audition_path_action = menu.addAction("设置 Adobe Audition 路径")
        
        # 显示菜单并获取选择的操作
        action = menu.exec_(QCursor.pos())
        
        if action == open_action:
            self.open_with_default_program(path)
        elif action == open_folder_action:
            self.open_containing_folder(path)
        elif action == send_to_audition_action:
            if self.audition_path:
                self.send_to_audition(path)
            else:
                if self.set_audition_path():
                    self.send_to_audition(path)
        elif action == set_audition_path_action:
            self.set_audition_path()
    
    def open_with_default_program(self, file_path):
        """使用系统默认程序打开文件"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS/Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.Popen(['open', file_path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', file_path])
        except Exception as e:
            QMessageBox.warning(
                self,
                "打开失败",
                f"无法使用系统默认程序打开文件: {str(e)}"
            )
    
    def open_containing_folder(self, file_path):
        """打开文件所在文件夹"""
        try:
            folder_path = os.path.dirname(file_path)
            if os.name == 'nt':  # Windows
                # 使用 explorer 并选中文件
                subprocess.Popen(f'explorer /select,"{file_path}"', shell=True)
            elif os.name == 'posix':  # macOS/Linux
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.Popen(['open', folder_path])
                else:  # Linux
                    subprocess.Popen(['xdg-open', folder_path])
        except Exception as e:
            QMessageBox.warning(
                self,
                "打开失败",
                f"无法打开文件所在文件夹: {str(e)}"
            )
    
    def send_to_audition(self, file_path):
        """发送文件到 Adobe Audition"""
        try:
            if not self.audition_path or not os.path.exists(self.audition_path):
                if not self.set_audition_path():
                    return
            
            # 使用subprocess打开Adobe Audition并加载文件
            subprocess.Popen([self.audition_path, file_path])
        except Exception as e:
            QMessageBox.warning(
                self,
                "发送失败",
                f"无法发送文件到 Adobe Audition: {str(e)}"
            )
    
    def get_project_root(self):
        """获取项目根目录"""
        # 获取当前文件所在目录，然后向上两级得到项目根目录
        current_dir = Path(os.path.abspath(__file__))
        # components目录的上级是gui，再上级是项目根目录
        return current_dir.parent.parent.parent
    
    def parse_filename(self, file_name):
        """解析文件名，提取时间戳、角色、情绪和文本"""
        # 尝试匹配新格式的文件名: 时间戳_角色_情绪_文本前10字.wav
        pattern = r'(\d{8}_\d{6})_([^_]*)_([^_]*)_([^.]*)'
        match = re.match(pattern, file_name)
        
        if match:
            timestamp_str, role, emotion, text_prefix = match.groups()
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                return {
                    "timestamp": timestamp,
                    "role": role.replace('_', ' '),
                    "emotion": emotion.replace('_', ' '),
                    "text_prefix": text_prefix.replace('_', ' ')
                }
            except ValueError:
                pass
        
        # 如果不是新格式，尝试旧格式: 时间戳_uuid.wav
        try:
            timestamp_part = file_name[:15]  # 取前15个字符 (YYYYMMDD_HHMMSS)
            timestamp = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
            return {
                "timestamp": timestamp,
                "role": "未知",
                "emotion": "未知",
                "text_prefix": ""
            }
        except (ValueError, IndexError):
            return None
    
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
            file_name = file_path.name
            
            # 解析文件名
            file_info = self.parse_filename(file_name)
            
            if file_info and file_info.get("timestamp"):
                timestamp = file_info["timestamp"]
                role = file_info.get("role", "未知")
                emotion = file_info.get("emotion", "未知")
                text_prefix = file_info.get("text_prefix", "")
                
                # 构建显示文本
                if role != "未知" and emotion != "未知" and text_prefix:
                    display_text = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {role}/{emotion} | {text_prefix}..."
                else:
                    display_text = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {file_name}"
            else:
                # 如果解析失败，直接使用文件名
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
            self.open_with_default_program(path) 