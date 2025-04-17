"""
词语替换视图
负责词语替换界面的显示和交互
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QPushButton, QTextEdit, QMessageBox, QLabel
)
from PySide6.QtCore import Signal, Qt


class WordReplaceView(QWidget):
    """词语替换视图类"""
    
    # 定义信号
    config_save = Signal(str)  # 保存配置信号
    config_reload = Signal()   # 重新加载配置信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建说明组
        instruction_group = QGroupBox("使用说明")
        instruction_layout = QVBoxLayout()
        
        # 说明文本 - 使用QLabel替代QTextEdit
        instruction_text = QLabel(
            "词语替换配置格式：\n"
            "替换前 替换后\n\n"
            "示例：\n"
            "hello 你好\n"
            "world 世界\n\n"
            "注意：\n"
            "- 每行一条规则\n"
            "- 以#开头的行为注释\n"
            "- 英文部分不区分大小写"
        )
        instruction_text.setWordWrap(True)  # 启用自动换行
        instruction_text.setTextFormat(Qt.TextFormat.PlainText)  # 使用纯文本格式
        instruction_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # 靠上靠左对齐
        instruction_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # 允许鼠标选择文本
        
        instruction_layout.addWidget(instruction_text)
        instruction_group.setLayout(instruction_layout)
        
        # 创建配置组
        config_group = QGroupBox("词语替换配置")
        config_layout = QVBoxLayout()
        
        # 配置文本编辑区
        self.config_text = QTextEdit()
        self.config_text.setMinimumHeight(400)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 保存按钮
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.on_save_clicked)
        
        # 刷新按钮
        self.reload_btn = QPushButton("重新加载")
        self.reload_btn.clicked.connect(self.on_reload_clicked)
        
        # 添加到按钮布局
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reload_btn)
        
        # 添加到配置布局
        config_layout.addWidget(self.config_text)
        config_layout.addLayout(button_layout)
        config_group.setLayout(config_layout)
        
        # 添加到主布局
        main_layout.addWidget(instruction_group)
        main_layout.addWidget(config_group)
    
    def on_save_clicked(self):
        """保存按钮点击事件"""
        config_text = self.config_text.toPlainText()
        self.config_save.emit(config_text)
    
    def on_reload_clicked(self):
        """重新加载按钮点击事件"""
        self.config_reload.emit()
    
    def set_config_text(self, text):
        """设置配置文本"""
        self.config_text.setText(text)
    
    def show_message(self, title, message, icon=QMessageBox.Information):
        """显示消息对话框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.exec_() 