"""
替换规则编辑标签页

提供编辑文本替换规则的功能
"""

import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QPlainTextEdit, QPushButton, 
    QMessageBox, QSplitter, QStatusBar
)


class ReplaceTab(QWidget):
    """替换规则编辑标签页，用于编辑word_replace.txt文件内容"""
    
    # 添加规则更新信号
    rules_updated = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 替换规则文件路径
        self.replace_file_path = None
        self.find_replace_file()
        
        # 初始化界面
        self.init_ui()
        
        # 加载替换规则
        self.load_rules()
    
    def find_replace_file(self):
        """查找替换规则文件路径"""
        # 获取项目根目录
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        replace_file = root_dir / "gui" / "word_replace.txt"
        
        if replace_file.exists():
            self.replace_file_path = str(replace_file)
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        
        # 说明标签
        info_label = QLabel("编辑替换规则 (每行一条规则，格式为'查找字符串|需修改字符串|替换后的字符串')")
        main_layout.addWidget(info_label)
        
        # 编辑区
        self.edit_area = QPlainTextEdit()
        self.edit_area.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.edit_area.setPlaceholderText("例如：\n操死|操|cao4\n干死|干|gan4\n远比|远|yuan2")
        main_layout.addWidget(self.edit_area)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪")
        main_layout.addWidget(self.status_bar)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("保存并应用")
        self.save_button.clicked.connect(self.save_rules)
        button_layout.addWidget(self.save_button)
        
        self.reload_button = QPushButton("重新加载")
        self.reload_button.clicked.connect(self.load_rules)
        button_layout.addWidget(self.reload_button)
        
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
    
    def load_rules(self):
        """加载替换规则文件内容"""
        if not self.replace_file_path or not os.path.exists(self.replace_file_path):
            self.status_bar.showMessage("警告: 找不到替换规则文件")
            return
        
        try:
            with open(self.replace_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.edit_area.setPlainText(content)
                
                # 计算行数
                lines = content.splitlines()
                valid_rules = 0
                for line in lines:
                    line = line.strip()
                    # 跳过空行和注释行
                    if line and not line.startswith('#') and '|' in line:
                        parts = line.split('|')
                        if len(parts) == 3:
                            valid_rules += 1
                        
                self.status_bar.showMessage(f"已加载 {valid_rules} 条有效规则，共 {len(lines)} 行")
        except Exception as e:
            self.status_bar.showMessage(f"错误: 加载替换规则文件失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"加载替换规则文件失败: {str(e)}")
    
    def validate_rules(self, content):
        """验证规则格式"""
        lines = content.splitlines()
        valid_rules = []
        invalid_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释行
                continue
                
            if '|' not in line:
                invalid_lines.append((i+1, line, "缺少'|'分隔符"))
            else:
                parts = line.split('|')
                if len(parts) != 3:
                    invalid_lines.append((i+1, line, "格式错误，应为'查找字符串|需修改字符串|替换后的字符串'"))
                elif not parts[0] or not parts[1] or not parts[2]:
                    invalid_lines.append((i+1, line, "三个部分都不能为空"))
                elif parts[1] not in parts[0]:
                    invalid_lines.append((i+1, line, "需修改字符串必须是查找字符串的一部分"))
                else:
                    valid_rules.append((parts[0], parts[1], parts[2]))
        
        return valid_rules, invalid_lines
    
    def save_rules(self):
        """保存并应用替换规则"""
        if not self.replace_file_path:
            self.status_bar.showMessage("警告: 无法确定替换规则文件路径")
            return
        
        try:
            # 获取编辑区内容
            content = self.edit_area.toPlainText()
            
            # 验证规则
            valid_rules, invalid_lines = self.validate_rules(content)
            
            # 如果有无效规则，提示用户
            if invalid_lines:
                error_msg = "以下行存在格式问题:\n"
                for line_num, line_content, reason in invalid_lines[:10]:  # 最多显示10条错误
                    error_msg += f"第 {line_num} 行: {line_content} - {reason}\n"
                    
                if len(invalid_lines) > 10:
                    error_msg += f"...以及其他 {len(invalid_lines) - 10} 行存在问题\n"
                    
                error_msg += "\n是否仍要保存？"
                
                reply = QMessageBox.question(self, "规则验证警告", error_msg,
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.status_bar.showMessage("已取消保存")
                    return
            
            # 保存到文件
            with open(self.replace_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 生成规则列表
            rules = []
            for search_string, target_part, replace_with in valid_rules:
                rules.append((search_string, target_part, replace_with))
            
            # 发送更新信号，通知主窗口更新规则
            self.rules_updated.emit(rules)
            
            self.status_bar.showMessage(f"已保存并应用 {len(rules)} 条有效替换规则")
            
        except Exception as e:
            self.status_bar.showMessage(f"错误: 保存替换规则文件失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"保存替换规则文件失败: {str(e)}") 