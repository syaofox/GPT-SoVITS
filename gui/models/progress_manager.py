"""
进度管理器模块

负责更新和显示进度信息
"""

from PySide6.QtCore import Qt


class ProgressManager:
    """进度管理器，负责更新和显示进度信息"""
    
    def __init__(self, progress_bar, progress_label, status_bar, generate_button=None):
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.status_bar = status_bar
        self.generate_button = generate_button
        self.is_inferencing = False
        self.main_window = None
        
    def set_main_window(self, main_window):
        """设置主窗口引用以便禁用/启用UI"""
        self.main_window = main_window
        
    def on_inference_started(self):
        """推理开始回调"""
        self.is_inferencing = True
        self.status_bar.showMessage("正在生成语音...")
        if self.generate_button:
            # 修改按钮文本为"停止生成"而不是禁用按钮
            self.generate_button.setText("停止生成")
        self.progress_label.setText("正在处理...")
        self.progress_bar.setValue(0)
        
        # 禁用主窗口中的大部分控件
        if self.main_window:
            # 禁用标签页
            self.main_window.tab_widget.setEnabled(False)
            # 禁用音频播放器
            if hasattr(self.main_window, 'audio_player'):
                self.main_window.audio_player.setEnabled(False)
            # 禁用历史记录列表
            if hasattr(self.main_window, 'history_list'):
                self.main_window.history_list.setEnabled(False)
    
    def on_inference_completed(self, file_path=None):
        """推理完成回调"""
        self.is_inferencing = False
        self.status_bar.showMessage("生成完成")
        if self.generate_button:
            # 恢复按钮文本
            self.generate_button.setText("生成语音")
        self.progress_label.setText("就绪")
        self.progress_bar.setValue(0)
        
        # 启用主窗口控件
        self._enable_ui()
    
    def on_inference_failed(self, error_msg):
        """推理失败回调"""
        self.is_inferencing = False
        self.status_bar.showMessage(f"生成失败: {error_msg}")
        if self.generate_button:
            # 恢复按钮文本
            self.generate_button.setText("生成语音")
        self.progress_label.setText(f"失败: {error_msg}")
        self.progress_bar.setValue(0)
        
        # 启用主窗口控件
        self._enable_ui()
    
    def on_inference_stopped(self):
        """推理被手动停止的回调"""
        self.is_inferencing = False
        self.status_bar.showMessage("生成已停止")
        if self.generate_button:
            self.generate_button.setText("生成语音")
        self.progress_label.setText("已停止")
        self.progress_bar.setValue(0)
        
        # 启用主窗口控件
        self._enable_ui()
    
    def _enable_ui(self):
        """启用主窗口中的控件"""
        if self.main_window:
            # 启用标签页
            self.main_window.tab_widget.setEnabled(True)
            # 启用音频播放器
            if hasattr(self.main_window, 'audio_player'):
                self.main_window.audio_player.setEnabled(True)
            # 启用历史记录列表
            if hasattr(self.main_window, 'history_list'):
                self.main_window.history_list.setEnabled(True)
    
    def is_running(self):
        """检查推理是否正在运行"""
        return self.is_inferencing
    
    def update_progress(self, message):
        """更新进度信息"""
        self.progress_label.setText(message)
        
        # 如果是合成进度信息，更新状态栏和进度条
        if "正在合成:" in message:
            self.status_bar.showMessage(message)
            
            # 从消息中提取进度信息
            try:
                # 解析"正在合成: X/Y"格式的消息
                parts = message.split(":")
                if len(parts) == 2:
                    numbers = parts[1].strip().split("/")
                    if len(numbers) == 2:
                        current = int(numbers[0])
                        total = int(numbers[1])
                        if total > 0:  # 防止除零错误
                            progress_value = int((current / total) * 100)
                            # 确保进度值在有效范围内
                            progress_value = max(0, min(100, progress_value))
                            self.progress_bar.setValue(progress_value)
            except Exception as e:
                print(f"解析进度信息失败: {e}")
        
        # 文本预处理完成，重置进度条为起始状态
        elif "文本已分割为" in message:
            try:
                # 从消息中提取片段数量，显示初始进度
                parts = message.split("文本已分割为")
                if len(parts) == 2:
                    segments_str = parts[1].strip().split(" ")[0]
                    if segments_str.isdigit() and int(segments_str) > 0:
                        # 设置一个较小的初始进度值，表示准备开始
                        self.progress_bar.setValue(1)
                    else:
                        self.progress_bar.setValue(0)
                else:
                    self.progress_bar.setValue(0)
            except Exception as e:
                print(f"解析文本分段信息失败: {e}")
                self.progress_bar.setValue(0)
        
        # 生成开始，设置初始进度
        elif "开始生成语音" in message:
            self.progress_bar.setValue(1)  # 设置为1%表示开始
        
        # 保存音频，设置进度条为完成状态
        elif "正在保存音频" in message:
            self.progress_bar.setValue(100)
    
    def show_status_message(self, message):
        """显示状态栏消息"""
        self.status_bar.showMessage(message) 