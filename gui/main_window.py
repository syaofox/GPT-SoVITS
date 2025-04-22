"""
主窗口模块

应用程序的主窗口界面
"""

import os
import glob
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox, QSplitter, QLabel, QPushButton, QListWidgetItem
from PySide6.QtCore import Qt, QSize

from gui.controllers import RoleController, InferenceController
from gui.tabs import ExperimentTab, RoleTab
from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化控制器
        self.role_controller = RoleController()
        self.inference_controller = InferenceController()
        
        # 扫描模型文件
        self.gpt_models = self.scan_models(["GPT_weights", "GPT_weights_v2", "GPT_weights_v3", "GPT_weights_v4"])
        self.sovits_models = self.scan_models(["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3", "SoVITS_weights_v4"])
        
        self.init_ui()
        self.connect_signals()
        
        # 加载模型到下拉框
        self.experiment_tab.load_gpt_models(self.gpt_models)
        self.experiment_tab.load_sovits_models(self.sovits_models)
    
    def scan_models(self, model_dirs):
        """扫描模型文件夹，返回模型名称和路径的字典"""
        models_dict = {}
        
        # 获取项目根目录
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        
        # 扫描每个模型目录
        for model_dir in model_dirs:
            dir_path = root_dir / model_dir
            if not dir_path.exists():
                continue
                
            # 查找所有模型文件 (.pth, .ckpt)
            for model_file in glob.glob(str(dir_path / "*.pth")) + glob.glob(str(dir_path / "*.ckpt")):
                model_path = Path(model_file)
                # 使用相对路径作为显示名称
                display_name = f"{model_dir}/{model_path.name}"
                # 使用绝对路径作为实际值
                models_dict[display_name] = str(model_path.absolute())
        
        return models_dict
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("GPT-SoVITS 语音合成")
        self.setMinimumSize(1100, 700)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧部分 - 选项卡
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # 实验选项卡 - 创建为共享控件版本
        self.experiment_tab = ExperimentTab(self.role_controller, self.inference_controller, shared_controls=True)
        self.tab_widget.addTab(self.experiment_tab, "试听配置")
        
        # 角色选项卡 - 创建为共享控件版本
        self.role_tab = RoleTab(self.role_controller, self.inference_controller, shared_controls=True)
        self.tab_widget.addTab(self.role_tab, "角色推理")
        
        left_layout.addWidget(self.tab_widget)
        
        # 右侧部分 - 共享控件
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 状态显示
        self.progress_label = QLabel("就绪")
        right_layout.addWidget(self.progress_label)
        
        # 生成语音按钮
        self.generate_button = QPushButton("生成语音")
        self.generate_button.clicked.connect(self.generate_speech)
        right_layout.addWidget(self.generate_button)
        
        # 音频播放器
        self.audio_player = AudioPlayer()
        right_layout.addWidget(self.audio_player)
        
        # 历史记录
        history_group_layout = QVBoxLayout()
        self.history_list = HistoryList()
        history_group_layout.addWidget(self.history_list)
        right_layout.addLayout(history_group_layout)
        
        # 添加到主布局的分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def connect_signals(self):
        """连接信号槽"""
        # 控制器信号
        self.role_controller.error_occurred.connect(self.show_error)
        self.inference_controller.error_occurred.connect(self.show_error)
        
        self.inference_controller.inference_started.connect(self.on_inference_started)
        self.inference_controller.inference_completed.connect(self.on_inference_completed)
        self.inference_controller.inference_failed.connect(self.on_inference_failed)
        self.inference_controller.progress_updated.connect(self.update_progress)
        
        # 共享组件信号
        self.inference_controller.inference_completed.connect(self.on_new_audio_generated)
        self.history_list.audio_selected.connect(self.audio_player.load_audio)
        
        # 标签页生成按钮信号
        self.experiment_tab.generate_requested.connect(self.on_generate_requested)
        self.role_tab.generate_requested.connect(self.on_generate_requested)
        
        # 连接角色选择信号
        self.role_tab.role_config_selected.connect(self.apply_role_config_to_experiment)
        # 连接角色信息选择信号
        self.role_tab.role_info_selected.connect(self.apply_role_info_to_experiment)
    
    def apply_role_config_to_experiment(self, role_config):
        """将角色配置应用到试听配置标签页"""
        
        # 设置参考音频和文本
        ref_audio = role_config.get("ref_audio", "")
        if ref_audio and os.path.exists(ref_audio):
            self.experiment_tab.ref_path_edit.setText(ref_audio)
        
        # 设置参考文本和语言
        self.experiment_tab.prompt_text_edit.setText(role_config.get("prompt_text", ""))
        
        prompt_lang = role_config.get("prompt_lang", "中文")
        index = self.experiment_tab.prompt_lang_combo.findText(prompt_lang)
        if index >= 0:
            self.experiment_tab.prompt_lang_combo.setCurrentIndex(index)
        
        # 设置文本语言
        text_lang = role_config.get("text_lang", "中文")
        index = self.experiment_tab.text_lang_combo.findText(text_lang)
        if index >= 0:
            self.experiment_tab.text_lang_combo.setCurrentIndex(index)
        
        # 设置文本切分方式
        how_to_cut = role_config.get("how_to_cut", "按句切")
        index = self.experiment_tab.cut_method_combo.findText(how_to_cut)
        if index >= 0:
            self.experiment_tab.cut_method_combo.setCurrentIndex(index)
        
        # 设置模型
        gpt_path = role_config.get("gpt_path")
        if gpt_path:
            # 尝试直接匹配完整路径
            gpt_matched = False
            for display_name, path in self.gpt_models.items():
                if path == gpt_path:
                    index = self.experiment_tab.gpt_model_combo.findText(display_name)
                    if index >= 0:
                        self.experiment_tab.gpt_model_combo.setCurrentIndex(index)
                        gpt_matched = True
                        break
            
            # 如果直接匹配失败，尝试匹配文件名
            if not gpt_matched:
                gpt_filename = os.path.basename(gpt_path)
                for display_name, path in self.gpt_models.items():
                    if os.path.basename(path) == gpt_filename:
                        index = self.experiment_tab.gpt_model_combo.findText(display_name)
                        if index >= 0:
                            self.experiment_tab.gpt_model_combo.setCurrentIndex(index)
                            break
        
        sovits_path = role_config.get("sovits_path")
        
        if sovits_path:
            # 尝试直接匹配完整路径
            sovits_matched = False
            for display_name, path in self.sovits_models.items():
                if path == sovits_path:
                    index = self.experiment_tab.sovits_model_combo.findText(display_name)
                    if index >= 0:
                        self.experiment_tab.sovits_model_combo.setCurrentIndex(index)
                        sovits_matched = True
                        break
            
            # 如果直接匹配失败，尝试匹配文件名
            if not sovits_matched:
                sovits_filename = os.path.basename(sovits_path)
                for display_name, path in self.sovits_models.items():
                    if os.path.basename(path) == sovits_filename:
                        index = self.experiment_tab.sovits_model_combo.findText(display_name)
                        if index >= 0:
                            self.experiment_tab.sovits_model_combo.setCurrentIndex(index)
                            break
        
        # 设置高级参数
        self.experiment_tab.speed_spin.setValue(role_config.get("speed", 1.0))
        self.experiment_tab.top_k_spin.setValue(role_config.get("top_k", 15))
        self.experiment_tab.top_p_spin.setValue(role_config.get("top_p", 1.0))
        self.experiment_tab.temperature_spin.setValue(role_config.get("temperature", 1.0))
        self.experiment_tab.sample_steps_spin.setValue(role_config.get("sample_steps", 8))
        self.experiment_tab.pause_spin.setValue(role_config.get("pause_second", 0.3))
        
        # 设置选项
        self.experiment_tab.ref_free_check.setChecked(role_config.get("ref_free", False))
        self.experiment_tab.sr_check.setChecked(role_config.get("if_sr", False))
        
        # 清空并添加辅助参考音频
        self.experiment_tab.aux_refs_list.clear()
        
        aux_refs = role_config.get("aux_refs", [])
        for aux_ref in aux_refs:
            if aux_ref and os.path.exists(aux_ref):
                # 添加到列表
                basename = os.path.basename(aux_ref)
                item = QListWidgetItem(basename)
                item.setData(Qt.UserRole, aux_ref)
                self.experiment_tab.aux_refs_list.addItem(item)
    
    def apply_role_info_to_experiment(self, role_name, emotion_name):
        """将角色名和情绪信息应用到试听配置标签页"""
        if not role_name or not emotion_name:
            return
        
        # 填充角色名和情绪到实验页面的输入框
        self.experiment_tab.role_name_edit.setText(role_name)
        self.experiment_tab.emotion_name_edit.setText(emotion_name)
    
    def on_new_audio_generated(self, file_path: str):
        """新音频生成后刷新历史列表"""
        self.history_list.load_output_files()
    
    def on_inference_started(self):
        """推理开始回调"""
        self.status_bar.showMessage("正在生成语音...")
        self.generate_button.setEnabled(False)
        self.progress_label.setText("正在处理...")
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        self.status_bar.showMessage("生成完成")
        self.generate_button.setEnabled(True)
        self.progress_label.setText("就绪")
        self.audio_player.load_audio(file_path)
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        self.status_bar.showMessage(f"生成失败: {error_msg}")
        self.generate_button.setEnabled(True)
        self.progress_label.setText(f"失败: {error_msg}")
    
    def update_progress(self, message: str):
        """更新进度信息"""
        self.status_bar.showMessage(message)
        self.progress_label.setText(message)
    
    def on_generate_requested(self, config=None, text="", is_role=False):
        """处理来自标签页的生成请求"""
        if is_role:
            # 角色推理
            if config and text:
                config["text"] = text
                self.inference_controller.generate_speech_async(config)
        else:
            # 实验模式推理
            if config:
                self.inference_controller.generate_speech_async(config)
    
    def show_error(self, message: str):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # InferenceController已经通过atexit注册了程序退出时的保存函数，不需要重复调用
        # 直接继续正常关闭
        event.accept()

    def generate_speech(self):
        """生成语音"""
        # 根据当前选中的标签页决定使用哪种生成方法
        current_index = self.tab_widget.currentIndex()
        
        if current_index == 0:  # 实验选项卡
            config = self.experiment_tab.get_inference_config()
            text = self.experiment_tab.text_edit.toPlainText()
            
            # 验证必要参数
            if not text:
                self.show_error("请输入要合成的文本")
                return
                
            if not config["gpt_path"]:
                self.show_error("请选择GPT模型")
                return
                
            if not config["sovits_path"]:
                self.show_error("请选择SoVITS模型")
                return
                
            if not config["ref_free"] and not config["ref_audio"]:
                self.show_error("请选择参考音频文件或勾选无参考文本")
                return
                
            if not config["ref_free"] and not os.path.exists(config["ref_audio"]):
                self.show_error(f"参考音频文件不存在: {config['ref_audio']}")
                return
            
            # 确保角色名和情绪名存在
            if not config.get("role_name"):
                config["role_name"] = self.experiment_tab.role_name_edit.text().strip() or "未知角色"
            if not config.get("emotion_name"):
                config["emotion_name"] = self.experiment_tab.emotion_name_edit.text().strip() or "未知情绪"
                
            # 验证辅助参考音频
            aux_refs = config.get("aux_refs", [])
            valid_aux_refs = []
            for aux_ref in aux_refs:
                if os.path.exists(aux_ref):
                    valid_aux_refs.append(aux_ref)
                else:
                    self.show_error(f"辅助参考音频文件不存在，已忽略: {aux_ref}")
            
            # 更新有效的辅助参考音频
            config["aux_refs"] = valid_aux_refs
                
            # 调用推理控制器
            self.inference_controller.generate_speech_async(config)
            
        elif current_index == 1:  # 角色选项卡
            role_name = self.role_tab.current_role
            emotion_name = self.role_tab.current_emotion
            text = self.role_tab.text_edit.toPlainText()
            
            if not role_name or not emotion_name:
                self.show_error("请先选择角色和情感")
                return
                
            if not text:
                self.show_error("请输入要合成的文本")
                return
                
            # 获取推理配置
            config = self.role_tab.get_inference_config()
            if not config:
                self.show_error("无法获取角色配置")
                return
            
            # 确保角色名和情绪名存在
            if not config.get("role_name"):
                config["role_name"] = role_name
            if not config.get("emotion_name"):
                config["emotion_name"] = emotion_name
                
            # 检查参考音频路径是否存在
            ref_audio = config.get("ref_audio", "")
            if not config.get("ref_free", False) and ref_audio and not os.path.exists(ref_audio):
                self.show_error(f"参考音频文件不存在: {ref_audio}")
                return
                
            # 检查辅助参考音频
            aux_refs = config.get("aux_refs", [])
            valid_aux_refs = []
            for aux_ref in aux_refs:
                if os.path.exists(aux_ref):
                    valid_aux_refs.append(aux_ref)
                else:
                    self.show_error(f"辅助参考音频文件不存在，已忽略: {aux_ref}")
            
            # 更新有效的辅助参考音频
            config["aux_refs"] = valid_aux_refs
                
            # 调用推理控制器
            self.inference_controller.generate_speech_async(config)
    
    def on_tab_changed(self, index):
        """处理标签页切换"""
        if index == 1:  # 切换到角色选项卡
            # 如果当前角色和情感有选择，则触发配置加载
            if self.role_tab.current_role and self.role_tab.current_emotion:
                self.role_tab.get_current_emotion_config() 