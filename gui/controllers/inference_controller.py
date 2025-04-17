"""
推理控制器
负责协调推理模型和视图
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QUrl

from gui.models.inference_model import InferenceThread

# 配置logger
logger = logging.getLogger("word_replace")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class InferenceController:
    """推理控制器类"""
    
    def __init__(self, role_model, inference_model, word_replace_model, view):
        """初始化控制器"""
        self.role_model = role_model
        self.inference_model = inference_model
        self.word_replace_model = word_replace_model  # 添加词语替换模型
        self.view = view
        self.current_role = None
        self.current_emotion = None
        self.current_config = {}
        self.inference_thread = None
        self.is_inferring = False
        
        # 创建日志目录
        os.makedirs("logs", exist_ok=True)
        
        # 连接视图的信号
        self.connect_signals()
        
        # 加载角色列表
        self.load_roles()
        
        # 加载历史记录
        self.load_history()
    
    def connect_signals(self):
        """连接视图的信号"""
        # 角色选择信号
        self.view.role_selected.connect(self.on_role_selected)
        self.view.emotion_selected.connect(self.on_emotion_selected)
        self.view.role_refresh.connect(self.refresh_roles)  # 连接角色刷新信号
        
        # 推理控制信号
        self.view.infer_start.connect(self.on_infer_start)
        self.view.infer_stop.connect(self.on_infer_stop)
        
        # 历史记录信号
        self.view.history_selected.connect(self.on_history_selected)
        self.view.history_clear.connect(self.on_history_clear)
        self.view.history_refresh.connect(self.load_history)
        
        # 辅助参考音频信号
        self.view.aux_ref_play.connect(self.on_aux_ref_play)
    
    def load_roles(self):
        """加载角色列表"""
        roles = self.role_model.get_role_list()
        self.view.update_role_list(roles)
    
    def refresh_roles(self):
        """刷新角色列表，从模型目录扫描角色"""
        # 显示加载状态
        self.view.role_list.setEnabled(False)
        self.view.refresh_role_btn.setEnabled(False)
        self.view.refresh_role_btn.setText("刷新中...")
        
        try:
            # 同步模型目录和角色配置
            result = self.role_model.sync_roles_with_models()
            
            # 更新视图中的角色列表
            self.view.update_role_list(result["all_roles"])
            
            # 如果存在新角色，显示提示信息
            if result["new_roles"]:
                new_roles_str = ", ".join(result["new_roles"])
                self.view.show_message(
                    "刷新成功", 
                    f"角色列表已刷新，发现{len(result['new_roles'])}个新角色: {new_roles_str}"
                )
            else:
                self.view.show_message("刷新成功", "角色列表已刷新，未发现新角色")
                
            # 如果当前没有选中角色且列表不为空，则选择第一个
            if self.view.role_list.currentRow() < 0 and self.view.role_list.count() > 0:
                self.view.role_list.setCurrentRow(0)
        except Exception as e:
            # 显示错误消息
            self.view.show_message("刷新失败", f"角色列表刷新失败: {str(e)}", QMessageBox.Warning)
        finally:
            # 恢复按钮状态
            self.view.role_list.setEnabled(True)
            self.view.refresh_role_btn.setEnabled(True)
            self.view.refresh_role_btn.setText("刷新角色")
    
    def load_history(self):
        """加载历史记录"""
        history_items = self.inference_model.get_history_list()
        self.view.update_history_list(history_items)
    
    def on_role_selected(self, role_name):
        """角色选择事件处理"""
        self.current_role = role_name
        self.current_config = self.role_model.get_role_config(role_name)
        
        if not self.current_config:
            self.view.show_message("错误", f"加载角色'{role_name}'配置失败!", QMessageBox.Warning)
            self.view.set_inference_widgets_enabled(False)
            return
        
        # 更新音色列表
        emotions = self.current_config.get("emotions", {})
        self.view.update_emotion_list(emotions.keys())
        
        # 设置参数
        self.view.load_parameters(self.current_config)
        
        # 启用控件
        self.view.set_inference_widgets_enabled(True)
    
    def on_emotion_selected(self, emotion_name):
        """音色选择事件处理"""
        if not self.current_role or not self.current_config:
            return
        
        self.current_emotion = emotion_name
        
        # 获取音色配置
        emotions = self.current_config.get("emotions", {})
        emotion_config = emotions.get(emotion_name, {})
        
        if not emotion_config:
            return
        
        # 更新参考音频信息
        ref_audio = emotion_config.get("ref_audio", "")
        prompt_text = emotion_config.get("prompt_text", "")
        self.view.load_reference_info(ref_audio, prompt_text)
        
        # 更新辅助参考音频列表
        aux_refs = emotion_config.get("aux_refs", [])
        self.view.update_aux_ref_list(aux_refs)
        
        # 加载参考音频
        if ref_audio:
            ref_path = self.role_model.get_ref_audio_path(self.current_role, ref_audio)
            if ref_path.exists():
                self.view.set_ref_audio(str(ref_path))
            else:
                print(f"参考音频文件不存在: {ref_path}")
    
    def on_aux_ref_play(self, ref_path):
        """播放辅助参考音频事件处理"""
        if not self.current_role:
            return
        
        # 构建全路径
        path = self.role_model.get_role_path(self.current_role) / ref_path
        if not path.exists() and Path(ref_path).exists():
            path = Path(ref_path)
            
        if path.exists():
            self.view.set_ref_audio(str(path))
        else:
            self.view.show_message("错误", f"音频文件 '{ref_path}' 不存在!", QMessageBox.Warning)
    
    def on_infer_start(self, params):
        """开始推理事件处理"""
        if self.is_inferring:
            return
        
        if not self.current_role or not self.current_emotion:
            self.view.show_message("错误", "请先选择角色和音色!", QMessageBox.Warning)
            return
        
        # 应用词语替换
        original_text = params["text"]
        replaced_text = self.word_replace_model.clean_text(original_text)
        
        # 记录替换日志
        has_change = original_text != replaced_text
        self._log_word_replace(original_text, replaced_text, has_change)
        
        # 使用替换后的文本
        params["text"] = replaced_text
        
        # 准备推理参数
        try:
            infer_params, error_msg = self.inference_model.prepare_inference_params(
                self.current_config,
                self.current_emotion,
                params["text"],
                params
            )
            
            if error_msg:
                self.view.show_message("参数错误", error_msg, QMessageBox.Warning)
                return
                
            if not infer_params:
                self.view.show_message("错误", "准备推理参数失败!", QMessageBox.Warning)
                return
        except Exception as e:
            self.view.show_message("错误", f"准备推理参数时发生异常: {str(e)}", QMessageBox.Warning)
            return
        
        # 设置状态
        self.is_inferring = True
        self.view.set_inferring_state(True)
        
        # 重置进度条
        self.view.set_progress(0)
        
        # 创建并启动推理线程
        try:
            self.inference_thread = InferenceThread(
                self.current_config.get("gpt_path", ""),
                self.current_config.get("sovits_path", ""),
                infer_params
            )
            
            # 连接信号
            self.inference_thread.progress_update.connect(self.on_progress_update)
            self.inference_thread.inference_complete.connect(self.on_inference_complete)
            self.inference_thread.inference_error.connect(self.on_inference_error)
            
            # 启动线程
            self.inference_thread.start()
        except Exception as e:
            self.is_inferring = False
            self.view.set_inferring_state(False)
            self.view.show_message("错误", f"启动推理线程时发生异常: {str(e)}", QMessageBox.Warning)
    
    def on_infer_stop(self):
        """停止推理事件处理"""
        if not self.is_inferring or not self.inference_thread:
            return
        
        # 终止线程
        self.inference_thread.terminate()
        self.inference_thread.wait()
        
        # 重置状态
        self.is_inferring = False
        self.view.set_inferring_state(False)
        self.view.set_progress(0)
    
    def on_progress_update(self, progress_data):
        """推理进度更新事件处理
        
        Args:
            progress_data: 包含进度值和段落信息的元组(progress_value, segment_info)
        """
        if isinstance(progress_data, tuple) and len(progress_data) == 2:
            progress_value, segment_info = progress_data
            self.view.set_progress(progress_value, segment_info)
        else:
            # 兼容处理旧版本的进度更新信号
            self.view.set_progress(progress_data)
    
    def on_inference_complete(self, output_path):
        """推理完成事件处理"""
        # 重置状态
        self.is_inferring = False
        self.view.set_inferring_state(False)
        
        # 重置进度条并显示100%
        self.view.set_progress(100)
        
        # 设置结果音频
        self.view.set_result_audio(output_path)
        
        # 刷新历史记录
        self.load_history()
        
        # 显示成功消息
        self.view.show_message("推理完成", f"音频已生成: {output_path}")
    
    def on_inference_error(self, error_msg):
        """推理错误事件处理"""
        # 重置状态
        self.is_inferring = False
        self.view.set_inferring_state(False)
        self.view.set_progress(0)
        
        # 显示错误消息
        self.view.show_message("推理失败", error_msg, QMessageBox.Critical)
    
    def on_history_selected(self, file_path):
        """历史记录选择事件处理"""
        self.view.set_result_audio(file_path)
    
    def on_history_clear(self):
        """清空历史记录事件处理"""
        if self.inference_model.clear_history():
            self.load_history()
            self.view.show_message("成功", "历史记录已清空!")
        else:
            self.view.show_message("错误", "清空历史记录失败!", QMessageBox.Warning)
    
    def _log_word_replace(self, original_text, replaced_text, has_change):
        """记录词语替换日志"""
        status = "有替换" if has_change else "无替换"
        
        # 记录基本信息
        logger.info(f"词语替换 - {status}")
        logger.info(f"原文本: {original_text}")
        
        if has_change:
            logger.info(f"替换后: {replaced_text}")
            
            # 查找并记录哪些词被替换了
            replace_dict = self.word_replace_model.get_replace_dict()
            found_replacements = []
            
            for src, dst in replace_dict.items():
                import re
                # 创建不区分大小写的模式
                pattern = ''.join(['[' + c.upper() + c.lower() + ']' if c.isascii() and c.isalpha() else re.escape(c) for c in src])
                if re.search(pattern, original_text):
                    found_replacements.append(f"{src} → {dst}")
            
            if found_replacements:
                logger.info("替换详情:")
                for repl in found_replacements:
                    logger.info(f"- {repl}")
                    
        logger.info("-" * 50) 