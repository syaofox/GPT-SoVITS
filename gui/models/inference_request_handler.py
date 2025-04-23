"""
推理请求处理器模块

负责处理和验证生成请求
"""

import os
from PySide6.QtWidgets import QMessageBox


class InferenceRequestHandler:
    """推理请求处理器，负责处理和验证生成请求"""
    
    def __init__(self, inference_controller, text_processor, parent_widget=None):
        self.inference_controller = inference_controller
        self.text_processor = text_processor
        self.parent_widget = parent_widget
    
    def handle_experiment_request(self, config, text):
        """处理实验选项卡的生成请求"""
        if not config:
            self.show_error("无效的配置")
            return False
        
        # 应用文字替换规则
        processed_text = self.text_processor.apply_word_replace(text) if text else ""
        config["text"] = processed_text
        
        # 验证必要参数
        if not processed_text:
            self.show_error("请输入要合成的文本")
            return False
            
        if not config.get("gpt_path"):
            self.show_error("请选择GPT模型")
            return False
            
        if not config.get("sovits_path"):
            self.show_error("请选择SoVITS模型")
            return False
            
        ref_free = config.get("ref_free", False)
        ref_audio = config.get("ref_audio", "")
        if not ref_free and not ref_audio:
            self.show_error("请选择参考音频文件或勾选无参考文本")
            return False
            
        if not ref_free and not os.path.exists(ref_audio):
            self.show_error(f"参考音频文件不存在: {ref_audio}")
            return False
        
        # 确保角色名和情绪名存在
        if not config.get("role_name"):
            config["role_name"] = "未知角色"
        if not config.get("emotion_name"):
            config["emotion_name"] = "未知情绪"
            
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
        return True
    
    def handle_role_request(self, role_name, emotion_name, text, config=None):
        """处理角色选项卡的生成请求"""
        if not config:
            self.show_error("无法获取角色配置")
            return False
        
        # 应用文字替换规则
        processed_text = self.text_processor.apply_word_replace(text) if text else ""
        
        if not role_name or not emotion_name:
            self.show_error("请先选择角色和情感")
            return False
            
        if not processed_text:
            self.show_error("请输入要合成的文本")
            return False
        
        # 更新处理后的文本
        config["text"] = processed_text
        
        # 确保角色名和情绪名存在
        if not config.get("role_name"):
            config["role_name"] = role_name
        if not config.get("emotion_name"):
            config["emotion_name"] = emotion_name
            
        # 检查参考音频路径是否存在
        ref_audio = config.get("ref_audio", "")
        if not config.get("ref_free", False) and ref_audio and not os.path.exists(ref_audio):
            self.show_error(f"参考音频文件不存在: {ref_audio}")
            return False
            
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
        return True
    
    def on_generate_requested(self, config=None, text="", is_role=False):
        """处理来自标签页的生成请求"""
        if is_role:
            # 角色推理
            if config and text:
                # 应用文字替换规则
                text = self.text_processor.apply_word_replace(text)
                config["text"] = text
                self.inference_controller.generate_speech_async(config)
        else:
            # 实验模式推理
            if config:
                # 应用文字替换规则
                text = config.get("text", "")
                if text:
                    text = self.text_processor.apply_word_replace(text)
                    config["text"] = text
                self.inference_controller.generate_speech_async(config)
    
    def show_error(self, message):
        """显示错误信息"""
        if self.parent_widget:
            QMessageBox.critical(self.parent_widget, "错误", message)
        else:
            print(f"错误: {message}") 