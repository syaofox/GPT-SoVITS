"""
配置应用器模块

负责将角色配置应用到UI组件
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem


class ConfigApplier:
    """配置应用器，负责将角色配置应用到UI组件"""
    
    def __init__(self, experiment_tab, gpt_models, sovits_models):
        self.experiment_tab = experiment_tab
        self.gpt_models = gpt_models
        self.sovits_models = sovits_models
    
    def apply_role_config(self, role_config):
        """将角色配置应用到试听配置标签页"""
        if not role_config:
            return
            
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
    
    def apply_role_info(self, role_name, emotion_name):
        """将角色名和情绪信息应用到试听配置标签页"""
        if not role_name or not emotion_name:
            return
        
        # 填充角色名和情绪到实验页面的输入框
        self.experiment_tab.role_name_edit.setText(role_name)
        self.experiment_tab.emotion_name_edit.setText(emotion_name) 