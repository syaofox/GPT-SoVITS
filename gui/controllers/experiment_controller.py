"""
实验控制器

处理语音合成实验标签页的业务逻辑
"""

import os
from typing import Dict, List, Optional, Tuple
from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import QMessageBox

from gui.controllers.base_controller import BaseController
from gui.models.experiment_model import ExperimentModel


class ExperimentController(BaseController):
    """实验控制器类，处理语音合成实验的业务逻辑"""
    
    # 信号定义
    models_loaded = Signal()
    config_created = Signal(dict)
    ref_path_changed = Signal(str)
    aux_refs_added = Signal(list)
    aux_refs_removed = Signal(list)
    
    def __init__(self, inference_controller, role_controller, parent=None):
        """
        初始化实验控制器
        
        参数:
            inference_controller: 推理控制器
            role_controller: 角色控制器
            parent: 父对象
        """
        super().__init__(parent)
        self.inference_controller = inference_controller
        self.role_controller = role_controller
        self.experiment_model = ExperimentModel()
        
        # 项目根目录缓存
        self._project_root = None
    
    @Slot(dict)
    def load_models(self, model_paths: Dict[str, Dict[str, str]]) -> None:
        """
        加载模型列表
        
        参数:
            model_paths: 包含gpt和sovits模型路径的字典
        """
        if "gpt" in model_paths:
            self.experiment_model.load_gpt_models(model_paths["gpt"])
        
        if "sovits" in model_paths:
            self.experiment_model.load_sovits_models(model_paths["sovits"])
            
        self.models_loaded.emit()
    
    def get_gpt_model_paths(self) -> Dict[str, str]:
        """获取GPT模型路径字典"""
        return self.experiment_model.get_gpt_model_paths()
    
    def get_sovits_model_paths(self) -> Dict[str, str]:
        """获取SoVITS模型路径字典"""
        return self.experiment_model.get_sovits_model_paths()
    
    @Slot(dict, bool, result=dict)
    def get_inference_config(self, view_data: Dict, is_save_role: bool = False) -> Optional[Dict]:
        """
        获取推理配置
        
        参数:
            view_data: 从视图层收集的数据
            is_save_role: 是否为保存角色而获取配置
            
        返回:
            推理配置字典，若配置无效则返回None
        """
        config = self.experiment_model.get_inference_config(view_data, is_save_role)
        if config:
            self.config_created.emit(config)
        return config
    
    @Slot(dict, result=bool)
    def generate_speech(self, view_data: Dict) -> bool:
        """
        生成语音
        
        参数:
            view_data: 从视图层收集的数据
            
        返回:
            是否成功启动推理
        """
        # 获取合成文本
        text = view_data.get("text", "").strip()
        if not text:
            self.error_occurred.emit("请输入要合成的文本")
            return False
            
        # 获取当前配置
        config = self.get_inference_config(view_data)
        
        # 检查配置是否有效
        if config is None:
            self.error_occurred.emit("当前配置无效，请检查必填参数")
            return False
            
        # 检查必要参数
        if not config["gpt_path"]:
            self.error_occurred.emit("请选择GPT模型")
            return False
            
        if not config["sovits_path"]:
            self.error_occurred.emit("请选择SoVITS模型")
            return False
            
        if not config["ref_free"] and not config["ref_audio"]:
            self.error_occurred.emit("请选择参考音频文件或勾选无参考文本")
            return False
            
        if not config["ref_free"] and not os.path.exists(config["ref_audio"]):
            self.error_occurred.emit(f"参考音频文件不存在: {config['ref_audio']}")
            return False
            
        # 调用推理控制器
        self.inference_controller.generate_speech_async(config)
        return True
    
    @Slot(str)
    def on_ref_audio_changed(self, path: str) -> None:
        """
        处理参考音频路径改变
        
        参数:
            path: 新的参考音频路径
        """
        # 只发出信号，不再调用视图的方法，避免递归
        if path:
            self.ref_path_changed.emit(path)
    
    @Slot(list)
    def add_aux_refs(self, file_paths: List[str]) -> None:
        """
        添加辅助参考音频
        
        参数:
            file_paths: 要添加的音频文件路径列表
        """
        self.aux_refs_added.emit(file_paths)
    
    @Slot(list)
    def remove_aux_refs(self, file_paths: List[str]) -> None:
        """
        移除辅助参考音频
        
        参数:
            file_paths: 要移除的音频文件路径列表
        """
        self.aux_refs_removed.emit(file_paths)
    
    def _get_project_root(self) -> str:
        """
        获取项目根目录
        
        返回:
            项目根目录路径
        """
        # 如果已经缓存了项目根目录，直接返回
        if self._project_root:
            return self._project_root
            
        # 从当前文件位置向上查找，直到找到项目根目录
        # 这里假设项目根目录是包含'gui'文件夹的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while True:
            parent_dir = os.path.dirname(current_dir)
            if os.path.basename(current_dir) == 'gui' and os.path.exists(os.path.join(parent_dir, 'config')):
                self._project_root = parent_dir
                return parent_dir
            if current_dir == parent_dir:  # 已到达系统根目录
                self._project_root = os.getcwd()  # 返回当前工作目录作为备选
                return self._project_root
            current_dir = parent_dir
    
    @Slot(str, str, dict, result=bool)
    def save_as_role(self, role_name: str, emotion_name: str, view_data: Dict) -> bool:
        """
        保存为角色配置
        
        参数:
            role_name: 角色名称
            emotion_name: 情感名称
            view_data: 从视图层收集的数据
            
        返回:
            是否成功保存角色
        """
        # 验证输入
        if not role_name:
            self.error_occurred.emit("请输入角色名称")
            return False
            
        if not emotion_name:
            self.error_occurred.emit("请输入情绪名称")
            return False
            
        # 获取当前配置
        config = self.get_inference_config(view_data, is_save_role=True)
        
        # 检查配置是否有效
        if config is None:
            self.error_occurred.emit("当前配置无效，请检查必填参数")
            return False
        
        try:
            # 获取项目根目录
            project_root = self._get_project_root()
            
            # 将模型路径转换为相对路径
            if "gpt_path" in config and config["gpt_path"]:
                config["gpt_path"] = self.experiment_model.convert_to_relative_path(
                    config["gpt_path"], project_root
                )
                
            if "sovits_path" in config and config["sovits_path"]:
                config["sovits_path"] = self.experiment_model.convert_to_relative_path(
                    config["sovits_path"], project_root
                )
                
            # 保存角色（音频文件的复制已经在role_model中处理了）
            success = self.role_controller.save_role(role_name, emotion_name, config)
            return success
            
        except Exception as e:
            self.error_occurred.emit(f"保存角色失败: {str(e)}")
            return False 