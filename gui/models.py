"""
GUI模型层

处理数据和业务逻辑
"""

import os
import json
import uuid
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference

# 导入多线程支持
from PySide6.QtCore import QObject, QThread, Signal

class InferenceWorker(QObject):
    """推理工作线程"""
    
    finished = Signal(bool, str)  # 成功标志，结果路径或错误信息
    progress = Signal(str)  # 进度信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self.engine = None
        self.output_dir = "output"
    
    def set_config(self, config: Dict, engine, output_dir: str):
        """设置推理配置和引擎"""
        self.config = config
        self.engine = engine
        self.output_dir = output_dir
    
    def run(self):
        """执行推理任务"""
        if not self.engine:
            self.finished.emit(False, "推理引擎未初始化")
            return
            
        try:
            self.progress.emit("正在生成语音...")
            
            # 生成语音
            sample_rate, audio_data = self.engine.generate_speech(
                ref_wav_path=self.config.get("ref_audio", ""),
                prompt_text=self.config.get("prompt_text", ""),
                prompt_language=self.config.get("prompt_lang", "中文"),
                text=self.config.get("text", ""),
                text_language=self.config.get("text_lang", "中文"),
                how_to_cut=self.config.get("how_to_cut", "凑四句一切"),
                top_k=self.config.get("top_k", 20),
                top_p=self.config.get("top_p", 0.6),
                temperature=self.config.get("temperature", 0.6),
                ref_free=self.config.get("ref_free", False),
                speed=self.config.get("speed", 1.0),
                inp_refs=self.config.get("aux_refs", []),
                sample_steps=self.config.get("sample_steps", 8),
                if_sr=self.config.get("if_sr", False),
                pause_second=self.config.get("pause_second", 0.3),
            )
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.wav"
            output_path = Path(self.output_dir) / filename
            
            # 保存音频
            self.progress.emit("正在保存音频...")
            sf.write(output_path, audio_data, sample_rate)
            
            self.finished.emit(True, str(output_path))
        except Exception as e:
            error_msg = f"生成语音失败: {str(e)}"
            print(error_msg)
            self.finished.emit(False, error_msg)


class RoleModel:
    """角色模型类，用于管理角色配置"""
    
    def __init__(self, roles_dir: str = "gui/roles"):
        """
        初始化角色模型
        
        参数:
            roles_dir: 角色配置目录
        """
        self.roles_dir = Path(roles_dir)
        self.roles_dir.mkdir(exist_ok=True)
        self.roles: Dict[str, Dict] = {}
        self.load_roles()
    
    def load_roles(self) -> Dict[str, Dict]:
        """加载所有角色配置"""
        self.roles = {}
        
        for role_dir in self.roles_dir.iterdir():
            if not role_dir.is_dir():
                continue
                
            config_file = role_dir / "config.json"
            if not config_file.exists():
                continue
                
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 将相对路径的音频文件转换为绝对路径
                for emotion_name, emotion_config in config.get("emotions", {}).items():
                    if "ref_audio" in emotion_config:
                        ref_audio = emotion_config["ref_audio"]
                        # 如果只是文件名而不是路径，则添加角色目录路径
                        if not os.path.isabs(ref_audio) and "/" not in ref_audio and "\\" not in ref_audio:
                            emotion_config["ref_audio"] = str(role_dir / ref_audio)
                        elif not os.path.isabs(ref_audio):
                            # 处理可能已包含部分路径的情况
                            emotion_config["ref_audio"] = str(role_dir / os.path.basename(ref_audio))
                    
                    # 处理辅助参考音频
                    if "aux_refs" in emotion_config:
                        aux_refs = []
                        for aux_ref in emotion_config["aux_refs"]:
                            if not os.path.isabs(aux_ref) and "/" not in aux_ref and "\\" not in aux_ref:
                                # 如果只是文件名，添加角色目录路径
                                aux_refs.append(str(role_dir / aux_ref))
                            elif not os.path.isabs(aux_ref):
                                # 处理可能已包含部分路径的情况
                                aux_refs.append(str(role_dir / os.path.basename(aux_ref)))
                            else:
                                aux_refs.append(aux_ref)
                        emotion_config["aux_refs"] = aux_refs
                
                self.roles[role_dir.name] = config
            except Exception as e:
                print(f"加载角色配置失败: {role_dir.name}, 错误: {str(e)}")
        
        return self.roles
    
    def get_role_emotions(self, role_name: str) -> Dict[str, Dict]:
        """获取角色的所有情感配置"""
        role_config = self.roles.get(role_name, {})
        return role_config.get("emotions", {})
    
    def get_emotion_config(self, role_name: str, emotion_name: str) -> Dict:
        """获取特定角色的特定情感配置"""
        emotions = self.get_role_emotions(role_name)
        return emotions.get(emotion_name, {})
    
    def save_role_config(self, role_name: str, config: Dict) -> bool:
        """
        保存角色配置
        
        如果角色已存在，会合并情感配置而不是完全覆盖
        """
        role_dir = self.roles_dir / role_name
        role_dir.mkdir(exist_ok=True)
        
        config_file = role_dir / "config.json"
        
        try:
            # 检查角色是否已存在，并读取现有配置
            existing_config = {}
            if role_name in self.roles:
                existing_config = self.roles[role_name]
            
            # 合并情感配置
            merged_config = existing_config.copy()
            
            # 确保emotions键存在
            if "emotions" not in merged_config:
                merged_config["emotions"] = {}
                
            # 深拷贝配置，以防意外修改
            new_emotions = json.loads(json.dumps(config.get("emotions", {})))
            
            # 合并新的情感配置到现有配置
            for emotion_name, emotion_config in new_emotions.items():
                merged_config["emotions"][emotion_name] = emotion_config
            
            # 处理音频文件路径，将绝对路径转为相对路径保存
            for emotion_name, emotion_config in merged_config.get("emotions", {}).items():
                if "ref_audio" in emotion_config:
                    ref_audio = emotion_config["ref_audio"]
                    if os.path.isabs(ref_audio):
                        # 获取文件名
                        ref_audio_name = os.path.basename(ref_audio)
                        # 复制音频文件到角色目录
                        if os.path.exists(ref_audio):
                            new_path = role_dir / ref_audio_name
                            if ref_audio != str(new_path):
                                import shutil
                                shutil.copy2(ref_audio, new_path)
                        # 保存为相对路径（只保存文件名，不包含目录）
                        emotion_config["ref_audio"] = ref_audio_name
                
                # 处理辅助参考音频
                if "aux_refs" in emotion_config:
                    aux_refs = []
                    for aux_ref in emotion_config["aux_refs"]:
                        if os.path.isabs(aux_ref):
                            # 获取文件名
                            aux_ref_name = os.path.basename(aux_ref)
                            # 复制音频文件到角色目录
                            if os.path.exists(aux_ref):
                                new_path = role_dir / aux_ref_name
                                if aux_ref != str(new_path):
                                    import shutil
                                    shutil.copy2(aux_ref, new_path)
                            # 保存为相对路径（只保存文件名，不包含目录）
                            aux_refs.append(aux_ref_name)
                        else:
                            # 对于已经是相对路径的文件，确保只保留文件名
                            aux_ref_name = os.path.basename(aux_ref)
                            aux_refs.append(aux_ref_name)
                    emotion_config["aux_refs"] = aux_refs
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, ensure_ascii=False, indent=4)
            
            # 更新内存中的配置
            self.roles[role_name] = merged_config
            return True
        except Exception as e:
            print(f"保存角色配置失败: {role_name}, 错误: {str(e)}")
            return False


class InferenceModel:
    """推理模型类，用于处理语音合成请求"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化推理模型
        
        参数:
            output_dir: 输出音频文件目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 历史记录文件
        self.history_file = Path("gui") / "history.json"
        
        self.inference_engine: Optional[GPTSoVITSInference] = None
        self.history: List[Dict] = []
        self.current_gpt_path: str = ""
        self.current_sovits_path: str = ""
        
        # 工作线程相关
        self.worker = None
        self.thread = None
        
        # 加载历史记录
        self.load_history()
    
    def reset_engine(self):
        """重置推理引擎"""
        if self.inference_engine is not None:
            del self.inference_engine
            self.inference_engine = None
            self.current_gpt_path = ""
            self.current_sovits_path = ""
            # 强制垃圾回收以释放GPU内存
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
    def initialize_engine(
        self, 
        gpt_path: str, 
        sovits_path: str, 
        device: str = None, 
        half: bool = True
    ) -> bool:
        """
        初始化推理引擎
        
        参数:
            gpt_path: GPT模型路径
            sovits_path: SoVITS模型路径
            device: 计算设备，默认自动选择
            half: 是否使用半精度
            
        返回:
            初始化是否成功
        """
        # 如果模型路径变化，需要重置引擎
        if (self.inference_engine is not None and 
            (gpt_path != self.current_gpt_path or sovits_path != self.current_sovits_path)):
            self.reset_engine()
            
        try:
            if self.inference_engine is None:
                import torch
                self.inference_engine = GPTSoVITSInference(
                    gpt_path=gpt_path,
                    sovits_path=sovits_path,
                    device=device,
                    half=half
                )
                # 保存当前使用的模型路径
                self.current_gpt_path = gpt_path
                self.current_sovits_path = sovits_path
            return True
        except Exception as e:
            print(f"初始化推理引擎失败: {str(e)}")
            return False
    
    def generate_speech_async(self, config: Dict, on_finished, on_progress=None) -> bool:
        """
        异步生成语音
        
        参数:
            config: 推理配置
            on_finished: 完成回调函数
            on_progress: 进度回调函数
            
        返回:
            是否成功启动推理
        """
        gpt_path = config.get("gpt_path")
        sovits_path = config.get("sovits_path")
        
        if not gpt_path or not sovits_path:
            return False
        
        # 检查模型路径是否变化
        if (self.inference_engine is not None and 
            (gpt_path != self.current_gpt_path or sovits_path != self.current_sovits_path)):
            self.reset_engine()
                
        # 初始化或重新初始化推理引擎
        success = self.initialize_engine(gpt_path, sovits_path)
        if not success:
            return False
        
        # 停止现有的工作线程（如果有）
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        
        # 创建新的工作线程
        self.thread = QThread()
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)
        
        # 设置推理配置
        self.worker.set_config(config, self.inference_engine, str(self.output_dir))
        
        # 连接信号
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_finished)
        if on_progress:
            self.worker.progress.connect(on_progress)
        self.worker.finished.connect(lambda: self.thread.quit())
        self.thread.finished.connect(lambda: self._cleanup_thread())
        
        # 保存回调函数
        self._on_finished_callback = on_finished
        
        # 启动线程
        self.thread.start()
        
        return True
    
    def _on_worker_finished(self, success: bool, result: str):
        """工作线程完成回调"""
        if success:
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 添加到历史记录
            history_entry = {
                "timestamp": timestamp,
                "path": result,
                "config": {k: v for k, v in self.worker.config.items() if k != "aux_refs"},
                "text": self.worker.config.get("text", "")
            }
            self.history.append(history_entry)
            
            # 保存历史记录
            self.save_history()
        
        # 调用外部回调
        if self._on_finished_callback:
            self._on_finished_callback(success, result)
    
    def _cleanup_thread(self):
        """清理线程资源"""
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None
    
    def generate_speech(self, config: Dict) -> Tuple[bool, str]:
        """
        同步生成语音（保留以兼容现有代码）
        
        参数:
            config: 推理配置
            
        返回:
            (是否成功, 输出文件路径)
        """
        gpt_path = config.get("gpt_path")
        sovits_path = config.get("sovits_path")
        
        if not gpt_path or not sovits_path:
            return False, "未指定模型路径"
        
        # 检查模型路径是否变化
        if (self.inference_engine is not None and 
            (gpt_path != self.current_gpt_path or sovits_path != self.current_sovits_path)):
            self.reset_engine()
                
        # 初始化或重新初始化推理引擎
        success = self.initialize_engine(gpt_path, sovits_path)
        if not success:
            return False, "初始化推理引擎失败"
        
        try:
            # 生成语音
            sample_rate, audio_data = self.inference_engine.generate_speech(
                ref_wav_path=config.get("ref_audio", ""),
                prompt_text=config.get("prompt_text", ""),
                prompt_language=config.get("prompt_lang", "中文"),
                text=config.get("text", ""),
                text_language=config.get("text_lang", "中文"),
                how_to_cut=config.get("how_to_cut", "凑四句一切"),
                top_k=config.get("top_k", 20),
                top_p=config.get("top_p", 0.6),
                temperature=config.get("temperature", 0.6),
                ref_free=config.get("ref_free", False),
                speed=config.get("speed", 1.0),
                inp_refs=config.get("aux_refs", []),
                sample_steps=config.get("sample_steps", 8),
                if_sr=config.get("if_sr", False),
                pause_second=config.get("pause_second", 0.3),
            )
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.wav"
            output_path = self.output_dir / filename
            
            # 保存音频
            sf.write(output_path, audio_data, sample_rate)
            
            # 添加到历史记录
            history_entry = {
                "timestamp": timestamp,
                "path": str(output_path),
                "config": {k: v for k, v in config.items() if k != "aux_refs"},
                "text": config.get("text", "")
            }
            self.history.append(history_entry)
            
            # 保存历史记录
            self.save_history()
            
            return True, str(output_path)
        except Exception as e:
            print(f"生成语音失败: {str(e)}")
            return False, f"生成语音失败: {str(e)}"
    
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.history
    
    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self.save_history()
        return True
    
    def save_history(self):
        """保存历史记录到文件"""
        try:
            # 确保目录存在
            self.history_file.parent.mkdir(exist_ok=True)
            
            # 验证文件路径
            valid_history = []
            for entry in self.history:
                # 检查音频文件是否仍然存在
                path = entry.get("path", "")
                if os.path.exists(path):
                    valid_history.append(entry)
            
            # 保存到文件
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(valid_history, f, ensure_ascii=False, indent=4)
                
            print(f"已保存{len(valid_history)}条历史记录")
            return True
        except Exception as e:
            print(f"保存历史记录失败: {str(e)}")
            return False
    
    def load_history(self):
        """从文件加载历史记录"""
        self.history = []
        
        if not self.history_file.exists():
            return
            
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                loaded_history = json.load(f)
            
            need_save = False
            
            # 验证文件路径
            for entry in loaded_history:
                path = entry.get("path", "")
                
                # 修复可能的路径错误（如多余的目录结构）
                if path and not os.path.exists(path):
                    # 尝试从路径中提取文件名，然后在output目录中查找
                    filename = os.path.basename(path)
                    fixed_path = str(self.output_dir / filename)
                    if os.path.exists(fixed_path):
                        print(f"已修复音频文件路径: {path} -> {fixed_path}")
                        entry["path"] = fixed_path
                        need_save = True
                
                # 如果路径存在（原始或修复后），添加到历史记录
                if os.path.exists(entry.get("path", "")):
                    self.history.append(entry)
            
            print(f"已加载{len(self.history)}条历史记录")
            
            # 如果有修复的路径，重新保存历史记录
            if need_save:
                self.save_history()
                print("已重新保存修复后的历史记录")
                
        except Exception as e:
            print(f"加载历史记录失败: {str(e)}") 