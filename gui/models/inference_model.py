"""
推理模型

管理语音合成推理和历史记录
"""

import os
import json
import uuid
import gc
import torch
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime

# 导入多线程支持
from PySide6.QtCore import QThread

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference
from gui.models.inference_worker import InferenceWorker


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
        self.current_gpt_model: str = ""
        self.current_sovits_model: str = ""
        
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
            self.current_gpt_model = ""
            self.current_sovits_model = ""
            # 强制垃圾回收以释放GPU内存
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
    def initialize_engine(
        self, 
        gpt_model: str, 
        sovits_model: str, 
        device: str = None, 
        half: bool = True
    ) -> bool:
        """
        初始化推理引擎
        
        参数:
            gpt_model: GPT模型路径
            sovits_model: SoVITS模型路径
            device: 计算设备，默认自动选择
            half: 是否使用半精度
            
        返回:
            初始化是否成功
        """
        # 如果模型路径变化，需要重置引擎
        if (self.inference_engine is not None and 
            (gpt_model != self.current_gpt_model or sovits_model != self.current_sovits_model)):
            self.reset_engine()
            
        try:
            if self.inference_engine is None:
                self.inference_engine = GPTSoVITSInference(
                    gpt_model=gpt_model,
                    sovits_model=sovits_model,
                    device=device,
                    half=half
                )
                # 保存当前使用的模型路径
                self.current_gpt_model = gpt_model
                self.current_sovits_model = sovits_model
            return True
        except Exception as e:
            print(f"初始化推理引擎失败: {str(e)}")
            return False
    
    def generate_speech_async(self, config: Dict, on_finished: Callable, on_progress=None) -> bool:
        """
        异步生成语音
        
        参数:
            config: 推理配置
            on_finished: 完成回调函数
            on_progress: 进度回调函数
            
        返回:
            是否成功启动推理
        """
        gpt_model = config.get("gpt_model")
        sovits_model = config.get("sovits_model")
        
        if not gpt_model or not sovits_model:
            return False
        
        # 检查模型路径是否变化
        if (self.inference_engine is not None and 
            (gpt_model != self.current_gpt_model or sovits_model != self.current_sovits_model)):
            self.reset_engine()
                
        # 初始化或重新初始化推理引擎
        success = self.initialize_engine(gpt_model, sovits_model)
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
        gpt_model = config.get("gpt_model")
        sovits_model = config.get("sovits_model")
        
        if not gpt_model or not sovits_model:
            return False, "未指定模型路径"
        
        # 检查模型路径是否变化
        if (self.inference_engine is not None and 
            (gpt_model != self.current_gpt_model or sovits_model != self.current_sovits_model)):
            self.reset_engine()
                
        # 初始化或重新初始化推理引擎
        success = self.initialize_engine(gpt_model, sovits_model)
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