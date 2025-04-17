"""
推理模型
负责处理语音合成的业务逻辑
"""
import os
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import QThread, Signal

class InferenceModel:
    """推理模型类"""
    
    def __init__(self):
        """初始化推理模型"""
        self.output_dir = Path("output")
        self.current_role = None
        self.current_emotion = None
        self.current_config = {}
        
        # 确保输出目录存在
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def get_history_list(self):
        """获取历史音频列表"""
        history_list = []
        if not self.output_dir.exists():
            return history_list
        
        # 加载所有WAV文件
        wav_files = list(self.output_dir.glob("*.wav"))
        wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for wav_file in wav_files:
            history_list.append({
                "name": wav_file.name,
                "path": str(wav_file),
                "mtime": wav_file.stat().st_mtime
            })
        return history_list
    
    def clear_history(self):
        """清空历史音频"""
        try:
            for wav_file in self.output_dir.glob("*.wav"):
                wav_file.unlink()
            return True
        except Exception as e:
            print(f"清空历史记录出错: {str(e)}")
            return False
    
    def generate_output_path(self, text):
        """生成输出文件路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_prefix = text[:10].replace(' ', '_')
        output_path = self.output_dir / f"{text_prefix}_{timestamp}.wav"
        return output_path
    
    def prepare_inference_params(self, config, emotion_name, text, user_params=None):
        """准备推理参数"""
        if not config:
            return None, "配置信息为空"
        
        emotions = config.get("emotions", {})
        emotion_config = emotions.get(emotion_name, {})
        
        if not emotion_config:
            return None, "未找到音色配置"
        
        # 基本参数
        params = {
            "prompt_language": config.get("prompt_lang", "中文"),
            "text": text,
            "text_language": config.get("text_lang", "中文"),
            "top_k": config.get("top_k", 15),
            "top_p": config.get("top_p", 1.0),
            "temperature": config.get("temperature", 1.0),
            "speed": config.get("speed", 1.0),
            "sample_steps": config.get("sample_steps", 32),
            "if_sr": config.get("if_sr", False),
            "pause_second": config.get("pause_second", 0.3),
            "ref_free": config.get("ref_free", False),
            "cut_punc": "。！？.!?",
            "audio_format": "wav",
            "bit_depth": "int16",
        }
        
        # 覆盖用户自定义参数
        if user_params:
            for key, value in user_params.items():
                params[key] = value
        
        # 参考音频和文本
        ref_audio = emotion_config.get("ref_audio", "")
        prompt_text = emotion_config.get("prompt_text", "")
        
        if ref_audio:
            role_dir = Path("configs/roles")
            role_name = config.get("name", "")
            
            # 确保使用完整的角色名称
            if not role_name:
                # 如果name字段为空，尝试从config中获取name
                for key, value in config.items():
                    if key not in ["emotions", "version", "text_lang", "prompt_lang", 
                                  "gpt_path", "sovits_path", "speed", "ref_free", 
                                  "if_sr", "top_k", "top_p", "temperature", 
                                  "sample_steps", "pause_second", "description"]:
                        role_name = key
                        break
            
            ref_path = role_dir / role_name / ref_audio
            if not ref_path.exists():
                # 尝试在configs/roles目录下查找正确的角色目录
                for role_dir_item in role_dir.iterdir():
                    if role_dir_item.is_dir():
                        potential_path = role_dir_item / ref_audio
                        if potential_path.exists():
                            ref_path = potential_path
                            break
                
                if not ref_path.exists():
                    return None, f"参考音频文件不存在: {ref_path}"
            
            params["ref_wav_path"] = str(ref_path)
        else:
            # 必须有参考音频才能推理
            return None, "未指定参考音频文件"
        
        params["prompt_text"] = prompt_text
        
        # 辅助参考音频
        aux_refs = emotion_config.get("aux_refs", [])
        if aux_refs:
            role_dir = Path("configs/roles")
            valid_aux_refs = []
            for aux_path in aux_refs:
                full_path = role_dir / config.get("name", "") / aux_path
                if not full_path.exists() and Path(aux_path).exists():
                    full_path = Path(aux_path)
                if full_path.exists():
                    valid_aux_refs.append(str(full_path))
            params["inp_refs"] = valid_aux_refs if valid_aux_refs else None
        else:
            params["inp_refs"] = None
        
        return params, None


class InferenceThread(QThread):
    """推理线程类"""
    
    progress_update = Signal(tuple)
    inference_complete = Signal(str)
    inference_error = Signal(str)
    
    def __init__(self, gpt_path, sovits_path, params):
        super().__init__()
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        self.params = params
    
    def hook_progress(self, current_index, total_segments):
        """处理段落进度回调
        
        Args:
            current_index: 当前处理的段落索引 (从0开始)
            total_segments: 总段落数
        """
        if total_segments <= 0:
            return
        
        # 计算进度百分比 (40%~80%的范围，因为前40%用于加载模型，最后20%用于保存音频)
        progress_percent = 40 + int((current_index / total_segments) * 40)
        
        # 构建段落信息字典
        segment_info = {
            'current_segment': current_index,
            'total_segments': total_segments
        }
        
        # 发出带有段落信息的进度更新信号
        self.progress_update.emit((progress_percent, segment_info))
    
    def run(self):
        """运行推理"""
        try:
            # 模拟进度更新
            self.progress_update.emit((10, {}))
            
            # 导入GPTSoVITS
            from gpt_sovits_lib import GPTSoVITS, GPTSoVITSConfig
            
            # 初始化GPTSoVITS
            config = GPTSoVITSConfig()
            config.gpt_path = self.gpt_path
            config.sovits_path = self.sovits_path
            
            self.progress_update.emit((20, {}))
            
            gpt_sovits = GPTSoVITS(config)
            gpt_sovits.load_models()
            
            self.progress_update.emit((30, {}))
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_prefix = self.params["text"][:10].replace(' ', '_').replace('\n', '')
            output_path = Path("output") / f"{text_prefix}_{timestamp}.wav"
            
            self.progress_update.emit((40, {}))
            
            # 执行推理
            tts_params = self.params.copy()
            if "spk" in tts_params:
                if tts_params["spk"] is None:
                    tts_params["spk"] = "default"
            
            result_audio_bytes = gpt_sovits.tts(
                ref_wav_path=tts_params.get("ref_wav_path"),
                prompt_text=tts_params.get("prompt_text"),
                prompt_language=tts_params.get("prompt_language"),
                text=tts_params.get("text"),
                text_language=tts_params.get("text_language"),
                top_k=tts_params.get("top_k"),
                top_p=tts_params.get("top_p"),
                temperature=tts_params.get("temperature"),
                speed=tts_params.get("speed"),
                inp_refs=tts_params.get("inp_refs"),
                sample_steps=tts_params.get("sample_steps"),
                if_sr=tts_params.get("if_sr"),
                pause_second=tts_params.get("pause_second"),
                cut_punc=tts_params.get("cut_punc"),
                audio_format=tts_params.get("audio_format"),
                bit_depth=tts_params.get("bit_depth"),
                progress_callback=self.hook_progress  # 添加进度回调
            )
            
            self.progress_update.emit((80, {}))
            
            # 保存音频字节流到WAV文件
            with open(str(output_path), 'wb') as f:
                f.write(result_audio_bytes)
            
            self.progress_update.emit((100, {}))
            
            # 发送完成信号
            self.inference_complete.emit(str(output_path))
            
        except Exception as e:
            self.inference_error.emit(f"推理失败: {str(e)}")
            print(f"推理线程错误: {str(e)}") 