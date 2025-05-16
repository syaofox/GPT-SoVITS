import os
import torch
import numpy as np
import torchaudio
import librosa
from typing import Tuple

from models.logger import debug, error, info
from models.model_loader import ModelLoader
from models.audio_processor import AudioProcessor
from models.text_processor import TextProcessor
from models.cache_manager import CacheManager


class GPTSoVITSServer:
    """
    GPT-SoVITS语音合成服务
    支持v1, v2, v3, v4版本模型的调用
    """

    def __init__(self, is_half: bool = True, device: str = None) -> None:
        """
        初始化GPT-SoVITS语音合成服务

        Args:
            is_half: 是否使用半精度
            device: 设备，默认自动选择
        """
        # 设置设备选择
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 是否使用半精度
        self.is_half = is_half and torch.cuda.is_available()

        # 初始化模块
        self.model_loader = ModelLoader(self.device, self.is_half)
        self.audio_processor = AudioProcessor(self.device, self.is_half)
        self.text_processor = TextProcessor(self.device, self.is_half)

        # 设置模型路径
        self.gpt_path = None
        self.sovits_path = None

        # 版本相关属性，委托给ModelLoader管理
        self.v3v4set = {"v3", "v4"}

    def load_models(self, gpt_path: str, sovits_path: str) -> None:
        """
        加载所有模型

        Args:
            gpt_path: GPT模型路径
            sovits_path: SoVITS模型路径
        """
        # 检查是否需要重新加载
        if (
            self.gpt_path == gpt_path
            and self.sovits_path == sovits_path
            and self.model_loader.vq_model is not None
            and self.model_loader.t2s_model is not None
        ):
            debug("使用缓存的模型，跳过加载过程")
            return

        # 记录模型路径
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path

        # 加载基础模块
        modules = self.model_loader.load_base_modules()

        # 设置辅助模块的函数引用
        if modules:
            self.text_processor.clean_text = modules.get("clean_text")
            self.text_processor.cleaned_text_to_sequence = modules.get(
                "cleaned_text_to_sequence"
            )
            self.text_processor.LangSegmenter = modules.get("LangSegmenter")
            self.audio_processor.mel_spectrogram_torch = modules.get(
                "mel_spectrogram_torch"
            )
            self.audio_processor.spectrogram_torch = modules.get("spectrogram_torch")

        # 加载BERT模型
        self.model_loader.load_bert()
        self.text_processor.bert_model = self.model_loader.bert_model
        self.text_processor.tokenizer = self.model_loader.tokenizer

        # 加载SSL模型
        self.model_loader.load_ssl()

        # 加载SoVITS模型
        self.model_loader.load_sovits_model(sovits_path)

        # 同步属性
        self.audio_processor.set_hps(self.model_loader.hps)
        self.text_processor.set_language_dict(self.model_loader.version)

        # 加载GPT模型
        self.model_loader.load_gpt_model(gpt_path)

        # 根据模型版本管理声码器
        self.model_loader.manage_vocoders()

    def generate_speech(
        self,
        text: str,  # 需要合成的文本
        text_language: str = "中文",  # 文本语言
        ref_wav_path: str = None,  # 参考音频路径
        prompt_text: str = None,  # 参考文本
        prompt_language: str = "中文",  # 参考文本语言
        how_to_cut: str = "凑四句一切",  # 文本切分方式
        top_k: int = 15,  # GPT采样参数top_k
        top_p: float = 1,  # GPT采样参数top_p
        temperature: float = 1,  # GPT采样参数temperature
        ref_free: bool = False,  # 是否无参考文本模式
        speed: float = 1.0,  # 语速
        if_sr: bool = False,  # 是否使用超分辨率模型 (仅v3支持)
        sample_steps: int = 8,  # 采样步数 (仅v3,v4支持)
        pause_second: float = 0.3,  # 句间停顿秒数
        inp_refs: list = None,  # 多个参考音频路径列表 (仅v1,v2支持)
        gpt_path: str = None,  # GPT模型路径，如果提供会覆盖初始化时设置的路径
        sovits_path: str = None,  # SoVITS模型路径
        process_callback: callable = None,  # 处理回调函数
        process_current_segment: int = 0,  # 处理段落
        process_total_segment: int = 0,  # 处理总段落
    ) -> Tuple[int, np.ndarray]:  # 返回值: 采样率和音频数据
        """
        生成语音

        Args:
            ref_wav_path: 参考音频路径
            text: 需要合成的文本
            text_language: 文本语言
            prompt_text: 参考文本
            prompt_language: 参考文本语言
            how_to_cut: 文本切分方式
            top_k: GPT采样参数
            top_p: GPT采样参数
            temperature: GPT采样参数
            ref_free: 是否无参考文本模式
            speed: 语速
            if_sr: 是否使用超分辨率模型 (仅v3支持)
            sample_steps: 采样步数 (仅v3,v4支持)
            pause_second: 句间停顿秒数
            inp_refs: 多个参考音频路径列表 (仅v1,v2支持)
            gpt_path: GPT模型路径，如果提供会覆盖初始化时设置的路径
            sovits_path: SoVITS模型路径，如果提供会覆盖初始化时设置的路径
            process_callback: 处理回调函数
            process_current_segment: 处理段落
            process_total_segment: 处理总段落

        Returns:
            采样率和音频数据
        """
        debug(
            f"\n>> 生成语音参数: \n"
            f">> text: {text}\n"
            f">> text_language: {text_language}\n"
            f">> ref_wav_path: {ref_wav_path}\n"
            f">> prompt_text: {prompt_text}\n"
            f">> prompt_language: {prompt_language}\n"
            f">> how_to_cut: {how_to_cut}\n"
            f">> top_k: {top_k}\n"
            f">> top_p: {top_p}\n"
            f">> temperature: {temperature}\n"
            f">> ref_free: {ref_free}\n"
            f">> speed: {speed}\n"
            f">> if_sr: {if_sr}\n"
            f">> sample_steps: {sample_steps}\n"
            f">> pause_second: {pause_second}\n"
            f">> inp_refs: {inp_refs}\n"
            f">> gpt_path: {gpt_path}\n"
            f">> sovits_path: {sovits_path}\n"
        )

        # 检查并加载模型
        if gpt_path is not None or sovits_path is not None:
            # 优先使用传入的模型路径
            current_gpt_path = gpt_path or self.gpt_path
            current_sovits_path = sovits_path or self.sovits_path

            # 加载模型（会使用缓存机制）
            self.load_models(current_gpt_path, current_sovits_path)
        elif self.model_loader.vq_model is None or self.model_loader.t2s_model is None:
            # 如果没有传入模型路径且当前没有加载模型
            if not self.gpt_path or not self.sovits_path:
                raise ValueError(
                    "未指定模型路径，请在初始化或调用generate_speech时提供gpt_path和sovits_path"
                )
            self.load_models(self.gpt_path, self.sovits_path)

        # 输入检查
        if not ref_wav_path:
            raise ValueError("请指定参考音频")
        if not text:
            raise ValueError("请指定需要合成的文本")

        # 获取当前模型属性的便捷引用
        hps = self.model_loader.hps
        vq_model = self.model_loader.vq_model
        t2s_model = self.model_loader.t2s_model
        ssl_model = self.model_loader.ssl_model
        hz = self.model_loader.hz
        max_sec = self.model_loader.max_sec
        model_version = self.model_loader.model_version
        version = self.model_loader.version
        dict_language = self.text_processor.dict_language
        splits = self.text_processor.splits

        # 设置语言代码
        prompt_language_code = dict_language[prompt_language]
        text_language_code = dict_language[text_language]

        # 模型兼容性检查
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        if model_version in self.v3v4set:
            ref_free = False  # v3,v4暂不支持ref_free
        else:
            if_sr = False  # 超分辨率仅在v3支持

        # 处理参考文本
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_language_code != "en" else "."
            info(f"实际输入的参考文本: {prompt_text}")

        # 处理目标文本

        # 替换文本
        debug("处理文本替换")
        count, text = self.text_processor._apply_replace_rules(text)
        info(f"文本替换完成，共替换 {count} 处")

        text = text.strip("\n")
        debug(f"实际输入的目标文本: {text}")

        # 空行替换成静音标记<BR>
        text = self.text_processor.replace_empty_lines_with_br(text)

        debug(f"实际输入的目标文本(空行替换成静音标记<BR>): {text}")

        # 创建零音频作为停顿
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * pause_second),
            dtype=np.float16 if self.is_half else np.float32,
        )
        zero_wav_torch = torch.from_numpy(zero_wav)
        if self.is_half:
            zero_wav_torch = zero_wav_torch.half().to(self.device)
        else:
            zero_wav_torch = zero_wav_torch.to(self.device)

        # 处理参考音频并提取语义信息
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 320000 or wav16k.shape[0] < 48000:
                    raise ValueError("参考音频在3~20秒范围外，请更换！")
                wav16k = torch.from_numpy(wav16k)
                if self.is_half:
                    wav16k = wav16k.half().to(self.device)
                else:
                    wav16k = wav16k.to(self.device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(1, 2)
                codes = vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(self.device)

        # 文本切分
        text = self.text_processor.cut_text(text, how_to_cut)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        debug(f"实际输入的目标文本(切句后): {text}")

        texts = text.split("\n")
        texts = self.text_processor.process_text(texts)
        audio_opt = []

        # 处理参考文本
        if not ref_free:
            phones1, bert1, norm_text1 = self.text_processor.get_phones_and_bert(
                prompt_text, prompt_language_code, version
            )

        # 构建缓存键，加入inp_refs参数
        cache_key_base = f"{ref_wav_path}_{prompt_text}_{prompt_language}_{text_language}_{top_k}_{top_p}_{temperature}_{speed}"
        # 处理inp_refs以生成缓存键的附加部分
        inp_refs_key = ""
        if inp_refs and model_version not in self.v3v4set:
            # 对inp_refs进行排序并连接以确保缓存键的一致性
            inp_refs_key = "_" + "_".join(
                sorted([os.path.basename(path) for path in inp_refs])
            )
        cache_key_base += inp_refs_key

        current_step = 0
        total_step = len(texts)

        # 对每个文本段落进行处理
        for i_text, text in enumerate(texts):
            if text == self.text_processor.BR_TAG:
                audio_opt.append(zero_wav_torch)  # 添加停顿
                debug(f"添加换行停顿{pause_second}秒")
                continue

            # 解决空行导致报错的问题
            if len(text.strip()) == 0:
                continue

            # 添加结尾标点符号
            if text[-1] not in splits:
                text += "。" if text_language_code != "en" else "."
            info(f"实际输入的目标文本(每句): {text}")

            if process_callback:
                process_callback(
                    process_current_segment / process_total_segment,
                    f"总进度 {process_current_segment}/{process_total_segment}，当前进度 {current_step}/{total_step}，文本: {text[:30] + '...' if len(text) > 30 else text}",
                )
            current_step += 1

            # 获取音素和BERT特征
            phones2, bert2, norm_text2 = self.text_processor.get_phones_and_bert(
                text, text_language_code, version
            )
            info(f"前端处理后的文本(每句): {norm_text2}")

            # 准备输入
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = (
                    torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
                )
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            # 生成语义表示
            with torch.no_grad():
                # 创建每句的缓存键
                current_cache_key = f"{cache_key_base}_{i_text}_{text}"
                # 从缓存管理器中获取
                cached_semantic = CacheManager.get(current_cache_key)

                if cached_semantic is not None:
                    pred_semantic = cached_semantic
                else:
                    pred_semantic, idx = t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        None if ref_free else prompt,
                        bert,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=hz * max_sec,
                    )
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    # 将结果存入缓存
                    CacheManager.set(current_cache_key, pred_semantic)

            # 根据模型版本进行处理
            if model_version not in self.v3v4set:
                # v1, v2模型
                # 处理多参考音频
                refers = []

                if inp_refs:
                    # 添加对多个参考音频的处理
                    for path in inp_refs:
                        try:
                            refer = (
                                self.audio_processor.get_spepc(
                                    path, self.audio_processor.spectrogram_torch
                                )
                                .to(self.audio_processor.dtype)
                                .to(self.device)
                            )
                            refers.append(refer)
                        except Exception as e:
                            error(f"处理参考音频 {path} 时出错: {e}")

                info(f"辅助参考音频: {inp_refs}")

                # 如果没有有效的多参考音频，则使用单个参考音频
                if len(refers) == 0:
                    refers = [
                        self.audio_processor.get_spepc(
                            ref_wav_path, self.audio_processor.spectrogram_torch
                        )
                        .to(self.audio_processor.dtype)
                        .to(self.device)
                    ]

                audio = vq_model.decode(
                    pred_semantic,
                    torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                    refers,
                    speed=speed,
                )[0][0]
            else:
                # v3, v4模型
                refer = (
                    self.audio_processor.get_spepc(
                        ref_wav_path, self.audio_processor.spectrogram_torch
                    )
                    .to(self.device)
                    .to(self.audio_processor.dtype)
                )

                phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

                fea_ref, ge = vq_model.decode_encp(
                    prompt.unsqueeze(0), phoneme_ids0, refer
                )
                ref_audio, sr = torchaudio.load(ref_wav_path)
                ref_audio = ref_audio.to(self.device).float()
                if ref_audio.shape[0] == 2:
                    ref_audio = ref_audio.mean(0).unsqueeze(0)

                tgt_sr = 24000 if model_version == "v3" else 32000
                if sr != tgt_sr:
                    ref_audio = self.audio_processor.resample(ref_audio, sr, tgt_sr)

                # 设置模型版本特定函数
                mel_fn = (
                    self.audio_processor.mel_fn_v3
                    if model_version == "v3"
                    else self.audio_processor.mel_fn_v4
                )

                mel2 = mel_fn(ref_audio)
                mel2 = self.audio_processor.norm_spec(mel2)
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]

                Tref = 468 if model_version == "v3" else 500
                Tchunk = 934 if model_version == "v3" else 1000

                if T_min > Tref:
                    mel2 = mel2[:, :, -Tref:]
                    fea_ref = fea_ref[:, :, -Tref:]
                    T_min = Tref

                chunk_len = Tchunk - T_min
                mel2 = mel2.to(self.audio_processor.dtype)
                fea_todo, ge = vq_model.decode_encp(
                    pred_semantic, phoneme_ids1, refer, ge, speed
                )

                cfm_resss = []
                idx = 0
                while 1:
                    fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                    if fea_todo_chunk.shape[-1] == 0:
                        break
                    idx += chunk_len
                    fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                    cfm_res = vq_model.cfm.inference(
                        fea,
                        torch.LongTensor([fea.size(1)]).to(fea.device),
                        mel2,
                        sample_steps,
                        inference_cfg_rate=0,
                    )
                    cfm_res = cfm_res[:, :, mel2.shape[2] :]
                    mel2 = cfm_res[:, :, -T_min:]
                    fea_ref = fea_todo_chunk[:, :, -T_min:]
                    cfm_resss.append(cfm_res)

                cfm_res = torch.cat(cfm_resss, 2)
                cfm_res = self.audio_processor.denorm_spec(cfm_res)

                # 确保声码器已经加载
                self.model_loader.manage_vocoders()

                # 使用对应的声码器
                with torch.inference_mode():
                    if model_version == "v3":
                        wav_gen = self.model_loader.bigvgan_model(cfm_res)
                    else:  # v4
                        wav_gen = self.model_loader.hifigan_model(cfm_res)
                    audio = wav_gen[0][0]

            # 简单防止16bit爆音
            max_audio = torch.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio

            audio_opt.append(audio)
            audio_opt.append(zero_wav_torch)  # 添加停顿

        # 合并所有音频段
        audio_opt = torch.cat(audio_opt, 0)

        # 确定采样率并进行后处理
        if model_version in {"v1", "v2"}:
            opt_sr = 32000
        elif model_version == "v3":
            opt_sr = 24000
        else:  # v4
            opt_sr = 48000

        # 音频超分
        if if_sr and opt_sr == 24000:
            info("音频超分中")
            from tools.audio_sr import AP_BWE

            try:
                audio_sr_model = AP_BWE(
                    self.device, self.model_loader.dict_to_attr_recursive
                )
                audio_opt, opt_sr = audio_sr_model(audio_opt.unsqueeze(0), opt_sr)
                max_audio = np.abs(audio_opt).max()
                if max_audio > 1:
                    audio_opt /= max_audio
            except Exception as e:
                error(f"音频超分失败，使用原始音频: {e}")
                audio_opt = audio_opt.cpu().detach().numpy()
        else:
            audio_opt = audio_opt.cpu().detach().numpy()

        return opt_sr, (audio_opt * 32767).astype(np.int16)
