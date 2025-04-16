"""
GPT-SoVITS 核心功能
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import logging
import re  # 添加正则表达式模块
from io import BytesIO
from time import time as ttime

import librosa
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from feature_extractor import cnhubert
from module.mel_processing import spectrogram_torch, mel_spectrogram_torch
from module.models import SynthesizerTrn, SynthesizerTrnV3
from peft import LoraConfig, PeftModel, get_peft_model
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from text.LangSegmenter import LangSegmenter
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .utils import (
    DictToAttrRecursive, find_custom_tone, revise_custom_tone, 
    dict_language, is_empty, is_full, only_punc, splits, pack_audio, convert_text,
    cut_text  # 添加这个函数
)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"  # 设置直连地址
os.environ["all_proxy"] = ""  # 清空全局代理

# 设置Python路径
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))


class Speaker:
    """说话人类，包含每个说话人的模型和配置"""
    def __init__(self, name, gpt, sovits, phones=None, bert=None, prompt=None):
        self.name = name
        self.sovits = sovits
        self.gpt = gpt
        self.phones = phones
        self.bert = bert
        self.prompt = prompt


class Sovits:
    """SoVITS模型封装类"""
    def __init__(self, vq_model, hps):
        self.vq_model = vq_model
        self.hps = hps


class Gpt:
    """GPT模型封装类"""
    def __init__(self, max_sec, t2s_model):
        self.max_sec = max_sec
        self.t2s_model = t2s_model


class GPTSoVITS:
    """GPT-SoVITS 主类"""
    def __init__(self, config):
        """初始化 GPT-SoVITS"""
        self.config = config
        self.device = config.infer_device
        self.is_half = config.is_half
        self.speakers = {}
        self.current_speaker = "default"
        
        # 初始化MeL处理相关
        self.hz = 50
        self.spec_min = -12
        self.spec_max = 2
        self.resample_transform_dict = {}
        
        # 其他组件
        self.bigvgan_model = None
        self.sr_model = None
        self.bert_model = None
        self.tokenizer = None
        self.ssl_model = None
        
        # 初始化模型组件
        self._init_models()
    
    def _init_models(self):
        """初始化基础模型组件"""
        logger.info("初始化模型组件...")
        
        # 初始化HuBERT模型
        cnhubert.cnhubert_base_path = self.config.cnhubert_path
        self.ssl_model = cnhubert.get_model()
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)
        
        # 初始化BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.config.bert_path)
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)
        
        logger.info("模型组件初始化完成")
    
    def _init_bigvgan(self):
        """初始化BigVGAN模型"""
        if self.bigvgan_model is not None:
            return
            
        from BigVGAN import bigvgan
        
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
            use_cuda_kernel=False,
        )
        # 移除权重标准化并设为评估模式
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval()
        
        if self.is_half:
            self.bigvgan_model = self.bigvgan_model.half().to(self.device)
        else:
            self.bigvgan_model = self.bigvgan_model.to(self.device)
    
    def load_models(self, sovits_path=None, gpt_path=None, speaker_name="default"):
        """加载SoVITS和GPT模型"""
        if sovits_path is None:
            sovits_path = self.config.sovits_path if self.config.sovits_path else self.config.pretrained_sovits_path
        
        if gpt_path is None:
            gpt_path = self.config.gpt_path if self.config.gpt_path else self.config.pretrained_gpt_path
        
        logger.info(f"加载模型:")
        logger.info(f"GPT 路径: {gpt_path}")
        logger.info(f"SoVITS 路径: {sovits_path}")
        
        try:
            # 加载GPT模型
            gpt = self._load_gpt_model(gpt_path)
            
            # 加载SoVITS模型
            sovits = self._load_sovits_model(sovits_path)
            
            # 保存到speakers
            self.speakers[speaker_name] = Speaker(name=speaker_name, gpt=gpt, sovits=sovits)
            self.current_speaker = speaker_name
            
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def _load_gpt_model(self, gpt_path):
        """加载GPT模型"""
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        
        if self.is_half:
            t2s_model = t2s_model.half()
        
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        
        return Gpt(max_sec, t2s_model)
    
    def _load_sovits_model(self, sovits_path):
        """加载SoVITS模型"""
        from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
        
        path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
        is_exist_s2gv3 = os.path.exists(path_sovits_v3)

        version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
        if if_lora_v3 and not is_exist_s2gv3:
            logger.warning("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")

        dict_s2 = load_sovits_new(sovits_path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps.model.version = "v2"  # v3model,v2sybomls
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"

        if model_version == "v3":
            hps.model.version = "v3"

        model_params_dict = vars(hps.model)
        if model_version != "v3":
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **model_params_dict,
            )
        else:
            vq_model = SynthesizerTrnV3(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **model_params_dict,
            )
            self._init_bigvgan()
            
        try:
            del vq_model.enc_q
        except:
            pass
            
        if self.is_half:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
            
        vq_model.eval()
        
        if not if_lora_v3:
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
        else:
            vq_model.load_state_dict(load_sovits_new(path_sovits_v3)["weight"], strict=False)
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
            vq_model.cfm = vq_model.cfm.merge_and_unload()
            vq_model.eval()

        return Sovits(vq_model, hps)
    
    def get_bert_feature(self, text, word2ph):
        """获取BERT特征"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        
        return phone_level_feature.T
    
    def clean_text_inf(self, text, language, version):
        """清理文本"""
        language = language.replace("all_", "")

        text, tone_data_list = find_custom_tone(text)
        if tone_data_list:
            print(f"tone_data_list: {tone_data_list}")

        phones, word2ph, norm_text = clean_text(text, language, version)
        # 修正多音字
        revise_custom_tone(phones, word2ph, tone_data_list)

        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text
    
    def get_bert_inf(self, phones, word2ph, norm_text, language):
        """获取BERT信息"""
        language = language.replace("all_", "")
        if language == "zh" or "<tone" in norm_text:
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)

        return bert
    
    def get_phones_and_bert(self, text, language, version, final=False):
        """获取音素和BERT特征"""
        from text import chinese
        
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")

            # 如果语言是all_zh，并且文本中包含<tone>，则将文本中的<tone>替换为空字符串
            if language == "all_zh" and "<tone" in formattext:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)

            elif language == "all_zh":
                if re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half else torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return (
            phones,
            bert.to(torch.float16 if self.is_half else torch.float32),
            norm_text,
        )
    
    def get_spepc(self, filename):
        """获取语谱图"""
        speaker = self.speakers[self.current_speaker]
        hps = speaker.sovits.hps
        
        audio, _ = librosa.load(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec
    
    def resample(self, audio_tensor, sr0):
        """重采样音频"""
        if sr0 not in self.resample_transform_dict:
            self.resample_transform_dict[sr0] = torchaudio.transforms.Resample(sr0, 24000).to(self.device)
        return self.resample_transform_dict[sr0](audio_tensor)
    
    def norm_spec(self, x):
        """正规化语谱图"""
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1
    
    def denorm_spec(self, x):
        """反正规化语谱图"""
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
    
    def audio_sr(self, audio, sr):
        """音频超分辨率"""
        if self.sr_model is None:
            from tools.audio_sr import AP_BWE
            try:
                self.sr_model = AP_BWE(self.device, DictToAttrRecursive)
            except FileNotFoundError:
                logger.info("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载")
                return audio.cpu().detach().numpy(), sr
        return self.sr_model(audio, sr)
    
    def tts(
        self,
        ref_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        top_k=15,
        top_p=0.6,
        temperature=0.6,
        speed=1,
        inp_refs=None,
        sample_steps=32,
        if_sr=False,
        spk="default",
        pause_second=0.3,
        cut_punc=",.。！？.!?",
        audio_format="wav",
        bit_depth="int16",
    ):
        """文本转语音"""
        if spk not in self.speakers:
            logger.error(f"未找到说话人 {spk}")
            return None
            
        # 文本预处理
        text = convert_text(text)
        text = cut_text(text, cut_punc)
        
        infer_sovits = self.speakers[spk].sovits
        vq_model = infer_sovits.vq_model
        hps = infer_sovits.hps
        version = vq_model.version

        infer_gpt = self.speakers[spk].gpt
        t2s_model = infer_gpt.t2s_model
        max_sec = infer_gpt.max_sec

        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
            
        prompt_language = dict_language[prompt_language.lower()]
        text_language = dict_language[text_language.lower()]
        
        dtype = torch.float16 if self.is_half else torch.float32
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * pause_second),
            dtype=np.float16 if self.is_half else np.float32,
        )
        
        all_audio = []
        
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            
            if self.is_half:
                wav16k = wav16k.half().to(self.device)
                zero_wav_torch = zero_wav_torch.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
                zero_wav_torch = zero_wav_torch.to(self.device)
                
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(1, 2)
            
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

            if version != "v3":
                refers = []
                if inp_refs:
                    for path in inp_refs:
                        try:
                            refer = self.get_spepc(path).to(dtype).to(self.device)
                            refers.append(refer)
                        except Exception as e:
                            logger.error(e)
                if len(refers) == 0:
                    refers = [self.get_spepc(ref_wav_path).to(dtype).to(self.device)]
            else:
                refer = self.get_spepc(ref_wav_path).to(self.device).to(dtype)

        phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language, version)
        texts = text.split("\n")
        
        mel_fn = lambda x: mel_spectrogram_torch(
            x,
            **{
                "n_fft": 1024,
                "win_size": 1024,
                "hop_size": 256,
                "num_mels": 100,
                "sampling_rate": 24000,
                "fmin": 0,
                "fmax": None,
                "center": False,
            },
        )

        for text in texts:
            if text == "<br>" or text == "<bt>":
                # 根据不同标记设置不同的停顿时长
                pause_duration = 0.4 if text == "<bt>" else 0.2
                print(f"插入{pause_duration}秒的空白")           
                silence_duration = int(hps.data.sampling_rate * pause_duration)
                silence_audio = np.zeros(
                    silence_duration,
                    dtype=np.float16 if self.is_half else np.float32,
                )
                all_audio.append(silence_audio)
                continue
                
            # 简单防止纯符号引发参考音频泄露
            if only_punc(text):
                continue

            if text[-1] not in splits:
                text += "。" if text_language != "en" else "."
                
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language, version)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

            if version != "v3":
                audio = (
                    vq_model.decode(
                        pred_semantic,
                        torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                        refers,
                        speed=speed,
                    )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
                )
            else:
                phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
                fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                ref_audio, sr = torchaudio.load(ref_wav_path)
                ref_audio = ref_audio.to(self.device).float()
                if ref_audio.shape[0] == 2:
                    ref_audio = ref_audio.mean(0).unsqueeze(0)
                if sr != 24000:
                    ref_audio = self.resample(ref_audio, sr)
                
                mel2 = mel_fn(ref_audio)
                mel2 = self.norm_spec(mel2)
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]
                if T_min > 468:
                    mel2 = mel2[:, :, -468:]
                    fea_ref = fea_ref[:, :, -468:]
                    T_min = 468
                chunk_len = 934 - T_min
                
                mel2 = mel2.to(dtype)
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
                    
                cmf_res = torch.cat(cfm_resss, 2)
                cmf_res = self.denorm_spec(cmf_res)
                if self.bigvgan_model is None:
                    self._init_bigvgan()
                    
                with torch.inference_mode():
                    wav_gen = self.bigvgan_model(cmf_res)
                    audio = wav_gen[0][0].cpu().detach().numpy()

            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio /= max_audio
                
            all_audio.append(audio)
            all_audio.append(zero_wav)
            
        audio_opt = np.concatenate(all_audio, 0)
        
        sr = hps.data.sampling_rate if version != "v3" else 24000
        if if_sr and sr == 24000:
            print("开始音频重采样")
            audio_opt = torch.from_numpy(audio_opt).float().to(self.device)
            audio_opt, sr = self.audio_sr(audio_opt.unsqueeze(0), sr)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
            sr = 48000
            print("音频重采样完成")
            
        # 打包音频
        # 注意：现在pack_audio函数已包含完整的类型转换逻辑，无需在此处额外转换
        audio_bytes = pack_audio(audio_opt, sr, audio_format, bit_depth)
        
        return audio_bytes.getvalue() 