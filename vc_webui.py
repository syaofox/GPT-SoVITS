

"""
受 GPT-SoVITS 启发
"""

import os
import os.path as osp
import re
import logging
from time import time as ttime
from warnings import warn
from pathlib import Path
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

import torch
from torch import nn
import torch.nn.functional as F
import librosa
import numpy as np
import LangSegment
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto
from tools import my_utils

def get_pretrain_model_path(env_name, log_file, def_path):
    """ 获取预训练模型路径
    env_name: 从环境变量获取，第一优先级
    log_file: 记录在文本文件内，第二优先级
    def_path: 传参，第三优先级
    """
    if osp.isfile(log_file):
        def_path = open(log_file, 'r', encoding="utf-8").read()
    pretrain_path = os.environ.get(env_name, def_path)
    return pretrain_path


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

gpt_path = get_pretrain_model_path('gpt_path', "./gweight.txt", 
                                   "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

sovits_path = get_pretrain_model_path('sovits_path', "./sweight.txt",
                                      "GPT_SoVITS/pretrained_models/s2G488k.pth")

cnhubert_base_path = get_pretrain_model_path("cnhubert_base_path", '', "GPT_SoVITS/pretrained_models/chinese-hubert-base")

bert_path = get_pretrain_model_path("bert_path", '', "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")

vc_webui_port = int(os.environ.get("vc_webui_port", 9888))  # specify gradio port
print(f'port: {vc_webui_port}')

is_share = eval(os.environ.get("is_share", "False"))

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

# is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
is_half = False

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

cnhubert.cnhubert_base_path = cnhubert_base_path

i18n = I18nAuto()

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0][1:-1]

        # 将结果移动到目标设备
        res = res.to(device)

    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def replace_chinese(text):
    pattern = r"([\u4e00-\u9fa5]{5}).*"
    result = re.sub(pattern, r"\1...", text)
    return result


def init_wav_list(sovits_path):
    wav_path = ""

    match = re.search(r"(.+)_e\d+_s\d+\.pth", sovits_path)
    if match:
        result = match.group(1).replace("SoVITS_weights/", "")
        wav_path = f"./sample/{result}/"

    else:
        return [], {}

    if not os.path.exists(wav_path):
        return [], {}

    res_wavs = {}

    res_text = ["请选择参考音频"]

    # 读取文本

    # 遍历目录
    for file_path in os.listdir(wav_path)[:200]:
        wfile_path = os.path.join(wav_path, file_path)
        if os.path.isfile(wfile_path):
            match = re.search(r"(_\d+秒).wav$", file_path)
            if match:
                # 提取主文本和后缀
                suffix = match.group(1)
                main_text = file_path[: match.start()]
            else:
                continue

            key = f"{main_text}{suffix}"

            res_text.append(key)

            # 情绪
            match2 = re.search(r"^【.+?】(.+)$", main_text)
            if match2:
                main_text = match2.group(1)

            res_wavs[key] = (
                wfile_path,
                main_text,
            )

    return res_text, res_wavs


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location=device)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)

    sovits_path_in = sovits_path.replace("SoVITS_weights/", "")

    print(sovits_path_in)

    global reference_wavs, reference_dict, max_textboxes
    reference_wavs, reference_dict = init_wav_list(sovits_path_in)
    max_textboxes = len(reference_wavs)

    return gr.update(choices=reference_wavs)

change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location=device)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
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


dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}


# def clean_text_inf(text, language):
#    phones, word2ph, norm_text = clean_text(text, language)
#    phones = cleaned_text_to_sequence(phones)
#    return phones, word2ph, norm_text


def clean_text_inf(text, language):
    """
    text: 字符串
    language: 所属语言
    
    return:
    phones: 音素 id 序列
    word2ph: 每个字转音素后，对应的个数，对于中文，就是声韵母，因此是全是 2 的 list
    norm_text: 归一化后文本
    """
    formattext = ""
    language = language.replace("all_","")
    for tmp in LangSegment.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    # print(f'音素: {phones}')
    phones = cleaned_text_to_sequence(phones)  # 统一了中、英、日等
    # print(f'音素 id: {phones}')
    return phones, word2ph, norm_text


dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()


@torch.no_grad()
def get_code_from_ssl(ssl):
    ssl = vq_model.ssl_proj(ssl)
    quantized, codes, commit_loss, quantized_list = vq_model.quantizer(ssl)
    # print(codes.shape, codes.dtype)  # [n_q, B, T]
    return codes.transpose(0, 1)  # [B, n_q, T]


@torch.no_grad()
def get_code_from_wav(wav_path):
    wav16k, sr = librosa.load(wav_path, sr=16000)
    if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
        # raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        warn(i18n("参考音频在3~10秒范围外，请更换！"))
    wav16k = torch.from_numpy(wav16k)
    if is_half == True:
        wav16k = wav16k.half().to(device)
    else:
        wav16k = wav16k.to(device)
    ssl_content = ssl_model.model(wav16k.unsqueeze(0))[ 
        "last_hidden_state"
    ].transpose(
        1, 2
    )  # .float()
    codes = get_code_from_ssl(ssl_content)  # [B, n_q, T]

    prompt_semantic = codes[0, 0] 
    return prompt_semantic


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist


def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def get_cleaned_text_final(text,language):
    if language in {"en","all_zh","all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text


@torch.no_grad()
def vc_main(wav_path, text, language, prompt_wav, noise_scale=0.5):
    """ Voice Conversion
    wav_path: 待变声的源音频
    text: 对应文本
    language: 对应语言
    prompt_wav: 目标人声
    """
    language = dict_language[language]

    phones, word2ph, norm_text = get_cleaned_text_final(text, language)

    spec = get_spepc(hps, prompt_wav).to(device)  # 将 spec 移动到 device
    codes = get_code_from_wav(wav_path)[None, None].to(device)  # 必须是 3D, [n_q, B, T]
    
    ge = vq_model.ref_enc(spec)  # [B, D, T/1]
    quantized = vq_model.quantizer.decode(codes)  # [B, D, T]
    
    if hps.model.semantic_frame_rate == "25hz":
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )
    
    _, m_p, logs_p, y_mask = vq_model.enc_p(
        quantized, 
        torch.LongTensor([quantized.shape[-1]]).to(device), 
        torch.LongTensor(phones)[None].to(device), 
        torch.LongTensor([len(phones)]).to(device), 
        ge
    )
    
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = vq_model.flow(z_p, y_mask, g=ge, reverse=True)
    o = vq_model.dec((z * y_mask)[:, :, :], g=ge)  # [B, D=1, T], torch.float32 (-1, 1)
    
    audio = o.detach().cpu().numpy()[0, 0]    
    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
    
    if max_audio > 1:
        audio /= max_audio
    
    yield hps.data.sampling_rate, (audio * 32768).astype(np.int16)





# 切换参考音频



# def change_sovits_weights(sovits_path_in):
#     sovits_path_in = sovits_path_in.replace("SoVITS_weights/", "")

#     print(sovits_path_in)

#     global reference_wavs, reference_dict, max_textboxes
#     reference_wavs, reference_dict = init_wav_list(sovits_path_in)
#     max_textboxes = len(reference_wavs)

#     return gr.update(choices=reference_wavs)

reference_wavs, reference_dict = init_wav_list(sovits_path)


def change_wav(audio_name):
    first_key = list(reference_dict.keys())[0]

    try:
        value = reference_dict[audio_name]
        return value[0]
    except Exception as e:
        return reference_dict[first_key][0]

def process_audio(audio_file_path):
    # 从文件路径中提取文件名

    main_text = os.path.basename(audio_file_path)


    main_text = re.sub(r"^【.+?】|_\d+秒|\.wav", '', main_text)
    return main_text.strip()


def merge_wav_files(input_dir):
    """
    Merge multiple WAV files located in the specified input directory into a single WAV file.

    Args:
        input_dir (str): The directory containing the input WAV files.
        output_file (str): The path to the output merged WAV file.
    """
    # Get a list of all WAV files in the input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    # Check if there are any WAV files in the directory
    if not wav_files:
        print("No WAV files found in the input directory.")
        return

    # Load each WAV file and concatenate them
    combined_audio = AudioSegment.empty()
    for wav_file in wav_files:
        wav_path = os.path.join(input_dir, wav_file)
        audio_segment = AudioSegment.from_file(wav_path, format="wav")
        combined_audio += audio_segment

   
    # Export the combined audio to a WAV file
    return combined_audio


def read_asr_list(src):
    with open(src, "r", encoding="utf-8") as file:
        # 逐行读取文件内容
        for line in file:
            # 去除行尾的换行符
            if line == "":
                continue
            line = line.strip()

            # 按 '|' 符号进行拆分
            parts = line.split("|")
            # 打印每一行的内容
            src_audio = parts[0]
            text = parts[3]

            yield src_audio, text

def change_audio(asr_list, text_language, inp_ref, output_path):
    asr_list=my_utils.clean_path(asr_list)
    output_path = my_utils.clean_path(output_path)

    for src_audio, text in tqdm(read_asr_list(asr_list)):        
        for sampling_rate, audio_data in vc_main(src_audio, text, text_language, inp_ref):
            tqdm.write(text)
            temp_path = Path(output_path) / Path(src_audio).name
            sf.write(temp_path, audio_data, sampling_rate)
            
    combined_audio =  merge_wav_files(output_path)

    combined_tempfile = os.path.join('.\TEMP', f"{Path(asr_list).stem}_combined_output.wav")
    combined_audio.export(combined_tempfile, format="wav")
    return combined_tempfile

def change_audio2(in_path, text_language, inp_ref, output_path):
    in_path=Path(my_utils.clean_path(in_path))
    output_path = my_utils.clean_path(output_path)

    for wav_file in tqdm(in_path.glob("*.wav")):
        text = wav_file.stem
        text = re.sub(r"^【.+?】|_\d+秒|\.wav", '', text)
        src_audio = str(wav_file)
    
        for sampling_rate, audio_data in vc_main(src_audio, text, text_language, inp_ref):
            tqdm.write(text)
            temp_path = Path(output_path) / Path(src_audio).name
            sf.write(temp_path, audio_data, sampling_rate)
            
    combined_audio =  merge_wav_files(output_path)

    combined_tempfile = os.path.join('.\TEMP', f"{in_path.stem}_combined_output.wav")
    combined_audio.export(combined_tempfile, format="wav")
    return combined_tempfile


def main():
    

    with gr.Blocks(title="GPT-SoVITS-VC WebUI") as app:
        
        gr.Markdown(
            value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
        )

        with gr.Group():
            gr.Markdown(value=i18n("模型切换"))

            with gr.Row():
                GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True)
                SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path, interactive=True)
                wavs_dropdown = gr.Dropdown(label=i18n("参考音频列表"),choices=reference_wavs, value="请选择参考音频", interactive=True)
                refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
                SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [wavs_dropdown])
                GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])
            
            gr.Markdown(value=i18n("* 请上传目标音色音频，要求说话人单一，声音干净"))
            with gr.Row():
                inp_ref = gr.Audio(label=i18n("请上传 3~10 秒内参考音频，超过会报警！"), type="filepath")
            
            wavs_dropdown.change(change_wav, [wavs_dropdown], [inp_ref])

            gr.Markdown(value=i18n("* 请填写需要变声/转换的源音频，以及对应文本"))
            with gr.Row():
                src_audio = gr.Audio(label=i18n('源音频'), type='filepath')
                text = gr.Textbox(label=i18n("源音频对应文本"), value="")
                text_language = gr.Dropdown(
                    label=i18n("文本语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
                )

                inference_button = gr.Button(i18n("合成语音"), variant="primary")
                output = gr.Audio(label=i18n("变声后"),show_download_button=True)

                src_audio.upload(process_audio, inputs=[src_audio], outputs=[text])

            inference_button.click(
                vc_main,
                [src_audio, text, text_language, inp_ref],
                [output],
            )

            gr.Markdown(value=i18n("* 根据标注文件批量变声"))
            with gr.Row():
                asr_list = gr.Textbox(label=i18n("*文本标注文件"),value=r"D:\aisound\sound_data\晓辰60\xiaochen60.list",interactive=True)
                out_wav_dir = gr.Textbox(label=i18n("*生成音频保存位置"), value=r"D:\aisound\temp", interactive=True)
                batch_inference_button = gr.Button(i18n("批量合成语音"), variant="primary")
            
            combined_audio = gr.Audio(label=i18n("变声后合并"),show_download_button=True)

            batch_inference_button.click(
                change_audio,
                [asr_list, text_language, inp_ref,out_wav_dir],
                [combined_audio],
            )


            gr.Markdown(value=i18n("* 输入文件夹批量变声"))
            with gr.Row():
                
                in_wav_dir2 = gr.Textbox(label=i18n("*原始音频保存位置"), value=r"D:\aisound\GPT-SoVITS\sample\阿醋", interactive=True)
                out_wav_dir2 = gr.Textbox(label=i18n("*生成音频保存位置"), value=r"D:\aisound\temp", interactive=True)
                batch_inference2_button = gr.Button(i18n("批量合成语音"), variant="primary")
            
            combined_audio2 = gr.Audio(label=i18n("变声后合并"),show_download_button=True)

            batch_inference2_button.click(
                change_audio2,
                [in_wav_dir2, text_language, inp_ref,out_wav_dir2],
                [combined_audio2],
            )

    app.queue(max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=vc_webui_port,
        quiet=True,
    )

if __name__ == '__main__':
    main()

