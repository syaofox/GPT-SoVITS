"""
GPT-SoVITS 工具函数
"""

import os
import re
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import subprocess
import librosa

# 公共变量
dict_language = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",  # 多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "en": "en",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}

# 特殊字符标点
splits = {
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"
}

class DictToAttrRecursive(dict):
    """将字典转换为可属性访问的类"""
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


def is_empty(*items):
    """判断是否为空（任意一项不为空返回False）"""
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):
    """判断是否为满（任意一项为空返回False）"""
    for item in items:
        if item is None or item == "":
            return False
    return True


def cut_text(text, punc):
    """根据标点符号分割文本"""
    punc_list = [
        p
        for p in punc
        if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}
    ]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items) % 2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    return text


def only_punc(text):
    """判断是否只包含标点符号"""
    return not any(t.isalnum() or t.isalpha() for t in text)


def find_custom_tone(text: str):
    """识别、提取文本中的多音字"""
    tone_list = []
    txts = []
    # 识别 tone 标记，形如<tone as=shu4>数</tone>或<tone as=\"shu3\">数</tone>或<tone as=\"shù\">数</tone>
    ptn1 = re.compile(r"<tone.*?>(.*?)</tone>")
    # 清除 tone 标记中不需要的部分
    ptn2 = re.compile(r"(</?tone)|(as)|([>\"'\s=])")
    matches = list(re.finditer(ptn1, text))
    offset = 0
    for match in matches:
        # tone 标记之前的文本
        pre = text[offset : match.start()]
        txts.append(pre)
        # tone 标签中的单个多音字
        tone_text = match.group(1)
        txts.append(tone_text)
        # 提取读音，支持识别 Style.TONE 和  Style.TONE3
        tone = match.group(0)
        tone = re.sub(ptn2, "", tone)
        tone = tone.replace(tone_text, "")
        # 多音字在当前文本中的索引位置
        pos = sum([len(s) for s in txts])
        offset = match.end()
        init, final = correct_initial_final(tone)
        data = [tone, init, final, pos]
        tone_list.append(data)
    # 不能忘了最后一个 tone 标签后面可能还有剩余的内容
    if offset < len(text):
        txts.append(text[offset:])

    text = "".join(str(i) for i in txts)
    text = text.replace(" ", "")  # 去除空格
    return text, tone_list


def correct_initial_final(tone):
    """校正声母韵母"""
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
    from text.symbols import punctuation
    from text.chinese2 import pinyin_to_symbol_map
    
    init = ""
    final = ""
    if tone[0].isalpha():
        init = to_initials(tone)
        final = to_finals_tone3(tone, neutral_tone_with_five=True)
    else:
        init = tone
        final = tone

    if init == final:
        assert init in punctuation
        return init, init
    else:
        v_without_tone = final[:-1]
        _tone = final[-1]

        pinyin = init + v_without_tone
        assert _tone in "12345"

        if init:
            # 多音节
            v_rep_map = {
                "uei": "ui",
                "iou": "iu",
                "uen": "un",
            }
            if v_without_tone in v_rep_map.keys():
                pinyin = init + v_rep_map[v_without_tone]
        else:
            # 单音节
            pinyin_rep_map = {
                "ing": "ying",
                "i": "yi",
                "in": "yin",
                "u": "wu",
            }
            if pinyin in pinyin_rep_map.keys():
                pinyin = pinyin_rep_map[pinyin]
            else:
                single_rep_map = {
                    "v": "yu",
                    "e": "e",
                    "i": "y",
                    "u": "w",
                }
                if pinyin[0] in single_rep_map.keys():
                    pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

        assert pinyin in pinyin_to_symbol_map.keys(), tone
        new_init, new_final = pinyin_to_symbol_map[pinyin].split(" ")
        new_final = new_final + _tone

        return new_init, new_final


def revise_custom_tone(phones, word2ph, tone_data_list):
    """修正自定义多音字"""
    for td in tone_data_list:
        tone = td[0]
        init = td[1]
        final = td[2]
        pos = td[3]
        if init == "" and final == "":
            # 如果匹配拼音的时候失败，这里保持模型中默认提供的读音
            continue

        wd_pos = 0
        for i in range(0, pos):
            wd_pos += word2ph[i]
        org_init = phones[wd_pos - 2]
        org_final = phones[wd_pos - 1]
        phones[wd_pos - 2] = init
        phones[wd_pos - 1] = final
        print(f"[+]成功修改读音: {org_init}{org_final} => {tone}")


def convert_text(text):
    """转换文本，处理标点符号和特殊格式"""
    # 定义字符转换映射
    # 阿拉伯数字到全角数字的映射
    digit_map = {str(i): chr(ord('０') + i) for i in range(10)}
    
    # 英文标点到中文标点的映射
    punctuation_map = {
        ',': '，',
        '.': '。',
        ':': '：',
        ';': '；',
        '?': '？',
        '!': '！',
        '(': '（',
        ')': '）',
        '[': '【',
        ']': '】',
        '{': '｛',
        '}': '｝',
        '<': '＜',
        '>': '＞',
        '"': '',
        "'": '',
        '@': '＠',
        '#': '＃',
        '$': '￥',
        '%': '％',
        '&': '＆',
        '*': '＊',
        '+': '＋',
        '-': '－',
        '=': '＝',
        '/': '／',
        '\\': '＼',
        '|': '｜',
        '_': '＿',
        '`': '｀'
    }
    
    # 英文字母到全角字母的映射
    alpha_map = {}
    for i in range(26):
        # 小写字母映射
        alpha_map[chr(ord('a') + i)] = chr(ord('ａ') + i)
        # 大写字母映射
        alpha_map[chr(ord('A') + i)] = chr(ord('Ａ') + i)
    
    result_list = []

    text_list = text.split("\n")
    for sub_text in text_list:

        # 将文本中的特殊符号替换为<br>标记
        if sub_text == "":
           sub_text = "<br>"
        # 如果匹配类似00:00:00.000开头的格式，则不进行转换
        elif re.match(r'^\d{2}:\d{2}:\d{2}.*', sub_text):
            print(f"匹配到00:00:00.000开头的格式: {sub_text}, 不进行转换")
            pass  
        # 如果不存在<>标记包裹的文本，则不进行转换
        elif not re.search(r'<[^>]*>', sub_text):
            print(f"不存在<>标记包裹的文本: {sub_text}, 不进行转换")
            pass
        else:
            # 处理非<>标记包裹的文本
            result = ""
            i = 0
            in_tag = False
            
            while i < len(sub_text):
                char = sub_text[i]
                
                # 处理标签开始
                if char == '<':
                    in_tag = True
                    result += char
                
                # 处理标签结束
                elif char == '>' and in_tag:
                    in_tag = False
                    result += char
                
                # 处理标签内部的字符
                elif in_tag:
                    result += char
                
                # 处理标签外部的字符（应用转换）
                else:
                    if char.isdigit():
                        result += digit_map.get(char, char)
                    elif char in punctuation_map:
                        result += punctuation_map.get(char, char)
                    elif char.isalpha():
                        result += alpha_map.get(char, char)
                    else:
                        result += char
                
                i += 1
                
            sub_text = result
        
        result_list.append(sub_text)       

    text = "\n".join(result_list)    
    return text


def pack_audio(audio_data, sample_rate, format='wav', bit_depth='int16'):
    """打包音频数据为指定格式"""
    audio_bytes = BytesIO()
    
    # 确保数据类型兼容 - 增强的数据类型转换代码
    import numpy as np
    import torch
    
    # 处理PyTorch张量
    if isinstance(audio_data, torch.Tensor):
        if audio_data.dtype == torch.float16:
            audio_data = audio_data.to(torch.float32)
        audio_data = audio_data.cpu().numpy()
    
    # 处理NumPy数组的float16类型
    if hasattr(audio_data, 'dtype') and str(audio_data.dtype) == 'float16':
        audio_data = audio_data.astype(np.float32)
    
    if format == 'wav':
        if bit_depth == 'int32':
            sf.write(audio_bytes, audio_data, sample_rate, format='WAV', subtype='PCM_32')
        else:  # int16
            sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
    elif format == 'ogg':
        sf.write(audio_bytes, audio_data, sample_rate, format='OGG')
    elif format == 'aac':
        # 使用 ffmpeg 将 PCM 转换为 AAC
        pcm_type = 's32le' if bit_depth == 'int32' else 's16le'
        bit_rate = '256k' if bit_depth == 'int32' else '128k'
        
        # 准备输入数据
        if bit_depth == 'int32':
            input_data = (audio_data * 2147483647).astype(np.int32)
        else:
            input_data = (audio_data * 32768).astype(np.int16)
        
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-f", pcm_type,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-i", "pipe:0",
                "-c:a", "aac",
                "-b:a", bit_rate,
                "-vn",
                "-f", "adts",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = process.communicate(input=input_data.tobytes())
        audio_bytes.write(out)
    
    audio_bytes.seek(0)
    return audio_bytes


def load_audio_from_file(file_path, target_sr=None):
    """从文件加载音频"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr 