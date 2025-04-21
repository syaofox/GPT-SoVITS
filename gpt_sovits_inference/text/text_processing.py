"""
文本处理和音素转换功能
"""

import re
import torch
from typing import List, Tuple, Dict, Any, Union

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 这些函数需要导入原始项目中的text模块
# 在实际使用时，需确保这些模块可用
# from text import chinese, cleaned_text_to_sequence
# from text.cleaner import clean_text
# from text.LangSegmenter import LangSegmenter


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
        logger.info(f"[+]成功修改读音: {org_init}{org_final} => {tone}")


def clean_text_inf(text: str, language: str, version: str) -> Tuple[List[int], List[int], str]:
    """
    清理文本并转换为音素序列
    
    参数:
        text: 输入文本
        language: 语言代码
        version: 模型版本
        
    返回:
        (phones, word2ph, norm_text): 音素序列、字到音素的映射、标准化文本
    """
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence
    
    language = language.replace("all_", "")

    text, tone_data_list = find_custom_tone(text)
    if tone_data_list:
        logger.info(f"tone_data_list: {tone_data_list}")


    phones, word2ph, norm_text = clean_text(text, language, version)

    revise_custom_tone(phones, word2ph, tone_data_list)

    phones = cleaned_text_to_sequence(phones, version)

    return phones, word2ph, norm_text


def get_bert_feature(tokenizer, bert_model, device, norm_text: str, word2ph: List[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    获取BERT特征
    
    参数:
        tokenizer: BERT分词器
        bert_model: BERT模型
        device: 计算设备
        norm_text: 标准化文本
        word2ph: 字到音素的映射
        dtype: 数据类型
        
    返回:
        BERT特征张量
    """
    with torch.no_grad():
        inputs = tokenizer(norm_text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        
    assert len(word2ph) == len(norm_text)
    phone_level_feature = []
    
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
        
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_bert_inf(tokenizer, bert_model, device, phones: List[int], word2ph: List[int], norm_text: str, language: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    根据语言获取BERT特征
    
    参数:
        tokenizer: BERT分词器
        bert_model: BERT模型
        device: 计算设备
        phones: 音素序列
        word2ph: 字到音素的映射
        norm_text: 标准化文本
        language: 语言代码
        dtype: 数据类型
        
    返回:
        BERT特征张量
    """
    language = language.replace("all_", "")
    if language == "zh" or "<tone" in norm_text:
        bert = get_bert_feature(tokenizer, bert_model, device, norm_text, word2ph, dtype).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=dtype,
        ).to(device)
    return bert


def get_phones_and_bert(
    tokenizer, bert_model, device, text: str, language: str, version: str, dtype: torch.dtype = torch.float32, final: bool = False
) -> Tuple[List[int], torch.Tensor, str]:
    """
    获取音素和BERT特征
    
    参数:
        tokenizer: BERT分词器
        bert_model: BERT模型
        device: 计算设备
        text: 输入文本
        language: 语言代码
        version: 模型版本
        dtype: 数据类型
        final: 是否为最终递归调用(用于处理短文本)
        
    返回:
        (phones, bert, norm_text): 音素序列、BERT特征、标准化文本
    """
    from text import chinese
    from text.LangSegmenter import LangSegmenter
    
    # 定义分隔符
    splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
    
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        # 处理单一语言文本
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh" and "<tone" in formattext:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = get_bert_feature(tokenizer, bert_model, device, norm_text, word2ph, dtype).to(device)

        elif language == "all_zh":
            if re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(tokenizer, bert_model, device, formattext, "zh", version, dtype)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(tokenizer, bert_model, device, norm_text, word2ph, dtype).to(device)
        elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(tokenizer, bert_model, device, formattext, "yue", version, dtype)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=dtype,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        # 处理混合语言文本
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
                
        print(textlist)
        print(langlist)
        
        phones_list = []
        bert_list = []
        norm_text_list = []
        
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(tokenizer, bert_model, device, phones, word2ph, norm_text, lang, dtype)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
            
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    # 如果文本太短，添加占位符
    if not final and len(phones) < 6:
        return get_phones_and_bert(tokenizer, bert_model, device, "." + text, language, version, dtype, final=True)

    return phones, bert.to(dtype), norm_text 