"""
文本处理和音素转换功能
"""

import re
import torch
from typing import List, Tuple, Dict, Any, Union

# 这些函数需要导入原始项目中的text模块
# 在实际使用时，需确保这些模块可用
# from text import chinese, cleaned_text_to_sequence
# from text.cleaner import clean_text
# from text.LangSegmenter import LangSegmenter


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
    phones, word2ph, norm_text = clean_text(text, language, version)
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
    if language == "zh":
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
            
        if language == "all_zh":
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