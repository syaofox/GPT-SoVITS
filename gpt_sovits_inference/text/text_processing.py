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

        # 计算音素位置 - 现在pos直接是"福"字的位置
        wd_pos = 0
        for i in range(0, pos):
            wd_pos += word2ph[i]
            
        # "福"字占位符本身的音素位置，需要加上"福"字对应的音素数
        wd_pos += word2ph[pos]
        
        # 修改"福"字的音素为指定拼音的音素
        phones[wd_pos - 2] = init
        phones[wd_pos - 1] = final
        logger.info(f"[+]成功修改读音: 福 => {tone}")


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
    if language == "zh" in norm_text:
        bert = get_bert_feature(tokenizer, bert_model, device, norm_text, word2ph, dtype).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=dtype,
        ).to(device)
    return bert


def find_pinyin_tone(text: str) -> Tuple[str, List]:
    """
    识别文本中的直接拼音输入（如liu4），并替换为占位符"福"
    
    参数:
        text: 输入文本
        
    返回:
        (processed_text, pinyin_list): 处理后的文本和拼音列表
    """
    # 匹配汉语拼音模式：字母+数字(1-5)
    # 如: liu4, ma3, hao3, a1, e2等
    pattern = re.compile(r'([a-zA-Z]+)([1-5])')
    
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text, []
    
    pinyin_list = []
    processed_text = ""
    last_end = 0
    
    for i, match in enumerate(matches):
        # 提取拼音和声调
        pinyin_text = match.group(1)
        tone_num = match.group(2)
        full_pinyin = pinyin_text + tone_num
        
        # 文本的前半部分
        processed_text += text[last_end:match.start()]
        
        # 替换为固定占位符"福"
        placeholder = "福"  # 统一使用"福"字作为占位符
        processed_text += placeholder
        
        # 保存拼音信息和位置
        position = len(processed_text) - len(placeholder)
        init, final = correct_initial_final(full_pinyin)
        pinyin_list.append([full_pinyin, init, final, position])
        
        last_end = match.end()
    
    # 添加最后一部分文本
    if last_end < len(text):
        processed_text += text[last_end:]
    
    if pinyin_list:
        logger.info(f"检测到直接拼音输入: {pinyin_list}")
    
    return processed_text, pinyin_list

def process_text_with_pinyin(text: str, language: str, version: str) -> Tuple[List[int], List[int], str]:
    """
    处理带有直接拼音输入的文本并转换为音素序列
    
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
    
    # 修复正则表达式的语法错误
    text = re.sub(r'[""\'\'《》【】]', "", text)
    text = re.sub(r"…+", "…", text)
    
    # 处理直接拼音输入
    processed_text, pinyin_list = find_pinyin_tone(text)
    
    # 清理文本并转换为音素
    phones, word2ph, norm_text = clean_text(processed_text, language, version)
    
    # 如果有拼音数据，则修正音素
    if pinyin_list:
        revise_custom_tone(phones, word2ph, pinyin_list)
    
    phones = cleaned_text_to_sequence(phones, version)
    
    return phones, word2ph, norm_text

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
    
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        # 处理单一语言文本
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if re.search(r'[a-zA-Z]+[1-5]', formattext):
            print("--------------------------------")
            print("检测到直接拼音输入")
            print(formattext)
            print("--------------------------------")
            phones, word2ph, norm_text = process_text_with_pinyin(formattext, language, version)
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
        
        # 修正拼音被错误识别为英文的问题
        # 检查每个被标记为英文的片段，如果符合拼音格式则修改其语言标记
        pinyin_pattern = re.compile(r'^([a-zA-Z]+[1-5])([,\.。，、；;:：!！?？\s]*)$')
        for i in range(len(textlist)):
            if langlist[i] == "en":
                match = pinyin_pattern.match(textlist[i])
                if match:
                    # 确定应该使用的语言 - 优先使用相邻的中文或粤语语言标签
                    target_lang = None
                    # 检查前面的语言
                    if i > 0 and langlist[i-1] in {"zh", "yue"}:
                        target_lang = langlist[i-1]
                    # 检查后面的语言
                    elif i < len(langlist)-1 and langlist[i+1] in {"zh", "yue"}:
                        target_lang = langlist[i+1]
                    # 如果没有相邻的中文或粤语，则使用当前设置的语言
                    else:
                        if language == "auto":
                            target_lang = "zh"  # 默认中文
                        elif language == "auto_yue":
                            target_lang = "yue"  # 默认粤语
                        else:
                            target_lang = language
                    
                    # 只修改标记为中文或粤语的目标语言
                    if target_lang in {"zh", "yue"}:
                        pinyin = match.group(1)  # 提取拼音部分
                        print(f"检测到拼音被错误识别为英文: {pinyin}, 将语言从'en'修改为'{target_lang}'")
                        langlist[i] = target_lang
                
        print(textlist)
        print(langlist)
        
        phones_list = []
        bert_list = []
        norm_text_list = []
        
        for i in range(len(textlist)):
            lang = langlist[i]
            # 检查当前语言片段是否包含拼音输入
            current_text = textlist[i]
            while "  " in current_text:
                current_text = current_text.replace("  ", " ")
                
            # 如果是中文相关语言，检查是否包含拼音输入
            if lang in {"zh", "yue"}:
                # 检查是否包含完整的拼音格式（如hao3）
                if re.search(r'[a-zA-Z]+[1-5]', current_text):
                    print("--------------------------------")
                    print(f"混合语言文本中检测到直接拼音输入 (语言: {lang})")
                    print(current_text)
                    print("--------------------------------")
                    phones, word2ph, norm_text = process_text_with_pinyin(current_text, lang, version)
                    bert = get_bert_feature(tokenizer, bert_model, device, norm_text, word2ph, dtype).to(device)
                # 如果整个片段就是一个拼音（来自前面的语言修正）
                elif re.match(r'^[a-zA-Z]+[1-5]$', current_text.strip()):
                    print("--------------------------------")
                    print(f"处理被识别为拼音的单独片段 (语言: {lang})")
                    print(current_text)
                    print("--------------------------------")
                    phones, word2ph, norm_text = process_text_with_pinyin(current_text, lang, version)
                    bert = get_bert_feature(tokenizer, bert_model, device, norm_text, word2ph, dtype).to(device)
                else:
                    # 使用原有处理逻辑
                    phones, word2ph, norm_text = clean_text_inf(current_text, lang, version)
                    bert = get_bert_inf(tokenizer, bert_model, device, phones, word2ph, norm_text, lang, dtype)
            else:
                # 非中文语言使用原有处理逻辑
                phones, word2ph, norm_text = clean_text_inf(current_text, lang, version)
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