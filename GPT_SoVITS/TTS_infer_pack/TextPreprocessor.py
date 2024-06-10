
import os, sys

from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
import LangSegment

from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
from tools.i18n.i18n import I18nAuto


pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(r"D:\aisound\GPT-SoVITS\GPT_SoVITS\text\opencpop-strict.txt").readlines()
}
i18n = I18nAuto()
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

def get_first(text:str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts:str, threshold:int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result






class TextPreprocessor:
    def __init__(self, bert_model:AutoModelForMaskedLM, 
                 tokenizer:AutoTokenizer, device:torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        
    def preprocess(self, text:str, lang:str, text_split_method:str)->List[Dict]:
        print(i18n("############ 切分文本 ############"))
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(i18n("############ 提取文本Bert特征 ############"))
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang)
            if phones is None:
                continue
            res={
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text:str, lang:str, text_split_method:str):
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): 
            text = "。" + text if lang != "en" else "." + text
        print(i18n("实际输入的目标文本:"))
        print(text)
        
        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)
        
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.process_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        
        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in splits): text += "。" if lang != "en" else "."
            
            # 解决句子过长导致Bert报错的问题
            if (len(text) > 510):
                texts.extend(split_big_text(text))
            else:
                texts.append(text)
            
        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts
    
    def segment_and_extract_feature_for_text(self, texts:list, language:str)->Tuple[list, torch.Tensor, str]:
        textlist, langlist = self.seg_text(texts, language)
        if len(textlist) == 0:
            return None, None, None
        
        phones, bert_features, norm_text = self.extract_bert_feature(textlist, langlist)
        return phones, bert_features, norm_text


    def seg_text(self, text:str, language:str)->Tuple[list, list]:

        textlist=[]
        langlist=[]
        if language in ["auto", "zh", "ja"]:
            LangSegment.setfilters(["zh","ja","en","ko"])
            for tmp in LangSegment.getTexts(text):
                if tmp["text"] == "":
                    continue
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                elif tmp["lang"] == "en":
                    langlist.append("en")
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language if language!="auto" else tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if formattext != "":
                textlist.append(formattext)
                langlist.append("en")
            
        elif language in ["all_zh","all_ja"]:

            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            language = language.replace("all_","")
            if text == "":
                return [],[]
            textlist.append(formattext)
            langlist.append(language) 
        
        else:
            raise ValueError(f"language {language} not supported")
        
        return textlist, langlist
    

    def extract_bert_feature(self, textlist:list, langlist:list):
        phones_list = []
        bert_feature_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
            _bert_feature = self.get_bert_inf(phones, word2ph, norm_text, lang)
            # phones_list.append(phones)
            phones_list.extend(phones)
            norm_text_list.append(norm_text)
            bert_feature_list.append(_bert_feature)
        bert_feature = torch.cat(bert_feature_list, dim=1)
        # phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
        return phones_list, bert_feature, norm_text


    def get_bert_feature(self, text:str, word2ph:list)->torch.Tensor:
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
    
    def parse_special_string(self, s):
        result = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                # 找到花括号的闭合部分
                j = i
                while j < len(s) and s[j] != '}':
                    j += 1
                if j < len(s):
                    result[-1] += s[i:j+1]  # 将花括号部分合并到前一个字符上
                    i = j + 1
                else:
                    # 如果没有闭合花括号，按普通字符处理
                    result.append(s[i])
                    i += 1
            else:
                result.append(s[i])
                i += 1
        return result
    
    def find_custom_tone(self, text, language):
        text = text.replace(" ", "") # 去除空格
        tone_list = []       

        texts = self.parse_special_string(text)

        pos = 0
        for i, sub_text in enumerate(texts):
            _, _, norm_text = clean_text(sub_text, language)
            if i > 0:
                if norm_text  in punctuation:
                    pos += 1
                else:
                    pos += 2
            _match = re.search(r"(.*?){(.*?)}",sub_text)
            if _match:
                seg = _match.group(1)
                content = _match.group(2)
                if content[0].isalpha():
                    sub_initials = to_initials(content)
                    sub_finals = to_finals_tone3(content,neutral_tone_with_five=True)
                else:
                    sub_initials = content
                    sub_finals = content

                raw_pinyin = sub_initials + sub_finals                
                v_without_tone = sub_finals[:-1]
                tone = sub_finals[-1]
                pinyin = sub_initials + v_without_tone
                assert tone in "12345"
                
                if sub_initials:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = sub_initials + v_rep_map[v_without_tone]
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

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                data = [pos, new_c,new_v]
                tone_list.append(data)           
        
        return re.sub(r"{.*?}", "", text), tone_list

    def clean_text_inf(self, text:str, language:str):
        text, tone_data_list = self.find_custom_tone(text,language)
        phones, word2ph, norm_text = clean_text(text, language)
        print(phones)
        for tone_data in tone_data_list:
            pos1 = tone_data[0]
            pos2 = pos1+1
            
            org_phones = phones[pos1]+phones[pos2]
            phones[pos1] = tone_data[1]
            phones[pos2] = tone_data[2]

            
            print(f"[+]成功修改读音: {org_phones} => {phones[pos1]+phones[pos2]}")
        # phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones:list, word2ph:list, norm_text:str, language:str):
        language=language.replace("all_","")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return feature
    
    def process_text(self,texts):
        _text=[]
        if all(text in [None, " ", "\n",""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in  [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text
    

    def replace_consecutive_punctuation(self,text):
        punctuations = ''.join(re.escape(p) for p in punctuation)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result



