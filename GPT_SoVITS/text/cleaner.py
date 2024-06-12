import re
from text import chinese, japanese, cleaned_text_to_sequence, symbols, english
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

language_module_map = {"zh": chinese, "ja": japanese, "en": english}
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(r"D:\aisound\GPT-SoVITS\GPT_SoVITS\text\opencpop-strict.txt").readlines()
    }

def g2pex(sub_initials, sub_finals, seg):
    
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
    return new_c,new_v
      


def parse_special_string(s):
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
    
def find_custom_tone(text, norm_text):  
    
    text = text.replace(" ", "") # 去除空格
    tone_data_list = []    

    texts = parse_special_string(text)

    pos = 0
    for i, (sub_text, norm_sub_text) in enumerate(zip(texts, norm_text)):
        if i > 0:
            if ord(norm_sub_text[0]) <= 127:
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
            
            new_c,new_v = g2pex(sub_initials,sub_finals,seg)
            data = [pos, new_c,new_v]
            tone_data_list.append(data) 

    return tone_data_list

def clean_text(text, language):
    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]

     

    notone_text = re.sub(r"{.*?}", "", text)
    norm_text = language_module.text_normalize(notone_text)
    tone_data_list = find_custom_tone(text, norm_text)

    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None

    for ph in phones:
        assert ph in symbols

    if tone_data_list:
        print(phones)
        for tone_data in tone_data_list:
            pos1 = tone_data[0]
            pos2 = pos1+1
            
            org_phones = phones[pos1]+phones[pos2]
            phones[pos1] = tone_data[1]
            phones[pos2] = tone_data[2]

            
            print(f"[+]成功修改读音: {org_phones} => {phones[pos1]+phones[pos2]}")
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
