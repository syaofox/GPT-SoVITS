import re
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(r"D:\aisound\GPT-SoVITS\GPT_SoVITS\text\opencpop-strict.txt").readlines()
    }
punctuation = {'。', '，', '；', '：', '？', '！', '（', '）', '“', '”', '‘', '’', '《', '》', '.', ',', ';', ':', '?', '!', '(', ')', '"', "'", '<', '>'}

additional_pinyin_dict = {
    "觉也不睡": ["jiao4", "ie3", "bu2", "shui4"],
    "嘚瑟": ["de4", "se5"],
    "载歌载舞": ["zai3", "ge1", "zai2", "wu3"],
    "这么着": ["zhe4", "me4", "zhao1"],
    "着急": ["zhao1", "ji2"],
    "正着": ["zheng4", "zhao2"],
    "萎缩": ["wei3", "suo1"],
    "想找找": ["xiang3", "zhao2", "zhao3"],
    "珍妮": ["zhen1", "ni2"],
    "安妮": ["an1", "ni2"],
    "鸡巴": ["ji1", "ba5"],
    "我得": ["wo2", "dei3"],
    "你得": ["ni2", "dei3"],
    "他得": ["ta1", "dei3"],
    "她得": ["ta1", "dei3"],
    "它得": ["ta1", "dei3"],
    "得劲": ["dei3", "jin4"],
    "得让": ["dei3", "rang4"],
    "得以": ["de2", "yi3"],
    "瞄上": ["miao1", "shang4"],
    "瞄一": ["miao1", "yi4"],
    "逮住": ["dei1", "zhu4"],
    "逮个": ["dei1", "ge4"],
    "絮叨": ["xu4", "dao1"],
    "慢慢": ["man4", "man1"],
    "一溜烟": ["yi2", "liu4", "yan1"],
    "背着黑锅": ["bei1", "zhe5", "hei1", "guo1"],
    "背黑锅": ["bei1", "hei1", "guo1"],
    "背锅": ["bei1", "guo1"],
    "嗷嗷": ["ao1", "ao1"],
    "怂": ["song2"],
    "好好": ["hao3", "hao1"],
    "嗲嗲": ["dia2", "dia3"],
    "嗲": ["dia3"],
    "发牌": ["fa1", "pai2"],
    "噼里啪啦": ["pi1", "li1", "pa1", "la1"],
    "长大": ["zhang3", "da4"],
    "薄薄": ["bao2", "bao2"],
    "强迫": ["qiang2", "po4"],
    "大汗淋漓": ["da4", "han4", "lin2", "li2"],
    "靓": ["liang4"],
    "懵": ["meng1"],
    "吐槽": ["tu3", "cao2"],
    "肏": ["cao4"],
    "很冲": ["hen3", "chong4"],
    "揶揄": ["ye2", "yu2"],
    "州长": ["zhou1", "zhang3"],
    "帧": ["zhen1"],
    "搭档": ["da1", "dang4"],
    "刹车": ["sha1", "che1"],
    "急刹": ["ji2", "sha1"],
    "调换": ["diao4", "huan4"],
    "勒死": ["lei1", "si3"]
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



# 计算字符的宽度，假设每个中文字符占2个位置，其他字符占1个位置
def calculate_position(seg, target, start_pos):
    position = seg.find(target, start_pos)
    if position == -1:
        return -1
    # 计算宽度
    width = 0
    for i in range(position):
        if ord(seg[i]) > 127:  # 非ASCII字符，假设为中文字符
            width += 2
        else:
            width += 1
    return width


def find_pinyinex(seg:str):
    start_pos = -1

    results = []
    
    for sub_text, pinyins in additional_pinyin_dict.items():
        start_pos = 0
        while True:
            start_pos = calculate_position(seg, sub_text, start_pos)
            if start_pos == -1:
                break
                # 计算子字符串的结束位置
            for i,pinyin in enumerate(pinyins):
                if pinyin[0].isalpha():
                    sub_initials = to_initials(pinyin)
                    sub_finals = to_finals_tone3(pinyin,neutral_tone_with_five=True)
                else:
                    sub_initials = pinyin
                    sub_finals = pinyin    

                new_c,new_v = g2pex(sub_initials,sub_finals,seg)
                results.append([start_pos + i*2,  new_c,new_v])

            start_pos += len(sub_text)

    return results


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

    output_list = []

# 遍历原始列表
    for item in result:
        output_list.append(item)
        if item not in punctuation:
            output_list.append('')
    return output_list

def find_custom_tone(text):  
    
    text = text.replace(" ", "") # 去除空格
    tone_data_list = [] 
    texts = parse_special_string(text)

    for i, sub_text in enumerate(texts): 
        _match = re.search(r"(.*?){(.*?)}",sub_text)
        if _match :
            seg = _match.group(1)
            content = _match.group(2)
            if content[0].isalpha():
                sub_initials = to_initials(content)
                sub_finals = to_finals_tone3(content,neutral_tone_with_five=True)
            else:
                sub_initials = content
                sub_finals = content               
            
            new_c,new_v = g2pex(sub_initials,sub_finals,seg)
            data = [i, new_c,new_v]
            tone_data_list.append(data) 

    return tone_data_list

if __name__ == "__main__":
    pinyin = 'wu'
    if pinyin[0].isalpha():
        sub_initials = to_initials(pinyin)
        sub_finals = to_finals_tone3(pinyin,neutral_tone_with_five=True)
    else:
        sub_initials = pinyin
        sub_finals = pinyin    
    print(sub_initials,sub_finals)