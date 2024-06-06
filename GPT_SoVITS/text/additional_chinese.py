from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

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
    "瞄了": ["miao1", "le5"],
    "逮住": ["dei1", "zhu4"],
    "逮个": ["dei1", "ge4"],
    "逮着机会": ["dai3", "zhao2", "ji1", "hui4"],
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
    # "一行": ["yi1", "hang2"],
    "强迫": ["qiang2", "po4"],
    "大汗淋漓": ["da4", "han4"],
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
    "勒死": ["lei1", "si3"],
    "好好吃": ["hao2", "hao3", "chi1"],
    "够呛": ["gou4", "qiang4"],
    "抗揍": ["kang2", "zou4"],
    "勾搭": ["gou1", "da5"]
}


def test_need_change_pos_pinyin(seg:str):
    start_pos = -1
    end_pos = -1
    results = []
    
    for sub_text, pinyins in additional_pinyin_dict.items():
        start_pos = 0
        while True:
            start_pos = seg.find(sub_text, start_pos)
            if start_pos == -1:
                break
                # 计算子字符串的结束位置
            end_pos = start_pos + len(sub_text)
            results.append([start_pos, end_pos, pinyins])
            start_pos += len(sub_text)

    return results

def change_pos_pinyin(initials, finals,results):
    for  start_pos, end_pos, pinyins in results:

        sub_initials = []
        sub_finals = []
        for pinyin in pinyins:   
            if pinyin[0].isalpha():
                sub_initials.append(to_initials(pinyin))
                sub_finals.append(to_finals_tone3(pinyin,neutral_tone_with_five=True))
            else:
                sub_initials.append(pinyin)
                sub_finals.append(pinyin)

        initials[start_pos:end_pos] = sub_initials
        finals[start_pos:end_pos] = sub_finals

    print(f'命中多音字{[x + y for x, y in zip(initials, finals)]}')
    return initials, finals