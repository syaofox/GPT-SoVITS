
additional_pinyin = {
    "靓":[["l"],["iang4"]],
    # "嘚":[["d"],["e4"]],
    # "载歌载舞":[["z", "g", "z", "w"],["ai3", "e1", "ai2", "u3"]],   
    "锝":[["d"],["ei3"]],# 得 dei3
    "笊":[["zh"],["ao2"]], # 着 zhao2
    # "这么着":[["zh", "m", "zh"],["e4", "e4", "ao1"]],
    # "珍妮":[["zh", "n"],["en1", "i2"]],
    # "安妮":[["", "n"],["an1", "i2"]],
    # "鸡巴":[["j", "b"],["i1", "a5"]],
    # "想找找":[["x", "zh", "zh"],["iang3", "ao2", "ao3"]],
    # "很着急":[["h", "zh", "j"],["en3", "ao1", "i2"]],
    # "一着急":[["y", "zh", "j"],["i4", "ao1", "i2"]],
    # "干着急":[["g", "zh", "j"],["an1", "ao1", "i2"]],
    
}

pos_pinyin = {
    "觉也不睡":[["j", "", "b", "sh"],["iao4", "ie3","u2","ui4"]],
    "嘚瑟":[["d", "s"],["e4", "e5"]],
    "载歌载舞":[["z", "g", "z", "w"],["ai3", "e1", "ai2", "u3"]],  
    "这么着":[["zh", "m", "zh"],["e4", "e4", "ao1"]], 
    "着急":[["zh", "j"],["ao1", "i2"]],
    "正着":[["zh", "zh"],["eng4", "ao2"]],
    "萎缩":[["w", "s"],["ei3", "uo1"]],
    "想找找":[["x", "zh", "zh"],["iang3", "ao2", "ao3"]],
    "珍妮":[["zh", "n"],["en1", "i2"]],
    "安妮":[["", "n"],["an1", "i2"]],
    "鸡巴":[["j", "b"],["i1", "a5"]],
    "我得":[["w", "d"],["o2", "ei3"]],
    "你得":[["n", "d"],["i2", "ei3"]],
    "他得":[["t", "d"],["a1", "ei3"]],
    "她得":[["t", "d"],["a1", "ei3"]],
    "它得":[["t", "d"],["a1", "ei3"]],
    "得劲":[["d", "j"],["ei3", "in4"]],
    "得让":[["d", "r"],["ei3", "ang4"]],
    "得以":[["d", ""],["e2", "i3"]],
    "瞄上":[["m", "sh"],["iao1", "ang4"]],
    "瞄一":[["m", ""],["iao1", "i4"]],
    "逮住":[["d", "zh"],["ei1", "u4"]],
    "逮个":[["d", "g"],["ei1", "e4"]],
    "絮叨":[["x", "d"],["u4", "ao1"]],
    "慢慢":[["m", "m"],["an4", "an1"]],
    "一溜烟":[["", "l", ""],["i2", "iu4", 'ian1']],
    "背着黑锅":[["b", "zh","h", "g"],["ei1", "e5", "ei1", "uo1"]], 
    "背黑锅":[["b", "h", "g"],["ei1", "ei1", "uo1"]], 
    "背锅":[["b",  "g"],["ei1", "uo1"]], 
    "嗷嗷":[["",  ""],["ao1", "ao1"]], 
    "怂":[["s"],["ong2"]], 
    "好好":[["h", "h"],["ao3", "ao1"]],    
    "嗲嗲":[["d","d"],["ia2", "ia3"]],
    "嗲":[["d"],["ia3"]],
}

def test_need_change_pos_pinyin(seg:str):
    start_pos = -1
    end_pos = -1
    results = []
    
    for sub_text, pinyins in pos_pinyin.items():
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
        initials[start_pos:end_pos] = pinyins[0] 
        finals[start_pos:end_pos] = pinyins[1] 
    return initials, finals