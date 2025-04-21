"""
文本切分功能
"""

import re
from typing import List


class TextCutter:
    """
    文本切分工具类
    提供各种文本切分策略
    """
    
    def __init__(self):
        """初始化切分器"""
        # 标点符号集合
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
        self.punctuation = self.splits.union({" "})
    
    def split(self, todo_text: str) -> List[str]:
        """将文本按标点符号分割"""
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
            
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        
        while True:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
                
        return todo_texts
    
    def cut1(self, inp: str) -> str:
        """凑四句一切"""
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
        else:
            opts = [inp]
            
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)
    
    def cut2(self, inp: str) -> str:
        """凑50字一切"""
        inp = inp.strip("\n")
        inps = self.split(inp)
        
        if len(inps) < 2:
            return inp
            
        opts = []
        summ = 0
        tmp_str = ""
        
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
                
        if tmp_str != "":
            opts.append(tmp_str)
            
        # 如果最后一个太短了，和前一个合一起
        if len(opts) > 1 and len(opts[-1]) < 50:
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
            
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)
    
    def cut3(self, inp: str) -> str:
        """按中文句号。切"""
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)
    
    def cut4(self, inp: str) -> str:
        """按英文句号.切"""
        inp = inp.strip("\n")
        opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)
    
    def cut5(self, inp: str) -> str:
        """按标点符号切"""
        inp = inp.strip("\n")
        punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item for item in mergeitems if not set(item).issubset(self.punctuation)]
        return "\n".join(opt)
    
    def process_text(self, texts: List[str]) -> List[str]:
        """处理文本列表，过滤无效文本"""
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("请输入有效文本")
            
        for text in texts:
            if text not in [None, " ", ""]:
                _text.append(text)
                
        return _text
    
    def merge_short_text(self, texts: List[str], threshold: int) -> List[str]:
        """合并短文本"""
        if len(texts) < 2:
            return texts
            
        result = []
        text = ""
        
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
                
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
                
        return result


def cut_text(inp: str, method: str = "凑四句一切") -> str:
    """
    根据指定方法切分文本
    
    参数:
        inp: 输入文本
        method: 切分方法，包括：
            - "凑四句一切"
            - "凑50字一切"
            - "按中文句号。切"
            - "按英文句号.切"
            - "按标点符号切"
            - "不切"
            
    返回:
        切分后的文本
    """
    cutter = TextCutter()
    
    if method == "凑四句一切":
        return cutter.cut1(inp)
    elif method == "凑50字一切":
        return cutter.cut2(inp)
    elif method == "按中文句号。切":
        return cutter.cut3(inp)
    elif method == "按英文句号.切":
        return cutter.cut4(inp)
    elif method == "按标点符号切":
        return cutter.cut5(inp)
    else:
        return inp  # 不切 