"""
文本处理器模块

负责加载和应用替换规则
"""

import os
import re
from pathlib import Path


class TextProcessor:
    """文本处理器，负责加载和应用替换规则"""
    
    def __init__(self):
        self.word_replace_rules = []
        # 首次初始化时自动加载规则
        self.load_word_replace_rules()
    
    def load_word_replace_rules(self):
        """加载文字替换规则
        
        新格式：查找字符串|需修改字符串|替换后的字符串
        例如：操死|操|gan4 意味着在"操死"这个词中，将"操"替换为"gan4"
        """
        replace_rules = []
        
        # 获取项目根目录
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        replace_file = root_dir / "gui" / "word_replace.txt"
        
        try:
            if replace_file.exists():
                with open(replace_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '|' in line:
                            parts = line.split('|')
                            if len(parts) == 3:
                                search_string, target_part, replace_with = parts
                                replace_rules.append((search_string, target_part, replace_with))
        except Exception as e:
            print(f"加载替换规则失败: {e}")
        
        self.word_replace_rules = replace_rules
        return replace_rules
    
    def apply_word_replace(self, text):
        """应用文字替换规则"""
        if not self.word_replace_rules:
            return text
            
        result = text
        for search_string, target_part, replace_with in self.word_replace_rules:
            # 只有完整查找字符串出现在文本中时才进行替换
            if search_string in result:
                # 替换查找字符串中的目标部分
                modified_string = search_string.replace(target_part, replace_with)
                # 用修改后的字符串替换原始的查找字符串
                result = result.replace(search_string, modified_string)
        
        return result
    
    def update_rules(self, new_rules):
        """更新替换规则"""
        self.word_replace_rules = new_rules
        return len(new_rules) 