"""
文本处理器模块

负责加载和应用替换规则
"""

import os
from pathlib import Path


class TextProcessor:
    """文本处理器，负责加载和应用替换规则"""
    
    def __init__(self):
        self.word_replace_rules = {}
        # 首次初始化时自动加载规则
        self.load_word_replace_rules()
    
    def load_word_replace_rules(self):
        """加载文字替换规则"""
        replace_rules = {}
        
        # 获取项目根目录
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        replace_file = root_dir / "gui" / "word_replace.txt"
        
        try:
            if replace_file.exists():
                with open(replace_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ' ' in line:
                            source, target = line.split(' ', 1)
                            replace_rules[source] = target
        except Exception as e:
            print(f"加载替换规则失败: {e}")
        
        self.word_replace_rules = replace_rules
        return replace_rules
    
    def apply_word_replace(self, text):
        """应用文字替换规则"""
        if not self.word_replace_rules:
            return text
            
        result = text
        for source, target in self.word_replace_rules.items():
            result = result.replace(source, target)
        
        return result
    
    def update_rules(self, new_rules):
        """更新替换规则"""
        self.word_replace_rules = new_rules
        return len(new_rules) 