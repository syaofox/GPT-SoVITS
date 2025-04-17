"""
词语替换模型
负责管理词语替换配置的加载和保存
"""
import os
from pathlib import Path


class WordReplaceModel:
    """词语替换模型类"""
    
    def __init__(self):
        """初始化模型"""
        self.config_file = Path("configs/word_replace.txt")
        self.word_replace_dict = {}
        self.load_config()
    
    def load_config(self):
        """加载词语替换配置"""
        try:
            # 确保配置目录存在
            os.makedirs("configs", exist_ok=True)
            
            # 如果文件不存在，创建空文件
            if not self.config_file.exists():
                with open(self.config_file, "w", encoding="utf-8") as f:
                    f.write("# 词语替换配置\n")
                    f.write("# 格式：替换前 替换后\n")
                    f.write("# 例如：hello 你好\n")
                return ""
                
            # 读取配置文件内容
            with open(self.config_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.update_replace_dict(content)
                return content
        except Exception as e:
            print(f"加载词语替换配置失败: {e}")
            return ""
    
    def save_config(self, text):
        """保存词语替换配置"""
        try:
            # 确保配置目录存在
            os.makedirs("configs", exist_ok=True)
            
            # 保存配置
            with open(self.config_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            # 更新替换字典
            self.update_replace_dict(text)
            return True
        except Exception as e:
            print(f"保存词语替换配置失败: {e}")
            return False
    
    def update_replace_dict(self, content):
        """更新替换字典"""
        replace_dict = {}
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):  # 跳过空行和注释
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            
            src, dst = parts
            # 英文不区分大小写，将英文部分转为小写作为键
            src_lower = ''.join([c.lower() if c.isascii() and c.isalpha() else c for c in src])
            replace_dict[src_lower] = dst
        
        self.word_replace_dict = replace_dict
    
    def get_replace_dict(self):
        """获取替换字典"""
        return self.word_replace_dict
    
    def clean_text(self, text):
        """清理文本，应用词语替换"""
        import re
        
        if not self.word_replace_dict:
            return text
        
        result = text
        
        # 对文本中的每个单词进行检查和替换
        for src, dst in self.word_replace_dict.items():
            # 创建一个模式，匹配源字符串，英文部分不区分大小写
            pattern = ''.join(['[' + c.upper() + c.lower() + ']' if c.isascii() and c.isalpha() else re.escape(c) for c in src])
            result = re.sub(pattern, dst, result)
        
        return result 