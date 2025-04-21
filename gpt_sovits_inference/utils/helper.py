"""
通用辅助函数
"""

import logging
import warnings


class DictToAttrRecursive(dict):
    """递归地将字典转换为可属性访问的对象"""
    
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def setup_logging():
    """设置日志级别，禁用不必要的警告"""
    loggers = [
        "markdown_it", "urllib3", "httpcore", "httpx", 
        "asyncio", "charset_normalizer", "torchaudio._extension", 
        "multipart.multipart"
    ]
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    warnings.simplefilter(action="ignore", category=FutureWarning) 