import re
from pathlib import Path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    # 移除非法字符
    filename = re.sub(r'[<>:"/\\|?*\n\r]', '', filename)

    # 去除首尾空格和点
    filename = filename.strip(' .')

    # 截断文件名
    filename = filename[:max_length]

    # 处理保留名称
    reserved_names = {'CON', 'PRN', 'AUX', 'NUL'} | {f'COM{i}' for i in range(1, 10)} | {f'LPT{i}' for i in range(1, 10)}
    if Path(filename).stem.upper() in reserved_names:
        filename = f'_{filename}'

    return filename
