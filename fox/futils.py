import re # type: ignore
from pathlib import Path # type: ignore

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\|?*\n\r]', '', filename)
    filename = filename.strip(' .')

    # 限制文件名长度
    max_length = 255
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # 确保文件名不是保留名称，例如CON, PRN, AUX, NUL等（Windows系统）
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
    base_filename = Path(filename).stem
    if base_filename.upper() in reserved_names:
        filename = f'_{filename}'
    
    return filename