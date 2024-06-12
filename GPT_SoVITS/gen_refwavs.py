import glob
import os
import re
import shutil
import sys

import librosa
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))


def replace_chinese(text):
    pattern = r"([\u4e00-\u9fa5]{5}).*"
    result = re.sub(pattern, r"\1...", text)
    return result


def init_wav_list(sovits_path):
    wav_path = ""

    match = re.search(r"(.+)_e\d+_s\d+\.pth", sovits_path)
    if match:
        result = match.group(1).replace("SoVITS_weights/", "")
        wav_path = f"./sample/{result}/"

    else:
        return [], {}

    if not os.path.exists(wav_path):
        return [], {}

    res_wavs = {}

    res_text = ["请选择参考音频"]

    # 读取文本

    # 遍历目录
    for file_path in os.listdir(wav_path)[:200]:
        wfile_path = os.path.join(wav_path, file_path)
        if os.path.isfile(wfile_path):
            match = re.search(r"(_\d+秒).wav$", file_path)
            if match:
                # 提取主文本和后缀
                suffix = match.group(1)
                main_text = file_path[: match.start()]
            else:
                continue

            key = f"{main_text}{suffix}"

            res_text.append(key)

            # 情绪
            match2 = re.search(r"^【.+?】(.+)$", main_text)
            if match2:
                main_text = match2.group(1)

            res_wavs[key] = (
                wfile_path,
                main_text,
            )

    return res_text, res_wavs


def check_files_starting_with(directory, prefix):
    # 构造搜索模式
    search_pattern = os.path.join(directory, f"*{prefix}*")

    # 获取目录下所有以prefix开头的文件
    files = glob.glob(search_pattern)

    # 判断是否有符合条件的文件存在
    if files:
        return True
    else:
        return False


def replace_punctuation(text):
    # 创建一个字典来映射英文标点到中文标点
    punctuation_map = {
        ",": "，",
        ".": "。",
        "?": "？",
        "!": "！",
        ":": "：",
        ";": "；",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
        "{": "｛",
        "}": "｝",
        '"': "“",
        "'": "‘",
    }

    # 使用字符串的 translate 方法进行替换
    # 需要将字典转换为一个 translation table
    translation_table = str.maketrans(punctuation_map)

    return text.translate(translation_table)


def gen_sample(sovits_path: str):
    match = re.search(r"(.+)_e\d+_s\d+\.pth", sovits_path)
    if match:
        wname = match.group(1)
    else:
        return
    # 创建新的目录
    wdir_path = f"./sample/{wname}/"
    os.makedirs(wdir_path, exist_ok=True)

    # 复制参考文本
    swav_dir = f"./logs/{wname}/5-wav32k"
    name2text_path = rf"./logs/{wname}/2-name2text.txt"
    if not os.path.exists(name2text_path):
        return
    with open(name2text_path, "r", encoding="utf-8") as f:
        texts = f.readlines()
        for text in tqdm(texts):
            match = re.search(rf"(.+?)\t(.+?)\t(.+?)\t(.+?)\n", text)
            if match:
                if check_files_starting_with(
                    wdir_path, f"{replace_punctuation(match.group(4))}"
                ):
                    continue

                source_file_path = os.path.join(swav_dir, match.group(1))

                # 传入音频文件路径，获取音频数据和采样率
                audio_data, sample_rate = librosa.load(source_file_path)
                # 使用librosa.get_duration函数计算音频文件的长度
                duration = librosa.get_duration(y=audio_data, sr=sample_rate)
                duration = int(duration)

                if duration > 2 and duration < 6:
                    destination_file_path = os.path.join(
                        wdir_path,
                        f"{replace_punctuation(match.group(4))}_{duration}秒.wav",
                    )

                    if os.path.exists(destination_file_path):
                        continue

                    tqdm.write(f"{source_file_path}-> {destination_file_path}")
                    shutil.copy(source_file_path, destination_file_path)


def gen_samples():
    sovits_path = "./SoVITS_weights"
    # 获取模型名
    for file_path in os.listdir(sovits_path):
        gen_sample(file_path)


if __name__ == "__main__":
    gen_samples()
