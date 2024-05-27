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


def check_files_starting_with(directory, prefix):
    # 构造搜索模式
    search_pattern = os.path.join(directory, f"*{prefix}*")

    # 获取目录下所有以prefix开头的文件
    files = glob.glob(search_pattern)

    # 判断是否有符合条件的文件存在
    if files:
        # print(f"Found files: {files}")
        return True
    else:
        # print("No files found starting with the given prefix.")
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

        # match = re.search(r"(.+)-e\d+.ckpt", file_path)
        # if match:
        #     wname = match.group(1)
        #     if wname not in audio_list:
        #         audio_list.append(wname)

        # # 创建新的目录
        # wdir_path = f"./sample/{wname}/"
        # os.makedirs(wdir_path, exist_ok=True)

        # # 复制参考文本
        # swav_dir = f"./logs/{wname}/5-wav32k"
        # name2text_path = rf"./logs/{wname}/2-name2text.txt"
        # if not os.path.exists(name2text_path):
        #     continue
        # with open(name2text_path, "r", encoding="utf-8") as f:
        #     texts = f.readlines()
        #     for text in tqdm(texts):
        #         match = re.search(rf"(.+?)\t(.+?)\t(.+?)\t(.+?)\n", text)
        #         if match:
        #             source_file_path = os.path.join(swav_dir, match.group(1))

        #             destination_file_path = os.path.join(
        #                 wdir_path, f"{replace_punctuation(match.group(4))}.wav"
        #             )

        #             if os.path.exists(destination_file_path):
        #                 continue

        #             # 传入音频文件路径，获取音频数据和采样率
        #             audio_data, sample_rate = librosa.load(source_file_path)
        #             # 使用librosa.get_duration函数计算音频文件的长度
        #             duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        #             duration = int(duration)

        #             if duration > 2 and duration < 6:
        #                 destination_file_path = os.path.join(
        #                     wdir_path,
        #                     f"{replace_punctuation(match.group(4))}_{duration}秒.wav",
        #                 )

        #                 if os.path.exists(destination_file_path):
        #                     continue

        #                 tqdm.write(f"{source_file_path}-> {destination_file_path}")
        #                 shutil.copy(source_file_path, destination_file_path)


if __name__ == "__main__":
    gen_samples()
