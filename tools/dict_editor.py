import os
import pickle
import re

import gradio as gr

# 获取词典文件路径
current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_path)
DICT_PATH = os.path.join(parent_dir, "GPT_SoVITS", "text", "engdict-hot.rep")
CACHE_PATH = os.path.join(parent_dir, "GPT_SoVITS", "text", "engdict_cache.pickle")


def load_dict():
    """加载词典文件"""
    dict_data = {}
    print(f"正在读取词典文件: {DICT_PATH}")  # 调试信息
    if not os.path.exists(DICT_PATH):
        print(f"词典文件不存在: {DICT_PATH}")  # 调试信息
        return dict_data

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                word, *phones = line.strip().split()
                dict_data[word.lower()] = phones
    print(f"已加载 {len(dict_data)} 个单词")  # 调试信息
    return dict_data


def save_dict(dict_data):
    """保存词典文件"""
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        for word, phones in sorted(dict_data.items()):
            f.write(f"{word.upper()} {' '.join(phones)}\n")
    # # 删除缓存文件
    # if os.path.exists(CACHE_PATH):
    #     os.remove(CACHE_PATH)
    # return "词典已保存,缓存已清除"


def add_word(word: str, phones: str, dict_content: str):
    """添加或更新单词"""
    current_dict = load_dict()

    if not word or not phones:
        return "单词和音素不能为空", dict_content

    word = word.strip().lower()
    phones = phones.strip().upper()

    # 验证音素格式
    phone_list = phones.split()
    valid_format = all(re.match(r"^[A-Z]+[0-2]?$", p) for p in phone_list)
    if not valid_format:
        return "音素格式错误,请使用大写字母+数字(0-2)的格式", dict_content

    current_dict[word] = phone_list
    save_dict(current_dict)
    return f"已添加/更新单词: {word}", list_dict(current_dict)


def delete_word(word: str, dict_content: str):
    """删除单词"""
    current_dict = load_dict()
    word = word.strip().lower()
    if word in current_dict:
        del current_dict[word]
        save_dict(current_dict)
        return f"已删除单词: {word}", list_dict(current_dict)
    return f"单词不存在: {word}", dict_content


def search_word(word: str):
    """查询单词"""
    current_dict = load_dict()
    word = word.strip().lower()
    if word in current_dict:
        return f"音素: {' '.join(current_dict[word])}"
    return "未找到该单词"


def list_dict(current_dict: dict = None):
    """列出词典内容"""
    if current_dict is None:
        current_dict = load_dict()
    if not current_dict:
        return "词典为空"
    result = []
    for word, phones in sorted(current_dict.items()):
        result.append(f"{word}: {' '.join(phones)}")
    return "\n".join(result)


def create_ui():
    with gr.Blocks(title="英文发音词典编辑器") as app:
        gr.Markdown(
            "## 英文发音词典编辑器\n用于编辑GPT-SoVITS的英文发音词典(engdict-hot.rep)"
        )

        with gr.Row():
            with gr.Column():
                word_input = gr.Textbox(label="单词", placeholder="输入单词")
                phones_input = gr.Textbox(
                    label="音素", placeholder="输入音素,用空格分隔"
                )

                with gr.Row():
                    add_btn = gr.Button("添加/更新")
                    del_btn = gr.Button("删除")
                    search_btn = gr.Button("查询")

                result_text = gr.Textbox(label="操作结果", interactive=False)

            dict_content = gr.Textbox(
                label="词典内容", value=list_dict(), interactive=False
            )

        add_btn.click(
            add_word,
            inputs=[word_input, phones_input, dict_content],
            outputs=[result_text, dict_content],
        )

        del_btn.click(
            delete_word,
            inputs=[word_input, dict_content],
            outputs=[result_text, dict_content],
        )

        search_btn.click(search_word, inputs=[word_input], outputs=[result_text])

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()
