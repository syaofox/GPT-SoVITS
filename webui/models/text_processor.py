import re
import os
import torch
from typing import List, Tuple, Set

from models.logger import debug, error, info, warning
from models.constand import BR_TAG


class TextProcessor:
    """
    文本处理器，负责文本分段和音素处理
    """

    BR_TAG = BR_TAG
    TEXT_REPLACE_RULES_FILE = "webui/text_replace_rules.txt"

    def __init__(self, device: str, is_half: bool):
        """
        初始化文本处理器

        Args:
            device: 运行设备 (cuda或cpu)
            is_half: 是否使用半精度
        """
        self.device = device
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32

        # 设置语言字典
        self.dict_language_v1 = {
            "中文": "all_zh",  # 全部按中文识别
            "英文": "en",  # 全部按英文识别
            "日文": "all_ja",  # 全部按日文识别
            "中英混合": "zh",  # 按中英混合识别
            "日英混合": "ja",  # 按日英混合识别
            "多语种混合": "auto",  # 多语种启动切分识别语种
        }
        self.dict_language_v2 = {
            "中文": "all_zh",  # 全部按中文识别
            "英文": "en",  # 全部按英文识别
            "日文": "all_ja",  # 全部按日文识别
            "粤语": "all_yue",  # 全部按中文识别
            "韩文": "all_ko",  # 全部按韩文识别
            "中英混合": "zh",  # 按中英混合识别
            "日英混合": "ja",  # 按日英混合识别
            "粤英混合": "yue",  # 按粤英混合识别
            "韩英混合": "ko",  # 按韩英混合识别
            "多语种混合": "auto",  # 多语种启动切分识别语种
            "多语种混合(粤语)": "auto_yue",  # 多语种启动切分识别语种
        }
        self.dict_language = self.dict_language_v1

        # 切分标记
        self.splits: Set[str] = {
            "，",
            "。",
            "？",
            "！",
            ",",
            ".",
            "?",
            "!",
            "~",
            ":",
            "：",
            "—",
            "…",
        }

        # 替换规则
        self.replace_rules: List[Tuple[str, str, str]] = []
        self.last_mtime: float = 0

        # 需要外部设置的函数
        self.clean_text = None
        self.cleaned_text_to_sequence = None
        self.LangSegmenter = None
        self.bert_model = None
        self.tokenizer = None

    def set_language_dict(self, version: str) -> None:
        """
        设置语言字典

        Args:
            version: 模型版本
        """
        self.dict_language = (
            self.dict_language_v1 if version == "v1" else self.dict_language_v2
        )

    def process_text(self, texts: List[str]) -> List[str]:
        """
        处理文本列表，过滤空文本

        Args:
            texts: 文本列表

        Returns:
            处理后的文本列表
        """
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("请输入有效文本")
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def split_text(self, todo_text: str) -> List[str]:
        """
        按标点符号分割文本

        Args:
            todo_text: 输入文本

        Returns:
            分割后的文本列表
        """
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut_text(self, text: str, how_to_cut: str = "不切") -> str:
        """
        文本切分

        Args:
            text: 输入文本
            how_to_cut: 切分方式

        Returns:
            切分后的文本
        """
        if how_to_cut == "不切":
            return text

        # 定义标点符号集合
        punctuation = {
            ",",
            ".",
            ";",
            "?",
            "!",
            "、",
            "，",
            "。",
            "？",
            "！",
            ";",
            "：",
            "…",
        }

        text = text.strip("\n")
        inps = self.split_text(text)

        if how_to_cut == "凑四句一切":
            # 每四个句子一段
            split_idx = list(range(0, len(inps), 4))
            if len(split_idx) > 0:
                split_idx.append(None)
                opts = []
                for idx in range(len(split_idx) - 1):
                    opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
            else:
                opts = [text]
            opts = [item for item in opts if not set(item).issubset(punctuation)]
            return "\n".join(opts)

        elif how_to_cut == "凑50字一切":
            # 每50字一段
            if len(inps) < 2:
                return text
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
            if (
                len(opts) > 1 and len(opts[-1]) < 50
            ):  # 如果最后一个太短了，和前一个合一起
                opts[-2] = opts[-2] + opts[-1]
                opts = opts[:-1]
            opts = [item for item in opts if not set(item).issubset(punctuation)]
            return "\n".join(opts)

        elif how_to_cut == "按中文句号。切":
            # 按中文句号切分
            opts = ["%s" % item for item in text.strip("。").split("。")]
            opts = [item for item in opts if not set(item).issubset(punctuation)]
            return "\n".join(opts)

        elif how_to_cut == "按英文句号.切":
            # 按英文句号切分
            opts = re.split(r"(?<!\d)\.(?!\d)", text.strip("."))
            opts = [item for item in opts if not set(item).issubset(punctuation)]
            return "\n".join(opts)

        elif how_to_cut == "按标点符号切":
            # 按标点符号切分
            punds = {
                ",",
                ".",
                ";",
                "?",
                "!",
                "、",
                "，",
                "。",
                "？",
                "！",
                ";",
                "：",
                "…",
            }
            mergeitems = []
            items = []

            for i, char in enumerate(text):
                if char in punds:
                    if (
                        char == "."
                        and i > 0
                        and i < len(text) - 1
                        and text[i - 1].isdigit()
                        and text[i + 1].isdigit()
                    ):
                        items.append(char)
                    else:
                        items.append(char)
                        mergeitems.append("".join(items))
                        items = []
                else:
                    items.append(char)

            if items:
                mergeitems.append("".join(items))

            opt = [item for item in mergeitems if not set(item).issubset(punds)]
            return "\n".join(opt)

        return text

    def get_bert_feature(self, text: str, word2ph: List[int]) -> torch.Tensor:
        """
        获取BERT特征

        Args:
            text: 文本
            word2ph: 词到音素的映射

        Returns:
            BERT特征
        """
        if not self.bert_model or not self.tokenizer:
            raise ValueError("请先设置bert_model和tokenizer")

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def clean_text_inf(
        self, text: str, language: str, version: str
    ) -> Tuple[List[str], List[int], str]:
        """
        清理文本

        Args:
            text: 输入文本
            language: 语言
            version: 模型版本

        Returns:
            音素序列、词到音素的映射、标准化后的文本
        """
        if not self.clean_text or not self.cleaned_text_to_sequence:
            raise ValueError("请先设置clean_text和cleaned_text_to_sequence函数")

        language = language.replace("all_", "")
        phones, word2ph, norm_text = self.clean_text(text, language, version)
        phones = self.cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(
        self, phones: List[int], word2ph: List[int], norm_text: str, language: str
    ) -> torch.Tensor:
        """
        获取BERT特征

        Args:
            phones: 音素序列
            word2ph: 词到音素的映射
            norm_text: 标准化后的文本
            language: 语言

        Returns:
            BERT特征
        """
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=self.dtype,
            ).to(self.device)

        return bert

    def get_phones_and_bert(
        self, text: str, language: str, version: str, final: bool = False
    ) -> Tuple[List[int], torch.Tensor, str]:
        """
        获取音素和BERT特征

        Args:
            text: 输入文本
            language: 语言
            version: 模型版本
            final: 是否最终调用

        Returns:
            音素序列、BERT特征、标准化后的文本
        """
        if not self.LangSegmenter:
            raise ValueError("请先设置LangSegmenter")

        # 单一语言处理
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")

            if language == "all_zh":
                # 如果文本中包含数字和字母的组合，则认为是拼音输入
                if re.search(r"[a-zA-Z]+[1-5]", formattext):
                    phones, word2ph, norm_text = self.process_text_with_pinyin(
                        formattext, language, version
                    )
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
                elif re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(
                        r"[a-z]", lambda x: x.group(0).upper(), formattext
                    )
                    from text import chinese

                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(
                        formattext, language, version
                    )
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                from text import chinese

                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(
                    formattext, language, version
                )
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=self.dtype,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            if language == "auto":
                for tmp in self.LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in self.LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in self.LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])

            debug(f"文本分段: {textlist}")
            debug(f"语言分段: {langlist}")

            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                # 如果文本中包含数字和字母的组合，则认为是拼音输入
                if re.search(r"[a-zA-Z]+[1-5]", textlist[i]):
                    lang = "all_zh"
                    phones, word2ph, norm_text = self.process_text_with_pinyin(
                        textlist[i], lang, version
                    )
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(
                        textlist[i], lang, version
                    )
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(self.dtype), norm_text

    def process_text_with_pinyin(
        self, text: str, language: str, version: str
    ) -> Tuple[List[str], List[int], str]:
        """
        处理带有直接拼音输入的文本并转换为音素序列

        Args:
            text: 输入文本
            language: 语言代码
            version: 模型版本

        Returns:
            音素序列、字到音素的映射、标准化文本
        """
        if not self.clean_text or not self.cleaned_text_to_sequence:
            raise ValueError("请先设置clean_text和cleaned_text_to_sequence函数")

        language = language.replace("all_", "")

        # 修复正则表达式的语法错误
        text = re.sub(r'[""\'\'《》【】]', "", text)
        text = re.sub(r"…+", "…", text)

        # 处理直接拼音输入
        processed_text, pinyin_list = self.find_pinyin_tone(text)

        # 清理文本并转换为音素
        phones, word2ph, norm_text = self.clean_text(processed_text, language, version)

        # 如果有拼音数据，则修正音素
        if pinyin_list:
            self.revise_custom_tone(phones, word2ph, pinyin_list)

        phones = self.cleaned_text_to_sequence(phones, version)

        return phones, word2ph, norm_text

    def find_pinyin_tone(self, text: str) -> Tuple[str, List]:
        """
        识别文本中的直接拼音输入（如liu4），并替换为占位符"福"

        Args:
            text: 输入文本

        Returns:
            处理后的文本和拼音列表
        """
        # 匹配汉语拼音模式：字母+数字(1-5)
        # 如: liu4, ma3, hao3, a1, e2等
        pattern = re.compile(r"([a-zA-Z]+)([1-5])")

        matches = list(re.finditer(pattern, text))
        if not matches:
            return text, []

        pinyin_list = []
        processed_text = ""
        last_end = 0

        for i, match in enumerate(matches):
            # 提取拼音和声调
            pinyin_text = match.group(1)
            tone_num = match.group(2)
            full_pinyin = pinyin_text + tone_num

            # 文本的前半部分
            processed_text += text[last_end : match.start()]

            # 替换为固定占位符"福"
            placeholder = "福"  # 统一使用"福"字作为占位符
            processed_text += placeholder

            # 保存拼音信息和位置
            position = len(processed_text) - len(placeholder)
            init, final = self.correct_initial_final(full_pinyin)
            pinyin_list.append([full_pinyin, init, final, position])

            last_end = match.end()

        # 添加最后一部分文本
        if last_end < len(text):
            processed_text += text[last_end:]

        return processed_text, pinyin_list

    def correct_initial_final(self, tone):
        """校正声母韵母"""
        from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
        from text.symbols import punctuation
        from text.chinese2 import pinyin_to_symbol_map

        init = ""
        final = ""
        if tone[0].isalpha():
            init = to_initials(tone)
            final = to_finals_tone3(tone, neutral_tone_with_five=True)
        else:
            init = tone
            final = tone

        if init == final:
            assert init in punctuation
            return init, init
        else:
            v_without_tone = final[:-1]
            _tone = final[-1]

            pinyin = init + v_without_tone
            assert _tone in "12345"

            if init:
                # 多音节
                v_rep_map = {
                    "uei": "ui",
                    "iou": "iu",
                    "uen": "un",
                }
                if v_without_tone in v_rep_map.keys():
                    pinyin = init + v_rep_map[v_without_tone]
            else:
                # 单音节
                pinyin_rep_map = {
                    "ing": "ying",
                    "i": "yi",
                    "in": "yin",
                    "u": "wu",
                }
                if pinyin in pinyin_rep_map.keys():
                    pinyin = pinyin_rep_map[pinyin]
                else:
                    single_rep_map = {
                        "v": "yu",
                        "e": "e",
                        "i": "y",
                        "u": "w",
                    }
                    if pinyin[0] in single_rep_map.keys():
                        pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

            assert pinyin in pinyin_to_symbol_map.keys(), tone
            new_init, new_final = pinyin_to_symbol_map[pinyin].split(" ")
            new_final = new_final + _tone

            return new_init, new_final

    def revise_custom_tone(
        self, phones: List[str], word2ph: List[int], tone_data_list: List
    ) -> None:
        """
        修正自定义多音字

        Args:
            phones: 音素序列
            word2ph: 字到音素的映射
            tone_data_list: 拼音数据列表
        """
        for td in tone_data_list:
            init = td[1]
            final = td[2]
            pos = td[3]
            if init == "" and final == "":
                # 如果匹配拼音的时候失败，这里保持模型中默认提供的读音
                continue

            # 计算音素位置 - 现在pos直接是"福"字的位置
            wd_pos = 0
            for i in range(0, pos):
                wd_pos += word2ph[i]

            # "福"字占位符本身的音素位置，需要加上"福"字对应的音素数
            wd_pos += word2ph[pos]

            # 修改"福"字的音素为指定拼音的音素
            phones[wd_pos - 2] = init
            phones[wd_pos - 1] = final
            info(f"[+]自定义读音: {init}{final}，位置: {wd_pos - 2}")

    # 空行替换成静音标记<BR>+换行
    def replace_empty_lines_with_br(self, text: str) -> str:
        """
        将空行替换为静音标记<BR>+换行，一个空行一个<BR>+换行

        Args:
            text: 输入文本

        Returns:
            替换后的文本
        """

        sub_texts = []
        for sub_text in text.split("\n"):
            sub_text = re.sub(r"\s+", "", sub_text)
            if sub_text == "":
                sub_text = self.BR_TAG
            sub_texts.append(sub_text)

        return "\n".join(sub_texts)

    def _load_replace_rules(self) -> List[Tuple[str, str, str]]:
        if not os.path.exists(self.TEXT_REPLACE_RULES_FILE):
            return []

        # 读取配置文件修改时间，如果修改时间大于last_mtime，则重新加载配置文件
        file_mtime = os.path.getmtime(self.TEXT_REPLACE_RULES_FILE)

        if file_mtime <= self.last_mtime:
            debug("文本替换规则文件未修改，跳过加载")
            return self.replace_rules

        self.last_mtime = file_mtime
        self.replace_rules = []

        try:
            with open(self.TEXT_REPLACE_RULES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split("|")  # 使用竖线分隔
                    if len(parts) == 3:
                        search_str, replace_from, replace_to = parts
                        self.replace_rules.append(
                            (search_str, replace_from, replace_to)
                        )
                    else:
                        warning(f"警告：配置行格式不正确，已跳过: {line}")

            if self.replace_rules:
                info(f"已加载 {len(self.replace_rules)} 条文本替换规则")
        except Exception as e:
            error(f"加载文本替换配置文件出错: {str(e)}")

        return self.replace_rules

    def _apply_replace_rules(self, text: str) -> Tuple[int, str]:
        replace_rules = self._load_replace_rules()

        result_text = text
        count = 0
        for search_str, replace_from, replace_to in replace_rules:
            # 在搜索字符串中查找需要修改的部分并替换
            if search_str in result_text:
                # 创建一个新字符串，将搜索字符串中的替换源替换为替换目标
                modified_search_str = search_str.replace(replace_from, replace_to)
                # 替换原文本中的搜索字符串为修改后的字符串
                result_text = result_text.replace(search_str, modified_search_str)
                debug(f"文本替换: {search_str} -> {modified_search_str}")
                count += 1

        return count, result_text
