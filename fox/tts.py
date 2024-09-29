from setenv import setenv

setenv()

import json
import os
import re
from pathlib import Path

import numpy as np
from futils import sanitize_filename
from langdetect import detect
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
from scipy.io import wavfile

from GPT_SoVITS import inference_webui as iw
from GPT_SoVITS import inference_webui_fast as iwf
from GPT_SoVITS.text.chinese2 import pinyin_to_symbol_map
from GPT_SoVITS.text.symbols import punctuation

iw.language = 'zh_CN'

current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, 'correct_words.json')
with open(json_file_path, 'r', encoding='utf-8') as file:
    correct_words_map = json.load(file)


def get_phones_and_bert(text, language, version, final=False):
    if language in {'en', 'all_zh', 'all_ja', 'all_ko', 'all_yue'}:
        language = language.replace('all_', '')
        if language == 'en':
            iw.LangSegment.setfilters(['en'])
            formattext = ' '.join(tmp['text'] for tmp in iw.LangSegment.getTexts(text))  # type: ignore
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while '  ' in formattext:
            formattext = formattext.replace('  ', ' ')
        if language == 'zh':
            if '<tone' in formattext:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = iw.get_bert_feature(norm_text, word2ph).to(iw.device)

            elif re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = iw.chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, 'zh', version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = iw.get_bert_feature(norm_text, word2ph).to(iw.device)
        elif language == 'yue' and re.search(r'[A-Za-z]', formattext):
            formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
            formattext = iw.chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, 'yue', version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = iw.torch.zeros(
                (1024, len(phones)),
                dtype=iw.torch.float16 if iw.is_half else iw.torch.float32,
            ).to(iw.device)
    elif language in {'zh', 'ja', 'ko', 'yue', 'auto', 'auto_yue'}:
        textlist = []
        langlist = []
        iw.LangSegment.setfilters(['zh', 'ja', 'en', 'ko'])
        if language == 'auto':
            for tmp in iw.LangSegment.getTexts(text):
                langlist.append(tmp['lang'])  # type: ignore
                textlist.append(tmp['text'])  # type: ignore
        elif language == 'auto_yue':
            for tmp in iw.LangSegment.getTexts(text):
                if tmp['lang'] == 'zh':  # type: ignore
                    tmp['lang'] = 'yue'  # type: ignore
                langlist.append(tmp['lang'])  # type: ignore
                textlist.append(tmp['text'])  # type: ignore
        else:
            for tmp in iw.LangSegment.getTexts(text):
                if tmp['lang'] == 'en':  # type: ignore
                    langlist.append(tmp['lang'])  # type: ignore
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp['text'])  # type: ignore
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = iw.get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = iw.torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert('.' + text, language, version, final=True)

    return phones, bert.to(iw.dtype), norm_text


# 拦截inference_webui中的get_phones_and_bert
iw.get_phones_and_bert = get_phones_and_bert


def find_custom_tone(text: str):
    """
    识别、提取文本中的多音字
    """
    tone_list = []
    txts = []
    # 识别 tone 标记，形如<tone as=shu4>数</tone>或<tone as=\"shu3\">数</tone>或<tone as=\"shù\">数</tone>
    ptn1 = re.compile(r'<tone.*?>(.*?)</tone>')
    # 清除 tone 标记中不需要的部分
    ptn2 = re.compile(r"(</?tone)|(as)|([>\"'\s=])")
    matches = list(re.finditer(ptn1, text))
    offset = 0
    for match in matches:
        # tone 标记之前的文本
        pre = text[offset : match.start()]
        txts.append(pre)
        # tone 标签中的单个多音字
        tone_text = match.group(1)
        txts.append(tone_text)
        # 提取读音，支持识别 Style.TONE 和  Style.TONE3
        tone = match.group(0)
        tone = re.sub(ptn2, '', tone)
        tone = tone.replace(tone_text, '')
        # 多音字在当前文本中的索引位置
        pos = sum([len(s) for s in txts])
        offset = match.end()
        init, final = correct_initial_final(tone)
        data = [tone, init, final, pos]
        tone_list.append(data)
    # 不能忘了最后一个 tone 标签后面可能还有剩余的内容
    if offset < len(text):
        txts.append(text[offset:])

    text = ''.join(str(i) for i in txts)
    text = text.replace(' ', '')  # 去除空格
    return text, tone_list


def replace_custom_words(text: str):
    tone_list = []
    txts = []
    offset = 0

    # 遍历文本中的每个字符
    for i, char in enumerate(text):
        if char in correct_words_map:
            # 将匹配到的字符及之前的文本添加到txts
            if offset < i:
                txts.append(text[offset:i])
            txts.append(char)
            tone = correct_words_map[char]
            pos = sum(len(s) for s in txts)
            init, final = correct_initial_final(tone)
            tone_list.append([tone, init, final, pos])

            offset = i + 1
    return tone_list


def correct_initial_final(tone):
    init = ''
    final = ''
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
        assert _tone in '12345'

        if init:
            # 多音节
            v_rep_map = {
                'uei': 'ui',
                'iou': 'iu',
                'uen': 'un',
            }
            if v_without_tone in v_rep_map.keys():
                pinyin = init + v_rep_map[v_without_tone]
        else:
            # 单音节
            pinyin_rep_map = {
                'ing': 'ying',
                'i': 'yi',
                'in': 'yin',
                'u': 'wu',
            }
            if pinyin in pinyin_rep_map.keys():
                pinyin = pinyin_rep_map[pinyin]
            else:
                single_rep_map = {
                    'v': 'yu',
                    'e': 'e',
                    'i': 'y',
                    'u': 'w',
                }
                if pinyin[0] in single_rep_map.keys():
                    pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

        assert pinyin in pinyin_to_symbol_map.keys(), tone
        new_init, new_final = pinyin_to_symbol_map[pinyin].split(' ')
        new_final = new_final + _tone

        return new_init, new_final


def revise_custom_tone(phones, word2ph, tone_data_list):
    """
    修正自定义多音字
    """
    for td in tone_data_list:
        tone = td[0]
        init = td[1]
        final = td[2]
        pos = td[3]
        if init == '' and final == '':
            # 如果匹配拼音的时候失败，这里保持模型中默认提供的读音
            continue

        wd_pos = 0
        for i in range(0, pos):
            wd_pos += word2ph[i]
        org_init = phones[wd_pos - 2]
        org_final = phones[wd_pos - 1]
        phones[wd_pos - 2] = init
        phones[wd_pos - 1] = final
        print(f'[+]成功修改读音: {org_init}{org_final} => {tone}')


def clean_text_inf(text, language, version):
    text, tone_data_list = find_custom_tone(text)
    replace_tone_data_list = replace_custom_words(text)

    phones, word2ph, norm_text = iw.clean_text(text, language, version)
    # 修正多音字
    if len(tone_data_list) > 0:
        revise_custom_tone(phones, word2ph, tone_data_list)

    if len(replace_tone_data_list) > 0:
        revise_custom_tone(phones, word2ph, replace_tone_data_list)

    phones = iw.cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


# 拦截inference_webui中的clean_text_inf
iw.clean_text_inf = clean_text_inf


def normalize_audio(file_path, sampling_rate, audio_data):
    if audio_data.dtype != np.int16:
        audio_data = np.int16(audio_data)
    wavfile.write(file_path, sampling_rate, audio_data)
    return audio_data


def save_audio(file_path, sampling_rate, audio_data):
    """保存音频数据到文件"""
    wavfile.write(file_path, sampling_rate, audio_data)


def load_ttschars(json_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        tts_chars_dict_loaded = dict(json.load(f))
        for k in tts_chars_dict_loaded.keys():
            tts_chars_dict_loaded[k]['feels'] = [d.name for d in Path(rf'sample\{k}').iterdir() if d.is_dir()]
        return tts_chars_dict_loaded


tts_chars_dict = load_ttschars(r'sample\tts_chars_dict.json')
char_names = list(tts_chars_dict.keys())


def change_char_ui(char_name):
    global current_feels
    current_feels = tts_chars_dict[char_name]['feels']

    return {'choices': char_names, '__type__': 'update'}, {
        'choices': current_feels,
        '__type__': 'update',
    }


change_char_ui(char_names[0])


def change_char(char_name):
    load_gpt_path = tts_chars_dict[char_name]['gpt_path']
    if load_gpt_path:
        gpt_path = f'GPT_weights_v2\\{load_gpt_path}'
    else:
        gpt_path = r'GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
    iw.change_gpt_weights(gpt_path)

    load_sovits_path = tts_chars_dict[char_name]['sovist_path']
    if load_sovits_path:
        sovits_path = f'SoVITS_weights_v2\\{load_sovits_path}'
    else:
        sovits_path = r'GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth'
    iw.change_sovits_weights(sovits_path)

    iw.config['char_name'] = char_name


def detect_language(text):
    try:
        # 检测文本语言
        lang = detect(text)
        if lang == 'zh-cn':
            return '中文'
        elif lang == 'en':
            return '英文'
        elif lang == 'ja':
            return '日文'
        elif lang == 'ko':
            return '韩文'
        else:
            return '中文'
    except Exception:
        return '中文'


# 附加参考音频，为了适应inference_webui中的格式
class RefsParam:
    def __init__(self, name_path) -> None:
        self.name = str(name_path)


def create_tts_params(char_name, feel):
    sample_path = Path(rf'sample\{char_name}\{feel}')
    sample_wav_files = list(sample_path.glob('*.wav'))
    assert sample_wav_files, '参考音频不存在'

    ref_wav_path = str(sample_wav_files[0])
    prompt_text = sample_wav_files[0].stem
    prompt_language = detect_language(prompt_text)

    refs_path = sample_path / 'refs'
    inp_refs = [RefsParam(x) for x in refs_path.glob('*.wav')]
    return ref_wav_path, prompt_text, prompt_language, inp_refs


def extract_all_parts(text, charname, feel):
    # 定义正则表达式模式
    pattern_full = r'(.*?)_(.*?)\|(.*)'
    pattern_front = r'(.*?)_(.*)'
    pattern_back = r'(.*?)\|(.*)'

    # 使用正则表达式进行匹配
    match_full = re.search(pattern_full, text)
    match_front = re.search(pattern_front, text)
    match_back = re.search(pattern_back, text)

    if match_full:
        _charname = match_full.group(1)
        _feel = match_full.group(2)
        _remaining_str = match_full.group(3)
    elif match_back:
        _charname = charname
        _feel = match_back.group(1)
        _remaining_str = match_back.group(2)
    elif match_front:
        _charname = match_front.group(1)
        _feel = feel
        _remaining_str = match_front.group(2)
    else:
        _charname = charname
        _feel = feel
        _remaining_str = text

    return _charname, _feel, _remaining_str


def get_tts(
    char_name,
    feel,
    text,
    text_language,
    how_to_cut=iw.i18n('按中文句号。切'),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1,
    if_freeze=False,
    if_single=True,
    if_fast=False,
    blank_ms=300,
):
    ref_text_free = False

    # 输出结果另外保存
    out_dir = Path('TEMP\\_output')
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(out_dir / f'{char_name}_{sanitize_filename(text[:30])}.wav')

    if iw.config.get('char_name', None) is None:
        change_char(char_name)

    if if_single:
        texts = text.splitlines()
        target_sample_rate = None
        combined_audio = np.array([], dtype=np.int16)

        for sub_text in texts:
            if not sub_text:
                zero_wav = np.zeros(
                    int(iw.hps.data.sampling_rate * (int(blank_ms) / 1000)),
                    dtype=np.float16 if iw.is_half is True else np.float32,
                )
                combined_audio = np.concatenate((combined_audio, zero_wav))
                print(f'空行添加{blank_ms}ms间隔')
                continue

            sub_char_name, sub_feel, left_text = extract_all_parts(sub_text, char_name, feel)

            model_charname = iw.config.get('char_name', None)
            assert model_charname, '模型角色未设置'

            if sub_char_name != model_charname:
                change_char(sub_char_name)
                ref_wav_path, prompt_text, prompt_language, inp_refs = create_tts_params(sub_char_name, sub_feel)

            else:
                ref_wav_path, prompt_text, prompt_language, inp_refs = create_tts_params(char_name, sub_feel)

            for sr, audio in iw.get_tts_wav(
                ref_wav_path,
                prompt_text,
                prompt_language,
                left_text,
                text_language,
                how_to_cut,
                top_k,
                top_p,
                temperature,
                ref_text_free,
                speed,
                if_freeze,
                inp_refs,  # type: ignore
            ):
                if target_sample_rate is None:
                    target_sample_rate = sr
                combined_audio = np.concatenate((combined_audio, audio))

        yield (
            target_sample_rate,
            normalize_audio(save_path, target_sample_rate, combined_audio),
        )
    elif if_fast:
        model_charname = iw.config['train']['exp_name']
        if char_name != model_charname:
            change_char(char_name)
        ref_wav_path, prompt_text, prompt_language, inp_refs = create_tts_params(char_name, feel)

        split_bucket = True
        fragment_interval = 0.3
        seed = -1
        keep_random = True
        parallel_infer = True
        repetition_penalty = 1.35
        batch_size = 20
        top_k = 5
        top_p = 1
        temperature = 1

        for item, actual_seed in iwf.inference(
            text, text_language, ref_wav_path, inp_refs, prompt_text, prompt_language, top_k, top_p, temperature, how_to_cut, batch_size, speed, ref_text_free, split_bucket, fragment_interval, seed, keep_random, parallel_infer, repetition_penalty
        ):
            if isinstance(item, tuple) and len(item) == 2:
                sr, audio_data = item
                yield sr, normalize_audio(save_path, sr, audio_data)
            else:
                print(f'Unexpected item format in fast inference: {item}')
    else:
        model_charname = iw.config['train']['exp_name']
        if char_name != model_charname:
            change_char(char_name)

        ref_wav_path, prompt_text, prompt_language, inp_refs = create_tts_params(char_name, feel)
        for sr, audio_data in iw.get_tts_wav(
            ref_wav_path,
            prompt_text,
            prompt_language,
            text,
            text_language,
            how_to_cut,
            top_k,
            top_p,
            temperature,
            ref_text_free,
            speed,
            if_freeze,
            inp_refs,  # type: ignore
        ):
            yield sr, normalize_audio(save_path, sr, audio_data)


def ui():
    with iw.gr.Blocks(title='GPT-SoVITS WebUI') as app:
        with iw.gr.Group():
            iw.gr.Markdown(iw.i18n('角色选择'))
            with iw.gr.Row():
                chars_dropdown = iw.gr.Dropdown(
                    label=iw.i18n('角色列表'),
                    choices=char_names,
                    value=char_names[0],
                    interactive=True,
                    scale=14,
                )
                feel_dropdown = iw.gr.Dropdown(
                    label=iw.i18n('情绪选择'),
                    choices=current_feels,
                    value=current_feels[0],
                    interactive=True,
                    scale=14,
                )

            with iw.gr.Row():
                with iw.gr.Column(scale=13):
                    text = iw.gr.Textbox(
                        label=iw.i18n('需要合成的文本'),
                        value='',
                        lines=34,
                        max_lines=34,
                    )
                    output = iw.gr.Audio(label=iw.i18n('输出的语音'), scale=14)
                with iw.gr.Column(scale=7):
                    text_language = iw.gr.Dropdown(
                        label=iw.i18n('需要合成的语种') + iw.i18n('.限制范围越小判别效果越好。'),
                        choices=list(iw.dict_language.keys()),
                        value=iw.i18n('中文'),
                        scale=1,
                    )
                    how_to_cut = iw.gr.Dropdown(
                        label=iw.i18n('怎么切'),
                        choices=[
                            iw.i18n('不切'),
                            iw.i18n('凑四句一切'),
                            iw.i18n('凑50字一切'),
                            iw.i18n('按中文句号。切'),
                            iw.i18n('按英文句号.切'),
                            iw.i18n('按标点符号切'),
                        ],
                        value=iw.i18n('按中文句号。切'),
                        interactive=True,
                        scale=1,
                    )
                    blank_ms = iw.gr.Slider(
                        minimum=0,
                        maximum=10000,
                        step=100,
                        label=iw.i18n('停顿'),
                        value=300,
                        interactive=True,
                        scale=1,
                    )

                    iw.gr.Markdown(value=iw.html_center(iw.i18n('语速调整，高为更快')))
                    if_freeze = iw.gr.Checkbox(
                        label=iw.i18n('是否直接对上次合成结果调整语速和音色。防止随机性。'),
                        value=False,
                        interactive=True,
                        show_label=True,
                        scale=1,
                    )
                    speed = iw.gr.Slider(
                        minimum=0.6,
                        maximum=1.65,
                        step=0.01,
                        label=iw.i18n('语速'),
                        value=1,
                        interactive=True,
                        scale=1,
                    )
                    iw.gr.Markdown(iw.html_center(iw.i18n('GPT采样参数(无参考文本时不要太低。不懂就用默认)：')))
                    top_k = iw.gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label=iw.i18n('top_k'),
                        value=15,
                        interactive=True,
                        scale=1,
                    )
                    top_p = iw.gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        label=iw.i18n('top_p'),
                        value=1,
                        interactive=True,
                        scale=1,
                    )
                    temperature = iw.gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        label=iw.i18n('temperature'),
                        value=1,
                        interactive=True,
                        scale=1,
                    )

                    iw.gr.Markdown(value=iw.html_center(iw.i18n('情感标签')))
                    if_single = iw.gr.Checkbox(
                        label=iw.i18n('是否启用情感标签(格式: 情绪|文本内容)'),
                        value=False,
                        interactive=True,
                        show_label=True,
                        scale=1,
                    )
                    if_fast = iw.gr.Checkbox(
                        label=iw.i18n('是否启用快速合成'),
                        value=False,
                        interactive=True,
                        show_label=True,
                        scale=1,
                    )

                    inference_button = iw.gr.Button(iw.i18n('合成语音'), variant='primary', size='lg', scale=2)

            inference_button.click(
                get_tts,
                [
                    chars_dropdown,
                    feel_dropdown,
                    text,
                    text_language,
                    how_to_cut,
                    top_k,
                    top_p,
                    temperature,
                    speed,
                    if_freeze,
                    if_single,
                    if_fast,
                    blank_ms,
                ],
                [output],
            )
            chars_dropdown.change(
                change_char_ui,
                [chars_dropdown],
                outputs=[chars_dropdown, feel_dropdown],
            )

    app.queue().launch(
        server_name='0.0.0.0',
        inbrowser=True,
        share=iw.is_share,
        server_port=iw.infer_ttswebui,
        quiet=True,
    )


if __name__ == '__main__':
    ui()
