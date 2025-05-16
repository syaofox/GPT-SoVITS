import re
from typing import List, Dict, Any
from models.constand import BR_TAG


def _split_text_by_speaker_and_lines(
    text: str, default_speaker: str, default_emotion: str = ""
) -> List[Dict[str, Any]]:
    """
    按角色分割文本，并返回字典列表。
    文本格式示例：
    ```
    <角色名1|情绪1>
    文本内容段落1

    文本内容段落2

    <角色名2|情绪2>
    文本内容段落1

    文本内容段落2
    ```

    如果第一行不是<角色名|情绪>格式，则使用default_speaker和default_emotion

    返回格式：
    [
        {"text": "文本内容段落1", "speaker": "角色名1", "emotion": "情绪1"},
        {"text": "<BR>", "speaker": "角色名1", "emotion": "情绪1"},
        ...
    ]
    """
    if not text:
        return []

    # 清除文本开头的空行
    text = re.sub(r"^[\n\r]+", "", text)

    # 清除中英文引号
    text = re.sub(r"[‘’“”\"\']", "", text)

    segments = []
    lines = text.split("\n")

    current_speaker = default_speaker
    current_emotion = default_emotion

    buffer = []  # 用于临时存储当前角色的文本行

    # 角色标记正则表达式，匹配<角色名|情绪>格式或<角色名>格式
    speaker_pattern = re.compile(r"^<([^|>]+)(?:\|([^>]+))?>\s*$")

    # 检查第一行是否是角色标记，如果不是，保持默认角色和情绪
    first_line_processed = False

    for line in lines:
        line = line.strip()

        # 检查是否是角色标记行
        speaker_match = speaker_pattern.match(line)
        if speaker_match:
            # 如果buffer中有内容，先处理之前角色的内容
            if buffer:
                for text_line in buffer:
                    if not text_line:
                        segments.append(
                            {
                                "text": BR_TAG,
                                "speaker": current_speaker,
                                "emotion": current_emotion,
                            }
                        )
                    else:
                        segments.append(
                            {
                                "text": text_line,
                                "speaker": current_speaker,
                                "emotion": current_emotion,
                            }
                        )
                buffer = []

            # 更新当前角色和情绪
            current_speaker = speaker_match.group(1)
            # 获取情绪，如果没有指定则使用默认情绪
            current_emotion = (
                speaker_match.group(2) if speaker_match.group(2) else default_emotion
            )
            first_line_processed = True
        else:
            # 普通文本行，添加到buffer
            buffer.append(line)
            # 如果这是第一行且不是角色标记，标记为已处理
            if not first_line_processed:
                first_line_processed = True

    # 处理最后一个角色的内容
    if buffer:
        for text_line in buffer:
            if not text_line:
                segments.append(
                    {
                        "text": BR_TAG,
                        "speaker": current_speaker,
                        "emotion": current_emotion,
                    }
                )
            else:
                segments.append(
                    {
                        "text": text_line,
                        "speaker": current_speaker,
                        "emotion": current_emotion,
                    }
                )

    return segments
