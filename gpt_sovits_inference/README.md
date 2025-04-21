# GPT-SoVITS 推理模块

本模块提供了GPT-SoVITS模型的推理功能，将原始Web UI中的核心推理逻辑封装为可编程调用的Python模块。

## 特性

- 支持v1-v4四个版本的GPT-SoVITS模型
- 支持中文、英文、日文、粤语、韩文等多种语言和混合语言
- 提供与原Web UI相同的所有功能
- 易于集成到其他应用中
- 代码模块化，结构清晰

## 安装

确保已安装原始GPT-SoVITS仓库的所有依赖，然后将本模块复制到GPT-SoVITS根目录下：

```bash
# 假设已经克隆了GPT-SoVITS仓库
cd GPT-SoVITS
# 将本模块文件夹复制到此目录下
```

## 基本用法

```python
from gpt_sovits_inference import GPTSoVITSInference

# 初始化推理模块
tts_model = GPTSoVITSInference(
    # 可以指定特定模型路径，不指定则使用配置文件中的
    # gpt_path="GPT_weights/your_model.ckpt",
    # sovits_path="SoVITS_weights/your_model.pth",
    half=True  # 使用半精度加速
)

# 生成语音
sample_rate, audio_data = tts_model.generate_speech(
    ref_wav_path="path/to/reference_audio.wav",  # 参考音频
    prompt_text="这是参考音频的文本",  # 参考音频的文本
    prompt_language="中文",  # 参考音频的语言
    text="这是需要合成的新文本，将使用参考音频的音色。",  # 需要合成的文本
    text_language="中文",  # 合成文本的语言
    how_to_cut="凑四句一切",  # 文本切分方式
    top_k=20,  # GPT采样参数
    top_p=0.6,  # GPT采样参数
    temperature=0.6,  # GPT采样参数
    speed=1.0,  # 语速
    sample_steps=8,  # v3/v4模型采样步数
    if_sr=False  # 是否使用超分辨率增强(仅v3模型支持)
)

# 保存音频
import soundfile as sf
sf.write("output_speech.wav", audio_data, sample_rate)
```

## 命令行示例

提供了一个简单的命令行示例程序：

```bash
python -m gpt_sovits_inference.example \
    --ref path/to/reference.wav \
    --text "这是一段示例文本，将使用参考音频的音色合成" \
    --prompt_text "参考音频中说的内容" \
    --output output.wav
```

更多参数：

```bash
python -m gpt_sovits_inference.example --help
```

## 支持的语言

- **单语种**：中文、英文、日文、粤语、韩文等
- **混合语种**：中英混合、日英混合、粤英混合、韩英混合、多语种混合等

## 模型版本特性

- **v1/v2模型**：支持无参考文本模式、多参考音频融合
- **v3模型**：支持音频超分、不支持无参考文本模式
- **v4模型**：HiFi-GAN声码器、更高采样率输出、不支持无参考文本模式

## 贡献和反馈

欢迎提交Pull Request或Issues，共同改进项目。

## 许可

与原GPT-SoVITS项目相同。

## 致谢

感谢原始[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)项目的所有贡献者。 