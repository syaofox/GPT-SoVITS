"""
GPT-SoVITS 推理模块使用示例
"""

import argparse
import soundfile as sf
from gpt_sovits_inference import GPTSoVITSInference


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GPT-SoVITS 语音合成示例")
    parser.add_argument("--ref", type=str, required=True, help="参考音频路径")
    parser.add_argument("--text", type=str, required=True, help="需要合成的文本")
    parser.add_argument("--prompt_text", type=str, default="", help="参考音频文本")
    parser.add_argument("--prompt_language", type=str, default="中文", help="参考音频语言")
    parser.add_argument("--text_language", type=str, default="中文", help="合成文本语言")
    parser.add_argument("--output", type=str, default="output.wav", help="输出音频路径")
    parser.add_argument("--gpt", type=str, default=None, help="GPT模型路径")
    parser.add_argument("--sovits", type=str, default=None, help="SoVITS模型路径")
    parser.add_argument("--device", type=str, default=None, help="计算设备，默认自动选择")
    parser.add_argument("--speed", type=float, default=1.0, help="语速，范围0.6-1.65")
    parser.add_argument("--steps", type=int, default=8, help="v3/v4模型的采样步数")
    parser.add_argument("--sr", action="store_true", help="是否使用超分(仅v3模型支持)")
    args = parser.parse_args()
    
    # 初始化推理模块
    tts_model = GPTSoVITSInference(
        gpt_path=args.gpt,
        sovits_path=args.sovits,
        device=args.device
    )
    
    # 生成语音
    sample_rate, audio_data = tts_model.generate_speech(
        ref_wav_path=args.ref,
        prompt_text=args.prompt_text,
        prompt_language=args.prompt_language,
        text=args.text,
        text_language=args.text_language,
        speed=args.speed,
        sample_steps=args.steps,
        if_sr=args.sr
    )
    
    # 保存音频
    sf.write(args.output, audio_data, sample_rate)
    print(f"语音生成完成，已保存至 {args.output}")


if __name__ == "__main__":
    main() 