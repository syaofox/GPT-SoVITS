#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-SoVITS 本地库使用示例
"""

import os
import argparse
import logging
from gpt_sovits_lib import GPTSoVITS, GPTSoVITSConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS Example")
    parser.add_argument("--sovits", type=str, default="", help="SoVITS模型路径")
    parser.add_argument("--gpt", type=str, default="", help="GPT模型路径")
    parser.add_argument("--ref", type=str, required=True, help="参考音频路径")
    parser.add_argument("--ref_text", type=str, required=True, help="参考音频文本")
    parser.add_argument("--ref_lang", type=str, default="zh", help="参考音频语言")
    parser.add_argument("--text", type=str, required=True, help="要合成的文本")
    parser.add_argument("--text_lang", type=str, default="zh", help="要合成的文本语言")
    parser.add_argument("--output", type=str, default="output.wav", help="输出文件路径")
    parser.add_argument("--device", type=str, default="", help="推理设备，'cuda'或'cpu'")
    parser.add_argument("--format", type=str, default="wav", help="输出音频格式，'wav'、'ogg'或'aac'")
    parser.add_argument("--speaker", type=str, default="default", help="说话人名称")
    parser.add_argument("--top_k", type=int, default=15, help="top_k参数")
    parser.add_argument("--top_p", type=float, default=0.6, help="top_p参数")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature参数")
    parser.add_argument("--speed", type=float, default=1.0, help="语速")
    parser.add_argument("--sample_steps", type=int, default=32, help="采样步数")
    args = parser.parse_args()

    # 创建配置对象
    config = GPTSoVITSConfig()
    
    # 设置模型路径
    if args.sovits:
        config.sovits_path = args.sovits
    if args.gpt:
        config.gpt_path = args.gpt
    if args.device:
        config.infer_device = args.device
        config.is_half = True if args.device == "cuda" else False
    
    # 初始化GPT-SoVITS
    logger.info(f"初始化GPT-SoVITS，使用设备: {config.infer_device}")
    tts = GPTSoVITS(config)
    
    # 加载模型
    logger.info("加载模型...")
    success = tts.load_models(speaker_name=args.speaker)
    if not success:
        logger.error("模型加载失败！")
        return
    
    # 文本转语音
    logger.info(f"开始合成语音: {args.text}")
    audio_data = tts.tts(
        ref_wav_path=args.ref,
        prompt_text=args.ref_text,
        prompt_language=args.ref_lang,
        text=args.text,
        text_language=args.text_lang,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
        sample_steps=args.sample_steps,
        audio_format=args.format,
    )
    
    # 保存音频
    if audio_data:
        with open(args.output, "wb") as f:
            f.write(audio_data)
        logger.info(f"语音合成完成，已保存到: {args.output}")
    else:
        logger.error("语音合成失败！")

if __name__ == "__main__":
    main() 