#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-SoVITS 简化版API服务

使用 FastAPI 框架，基于本地库实现
"""

import os
import argparse
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
from pydantic import BaseModel

from gpt_sovits_lib import GPTSoVITS, GPTSoVITSConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpt_sovits_api")

# API 请求模型
class TTSRequest(BaseModel):
    refer_wav_path: Optional[str] = None
    prompt_text: Optional[str] = None
    prompt_language: Optional[str] = None
    text: str
    text_language: str
    top_k: Optional[int] = 15
    top_p: Optional[float] = 0.6
    temperature: Optional[float] = 0.6
    speed: Optional[float] = 1.0
    inp_refs: Optional[List[str]] = []
    sample_steps: Optional[int] = 32
    if_sr: Optional[bool] = False
    cut_punc: Optional[str] = ",.，。"
    pause_second: Optional[float] = 0.3


class ChangeReferRequest(BaseModel):
    refer_wav_path: str
    prompt_text: str
    prompt_language: str


# 全局变量
config = GPTSoVITSConfig()
tts_model = None
default_refer_path = ""
default_refer_text = ""
default_refer_language = ""

# 创建FastAPI应用
app = FastAPI(title="GPT-SoVITS API", description="GPT-SoVITS 文本到语音合成服务")


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    global tts_model
    logger.info(f"初始化GPT-SoVITS，使用设备: {config.infer_device}")
    tts_model = GPTSoVITS(config)
    
    # 加载模型
    logger.info("加载模型...")
    success = tts_model.load_models()
    if not success:
        logger.error("模型加载失败！")


@app.post("/set_model")
async def set_model(request: Request):
    """设置模型路径"""
    global tts_model
    
    json_post_raw = await request.json()
    gpt_path = json_post_raw.get("gpt_model_path")
    sovits_path = json_post_raw.get("sovits_model_path")
    
    if not gpt_path and not sovits_path:
        return JSONResponse({"code": 400, "message": "至少需要提供一个模型路径"}, status_code=400)
    
    if gpt_path:
        config.gpt_path = gpt_path
    if sovits_path:
        config.sovits_path = sovits_path
    
    # 重新加载模型
    logger.info(f"重新加载模型:")
    logger.info(f"GPT 路径: {config.gpt_path}")
    logger.info(f"SoVITS 路径: {config.sovits_path}")
    
    # 如果已经初始化过模型，先销毁旧模型
    if tts_model:
        del tts_model
        
    tts_model = GPTSoVITS(config)
    success = tts_model.load_models()
    
    if success:
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        return JSONResponse({"code": 400, "message": "模型加载失败"}, status_code=400)


@app.get("/set_model")
async def set_model_get(
    gpt_model_path: Optional[str] = None,
    sovits_model_path: Optional[str] = None,
):
    """设置模型路径 (GET方法)"""
    global tts_model
    
    if not gpt_model_path and not sovits_model_path:
        return JSONResponse({"code": 400, "message": "至少需要提供一个模型路径"}, status_code=400)
    
    if gpt_model_path:
        config.gpt_path = gpt_model_path
    if sovits_model_path:
        config.sovits_path = sovits_model_path
    
    # 重新加载模型
    logger.info(f"重新加载模型:")
    logger.info(f"GPT 路径: {config.gpt_path}")
    logger.info(f"SoVITS 路径: {config.sovits_path}")
    
    # 如果已经初始化过模型，先销毁旧模型
    if tts_model:
        del tts_model
        
    tts_model = GPTSoVITS(config)
    success = tts_model.load_models()
    
    if success:
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        return JSONResponse({"code": 400, "message": "模型加载失败"}, status_code=400)


@app.post("/change_refer")
async def change_refer(request: Request):
    """更改默认参考音频"""
    global default_refer_path, default_refer_text, default_refer_language
    
    json_post_raw = await request.json()
    path = json_post_raw.get("refer_wav_path")
    text = json_post_raw.get("prompt_text")
    language = json_post_raw.get("prompt_language")
    
    if not path or not text or not language:
        return JSONResponse(
            {
                "code": 400,
                "message": '缺少任意一项以下参数: "path", "text", "language"',
            },
            status_code=400,
        )
    
    default_refer_path = path
    default_refer_text = text
    default_refer_language = language
    
    logger.info(f"当前默认参考音频路径: {default_refer_path}")
    logger.info(f"当前默认参考音频文本: {default_refer_text}")
    logger.info(f"当前默认参考音频语种: {default_refer_language}")
    
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


@app.get("/change_refer")
async def change_refer_get(
    refer_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_language: Optional[str] = None,
):
    """更改默认参考音频 (GET方法)"""
    global default_refer_path, default_refer_text, default_refer_language
    
    if not refer_wav_path or not prompt_text or not prompt_language:
        return JSONResponse(
            {
                "code": 400,
                "message": '缺少任意一项以下参数: "refer_wav_path", "prompt_text", "prompt_language"',
            },
            status_code=400,
        )
    
    default_refer_path = refer_wav_path
    default_refer_text = prompt_text
    default_refer_language = prompt_language
    
    logger.info(f"当前默认参考音频路径: {default_refer_path}")
    logger.info(f"当前默认参考音频文本: {default_refer_text}")
    logger.info(f"当前默认参考音频语种: {default_refer_language}")
    
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


@app.post("/")
async def tts_endpoint(request: Request):
    """TTS合成接口 (POST方法)"""
    global tts_model, default_refer_path, default_refer_text, default_refer_language
    
    # 检查模型是否已初始化
    if not tts_model:
        return JSONResponse({"code": 400, "message": "模型未初始化"}, status_code=400)
    
    try:
        json_post_raw = await request.json()
        
        # 获取参数
        refer_wav_path = json_post_raw.get("refer_wav_path", default_refer_path)
        prompt_text = json_post_raw.get("prompt_text", default_refer_text)
        prompt_language = json_post_raw.get("prompt_language", default_refer_language)
        text = json_post_raw.get("text")
        text_language = json_post_raw.get("text_language")
        cut_punc = json_post_raw.get("cut_punc", ",.，。")
        top_k = json_post_raw.get("top_k", 15)
        top_p = json_post_raw.get("top_p", 0.6)
        temperature = json_post_raw.get("temperature", 0.6)
        speed = json_post_raw.get("speed", 1.0)
        inp_refs = json_post_raw.get("inp_refs", [])
        sample_steps = json_post_raw.get("sample_steps", 32)
        if_sr = json_post_raw.get("if_sr", False)
        pause_second = json_post_raw.get("pause_second", 0.3)
        
        # 验证必要参数
        if not text or not text_language:
            return JSONResponse(
                {"code": 400, "message": "缺少必要参数: text, text_language"},
                status_code=400,
            )
        
        # 验证参考音频
        if not refer_wav_path or not prompt_text or not prompt_language:
            if not default_refer_path or not default_refer_text or not default_refer_language:
                return JSONResponse(
                    {"code": 400, "message": "未指定参考音频且接口无预设"},
                    status_code=400,
                )
            refer_wav_path = default_refer_path
            prompt_text = default_refer_text
            prompt_language = default_refer_language
            
        # 样本步数校验
        if not sample_steps in [4, 8, 16, 32, 64, 128]:
            sample_steps = 32
            
        # 合成音频
        audio_data = tts_model.tts(
            ref_wav_path=refer_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            inp_refs=inp_refs,
            sample_steps=sample_steps,
            if_sr=if_sr,
            pause_second=pause_second,
            cut_punc=cut_punc,
            audio_format="wav",  # 非流式返回使用WAV格式
        )
        
        if not audio_data:
            return JSONResponse(
                {"code": 400, "message": "音频合成失败"},
                status_code=400,
            )
            
        # 返回音频流
        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/wav",
        )
        
    except Exception as e:
        logger.error(f"TTS合成出错: {str(e)}")
        return JSONResponse(
            {"code": 500, "message": f"服务器错误: {str(e)}"},
            status_code=500,
        )


@app.get("/")
async def tts_endpoint_get(
    refer_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_language: Optional[str] = None,
    text: str = None,
    text_language: str = None,
    cut_punc: Optional[str] = ",.，。",
    top_k: Optional[int] = 15,
    top_p: Optional[float] = 0.6,
    temperature: Optional[float] = 0.6,
    speed: Optional[float] = 1.0,
    inp_refs: List[str] = Query(default=[]),
    sample_steps: Optional[int] = 32,
    if_sr: Optional[bool] = False,
    pause_second: Optional[float] = 0.3,
):
    """TTS合成接口 (GET方法)"""
    global tts_model, default_refer_path, default_refer_text, default_refer_language
    
    # 检查模型是否已初始化
    if not tts_model:
        return JSONResponse({"code": 400, "message": "模型未初始化"}, status_code=400)
    
    try:
        # 验证必要参数
        if not text or not text_language:
            return JSONResponse(
                {"code": 400, "message": "缺少必要参数: text, text_language"},
                status_code=400,
            )
        
        # 验证参考音频
        if not refer_wav_path or not prompt_text or not prompt_language:
            if not default_refer_path or not default_refer_text or not default_refer_language:
                return JSONResponse(
                    {"code": 400, "message": "未指定参考音频且接口无预设"},
                    status_code=400,
                )
            refer_wav_path = default_refer_path
            prompt_text = default_refer_text
            prompt_language = default_refer_language
            
        # 样本步数校验
        if not sample_steps in [4, 8, 16, 32, 64, 128]:
            sample_steps = 32
            
        # 合成音频
        audio_data = tts_model.tts(
            ref_wav_path=refer_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            inp_refs=inp_refs,
            sample_steps=sample_steps,
            if_sr=if_sr,
            pause_second=pause_second,
            cut_punc=cut_punc,
            audio_format="wav",  # 非流式返回使用WAV格式
        )
        
        if not audio_data:
            return JSONResponse(
                {"code": 400, "message": "音频合成失败"},
                status_code=400,
            )
            
        # 返回音频流
        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/wav",
        )
        
    except Exception as e:
        logger.error(f"TTS合成出错: {str(e)}")
        return JSONResponse(
            {"code": 500, "message": f"服务器错误: {str(e)}"},
            status_code=500,
        )


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="GPT-SoVITS API服务")
    parser.add_argument("-s", "--sovits_path", type=str, default=config.sovits_path, help="SoVITS模型路径")
    parser.add_argument("-g", "--gpt_path", type=str, default=config.gpt_path, help="GPT模型路径")
    parser.add_argument("-d", "--device", type=str, default=config.infer_device, help="推理设备，'cuda'或'cpu'")
    parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="绑定地址")
    parser.add_argument("-p", "--port", type=int, default=9880, help="绑定端口")
    parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
    parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
    parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")
    
    args = parser.parse_args()
    
    # 设置配置
    config.sovits_path = args.sovits_path
    config.gpt_path = args.gpt_path
    config.infer_device = args.device
    
    # 设置默认参考音频
    default_refer_path = args.default_refer_path
    default_refer_text = args.default_refer_text
    default_refer_language = args.default_refer_language
    
    # 打印配置
    logger.info("GPT-SoVITS API服务配置:")
    logger.info(f"SoVITS模型路径: {config.sovits_path}")
    logger.info(f"GPT模型路径: {config.gpt_path}")
    logger.info(f"推理设备: {config.infer_device}")
    logger.info(f"半精度推理: {config.is_half}")
    
    if default_refer_path and default_refer_text and default_refer_language:
        logger.info(f"默认参考音频路径: {default_refer_path}")
        logger.info(f"默认参考音频文本: {default_refer_text}")
        logger.info(f"默认参考音频语种: {default_refer_language}")
    else:
        logger.info("未设置默认参考音频")
    
    # 启动FastAPI服务
    uvicorn.run(app, host=args.bind_addr, port=args.port) 