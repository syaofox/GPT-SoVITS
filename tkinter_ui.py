#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 基于Tkinter的UI界面
实现与原Gradio UI类似的功能
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.utils import LANGUAGE_OPTIONS, g_default_role, clean_file_path, parse_line
from ui.roles import list_roles, update_default_emotions, get_emotions, get_role_config, load_and_process_role_config, delete_role_config, save_role_config
from ui.text_processing import preprocess_text
from ui.api_client import call_api, merge_audio_segments
from ui.models import init_models, get_model_lists


class TkinterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-SoVITS 文本转语音 (Tkinter版)")
        self.root.geometry("1200x800")
        
        # 获取角色列表
        self.roles = list_roles()
        
        # 确保默认角色在角色列表中，否则使用第一个角色
        self.default_role = g_default_role
        if self.default_role not in self.roles:
            self.default_role = self.roles[0]
        
        # 初始化默认角色的模型
        try:
            gpt_path, sovits_path = init_models(self.default_role)
            print(f"初始化模型成功: {gpt_path}, {sovits_path}")
        except Exception as e:
            messagebox.showerror("初始化模型失败", str(e))
        
        # 创建标签页控件
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建各个标签页
        self.create_text_file_tab()
        self.create_role_management_tab()
        self.create_config_tab()
    
    def create_text_file_tab(self):
        """创建文本文件标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="文本文件")
        
        # 处理模式选择区域
        mode_frame = ttk.LabelFrame(tab, text="选择处理模式")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.process_mode = tk.StringVar(value="全文本处理")
        modes = ["逐行处理", "全文本处理", "分段处理"]
        for mode in modes:
            ttk.Radiobutton(mode_frame, text=mode, variable=self.process_mode, value=mode).pack(side=tk.LEFT, padx=10)
        
        # 模式说明文本
        mode_info_text = """
        模式说明:
        - 逐行处理: 逐行分析角色和情绪标记，适用于对话场景，每行文本单独处理
        - 全文本处理: 使用默认角色和情绪处理整个文本，适用于大段独白
        - 分段处理: 将文本分成多个段落(不超过500字)，分别处理后合并，解决长文本问题
        """
        mode_info = ttk.Label(mode_frame, text=mode_info_text, justify=tk.LEFT)
        mode_info.pack(padx=10, pady=5, anchor=tk.W)
        
        # 主要内容区域
        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 文本输入区域
        text_frame = ttk.LabelFrame(content_frame, text="文本输入")
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_content = scrolledtext.ScrolledText(text_frame, height=15)
        self.text_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_content.insert(tk.END, "每行一句话，支持以下格式：\n(角色)文本内容\n(角色|情绪)文本内容\n直接输入文本")
        
        # 文件上传和音频预览区域
        file_frame = ttk.LabelFrame(content_frame, text="文件操作")
        file_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        ttk.Button(file_frame, text="上传文本文件", command=self.upload_text_file).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="预处理文本", command=self.preprocess_text).pack(fill=tk.X, padx=5, pady=5)
        
        self.audio_output_path = tk.StringVar()
        ttk.Label(file_frame, text="输出音频:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        ttk.Entry(file_frame, textvariable=self.audio_output_path, state="readonly").pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="播放音频", command=self.play_audio).pack(fill=tk.X, padx=5, pady=5)
        
        # 角色和情绪设置区域
        settings_frame = ttk.LabelFrame(tab, text="角色和情绪设置")
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建一个包含两列的框架
        roles_frame = ttk.Frame(settings_frame)
        roles_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 第一列：默认角色和情绪
        default_frame = ttk.LabelFrame(roles_frame, text="默认角色和情绪")
        default_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(default_frame, text="默认角色(必选):").pack(anchor=tk.W, padx=5, pady=2)
        self.default_role_var = tk.StringVar(value=self.default_role)
        self.default_role_combo = ttk.Combobox(default_frame, textvariable=self.default_role_var, values=self.roles)
        self.default_role_combo.pack(fill=tk.X, padx=5, pady=2)
        self.default_role_combo.bind("<<ComboboxSelected>>", self.update_default_emotions)
        
        ttk.Label(default_frame, text="默认情绪:").pack(anchor=tk.W, padx=5, pady=2)
        emotions = get_emotions(self.default_role)
        self.default_emotion_var = tk.StringVar(value=emotions[0] if emotions else "")
        self.default_emotion_combo = ttk.Combobox(default_frame, textvariable=self.default_emotion_var, values=emotions)
        self.default_emotion_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # 第二列：强制角色和情绪
        force_frame = ttk.LabelFrame(roles_frame, text="强制角色和情绪")
        force_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(force_frame, text="强制使用角色(可选):").pack(anchor=tk.W, padx=5, pady=2)
        self.force_role_var = tk.StringVar(value="无")
        role_choices = ["无"] + self.roles
        self.force_role_combo = ttk.Combobox(force_frame, textvariable=self.force_role_var, values=role_choices)
        self.force_role_combo.pack(fill=tk.X, padx=5, pady=2)
        self.force_role_combo.bind("<<ComboboxSelected>>", self.update_force_emotions)
        
        ttk.Label(force_frame, text="强制使用情绪(可选):").pack(anchor=tk.W, padx=5, pady=2)
        self.force_emotion_var = tk.StringVar(value="无")
        self.force_emotion_combo = ttk.Combobox(force_frame, textvariable=self.force_emotion_var, values=["无"] + emotions)
        self.force_emotion_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # 第三列：语言和切分符号
        lang_frame = ttk.LabelFrame(roles_frame, text="语言和切分设置")
        lang_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(lang_frame, text="文本语言:").pack(anchor=tk.W, padx=5, pady=2)
        language_options = [lang[0] for lang in LANGUAGE_OPTIONS]
        self.text_lang_var = tk.StringVar(value="中文")
        self.text_lang_combo = ttk.Combobox(lang_frame, textvariable=self.text_lang_var, values=language_options)
        self.text_lang_combo.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(lang_frame, text="切分符号(可选):").pack(anchor=tk.W, padx=5, pady=2)
        self.cut_punc_var = tk.StringVar(value="。！？：.!?:")
        ttk.Entry(lang_frame, textvariable=self.cut_punc_var).pack(fill=tk.X, padx=5, pady=2)
        
        # 按钮区域
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="处理文本", command=self.process_text).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="刷新角色列表", command=self.refresh_roles).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 使用说明区域
        help_frame = ttk.LabelFrame(tab, text="使用说明")
        help_frame.pack(fill=tk.X, padx=10, pady=5)
        
        help_text = """
        使用说明:
        1. 文本输入：可以选择上传文本文件或直接在文本框中输入
        2. 处理模式：选择合适的处理模式
           - 逐行处理：每行文本单独处理，支持行内角色和情绪标记
           - 全文本处理：使用默认角色和情绪处理整个文本
           - 分段处理：将文本分段处理后合并
        3. 角色设置：
           - 默认角色：当文本没有指定角色时使用的角色
           - 强制角色：忽略文本中的角色标记，全部使用指定角色
        4. 预处理文本：将双引号内的文本作为对白（使用默认情绪），其他文本作为叙述
        """
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(padx=10, pady=5, anchor=tk.W)
    
    def create_role_management_tab(self):
        """创建角色管理标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="角色管理")
        
        # 角色列表区域
        roles_list_frame = ttk.LabelFrame(tab, text="角色列表")
        roles_list_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建角色列表框架
        roles_frame = ttk.Frame(roles_list_frame)
        roles_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 角色列表显示
        ttk.Label(roles_frame, text="已配置角色:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.roles_listbox = tk.Listbox(roles_frame, height=5, width=30)
        self.roles_listbox.grid(row=1, column=0, rowspan=4, padx=5, pady=2, sticky=tk.NSEW)
        
        # 添加滚动条
        roles_scrollbar = ttk.Scrollbar(roles_frame, orient=tk.VERTICAL, command=self.roles_listbox.yview)
        roles_scrollbar.grid(row=1, column=1, rowspan=4, sticky=tk.NS)
        self.roles_listbox.configure(yscrollcommand=roles_scrollbar.set)
        
        # 更新角色列表
        for role in self.roles:
            self.roles_listbox.insert(tk.END, role)
        
        # 角色操作按钮
        buttons_frame = ttk.Frame(roles_frame)
        buttons_frame.grid(row=1, column=2, padx=5, pady=2, sticky=tk.NSEW)
        
        ttk.Button(buttons_frame, text="加载角色", command=self.load_role).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="删除角色", command=self.delete_role).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="刷新角色列表", command=self.refresh_role_list).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="保存当前角色", command=self.save_current_role).pack(fill=tk.X, padx=5, pady=2)
        
        # 绑定选择事件
        self.roles_listbox.bind('<<ListboxSelect>>', self.on_role_select)
        
        # 获取模型列表
        gpt_models, sovits_models = get_model_lists()
        
        # 模型选择区域
        model_frame = ttk.LabelFrame(tab, text="模型选择")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="GPT模型:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.gpt_model_var = tk.StringVar(value=gpt_models[0] if gpt_models else "")
        self.gpt_model_combo = ttk.Combobox(model_frame, textvariable=self.gpt_model_var, values=gpt_models, width=50)
        self.gpt_model_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(model_frame, text="SoVITS模型:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.sovits_model_var = tk.StringVar(value=sovits_models[0] if sovits_models else "")
        self.sovits_model_combo = ttk.Combobox(model_frame, textvariable=self.sovits_model_var, values=sovits_models, width=50)
        self.sovits_model_combo.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Button(model_frame, text="刷新模型列表", command=self.refresh_models).grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        
        # 参考音频区域
        ref_frame = ttk.LabelFrame(tab, text="参考音频")
        ref_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ref_frame, text="参考音频文件路径:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ref_audio_var = tk.StringVar()
        ttk.Entry(ref_frame, textvariable=self.ref_audio_var, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(ref_frame, text="浏览...", command=self.browse_ref_audio).grid(row=0, column=2, padx=5, pady=2)
        
        # 添加辅助参考音频
        ttk.Label(ref_frame, text="辅助参考音频:").grid(row=1, column=0, sticky=tk.W+tk.N, padx=5, pady=2)
        self.aux_refs_text = scrolledtext.ScrolledText(ref_frame, height=3, width=50)
        self.aux_refs_text.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(ref_frame, text="添加", command=self.add_aux_ref).grid(row=1, column=2, padx=5, pady=2)
        
        self.ref_free_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ref_frame, text="开启无参考文本模式", variable=self.ref_free_var).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(ref_frame, text="参考音频的文本:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.prompt_text = scrolledtext.ScrolledText(ref_frame, height=4, width=50)
        self.prompt_text.grid(row=3, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(ref_frame, text="参考音频的语种:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        language_options = [lang[0] for lang in LANGUAGE_OPTIONS]
        self.prompt_lang_var = tk.StringVar(value="中文")
        ttk.Combobox(ref_frame, textvariable=self.prompt_lang_var, values=language_options).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(ref_frame, text="目标合成的语种:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.text_lang_role_var = tk.StringVar(value="中文")
        ttk.Combobox(ref_frame, textvariable=self.text_lang_role_var, values=language_options).grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # 情绪列表区域
        emotion_frame = ttk.LabelFrame(tab, text="情绪设置")
        emotion_frame.pack(fill=tk.X, padx=10, pady=5)
        
        emotions_grid = ttk.Frame(emotion_frame)
        emotions_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(emotions_grid, text="情绪名称:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.emotion_name_var = tk.StringVar()
        ttk.Entry(emotions_grid, textvariable=self.emotion_name_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(emotions_grid, text="说话风格(s1):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.style_s1_var = tk.DoubleVar(value=0.3)
        ttk.Scale(emotions_grid, from_=0.0, to=1.0, variable=self.style_s1_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(emotions_grid, textvariable=self.style_s1_var).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(emotions_grid, text="情感程度(s2):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.style_s2_var = tk.DoubleVar(value=0.3)
        ttk.Scale(emotions_grid, from_=0.0, to=1.0, variable=self.style_s2_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(emotions_grid, textvariable=self.style_s2_var).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(emotions_grid, text="音高(s3):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.style_s3_var = tk.DoubleVar(value=0.3)
        ttk.Scale(emotions_grid, from_=0.0, to=1.0, variable=self.style_s3_var, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(emotions_grid, textvariable=self.style_s3_var).grid(row=3, column=2, padx=5, pady=2)
        
        ttk.Button(emotions_grid, text="添加情绪", command=self.add_emotion).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(emotions_grid, text="删除情绪", command=self.delete_emotion).grid(row=4, column=1, padx=5, pady=5)
        
        # 情绪列表
        ttk.Label(emotions_grid, text="当前角色情绪列表:").grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        self.emotions_listbox = tk.Listbox(emotions_grid, height=6, width=30)
        self.emotions_listbox.grid(row=1, column=3, rowspan=4, padx=5, pady=2, sticky=tk.NSEW)
        
        # 添加滚动条
        emotions_scrollbar = ttk.Scrollbar(emotions_grid, orient=tk.VERTICAL, command=self.emotions_listbox.yview)
        emotions_scrollbar.grid(row=1, column=4, rowspan=4, sticky=tk.NS)
        self.emotions_listbox.configure(yscrollcommand=emotions_scrollbar.set)
        
        # 绑定情绪选择事件
        self.emotions_listbox.bind('<<ListboxSelect>>', self.on_emotion_select)
        
        # 合成区域
        synth_frame = ttk.LabelFrame(tab, text="合成设置")
        synth_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧放置文本区域
        ttk.Label(synth_frame, text="需要合成的文本:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.target_text = scrolledtext.ScrolledText(synth_frame, height=10, width=50)
        self.target_text.grid(row=1, column=0, rowspan=10, sticky=tk.NSEW, padx=5, pady=2)
        
        # 右侧放置参数区域
        param_frame = ttk.Frame(synth_frame)
        param_frame.grid(row=1, column=1, sticky=tk.NSEW, padx=5, pady=2)
        
        ttk.Label(param_frame, text="断句符号:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.cut_punc_role_var = tk.StringVar(value="。！？：.!?:")
        ttk.Entry(param_frame, textvariable=self.cut_punc_role_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(param_frame, text="语速:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(param_frame, from_=0.6, to=1.65, variable=self.speed_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(param_frame, textvariable=self.speed_var).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(param_frame, text="句间停顿秒数:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pause_var = tk.DoubleVar(value=0.3)
        ttk.Scale(param_frame, from_=0.1, to=0.5, variable=self.pause_var, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Label(param_frame, textvariable=self.pause_var).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(param_frame, text="采样步数:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.sample_steps_var = tk.IntVar(value=32)
        ttk.Combobox(param_frame, textvariable=self.sample_steps_var, values=[4, 8, 16, 32, 64, 128]).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        self.if_sr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="启用超采样提高语音质量", variable=self.if_sr_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # 合成按钮和输出区域
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="创建新角色", command=self.create_role).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(btn_frame, text="测试合成", command=self.test_synthesis).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 输出音频显示
        ttk.Label(btn_frame, text="输出音频:").pack(side=tk.LEFT, padx=(20, 5), pady=5)
        self.role_audio_output_var = tk.StringVar()
        ttk.Entry(btn_frame, textvariable=self.role_audio_output_var, width=50, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(btn_frame, text="播放", command=self.play_role_audio).pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_config_tab(self):
        """创建配置标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="配置替换音")
        
        # 词语替换配置区域
        config_frame = ttk.LabelFrame(tab, text="词语替换配置")
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.word_replace_text = scrolledtext.ScrolledText(config_frame, height=20)
        self.word_replace_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 尝试加载词语替换配置
        try:
            from ui.utils import load_word_replace_config
            config_text = load_word_replace_config()
            self.word_replace_text.insert(tk.END, config_text)
        except Exception as e:
            self.word_replace_text.insert(tk.END, f"加载配置失败: {str(e)}\n\n# 格式：\n# 替换前 替换后\n# 例如：\ntest 测试")
        
        # 按钮区域
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="保存词语替换配置", command=self.save_word_replace).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="刷新", command=self.refresh_word_replace).pack(side=tk.LEFT, padx=5, pady=5)

    # 文本文件标签页函数
    def upload_text_file(self):
        """上传文本文件"""
        file_path = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_content.delete(1.0, tk.END)
                self.text_content.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("文件读取错误", str(e))
    
    def preprocess_text(self):
        """预处理文本"""
        content = self.text_content.get(1.0, tk.END)
        default_role = self.default_role_var.get()
        default_emotion = self.default_emotion_var.get()
        
        try:
            processed_text = preprocess_text(content, default_role, default_emotion)
            self.text_content.delete(1.0, tk.END)
            self.text_content.insert(tk.END, processed_text)
        except Exception as e:
            messagebox.showerror("预处理失败", str(e))
    
    def update_default_emotions(self, event=None):
        """更新默认情绪下拉列表"""
        selected_role = self.default_role_var.get()
        emotions = get_emotions(selected_role)
        
        self.default_emotion_combo["values"] = emotions
        if emotions:
            self.default_emotion_var.set(emotions[0])
        else:
            self.default_emotion_var.set("")
    
    def update_force_emotions(self, event=None):
        """更新强制情绪下拉列表"""
        selected_role = self.force_role_var.get()
        if selected_role == "无":
            self.force_emotion_combo["values"] = ["无"]
            self.force_emotion_var.set("无")
        else:
            emotions = get_emotions(selected_role)
            self.force_emotion_combo["values"] = ["无"] + emotions
            self.force_emotion_var.set("无")
    
    def refresh_roles(self):
        """刷新角色列表"""
        self.roles = list_roles()
        
        # 更新默认角色下拉列表
        self.default_role_combo["values"] = self.roles
        if not self.default_role_var.get() in self.roles and self.roles:
            self.default_role_var.set(self.roles[0])
        
        # 更新强制角色下拉列表
        self.force_role_combo["values"] = ["无"] + self.roles
        
        # 更新对应的情绪列表
        self.update_default_emotions()
        self.update_force_emotions()
    
    def process_text(self):
        """处理文本生成音频"""
        text_content = self.text_content.get(1.0, tk.END)
        force_role = "" if self.force_role_var.get() == "无" else self.force_role_var.get()
        default_role = self.default_role_var.get()
        force_emotion = "" if self.force_emotion_var.get() == "无" else self.force_emotion_var.get()
        default_emotion = self.default_emotion_var.get()
        text_lang = self.text_lang_var.get()
        cut_punc = self.cut_punc_var.get()
        process_mode = self.process_mode.get()
        
        if not text_content.strip():
            messagebox.showerror("错误", "请输入文本内容")
            return
        
        if not default_role:
            messagebox.showerror("错误", "请选择默认角色")
            return
        
        # 显示处理中对话框
        progress_window = tk.Toplevel(self.root)
        progress_window.title("处理中")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="正在处理文本生成音频，请稍候...")
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        progress_bar.start()
        
        # 在新线程中处理音频以避免UI冻结
        def process_thread():
            try:
                from ui.tabs.text_file import process_text_content
                
                output_path = process_text_content(
                    text_content,
                    force_role,
                    default_role,
                    force_emotion,
                    default_emotion,
                    text_lang,
                    cut_punc,
                    process_mode,
                    "output"
                )
                
                self.audio_output_path.set(output_path)
                
                # 更新UI（必须在主线程中完成）
                self.root.after(0, lambda: messagebox.showinfo("处理完成", f"音频已生成：{output_path}"))
                self.root.after(0, progress_window.destroy)
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("处理失败", error_msg))
                self.root.after(0, progress_window.destroy)
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def play_audio(self):
        """播放生成的音频"""
        audio_path = self.audio_output_path.get()
        if not audio_path or not os.path.exists(audio_path):
            messagebox.showerror("错误", "没有可播放的音频文件")
            return
        
        import platform
        import subprocess
        
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(audio_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", audio_path])
            else:  # Linux
                subprocess.call(["xdg-open", audio_path])
        except Exception as e:
            messagebox.showerror("播放失败", str(e))
    
    # 角色管理标签页函数
    def refresh_models(self):
        """刷新模型列表"""
        try:
            gpt_models, sovits_models = get_model_lists()
            
            self.gpt_model_combo["values"] = gpt_models
            if gpt_models and not self.gpt_model_var.get() in gpt_models:
                self.gpt_model_var.set(gpt_models[0])
                
            self.sovits_model_combo["values"] = sovits_models
            if sovits_models and not self.sovits_model_var.get() in sovits_models:
                self.sovits_model_var.set(sovits_models[0])
                
            messagebox.showinfo("刷新成功", "模型列表已更新")
        except Exception as e:
            messagebox.showerror("刷新失败", str(e))
    
    def browse_ref_audio(self):
        """浏览选择参考音频文件"""
        file_path = filedialog.askopenfilename(
            title="选择参考音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.ogg"), ("所有文件", "*.*")]
        )
        if file_path:
            self.ref_audio_var.set(file_path)
            # 尝试从文件名提取文本作为参考文本
            try:
                file_name = os.path.basename(file_path)
                name_without_ext = os.path.splitext(file_name)[0]
                self.prompt_text.delete(1.0, tk.END)
                self.prompt_text.insert(tk.END, name_without_ext)
            except:
                pass
    
    def refresh_role_list(self):
        """刷新角色列表"""
        self.roles = list_roles()
        self.roles_listbox.delete(0, tk.END)
        for role in self.roles:
            self.roles_listbox.insert(tk.END, role)
        messagebox.showinfo("刷新成功", "角色列表已刷新")
    
    def on_role_select(self, event=None):
        """处理角色选择事件"""
        if not self.roles_listbox.curselection():
            return
        
        selected_idx = self.roles_listbox.curselection()[0]
        selected_role = self.roles_listbox.get(selected_idx)
        
        try:
            # 加载角色配置
            role_config = load_and_process_role_config(selected_role)
            
            # 更新模型选择
            if "gpt_path" in role_config and role_config["gpt_path"]:
                self.gpt_model_var.set(role_config["gpt_path"])
            if "sovits_path" in role_config and role_config["sovits_path"]:
                self.sovits_model_var.set(role_config["sovits_path"])
                
            # 更新参考音频（可能需要从情绪中获取）
            if "reference_audio" in role_config and role_config["reference_audio"]:
                self.ref_audio_var.set(role_config["reference_audio"])
            
            # 更新引用文本
            if "reference_text" in role_config and role_config["reference_text"]:
                self.prompt_text.delete(1.0, tk.END)
                self.prompt_text.insert(tk.END, role_config["reference_text"])
            
            # 更新语言设置
            if "prompt_lang" in role_config and role_config["prompt_lang"]:
                self.prompt_lang_var.set(role_config["prompt_lang"])
            elif "prompt_language" in role_config and role_config["prompt_language"]:
                self.prompt_lang_var.set(role_config["prompt_language"])
                
            if "text_lang" in role_config and role_config["text_lang"]:
                self.text_lang_role_var.set(role_config["text_lang"])
            elif "text_language" in role_config and role_config["text_language"]:
                self.text_lang_role_var.set(role_config["text_language"])
            
            # 更新无参考文本模式设置
            if "ref_free" in role_config:
                self.ref_free_var.set(role_config["ref_free"])
            
            # 更新其他合成参数
            if "speed" in role_config:
                self.speed_var.set(role_config["speed"])
            if "pause_second" in role_config:
                self.pause_var.set(role_config["pause_second"])
            if "sample_steps" in role_config:
                self.sample_steps_var.set(role_config["sample_steps"])
            if "if_sr" in role_config:
                self.if_sr_var.set(role_config["if_sr"])
            
            # 清空情绪列表
            self.emotions_listbox.delete(0, tk.END)
            
            # 处理情绪配置
            if "emotions" in role_config and role_config["emotions"]:
                # 检查情绪格式并添加到列表
                for emotion_name, emotion_data in role_config["emotions"].items():
                    self.emotions_listbox.insert(tk.END, emotion_name)
                
                # 默认选择第一个情绪并显示其参数
                if self.emotions_listbox.size() > 0:
                    self.emotions_listbox.selection_set(0)
                    self.on_emotion_select()
            
            messagebox.showinfo("加载成功", f"已加载角色配置: {selected_role}")
        except Exception as e:
            messagebox.showerror("加载失败", f"加载角色配置失败: {str(e)}")
    
    def on_emotion_select(self, event=None):
        """处理情绪选择事件"""
        if not self.roles_listbox.curselection() or not self.emotions_listbox.curselection():
            return
        
        selected_role_idx = self.roles_listbox.curselection()[0]
        selected_role = self.roles_listbox.get(selected_role_idx)
        
        selected_emotion_idx = self.emotions_listbox.curselection()[0]
        selected_emotion = self.emotions_listbox.get(selected_emotion_idx)
        
        try:
            # 加载角色配置
            role_config = load_and_process_role_config(selected_role)
            
            # 更新情绪名称
            self.emotion_name_var.set(selected_emotion)
            
            # 更新情绪参数
            if "emotions" in role_config and selected_emotion in role_config["emotions"]:
                emotion_config = role_config["emotions"][selected_emotion]
                
                # 处理不同格式的情绪配置
                if isinstance(emotion_config, dict):
                    # 检查是否有s1、s2、s3格式的参数
                    if "s1" in emotion_config:
                        self.style_s1_var.set(emotion_config["s1"])
                    if "s2" in emotion_config:
                        self.style_s2_var.set(emotion_config["s2"])
                    if "s3" in emotion_config:
                        self.style_s3_var.set(emotion_config["s3"])
                    
                    # 检查是否有参考音频格式的参数
                    if "ref_audio" in emotion_config:
                        self.ref_audio_var.set(emotion_config["ref_audio"])
                    if "prompt_text" in emotion_config:
                        self.prompt_text.delete(1.0, tk.END)
                        self.prompt_text.insert(tk.END, emotion_config["prompt_text"])
                    
                    # 检查是否有辅助参考音频
                    if "aux_refs" in emotion_config and emotion_config["aux_refs"]:
                        self.aux_refs_text.delete(1.0, tk.END)
                        for aux_ref in emotion_config["aux_refs"]:
                            self.aux_refs_text.insert(tk.END, aux_ref + "\n")
        except Exception as e:
            messagebox.showerror("加载失败", f"加载情绪参数失败: {str(e)}")
    
    def load_role(self):
        """加载选中的角色"""
        if not self.roles_listbox.curselection():
            messagebox.showwarning("未选择角色", "请先选择要加载的角色")
            return
        
        selected_idx = self.roles_listbox.curselection()[0]
        selected_role = self.roles_listbox.get(selected_idx)
        
        try:
            # 先调用角色选择事件处理函数以填充界面
            self.on_role_select()
            
            # 初始化模型
            gpt_path, sovits_path = init_models(selected_role)
            messagebox.showinfo("加载成功", f"角色 {selected_role} 已加载，模型路径:\nGPT: {gpt_path}\nSoVITS: {sovits_path}")
        except Exception as e:
            messagebox.showerror("加载失败", f"加载角色模型失败: {str(e)}")
    
    def delete_role(self):
        """删除选中的角色"""
        if not self.roles_listbox.curselection():
            messagebox.showwarning("未选择角色", "请先选择要删除的角色")
            return
        
        selected_idx = self.roles_listbox.curselection()[0]
        selected_role = self.roles_listbox.get(selected_idx)
        
        # 确认删除
        if not messagebox.askyesno("确认删除", f"确定要删除角色 {selected_role} 吗？此操作不可恢复。"):
            return
        
        try:
            # 删除角色配置
            delete_role_config(selected_role)
            
            # 刷新角色列表
            self.refresh_role_list()
            messagebox.showinfo("删除成功", f"角色 {selected_role} 已删除")
        except Exception as e:
            messagebox.showerror("删除失败", f"删除角色失败: {str(e)}")
    
    def save_current_role(self):
        """保存当前角色配置"""
        # 获取角色名
        role_name = simpledialog.askstring("角色名称", "请输入角色名称:", initialvalue="")
        if not role_name:
            return
        
        # 收集角色配置信息
        try:
            # 首先检查是否已有角色配置，如果有则尝试继承原有配置
            try:
                config = load_and_process_role_config(role_name)
            except:
                # 如果没有现有配置，创建新配置
                config = {
                    "version": "v3",
                    "emotions": {}
                }
            
            # 更新基本配置
            config["gpt_path"] = self.gpt_model_var.get()
            config["sovits_path"] = self.sovits_model_var.get()
            config["reference_audio"] = self.ref_audio_var.get()
            config["reference_text"] = self.prompt_text.get(1.0, tk.END).strip()
            config["prompt_lang"] = self.prompt_lang_var.get()
            config["text_lang"] = self.text_lang_role_var.get()
            config["ref_free"] = self.ref_free_var.get()
            
            # 更新合成参数
            config["speed"] = self.speed_var.get()
            config["pause_second"] = self.pause_var.get()
            config["sample_steps"] = self.sample_steps_var.get()
            config["if_sr"] = self.if_sr_var.get()
            
            # 保存当前界面上的情绪
            current_emotion = self.emotion_name_var.get()
            if current_emotion:
                # 尝试保留现有情绪配置
                if current_emotion not in config["emotions"]:
                    config["emotions"][current_emotion] = {}
                
                # 更新情绪参数
                current_emotion_config = config["emotions"][current_emotion]
                # 如果是字典就更新，否则创建新字典
                if not isinstance(current_emotion_config, dict):
                    current_emotion_config = {}
                
                # 保存StyleS值
                current_emotion_config["s1"] = self.style_s1_var.get()
                current_emotion_config["s2"] = self.style_s2_var.get() 
                current_emotion_config["s3"] = self.style_s3_var.get()
                
                # 保存参考音频参数
                current_emotion_config["ref_audio"] = self.ref_audio_var.get()
                current_emotion_config["prompt_text"] = self.prompt_text.get(1.0, tk.END).strip()
                
                # 保存辅助参考音频
                aux_refs_text = self.aux_refs_text.get(1.0, tk.END).strip()
                if aux_refs_text:
                    current_emotion_config["aux_refs"] = [line.strip() for line in aux_refs_text.split("\n") if line.strip()]
                
                # 更新到配置
                config["emotions"][current_emotion] = current_emotion_config
                
            # 保存配置
            save_role_config(role_name, config)
            
            # 刷新角色列表
            self.refresh_role_list()
            messagebox.showinfo("保存成功", f"角色 {role_name} 配置已保存")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存角色配置失败: {str(e)}")
    
    def add_emotion(self):
        """添加情绪到当前角色"""
        emotion_name = self.emotion_name_var.get()
        if not emotion_name:
            messagebox.showwarning("未指定情绪", "请输入情绪名称")
            return
        
        # 检查是否已存在
        for i in range(self.emotions_listbox.size()):
            if self.emotions_listbox.get(i) == emotion_name:
                messagebox.showwarning("情绪已存在", f"情绪 {emotion_name} 已存在")
                return
        
        # 添加到列表
        self.emotions_listbox.insert(tk.END, emotion_name)
        messagebox.showinfo("添加成功", f"情绪 {emotion_name} 已添加")
    
    def delete_emotion(self):
        """删除情绪"""
        if not self.emotions_listbox.curselection():
            messagebox.showwarning("未选择情绪", "请先选择要删除的情绪")
            return
        
        selected_idx = self.emotions_listbox.curselection()[0]
        selected_emotion = self.emotions_listbox.get(selected_idx)
        
        # 确认删除
        if not messagebox.askyesno("确认删除", f"确定要删除情绪 {selected_emotion} 吗？"):
            return
        
        # 删除选中的情绪
        self.emotions_listbox.delete(selected_idx)
        messagebox.showinfo("删除成功", f"情绪 {selected_emotion} 已删除")
    
    def create_role(self):
        """创建新角色"""
        # 获取角色名
        role_name = simpledialog.askstring("角色名称", "请输入新角色名称:", initialvalue="")
        if not role_name:
            return
        
        # 检查参考音频
        ref_audio = self.ref_audio_var.get()
        if not ref_audio or not os.path.exists(ref_audio):
            messagebox.showwarning("参考音频缺失", "请指定有效的参考音频文件")
            return
            
        # 检查模型
        gpt_model = self.gpt_model_var.get()
        sovits_model = self.sovits_model_var.get()
        if not gpt_model or not sovits_model:
            messagebox.showwarning("模型缺失", "请选择GPT和SoVITS模型")
            return
        
        try:
            # 基本配置
            config = {
                "version": "v3",
                "gpt_path": gpt_model,
                "sovits_path": sovits_model,
                "reference_audio": ref_audio,
                "reference_text": self.prompt_text.get(1.0, tk.END).strip(),
                "prompt_lang": self.prompt_lang_var.get(),
                "text_lang": self.text_lang_role_var.get(),
                "ref_free": self.ref_free_var.get(),
                "speed": self.speed_var.get(),
                "pause_second": self.pause_var.get(),
                "sample_steps": self.sample_steps_var.get(),
                "if_sr": self.if_sr_var.get(),
                "emotions": {}
            }
            
            # 获取当前情绪参数
            current_emotion = self.emotion_name_var.get() or "默认"
            config["emotions"][current_emotion] = {
                "s1": self.style_s1_var.get(),
                "s2": self.style_s2_var.get(),
                "s3": self.style_s3_var.get(),
                "ref_audio": ref_audio,
                "prompt_text": self.prompt_text.get(1.0, tk.END).strip()
            }
            
            # 保存配置
            save_role_config(role_name, config)
            
            # 刷新角色列表并选择新创建的角色
            self.refresh_role_list()
            # 查找并选择新角色
            for i in range(self.roles_listbox.size()):
                if self.roles_listbox.get(i) == role_name:
                    self.roles_listbox.selection_set(i)
                    self.on_role_select()
                    break
                    
            messagebox.showinfo("创建成功", f"角色 {role_name} 已创建")
        except Exception as e:
            messagebox.showerror("创建失败", f"创建角色失败: {str(e)}")
    
    def test_synthesis(self):
        """测试合成音频"""
        # 检查文本
        text = self.target_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("文本为空", "请输入需要合成的文本")
            return
        
        # 检查角色
        if not self.roles_listbox.curselection():
            messagebox.showwarning("未选择角色", "请先选择要使用的角色")
            return
        
        selected_idx = self.roles_listbox.curselection()[0]
        selected_role = self.roles_listbox.get(selected_idx)
        
        # 检查情绪
        if not self.emotions_listbox.curselection():
            messagebox.showwarning("未选择情绪", "请先选择要使用的情绪")
            return
        
        emotion_idx = self.emotions_listbox.curselection()[0]
        emotion = self.emotions_listbox.get(emotion_idx)
        
        # 显示处理中对话框
        progress_window = tk.Toplevel(self.root)
        progress_window.title("处理中")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="正在合成语音，请稍候...")
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        progress_bar.start()
        
        # 在新线程中处理音频以避免UI冻结
        def synthesis_thread():
            try:
                from ui.roles import test_role_synthesis
                
                output_path = test_role_synthesis(
                    selected_role,
                    emotion,
                    text,
                    self.cut_punc_role_var.get(),
                    self.speed_var.get(),
                    self.pause_var.get(),
                    sample_steps=self.sample_steps_var.get(),
                    if_sr=self.if_sr_var.get()
                )
                
                # 更新输出路径
                self.role_audio_output_var.set(output_path)
                
                # 更新UI（必须在主线程中完成）
                self.root.after(0, lambda: messagebox.showinfo("合成完成", f"音频已生成：{output_path}"))
                self.root.after(0, progress_window.destroy)
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("合成失败", error_msg))
                self.root.after(0, progress_window.destroy)
        
        threading.Thread(target=synthesis_thread, daemon=True).start()
    
    def play_role_audio(self):
        """播放角色管理页面生成的音频"""
        audio_path = self.role_audio_output_var.get()
        if not audio_path or not os.path.exists(audio_path):
            messagebox.showerror("错误", "没有可播放的音频文件")
            return
        
        import platform
        import subprocess
        
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(audio_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", audio_path])
            else:  # Linux
                subprocess.call(["xdg-open", audio_path])
        except Exception as e:
            messagebox.showerror("播放失败", str(e))
    
    # 配置标签页函数
    def save_word_replace(self):
        """保存词语替换配置"""
        config_text = self.word_replace_text.get(1.0, tk.END)
        try:
            from ui.utils import save_word_replace_config
            result = save_word_replace_config(config_text)
            messagebox.showinfo("保存结果", result)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
    
    def refresh_word_replace(self):
        """刷新词语替换配置"""
        try:
            from ui.utils import load_word_replace_config
            config_text = load_word_replace_config()
            self.word_replace_text.delete(1.0, tk.END)
            self.word_replace_text.insert(tk.END, config_text)
            messagebox.showinfo("刷新成功", "词语替换配置已刷新")
        except Exception as e:
            messagebox.showerror("刷新失败", str(e))

    def add_aux_ref(self):
        """添加辅助参考音频"""
        file_path = filedialog.askopenfilename(
            title="选择辅助参考音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.ogg"), ("所有文件", "*.*")]
        )
        if file_path:
            current_text = self.aux_refs_text.get(1.0, tk.END).strip()
            if current_text:
                self.aux_refs_text.insert(tk.END, f"\n{file_path}")
            else:
                self.aux_refs_text.insert(tk.END, file_path)
                

if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("configs/roles", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    root = tk.Tk()
    app = TkinterApp(root)
    root.mainloop() 