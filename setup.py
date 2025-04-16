#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="gpt_sovits_lib",
    version="0.1.0",
    description="GPT-SoVITS local library for direct integration in Python projects",
    author="GPT-SoVITS Community",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "numpy>=1.20.0",
        "librosa>=0.9.2",
        "soundfile>=0.10.0",
        "transformers>=4.20.0",
        "peft>=0.2.0",
        "fastapi>=0.95.0",
        "pydantic>=1.10.0",
        "uvicorn>=0.20.0",
        "pypinyin>=0.47.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 