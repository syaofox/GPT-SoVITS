#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将旧的角色配置文件格式（*.json）转换为新的目录结构格式

旧格式: configs/roles/角色名.json
新格式:
configs/roles/
   │
   ├── 角色名1/
   │   ├── config.json           # 角色配置文件
   │   ├── 参考音频1.wav          # 参考音频文件
   │   ├── 参考音频2.wav          # 更多参考音频
   │   └── ...
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple


def migrate_roles() -> List[str]:
    """将旧式角色配置迁移到新的目录结构"""
    roles_dir = Path("configs/roles")
    ref_audio_dir = Path("configs/ref_audio")
    
    # 确保目录存在
    roles_dir.mkdir(parents=True, exist_ok=True)
    ref_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录迁移结果
    migrated_roles = []
    failed_roles = []
    
    # 查找所有角色配置文件
    role_files = list(roles_dir.glob("*.json"))
    if not role_files:
        print("未找到任何旧式角色配置文件")
        return migrated_roles
    
    print(f"找到 {len(role_files)} 个旧式角色配置文件")
    
    # 处理每个角色
    for role_file in role_files:
        role_name = role_file.stem
        print(f"\n正在迁移角色 {role_name}...")
        
        try:
            # 读取角色配置
            with open(role_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 创建角色目录
            role_dir = roles_dir / role_name
            role_dir.mkdir(exist_ok=True)
            
            # 处理每个情绪下的参考音频和辅助音频
            audio_paths_to_process = []  # (源路径, 目标路径) 的列表
            
            if "emotions" in config:
                for emotion, emotion_config in config["emotions"].items():
                    # 处理主参考音频
                    if "ref_audio" in emotion_config:
                        ref_audio = emotion_config["ref_audio"]
                        if os.path.exists(ref_audio):
                            # 复制音频到角色目录
                            filename = os.path.basename(ref_audio)
                            target_path = role_dir / filename
                            audio_paths_to_process.append((ref_audio, target_path))
                            # 更新配置中的路径为相对路径
                            emotion_config["ref_audio"] = filename
                        else:
                            # 检查旧的音频存储位置
                            old_ref_dirs = [
                                ref_audio_dir / role_name,
                                Path("configs/refsounds")
                            ]
                            found = False
                            filename = os.path.basename(ref_audio)
                            for old_dir in old_ref_dirs:
                                old_path = old_dir / filename
                                if old_path.exists():
                                    target_path = role_dir / filename
                                    audio_paths_to_process.append((old_path, target_path))
                                    emotion_config["ref_audio"] = filename
                                    found = True
                                    break
                            
                            if not found:
                                print(f"  警告: 找不到参考音频 {ref_audio}")
                    
                    # 处理辅助参考音频
                    if "aux_refs" in emotion_config:
                        aux_refs = emotion_config["aux_refs"]
                        if isinstance(aux_refs, list):
                            new_aux_refs = []
                            for aux_ref in aux_refs:
                                if os.path.exists(aux_ref):
                                    # 复制音频到角色目录
                                    filename = os.path.basename(aux_ref)
                                    target_path = role_dir / filename
                                    audio_paths_to_process.append((aux_ref, target_path))
                                    # 更新为相对路径
                                    new_aux_refs.append(filename)
                                else:
                                    # 检查旧的音频存储位置
                                    old_ref_dirs = [
                                        ref_audio_dir / role_name,
                                        Path("configs/refsounds")
                                    ]
                                    found = False
                                    filename = os.path.basename(aux_ref)
                                    for old_dir in old_ref_dirs:
                                        old_path = old_dir / filename
                                        if old_path.exists():
                                            target_path = role_dir / filename
                                            audio_paths_to_process.append((old_path, target_path))
                                            new_aux_refs.append(filename)
                                            found = True
                                            break
                                    
                                    if not found:
                                        print(f"  警告: 找不到辅助参考音频 {aux_ref}")
                            
                            emotion_config["aux_refs"] = new_aux_refs
            
            # 复制所有音频文件
            for src_path, dst_path in audio_paths_to_process:
                if not dst_path.exists():
                    print(f"  复制音频: {src_path} -> {dst_path}")
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"  音频已存在: {dst_path}")
            
            # 保存新的配置文件
            config_path = role_dir / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            print(f"  角色 {role_name} 迁移成功")
            migrated_roles.append(role_name)
            
            # 重命名旧的配置文件为.bak
            bak_file = role_file.with_suffix(".json.bak")
            if bak_file.exists():
                os.remove(bak_file)
            role_file.rename(bak_file)
            print(f"  已将原配置文件备份为 {bak_file.name}")
            
        except Exception as e:
            print(f"  角色 {role_name} 迁移失败: {str(e)}")
            failed_roles.append(role_name)
    
    # 汇总结果
    print("\n迁移完成！")
    print(f"成功迁移 {len(migrated_roles)} 个角色: {', '.join(migrated_roles)}")
    if failed_roles:
        print(f"迁移失败 {len(failed_roles)} 个角色: {', '.join(failed_roles)}")
    
    return migrated_roles


if __name__ == "__main__":
    migrate_roles() 