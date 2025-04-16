"""
角色配置控制器
负责协调角色配置模型和视图
"""
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Qt
import os

class RoleController:
    """角色配置控制器类"""
    
    def __init__(self, model, view):
        """初始化控制器"""
        self.model = model
        self.view = view
        self.current_role = None
        self.current_emotion = None
        
        # 连接视图的信号
        self.connect_signals()
        
        # 加载角色列表
        self.load_roles()
    
    def connect_signals(self):
        """连接视图的信号"""
        # 角色操作信号
        self.view.role_selected.connect(self.on_role_selected)
        self.view.role_add.connect(self.on_role_add)
        self.view.role_delete.connect(self.on_role_delete)
        self.view.role_import.connect(self.on_role_import)
        self.view.role_export.connect(self.on_role_export)
        
        # 音色操作信号
        self.view.emotion_selected.connect(self.on_emotion_selected)
        self.view.emotion_add.connect(self.on_emotion_add)
        self.view.emotion_delete.connect(self.on_emotion_delete)
        self.view.aux_ref_add.connect(self.on_aux_ref_add)
        self.view.aux_ref_delete.connect(self.on_aux_ref_delete)
        
        # 配置保存信号
        self.view.config_save.connect(self.on_config_save)
    
    def load_roles(self):
        """加载角色列表"""
        roles = self.model.get_role_list()
        self.view.update_role_list(roles)
    
    def on_role_selected(self, role_name):
        """角色选择事件处理"""
        self.current_role = role_name
        config = self.model.get_role_config(role_name)
        if config:
            self.view.load_config_to_ui(config)
            self.view.set_config_widgets_enabled(True)
        else:
            self.view.show_message("错误", f"加载角色'{role_name}'配置失败!", QMessageBox.Warning)
            self.view.set_config_widgets_enabled(False)
    
    def on_role_add(self, role_name):
        """添加角色事件处理"""
        if self.model.create_role(role_name):
            self.load_roles()
            
            # 选中新角色
            items = self.view.role_list.findItems(role_name, Qt.MatchExactly)
            if items:
                self.view.role_list.setCurrentItem(items[0])
            
            self.view.show_message("成功", f"角色'{role_name}'创建成功!")
        else:
            self.view.show_message("错误", f"角色'{role_name}'已存在!", QMessageBox.Warning)
    
    def on_role_delete(self, role_name):
        """删除角色事件处理"""
        if self.model.delete_role(role_name):
            self.load_roles()
            self.view.show_message("成功", f"角色'{role_name}'已移到回收站!")
        else:
            self.view.show_message("错误", f"删除角色'{role_name}'失败!", QMessageBox.Warning)
    
    def on_role_import(self, file_path):
        """导入角色事件处理"""
        result = self.model.import_role(file_path)
        if result:
            self.load_roles()
            
            # 选中导入的角色
            items = self.view.role_list.findItems(result["role_name"], Qt.MatchExactly)
            if items:
                self.view.role_list.setCurrentItem(items[0])
            
            if result["exists"]:
                self.view.show_message("成功", f"角色'{result['role_name']}'已覆盖导入!")
            else:
                self.view.show_message("成功", f"角色'{result['role_name']}'导入成功!")
        else:
            self.view.show_message("错误", "导入角色失败!", QMessageBox.Warning)
    
    def on_role_export(self, role_name, file_path):
        """导出角色事件处理"""
        if self.model.export_role(role_name, file_path):
            self.view.show_message("成功", f"角色'{role_name}'导出成功!")
        else:
            self.view.show_message("错误", f"导出角色'{role_name}'失败!", QMessageBox.Warning)
    
    def on_emotion_selected(self, emotion_name):
        """音色选择事件处理"""
        if not self.current_role:
            return
        
        self.current_emotion = emotion_name
        
        # 获取音色配置
        emotions = self.model.current_config.get("emotions", {})
        emotion_config = emotions.get(emotion_name, {})
        
        # 更新界面
        self.view.load_emotion_to_ui(emotion_name, emotion_config)
    
    def on_emotion_add(self, emotion_name):
        """添加音色事件处理"""
        if not self.current_role:
            self.view.show_message("错误", "请先选择一个角色!", QMessageBox.Warning)
            return
        
        # 获取当前音色配置
        emotion_config = self.view.get_current_emotion_config()
        
        # 处理参考音频复制
        ref_audio = emotion_config.get("ref_audio", "")
        if ref_audio and os.path.isfile(ref_audio) and not ref_audio.startswith(str(self.model.role_dir)):
            # 参考音频是外部文件路径，需要复制到角色目录
            try:
                role_path = self.model.get_role_path(self.current_role)
                file_name = os.path.basename(ref_audio)
                target_path = os.path.join(role_path, file_name)
                
                # 复制文件
                import shutil
                shutil.copy2(ref_audio, target_path)
                
                # 更新配置中的路径为相对路径
                emotion_config["ref_audio"] = file_name
            except Exception as e:
                print(f"复制参考音频出错: {str(e)}")
                self.view.show_message("错误", f"复制参考音频失败: {str(e)}", QMessageBox.Warning)
                return
        
        # 处理辅助参考音频复制
        new_aux_refs = []
        for aux_ref in emotion_config.get("aux_refs", []):
            if aux_ref and os.path.isfile(aux_ref) and not aux_ref.startswith(str(self.model.role_dir)):
                # 辅助参考音频是外部文件路径，需要复制到角色目录
                try:
                    role_path = self.model.get_role_path(self.current_role)
                    file_name = os.path.basename(aux_ref)
                    target_path = os.path.join(role_path, file_name)
                    
                    # 复制文件
                    import shutil
                    shutil.copy2(aux_ref, target_path)
                    
                    # 更新为相对路径
                    new_aux_refs.append(file_name)
                except Exception as e:
                    print(f"复制辅助参考音频出错: {str(e)}")
                    continue
            else:
                new_aux_refs.append(aux_ref)
        emotion_config["aux_refs"] = new_aux_refs
        
        # 同时更新基本参数
        config = self.view.get_current_config()
        for key in config:
            if key != "emotions":  # 不更新音色列表
                self.model.current_config[key] = config[key]
        
        # 使用模型添加/更新音色
        if self.model.add_emotion(self.current_role, emotion_name, emotion_config):
            # 保存配置到文件
            self.model.save_role_config(self.current_role, self.model.current_config)
            
            # 更新音色列表
            self.view.update_emotion_list(self.model.current_config.get("emotions", {}).keys())
            
            # 选中新音色
            items = self.view.emotion_list.findItems(emotion_name, Qt.MatchExactly)
            if items:
                self.view.emotion_list.setCurrentItem(items[0])
            
            self.view.show_message("成功", f"音色'{emotion_name}'保存成功!")
        else:
            self.view.show_message("错误", f"保存音色'{emotion_name}'失败!", QMessageBox.Warning)
    
    def on_emotion_delete(self, emotion_name):
        """删除音色事件处理"""
        if not self.current_role:
            self.view.show_message("错误", "请先选择一个角色!", QMessageBox.Warning)
            return
        
        if self.model.delete_emotion(emotion_name):
            # 更新音色列表
            self.view.update_emotion_list(self.model.current_config.get("emotions", {}).keys())
            self.view.emotion_name.clear()
            self.view.ref_audio.clear()
            self.view.prompt_text.clear()
            self.view.aux_ref_list.clear()
        else:
            self.view.show_message("错误", f"删除音色'{emotion_name}'失败!", QMessageBox.Warning)
    
    def on_aux_ref_add(self, ref_path):
        """添加辅助参考音频事件处理"""
        # 直接由视图处理
        pass
    
    def on_aux_ref_delete(self, ref_path):
        """删除辅助参考音频事件处理"""
        # 直接由视图处理
        pass
    
    def on_config_save(self, config):
        """保存配置事件处理"""
        if not self.current_role:
            self.view.show_message("错误", "请先选择一个角色!", QMessageBox.Warning)
            return
        
        # 保留原来的音色配置
        emotions = self.model.current_config.get("emotions", {})
        
        # 更新基本参数（从UI表单获取所有设置）
        for key, value in config.items():
            if key != "emotions":  # 不覆盖音色列表
                self.model.current_config[key] = value
        
        # 如果当前有选中音色，更新该音色的配置
        if self.current_emotion:
            # 获取当前音色的编辑内容
            emotion_config = self.view.get_current_emotion_config()
            emotion_name = self.view.emotion_name.text().strip()
            
            # 处理参考音频复制
            ref_audio = emotion_config.get("ref_audio", "")
            if ref_audio and os.path.isfile(ref_audio) and not ref_audio.startswith(str(self.model.role_dir)):
                try:
                    role_path = self.model.get_role_path(self.current_role)
                    file_name = os.path.basename(ref_audio)
                    target_path = os.path.join(role_path, file_name)
                    
                    # 复制文件
                    import shutil
                    shutil.copy2(ref_audio, target_path)
                    
                    # 更新配置中的路径为相对路径
                    emotion_config["ref_audio"] = file_name
                except Exception as e:
                    print(f"复制参考音频出错: {str(e)}")
                    self.view.show_message("错误", f"复制参考音频失败: {str(e)}", QMessageBox.Warning)
                    return
            
            # 处理辅助参考音频复制
            new_aux_refs = []
            for aux_ref in emotion_config.get("aux_refs", []):
                if aux_ref and os.path.isfile(aux_ref) and not aux_ref.startswith(str(self.model.role_dir)):
                    try:
                        role_path = self.model.get_role_path(self.current_role)
                        file_name = os.path.basename(aux_ref)
                        target_path = os.path.join(role_path, file_name)
                        
                        # 复制文件
                        import shutil
                        shutil.copy2(aux_ref, target_path)
                        
                        # 更新为相对路径
                        new_aux_refs.append(file_name)
                    except Exception as e:
                        print(f"复制辅助参考音频出错: {str(e)}")
                        continue
                else:
                    new_aux_refs.append(aux_ref)
            emotion_config["aux_refs"] = new_aux_refs
            
            # 如果音色名称已改变
            if self.current_emotion != emotion_name and emotion_name:
                # 检查新名称是否与其他音色冲突
                if emotion_name in emotions and emotion_name != self.current_emotion:
                    reply = QMessageBox.question(
                        self.view, 
                        "音色名称冲突", 
                        f"音色'{emotion_name}'已存在，是否覆盖?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        # 恢复原名称
                        self.view.emotion_name.setText(self.current_emotion)
                        return
                
                # 删除旧音色，添加新音色
                if self.current_emotion in emotions:
                    del emotions[self.current_emotion]
                emotions[emotion_name] = emotion_config
                self.current_emotion = emotion_name  # 更新当前音色名
            else:
                # 更新当前音色配置
                emotions[self.current_emotion] = emotion_config
        
        # 更新音色列表
        self.model.current_config["emotions"] = emotions
        
        # 保存配置到文件
        if self.model.save_role_config(self.current_role, self.model.current_config):
            # 更新音色列表（以防音色名称变更）
            self.view.update_emotion_list(emotions.keys())
            # 如果有选中的音色，确保它在列表中被选中
            if self.current_emotion:
                items = self.view.emotion_list.findItems(self.current_emotion, Qt.MatchExactly)
                if items:
                    self.view.emotion_list.setCurrentItem(items[0])
                    
            self.view.show_message("成功", "配置保存成功!")
        else:
            self.view.show_message("错误", "配置保存失败!", QMessageBox.Warning) 