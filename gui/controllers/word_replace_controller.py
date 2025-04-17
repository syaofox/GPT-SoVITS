"""
词语替换控制器
负责协调词语替换模型和视图
"""
from PySide6.QtWidgets import QMessageBox


class WordReplaceController:
    """词语替换控制器类"""
    
    def __init__(self, model, view):
        """初始化控制器"""
        self.model = model
        self.view = view
        
        # 连接视图的信号
        self.connect_signals()
        
        # 加载配置到视图
        self.load_config()
    
    def connect_signals(self):
        """连接视图的信号"""
        self.view.config_save.connect(self.on_config_save)
        self.view.config_reload.connect(self.on_config_reload)
    
    def load_config(self):
        """加载配置到视图"""
        config_text = self.model.load_config()
        self.view.set_config_text(config_text)
    
    def on_config_save(self, config_text):
        """保存配置事件处理"""
        if self.model.save_config(config_text):
            self.view.show_message("成功", "词语替换配置保存成功!")
        else:
            self.view.show_message("错误", "保存词语替换配置失败!", QMessageBox.Warning)
    
    def on_config_reload(self):
        """重新加载配置事件处理"""
        self.load_config()
        self.view.show_message("成功", "词语替换配置重新加载完成!") 