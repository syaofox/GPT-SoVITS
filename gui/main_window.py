"""
主窗口
用于组织应用程序的主界面
"""
from PySide6.QtWidgets import QMainWindow, QTabWidget

from gui.models.role_model import RoleModel
from gui.models.inference_model import InferenceModel
from gui.models.word_replace_model import WordReplaceModel
from gui.views.role_view import RoleConfigView
from gui.views.inference_view import InferenceView
from gui.views.word_replace_view import WordReplaceView
from gui.controllers.role_controller import RoleController
from gui.controllers.inference_controller import InferenceController
from gui.controllers.word_replace_controller import WordReplaceController

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-SoVITS 语音合成")
        self.setMinimumSize(1024, 768)
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建标签页
        self.tabs = QTabWidget()
        
        # 创建MVC组件
        self.init_components()
        
        # 添加标签页
        self.tabs.addTab(self.inference_view, "音频推理")
        self.tabs.addTab(self.word_replace_view, "词语替换")
        self.tabs.addTab(self.role_view, "角色配置")
        
        
        
        self.setCentralWidget(self.tabs)
    
    def init_components(self):
        """初始化MVC组件"""
        # 创建模型
        self.role_model = RoleModel()
        self.inference_model = InferenceModel()
        self.word_replace_model = WordReplaceModel()
        
        # 创建视图
        self.role_view = RoleConfigView()
        self.inference_view = InferenceView()
        self.word_replace_view = WordReplaceView()
        
        # 创建控制器
        self.role_controller = RoleController(self.role_model, self.role_view)
        self.inference_controller = InferenceController(
            self.role_model, 
            self.inference_model,
            self.word_replace_model,
            self.inference_view
        )
        self.word_replace_controller = WordReplaceController(
            self.word_replace_model,
            self.word_replace_view
        ) 