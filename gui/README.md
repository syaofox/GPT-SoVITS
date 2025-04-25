# GPT-SoVITS GUI 模块说明

本目录包含了GPT-SoVITS项目的图形用户界面实现，采用模块化设计以便于维护和扩展。

## 目录结构

### 核心文件
- `main.py` - 应用程序入口点，负责启动GUI界面
- `main_window.py` - 主窗口类实现，包含界面布局和基本事件处理
- `__init__.py` - GUI包初始化文件
- `word_replace.txt` - 文本替换配置文件
- `requirements-gui.txt` - GUI模块依赖的Python包列表

### 模块目录

#### `components/` - 共享GUI组件
- `audio_player.py` - 音频播放器组件，用于预览和播放生成的音频
- `history_list.py` - 历史记录列表组件，用于管理和显示历史操作
- `__init__.py` - 组件包初始化文件

#### `views/` - 不同的标签页
- `experiment_tab.py` - 试听配置标签页，用于实时调整参数并试听效果
- `role_tab.py` - 角色推理标签页，管理和使用已训练的角色模型
- `__init__.py` - 标签页包初始化文件

#### `controllers/` - 控制器
- `base_controller.py` - 基础控制器，提供共用功能
- `role_controller.py` - 角色控制器，管理角色相关的业务逻辑
- `inference_controller.py` - 推理控制器，处理语音合成请求
- `__init__.py` - 控制器包初始化文件

#### `models/` - 模型
- `inference_worker.py` - 推理工作线程，处理异步推理任务
- `role_model.py` - 角色模型，定义角色的数据结构和行为
- `inference_model.py` - 推理模型，封装语音合成的核心逻辑
- `__init__.py` - 模型包初始化文件

#### `roles/` - 角色资源目录
包含预设和用户创建的角色配置，每个角色有独立的文件夹：

## 架构说明

本GUI采用MVC(Model-View-Controller)架构设计:
- Model层(models/): 负责数据处理和业务逻辑
- View层(components/, views/): 负责用户界面展示
- Controller层(controllers/): 连接Model和View，处理用户输入并更新界面

## 使用方法

通过`main.py`启动GUI应用，或者通过项目根目录的启动脚本间接启动。