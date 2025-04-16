# GPT-SoVITS GUI应用 - MVC重构版

本项目是对原始GPT-SoVITS GUI应用的重构版本，使用MVC（Model-View-Controller）设计模式进行了结构化重组，旨在提高代码的可维护性和可扩展性。

## 目录结构

```
gui/
  ├── __init__.py            # 包初始化文件
  ├── main_window.py         # 主窗口类
  ├── models/                # 模型层 - 数据和业务逻辑
  │   ├── __init__.py
  │   ├── role_model.py      # 角色配置模型
  │   └── inference_model.py # 推理模型
  ├── views/                 # 视图层 - 用户界面
  │   ├── __init__.py
  │   ├── role_view.py       # 角色配置视图
  │   ├── inference_view.py  # 推理视图
  │   └── common/            # 通用视图组件
  │       ├── __init__.py
  │       └── waveform_canvas.py # 波形图组件
  └── controllers/           # 控制器层 - 协调模型和视图
      ├── __init__.py
      ├── role_controller.py      # 角色配置控制器
      └── inference_controller.py # 推理控制器

gpt_sovits_gui.py            # 应用程序入口
```

## 设计模式说明

本项目采用MVC设计模式进行组织：

1. **Model（模型）**：负责处理数据和业务逻辑
   - `RoleModel`：处理角色配置的读取、保存、导入导出等
   - `InferenceModel`：处理语音合成、历史记录管理等

2. **View（视图）**：负责用户界面的显示和基本交互
   - `RoleConfigView`：角色和音色配置界面
   - `InferenceView`：语音合成推理界面
   - `WaveformCanvas`：音频波形显示组件

3. **Controller（控制器）**：负责协调模型和视图
   - `RoleController`：处理角色配置相关的用户操作
   - `InferenceController`：处理语音合成相关的用户操作

## 特点与优势

1. **关注点分离**：每一层都专注于自己的职责，代码结构更清晰
2. **可维护性**：模块化设计使得修改和调试更加容易
3. **可扩展性**：新功能可以方便地添加到适当的层次
4. **可测试性**：各层可以独立测试，提高代码质量

## 如何使用

1. 启动应用：
```bash
python gpt_sovits_gui.py
```

2. 应用包含两个主要选项卡：
   - **角色配置**：管理语音合成角色和音色
   - **音频推理**：使用配置好的角色进行语音合成

## 开发指南

如果需要修改或扩展功能，请遵循以下原则：

1. **添加新的模型功能**：在`models/`目录中修改或添加相应的模型类
2. **修改用户界面**：在`views/`目录中修改或添加相应的视图类
3. **添加新的交互逻辑**：在`controllers/`目录中修改或添加相应的控制器类

例如，如果要添加一个新的语音处理功能：
1. 在`models/`中添加相关的数据处理逻辑
2. 在`views/`中添加用户界面元素
3. 在`controllers/`中添加协调逻辑，连接模型和视图

## 注意事项

- 本重构版本保持了原始功能的完整性，只是对代码结构进行了优化
- 初次使用时，需要配置角色，设置GPT和SoVITS模型路径 