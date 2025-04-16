"""
波形图画布组件
用于显示音频波形
"""
import os
import sys
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSizePolicy

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    import librosa
    HAS_LIBROSA = True
except ImportError:
    print("警告: librosa 未安装，波形图显示功能将不可用")
    HAS_LIBROSA = False

class WaveformCanvas(FigureCanvas):
    """波形图画布类"""
    
    # 添加自定义信号用于点击跳转
    playback_position_changed = Signal(float)
    
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        try:
            self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            self.axes.set_facecolor('#282828')
            self.fig.patch.set_facecolor('#282828')
            self.axes.set_ylim([-1.1, 1.1])
            self.axes.set_yticks([])
            self.axes.set_xticks([])
            
            FigureCanvas.__init__(self, self.fig)
            self.setParent(parent)
            
            FigureCanvas.setSizePolicy(self,
                                      QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
            FigureCanvas.updateGeometry(self)
            
            # 存储音频信息
            self.audio_data = None
            self.audio_sr = None
            self.audio_duration = 0
            
            # 添加鼠标点击事件
            self.mpl_connect('button_press_event', self.on_click)
            
            # 添加当前位置标记线
            self.position_line = None
            
        except Exception as e:
            print(f"初始化波形图失败: {str(e)}")
            # 创建一个替代的占位符
            self.setParent(parent)
            self.setMinimumHeight(80)
            self.setStyleSheet("background-color: #282828;")
    
    def on_click(self, event):
        """鼠标点击事件处理"""
        try:
            if event.xdata is not None and self.audio_duration > 0:
                # 计算点击位置对应的时间(秒)
                pos_ratio = event.xdata / self.audio_duration
                # 发出信号
                self.playback_position_changed.emit(pos_ratio)
                # 更新位置标记线
                self.update_position_line(event.xdata)
        except Exception as e:
            print(f"处理波形图点击事件出错: {str(e)}")
    
    def update_position_line(self, position):
        """更新位置标记线"""
        try:
            # 移除旧线
            if self.position_line:
                self.position_line.remove()
            
            # 添加新线
            self.position_line = self.axes.axvline(x=position, color='red', linewidth=1)
            self.draw()
        except Exception as e:
            print(f"更新位置标记线出错: {str(e)}")
    
    def set_playback_position(self, position_ms):
        """更新播放位置标记线（由播放器进度更新触发）"""
        try:
            if self.audio_duration > 0:
                position_sec = position_ms / 1000.0
                if 0 <= position_sec <= self.audio_duration:
                    self.update_position_line(position_sec)
        except Exception as e:
            print(f"设置播放位置出错: {str(e)}")
    
    def plot_waveform(self, audio_path):
        """绘制波形图"""
        try:
            if not os.path.exists(audio_path):
                return
            
            # 检查是否有librosa库，如果没有则返回
            if not HAS_LIBROSA:
                print("缺少librosa库，无法绘制波形图")
                return
                
            # 清除当前图形
            self.axes.clear()
            self.axes.set_facecolor('#282828')
            self.axes.set_ylim([-1.1, 1.1])
            self.axes.set_yticks([])
            self.axes.set_xticks([])
            
            # 加载音频
            self.audio_data, self.audio_sr = librosa.load(audio_path, sr=None)
            
            # 计算时间轴和音频时长
            self.audio_duration = len(self.audio_data) / self.audio_sr
            time = np.arange(0, len(self.audio_data)) / self.audio_sr
            
            # 绘制波形
            self.axes.plot(time, self.audio_data, color='#00BFFF', linewidth=0.5)
            
            # 添加时间轴标签（每隔几秒一个）
            if self.audio_duration > 0:
                num_ticks = min(5, int(self.audio_duration) + 1)
                tick_positions = np.linspace(0, self.audio_duration, num_ticks)
                tick_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in tick_positions]
                self.axes.set_xticks(tick_positions)
                self.axes.set_xticklabels(tick_labels, fontsize=8, color='white')
            
            # 重置位置标记线
            self.position_line = None
            
            # 刷新图形
            self.fig.tight_layout()
            self.draw()
        except Exception as e:
            print(f"绘制波形图出错: {str(e)}") 