"""
波形图画布组件
用于显示音频波形，针对长音频文件进行了性能优化
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
    
    def __init__(self, parent=None, width=5, height=2, dpi=100, max_points=10000):
        try:
            self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            self.axes.set_facecolor('#282828')
            self.fig.patch.set_facecolor('#282828')
            self.axes.set_ylim([-1.1, 1.1])
            self.axes.set_yticks([])
            self.axes.set_xticks([])
            
            # 最大显示点数，用于长音频降采样
            self.max_points = max_points
            
            FigureCanvas.__init__(self, self.fig)
            self.setParent(parent)
            
            # 设置尺寸策略为可扩展，但优先遵循固定高度
            FigureCanvas.setSizePolicy(self,
                                      QSizePolicy.Expanding,
                                      QSizePolicy.Fixed)
            FigureCanvas.updateGeometry(self)
            
            # 存储音频信息
            self.audio_data = None
            self.audio_sr = None
            self.audio_duration = 0
            self.audio_path = None
            
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
    
    def downsample_audio(self, audio_data, target_length):
        """对音频数据进行降采样，减少绘图点数"""
        if len(audio_data) <= target_length:
            return audio_data
        
        # 计算降采样倍率
        step = len(audio_data) // target_length
        
        # 方法1: 简单抽样（更快但可能丢失峰值）
        # return audio_data[::step]
        
        # 方法2: 使用峰值折叠（保留音频特征，显示包络线）
        # 将数据分成多个块，每个块取最大值和最小值
        result = np.zeros(target_length * 2)
        for i in range(target_length):
            start = i * step
            end = min(start + step, len(audio_data))
            if start < end:
                block = audio_data[start:end]
                result[i*2] = np.max(block)
                result[i*2+1] = np.min(block)
            else:
                result[i*2] = 0
                result[i*2+1] = 0
        
        return result
    
    def load_audio_efficient(self, audio_path, max_duration=None):
        """高效加载音频文件，支持限制加载时长"""
        try:
            # 对于非常长的音频，可以选择只加载前几分钟
            if max_duration is not None:
                # 使用duration参数只加载指定长度的音频
                audio_data, sr = librosa.load(audio_path, sr=None, duration=max_duration)
                actual_duration = max_duration
            else:
                # 正常加载整个音频
                audio_data, sr = librosa.load(audio_path, sr=None)
                actual_duration = len(audio_data) / sr
            
            return audio_data, sr, actual_duration
        except Exception as e:
            print(f"加载音频失败: {str(e)}")
            return np.array([]), 44100, 0
    
    def plot_waveform(self, audio_path, max_duration=None):
        """绘制波形图，支持长音频优化"""
        try:
            if not os.path.exists(audio_path):
                return
            
            # 检查是否有librosa库，如果没有则返回
            if not HAS_LIBROSA:
                print("缺少librosa库，无法绘制波形图")
                return
            
            # 保存音频路径
            self.audio_path = audio_path
                
            # 清除当前图形
            self.axes.clear()
            self.axes.set_facecolor('#282828')
            self.axes.set_ylim([-1.1, 1.1])
            self.axes.set_yticks([])
            self.axes.set_xticks([])
            
            # 高效加载音频
            self.audio_data, self.audio_sr, self.audio_duration = self.load_audio_efficient(
                audio_path, max_duration
            )
            
            if len(self.audio_data) == 0:
                return
            
            # 对长音频进行降采样
            if len(self.audio_data) > self.max_points:
                # 降采样后的数据
                display_data = self.downsample_audio(self.audio_data, self.max_points // 2)
                
                # 创建对应的时间轴
                # 由于使用的是峰值折叠方法，时间点数量是音频点的两倍
                time = np.linspace(0, self.audio_duration, len(display_data))
            else:
                # 对于短音频，直接使用原始数据
                display_data = self.audio_data
                time = np.arange(0, len(display_data)) / self.audio_sr
            
            # 绘制波形
            self.axes.plot(time, display_data, color='#00BFFF', linewidth=0.5)
            
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