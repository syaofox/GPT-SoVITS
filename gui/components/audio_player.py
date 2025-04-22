"""
音频播放器组件

提供音频播放和控制功能
"""

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


class AudioPlayer(QWidget):
    """音频播放器组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建媒体播放器
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # 创建UI组件
        layout = QVBoxLayout(self)
        
        # 播放控制按钮
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_button)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.progress_slider)
        
        # 音量控制
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)
        controls_layout.addWidget(self.volume_slider)
        
        layout.addLayout(controls_layout)
        
        # 设置信号连接
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.playbackStateChanged.connect(self.state_changed)
        
        # 设置初始音量
        self.set_volume(70)
    
    def load_audio(self, file_path: str):
        """加载音频文件"""
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.play_button.setText("播放")
    
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()
    
    def stop_playback(self):
        """停止播放"""
        self.player.stop()
    
    def set_position(self, position):
        """设置播放位置"""
        self.player.setPosition(position)
    
    def set_volume(self, volume):
        """设置音量"""
        self.audio_output.setVolume(volume / 100.0)
    
    def position_changed(self, position):
        """播放位置变化回调"""
        self.progress_slider.setValue(position)
    
    def duration_changed(self, duration):
        """音频时长变化回调"""
        self.progress_slider.setRange(0, duration)
    
    def state_changed(self, state):
        """播放状态变化回调"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("暂停")
        else:
            self.play_button.setText("播放") 