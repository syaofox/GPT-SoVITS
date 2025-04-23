"""
共用GUI组件包

包含可以在多个标签页中使用的GUI组件
"""

from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList
from gui.components.draggable_widgets import DraggableLineEdit, DraggableListWidget

__all__ = ['AudioPlayer', 'HistoryList', 'DraggableLineEdit', 'DraggableListWidget'] 