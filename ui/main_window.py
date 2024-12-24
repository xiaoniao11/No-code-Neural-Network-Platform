from PyQt5.QtWidgets import QMainWindow, QTabWidget
from .data_analysis_page import DataAnalysisPage
from .model_builder_page import ModelBuilderPage
from .training_page import TrainingPage
from .inference_page import InferencePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("神经网络可视化编程平台")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # 添加四个主要页面
        self.data_analysis_page = DataAnalysisPage()
        self.model_builder_page = ModelBuilderPage()
        self.training_page = TrainingPage()
        self.inference_page = InferencePage()

        
        self.tabs.addTab(self.data_analysis_page, "数据分析")
        self.tabs.addTab(self.model_builder_page, "模型搭建")
        self.tabs.addTab(self.training_page, "模型训练")
        self.tabs.addTab(self.inference_page, "模型应用") 
