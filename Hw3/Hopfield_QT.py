from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QPushButton,QButtonGroup,QTextEdit
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import os
import network
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QMessageBox


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Hopfield神經網路"
        self.top = 35
        self.left = 35
        self.width = 1500
        self.height = 950
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.train_image = QLabel(self)
        self.train_image.move(500, 20)
        self.train_image.setGeometry(450,20,900,850)

        self.ComboBox()
        self.Test_ComboBox()
        self.initUI()
        self.train_btn = QPushButton('開始訓練', self)
        self.train_btn.move(20, 260)
        self.train_btn.clicked.connect(self.train_showDialog)
        
        #self.train_btn.setEnabled(False)

        self.show()

    def train_showDialog(self):
        self.train_file = self.qlabel.text() 
        self.test_file = self.res_qlabel.text()
        self.threshold = int(self.hidden_layer_le.text())
        self.asyn = int(self.epoch_le.text())
        if self.train_file[:5] == self.test_file[:5]:
            network.main(self.train_file,
                    self.test_file,self.threshold,self.asyn)
            if os.path.exists('result.png'):
                pix = QPixmap('result.png')
                self.train_image.setPixmap(pix)
        else:
            reply = QMessageBox.information(self,                         #使用infomation信息框
                                    "錯誤",
                                    "訓練檔案和測試檔案不一致",
                                    QMessageBox.Ok )

    def initUI(self): 
        
        self.hidden_layer_btn = QPushButton('設定threshold值(0~25)', self)
        self.hidden_layer_btn.move(20, 140)
        self.hidden_layer_btn.clicked.connect(self.hidden_layer_showDialog)

        self.hidden_layer_le = QLineEdit(self)
        self.hidden_layer_le.move(240, 142)
        self.hidden_layer_le.setText(str(3))

        self.epoch_btn = QPushButton('是否同步(0不同步,1同步)', self)
        self.epoch_btn.move(20, 180)
        self.epoch_btn.clicked.connect(self.epoch_showDialog)
        
        self.epoch_le = QLineEdit(self)
        self.epoch_le.move(240, 182)
        self.epoch_le.setText(str(1))

    def hidden_layer_showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入threshold值:')
        if ok:
            self.hidden_layer_le.setText(str(text))

    def epoch_showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入是否同步(0不同步,1同步):')
        if ok:
            self.epoch_le.setText(str(text))

    def ComboBox(self):
        self.train_qlabel = QLabel(self)
        self.train_qlabel.setText("訓練資料集")
        self.train_qlabel.move(20,20)
        self.combo = QComboBox(self)
        string_file =""
        for dirName, sub_dirNames, fileNames in os.walk('train'):
            for file in fileNames: 
                self.combo.addItem(file)
            string_file += fileNames[0]
        self.combo.move(120, 20)
        self.qlabel = QLabel(self)
        self.qlabel.setText(string_file)
        self.qlabel.move(300,20)

        self.combo.activated[str].connect(self.onChanged) #Signal
    
    def Test_ComboBox(self):
        self.test_qlabel = QLabel(self)
        self.test_qlabel.setText("測試資料集")
        self.test_qlabel.move(20,50)
        self.test_combo = QComboBox(self)
        string_file =""
        for dirName, sub_dirNames, fileNames in os.walk('test'):
            for file in fileNames: 
                self.test_combo.addItem(file)
            string_file += fileNames[0]
        self.test_combo.move(120, 50)
        self.res_qlabel = QLabel(self)
        self.res_qlabel.setText(string_file)
        self.res_qlabel.move(300,50)

        self.test_combo.activated[str].connect(self.test_onChanged) #Signal

    def onChanged(self, text):
        self.qlabel.setText(text)
        self.qlabel.adjustSize()

    def test_onChanged(self,text):
        self.res_qlabel.setText(text)
        self.res_qlabel.adjustSize()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())
