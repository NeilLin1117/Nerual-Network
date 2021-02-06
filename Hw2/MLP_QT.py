from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QPushButton,QButtonGroup,QTextEdit
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import os
import MLP
from PyQt5.QtGui import  QPixmap
 
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "多層感知器神經網路"
        self.top = 35
        self.left = 35
        self.width = 1800
        self.height = 900
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.ComboBox()
        self.initUI()
        self.train_btn = QPushButton('開始訓練', self)
        self.train_btn.move(20, 260)
        self.train_btn.clicked.connect(self.train_showDialog)
        #self.train_btn.setEnabled(False)

        self.train_qlabel = QLabel(self)
        self.train_qlabel.move(400,20)
        self.train_result = QLabel(self)
        self.train_result.move(400,270)
        self.test_qlabel = QLabel(self)
        self.test_qlabel.move(900,20)
        self.test_result = QLabel(self)
        self.test_result.move(900,270)
        self.train_image = QLabel(self)
        self.train_image.setGeometry(400,50,400,200)
        self.test_image = QLabel(self)
        self.test_image.setGeometry(900,50,400,200)
        self.train_qlabel.setText("                                                                                                                ")
        self.test_qlabel.setText("                                                                                                                 ")
        self.train_result.setText("                                                       ")
        self.test_result.setText("                                                        ")
        self.show_result()
        self.textEdit1 = QTextEdit(self)
        self.textEdit1.setGeometry(20,360,560,300)
        self.textEdit2 = QTextEdit(self)
        self.textEdit2.setGeometry(600,360,560,300)
        self.show()

    def train_showDialog(self):
        self.file = self.qlabel.text() 
        self.epochs = int (self.epoch_le.text())
        self.eta = float(self.rate_le.text())
        self.hidden = int(self.hidden_le.text())
        self.hidden_layer = int(self.hidden_layer_le.text())
        self.seed = int(self.seed_le.text())
        self.train_acc , self.test_acc = MLP.main_program(self.file,
                self.epochs,self.hidden,self.hidden_layer,self.eta,self.seed)

        if os.path.exists('./images/'+self.file[:-4]+'_train.png'):
            pix = QPixmap('./images/'+self.file[:-4]+'_train.png')
            #self.train_image.setStyleSheet("border: 1px solid black")
            self.train_image.setPixmap(pix)
            pix = QPixmap('./images/'+self.file[:-4]+'_test.png')
            #self.test_image.setStyleSheet("border: 1px solid black")
            self.test_image.setPixmap(pix)
            self.train_qlabel.setText("Train data 視覺化結果:")
            self.test_qlabel.setText("Test data 視覺化結果:")
            self.train_result.setText('Train accuracy: %.2f%%' % (self.train_acc * 100))
            self.test_result.setText('Test accuracy: %.2f%%' % (self.test_acc * 100))
        else:
            if os.path.exists('./sorry1.png'):
                pix = QPixmap('./sorry1.png')
                self.train_image.setPixmap(pix)
            if os.path.exists('./sorry2.png'):
                pix = QPixmap('./sorry2.png')
                self.test_image.setPixmap(pix)
            self.train_qlabel.setText("因為維度資料超過二維,很抱歉無法視覺化呈現train data!")
            self.test_qlabel.setText("因為維度資料超過二維,很抱歉無法視覺化呈現test data!")
            self.train_result.setText('Train accuracy: %.2f%%' % (self.train_acc * 100))
            self.test_result.setText('Test accuracy: %.2f%%' % (self.test_acc * 100))
        self.train_correct.setEnabled(True)
        self.train_error.setEnabled(True)
        self.test_correct.setEnabled(True)
        self.test_error.setEnabled(True)

    def show_result(self):
        self.train_correct = QPushButton('顯示train data預測正確資料', self)
        self.train_correct.move(20, 320)
        self.train_error = QPushButton('顯示train data預測錯誤資料', self)
        self.train_error.move(320, 320)
        self.test_correct = QPushButton('顯示test data預測正確資料', self)
        self.test_correct.move(620, 320)
        self.test_error = QPushButton('顯示test data預測錯誤資料', self)
        self.test_error.move(920, 320)
        self.train_correct.clicked.connect(self.train_correct_event)
        self.train_error.clicked.connect(self.train_error_event)
        self.test_correct.clicked.connect(self.test_correct_event)
        self.test_error.clicked.connect(self.test_error_event)
        self.train_correct.setEnabled(False)
        self.train_error.setEnabled(False)
        self.test_correct.setEnabled(False)
        self.test_error.setEnabled(False)
        

    def train_correct_event(self):
        f = open('./result/'+self.file[:-4]+'_train_correct.txt', 'r')
        with f:
            data = f.read()
            self.textEdit1.setText(data)

    def train_error_event(self):
        f = open('./result/'+self.file[:-4]+'_train_error.txt', 'r')
        with f:
            data = f.read()
            self.textEdit1.setText(data)
    def test_correct_event(self):
        f = open('./result/'+self.file[:-4]+'_test_correct.txt', 'r')
        with f:
            data = f.read()
            self.textEdit2.setText(data)

    def test_error_event(self):
        f = open('./result/'+self.file[:-4]+'_test_error.txt', 'r')
        with f:
            data = f.read()
            self.textEdit2.setText(data)

    def initUI(self): 
             
        self.rate_btn = QPushButton('設定學習率', self)
        self.rate_btn.move(20, 70)
        self.rate_btn.clicked.connect(self.rate_showDialog)
        
        self.rate_le = QLineEdit(self)
        self.rate_le.move(170, 72)
        self.rate_le.setText(str(0.005))

        self.hidden_btn = QPushButton('隱藏層神經元數', self)
        self.hidden_btn.move(20, 100)
        self.hidden_btn.clicked.connect(self.hidden_showDialog)

        self.hidden_le = QLineEdit(self)
        self.hidden_le.move(170, 102)
        self.hidden_le.setText(str(500))
        
        self.hidden_layer_btn = QPushButton('隱藏層數目', self)
        self.hidden_layer_btn.move(20, 140)
        self.hidden_layer_btn.clicked.connect(self.hidden_layer_showDialog)

        self.hidden_layer_le = QLineEdit(self)
        self.hidden_layer_le.move(170, 142)
        self.hidden_layer_le.setText(str(3))

        self.epoch_btn = QPushButton('設定epoch數', self)
        self.epoch_btn.move(20, 180)
        self.epoch_btn.clicked.connect(self.epoch_showDialog)
        
        self.epoch_le = QLineEdit(self)
        self.epoch_le.move(170, 182)
        self.epoch_le.setText(str(20))

        self.seed_btn = QPushButton('設定seed(0~10)', self)
        self.seed_btn.move(20, 220)
        self.seed_btn.clicked.connect(self.seed_showDialog)
        
        self.seed_le = QLineEdit(self)
        self.seed_le.move(170, 222)
        self.seed_le.setText(str(1))

    def seed_showDialog(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入seed(0~10):')
        
        if ok:
            self.seed_le.setText(str(text))

    def epoch_showDialog(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入epoch數:')
        
        if ok:
            self.epoch_le.setText(str(text))

    def rate_showDialog(self):
        
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入學習率:')
        
        if ok:
            self.rate_le.setText(str(text))

    def hidden_layer_showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入隱藏層數目:')
        if ok:
            self.hidden_layer_le.setText(str(text))

    def hidden_showDialog(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 
            '請輸入隱藏層神經元數目:')
        if ok:
            self.hidden_le.setText(str(text))

    def ComboBox(self):
        self.combo = QComboBox(self)
        string_file =""
        for dirName, sub_dirNames, fileNames in os.walk('datasets'):
            for file in fileNames: 
                self.combo.addItem(file)
            string_file += fileNames[0]
        self.combo.move(20, 20)
        self.qlabel = QLabel(self)
        self.qlabel.setText(string_file)
        self.qlabel.move(170,20)

        self.combo.activated[str].connect(self.onChanged) #Signal

    def onChanged(self, text):
        self.qlabel.setText(text)
        self.qlabel.adjustSize()
        


if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())
