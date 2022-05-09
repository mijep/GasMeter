from fileinput import filename
import json
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import (QWidget, QGridLayout,QPushButton, QApplication, QVBoxLayout, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage, QColor,  QIntValidator
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import os, uuid, sys
from matplotlib.font_manager import json_dump
import numpy as np
import json
from random import choice
from pathlib import Path





from GasMeterImageProcessor import ProcessMeterImage
from trainDigits import TrainDigits


class App(QWidget):
    def __init__(self):
        super().__init__()

        self._picFolder = r'c:\Users\mcj\Projects\gas_data'
        self._dataFolder = r"c:\Users\mcj\Projects\GasMeter\data"

        self._train = TrainDigits()
        self._numDigits = 0 
        self.initLayout()        

        self.OnNewFile()


    def pickRandomFile(self):
        path: Path = Path(self._picFolder)
        # The Path.iterdir method returns a generator, so we must convert it to a list
        # before passing it to random.choice, which expects an iterable.
        random_path = choice(list(path.iterdir()))

        pass
        return random_path.as_posix()
        

    def readFile(self, filename):
        self._activeFile = filename
        self._filenameLabel.setText(filename)

        try: 
            self.processor = ProcessMeterImage()        
            self.processor.process(filename)
        except:
            return False

        if len(self.processor._scaledDigits) != 7:
            return False


        disply_width = 640
        display_height = 480
        qt_img = self.convert_cv_qt(self.processor._cropedImage, disply_width, display_height)
        self._gasMeterImageLabel.setPixmap(qt_img)
        
        disply_width = 64*5
        display_height = 48*5
        for i, img in enumerate(self.processor._scaledDigits):
            qt_img = self.convert_cv_qt(img, disply_width, display_height)
            # display it            
            self._digitsImageLabel[i].setPixmap(qt_img)
            self._digitsEdit[i].setText('')

            ret,result,neighbours,dist =  self._train.testData(img) 
            self._digitsGuessLabel[i].setText(str(result[0,0]))

            pass

        self._numDigits = len(self.processor._scaledDigits)
        
        return True

    def initLayout(self):
        self._digitsGuessLabel = []
        self._digitsImageLabel = []
        self._digitsEdit = []

        
        vertical = QVBoxLayout()
        self.setLayout(vertical)
        disply_width = 640
        display_height = 480
                
        # display it
        horizontal = QHBoxLayout()

        self.shortcutNewFile = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.CTRL + Qt.Key_N), self)
        self.shortcutNewFile.activated.connect(self.OnNewFile)

        self.shortcutSave = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.CTRL + Qt.Key_S), self)
        self.shortcutSave.activated.connect(self.OnSave)


        self._filenameLabel = QLabel(self)
        horizontal.addWidget(self._filenameLabel)
        
        label = QLabel()
        label.setText("CTRL+N to Load New File")
        horizontal.addWidget(label)

        label = QLabel()
        label.setText("CTRL+S to save")
        horizontal.addWidget(label)

        vertical.addLayout(horizontal)


        self._gasMeterImageLabel  = QLabel(self)
        #self._gasMeterImageLabel.setPixmap(qt_img)        
        vertical.addWidget(self._gasMeterImageLabel)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)
        vertical.addLayout(grid_layout)

        
        disply_width = 64*5
        display_height = 48*5
        

        for y in range(7):
            guess_label = QLabel(self)                                
            f = guess_label.font()
            f.setPointSize(27) # sets the size to 27
            guess_label.setFont(f)            
            guess_label.setText("*")
            grid_layout.addWidget(guess_label, 0, y)
            self._digitsGuessLabel.append(guess_label)

        for y in range(7):
            image_label = QLabel(self)                                
            grid_layout.addWidget(image_label, 1, y)
            self._digitsImageLabel.append(image_label)
        
        for y in range(7):
            edit = QLineEdit(self)
            f = edit.font()
            f.setPointSize(27) # sets the size to 27
            edit.setFont(f)            
            
            self._digitsEdit.append(edit)

            grid_layout.addWidget(edit, 2, y)
            #edit.textChanged.connect(self.onChanged)
            edit.textChanged.connect(lambda text,id=y: self.onChanged(text, id))
            edit.setValidator(QIntValidator(0,9))
            #edit.setInputMask('9')            

        
        self.setWindowTitle('Basic Grid Layout')

    def onChanged(self, text, id):
        if len(text) == 1:
            print(f'id:{id}   text:{text}')
            if self._numDigits > 0:
                next = (id+1) % self._numDigits
            else:
                next = id
            
            self._digitsEdit[next].setFocus()

    def OnNewFile(self):
        while True:
            filename = self.pickRandomFile()
            isOk = self.readFile(filename)
            if isOk:
                break
        pass

    def OnSave(self):
        digits = []
        for index, edit in enumerate(self._digitsEdit):
            digitText = edit.text()
            try:            
                digit = int(digitText)
            except Exception as e:
                edit.setFocus()
                return
            digits.append(digit)

        self.saveDigits(digits)
        self.OnNewFile()
        pass


    def saveDigits(self, digitsList):

        #jsonData = {}    
        #with open(r'C:\Users\mcj\Projects\GasMeter\src\dgits.json') as file:
        #    jsonData = json.load(file)
        
        dataFolder = self._dataFolder
        
        digitsFolder = os.path.join(dataFolder, "digits")        
        picFolder =   os.path.join(dataFolder, "gasMeterPic")      

        if not os.path.exists(digitsFolder):
            os.makedirs(digitsFolder)

        digitsFileList = {
        }

        fileList = []
        if len(digitsList) == len(self.processor._scaledDigits):
            for i, img in enumerate(self.processor._scaledDigits):
                digit = digitsList[i]
                folder = os.path.join(digitsFolder, f'{digit}')
                if not os.path.exists(folder):
                    os.makedirs(os.path.join(folder, f'{digit}'))
                
                filename = f'{digit}_{uuid.uuid4().hex}.png'

                digitsFileList[filename] = digit
                digitsPath = os.path.join(folder, filename)
                
                cv2.imwrite(digitsPath, np.array(img))                                                
                fileList.append((os.path.relpath(digitsPath, digitsFolder), digit))
            
        print(fileList)

        jsonfile = r'c:\Users\mcj\Projects\GasMeter\data\dgits.json'

        self.updateFile(jsonfile, self._activeFile, fileList)
                

    def updateFile(self, jsonFilename, gasMeterPicfile, fileList):
        
        data = {
            "filelist" : {            
            }
        }
        try:
            with open(jsonFilename, 'r') as fp:
                data = json.load(fp)
                pass
        except:
            pass
        
        dict1 = {}
        for file, digit in fileList:
            dict1[file] = digit

        fileName = os.path.relpath(gasMeterPicfile, self._picFolder)
        dataNew = {
            "filelist" : {
                fileName : dict1
            }
        }

        dataNew['filelist'].update(data['filelist'])

        with open(jsonFilename, "w") as fp:
            json.dump(dataNew, fp, indent=2)


        print(data)

        pass


        fileList= [
            ('0_857c901e61d348009884eb926037ecb0.jpeg', 0), 
            ('8_668fb59a2dab4ccf89cd57c19de9ae38.jpeg', 8), 
            ('2_8200e1aa510e446d8dacd98d4df36c9d.jpeg', 2), 
            ('5_2d5ecc36de974120b0a43e2251663f16.jpeg', 5), 
            ('3_29b9a80ab32d4ea788fbfa3dece55c52.jpeg', 3), 
            ('6_68cdbe0e22c741e0bdc9dbf6da7d3c37.jpeg', 6), 
            ('3_266500dc3a824eb7965fcdc723897533.jpeg', 3)]

        

        





    


    
    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    


if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())