import sys
from PyQt5 import QtWidgets
import design
import os, shutil, glob
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np


CLASSES = ['animals', 'architecture', 'car', 'food', 'nature', 'people']
MODEL = load_model('CNN.h5')


class Application(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btnOpenFolder.clicked.connect(self.browse_folder)
        self.btnClassification.clicked.connect(self.classification)
        

    def browse_folder(self):
        self.listFolder.clear()
        self.linePath.clear()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Открыть папку")

        if directory:  
            for file_name in os.listdir(directory): 
                if file_name.endswith(".jpg"):
                    self.listFolder.addItem(file_name) 
            self.linePath.setText(directory)
            self.progressBar.setValue(0)
            listOfFiles = glob.glob(os.path.join(directory, "*.jpg"))
            self.label.setText('Обнаружено изображений: ' + str(len(listOfFiles)))


  	def classification(self):
        Path = self.linePath.text()
        if not Path:
            self.label.setText('Папка не открыта') 
        else:  
            listOfFiles = glob.glob(os.path.join(Path, "*.jpg"))
            if not len(listOfFiles):
                self.label.setText('Изображения отсутствуют') 
            else:    
                create_folders(Path)
                self.progressBar.setRange(0, len(listOfFiles)-1)    
                for i in range(0, len(listOfFiles)):
                    pred = predict(listOfFiles[i])
                    if pred == 'animals':
                        shutil.move(listOfFiles[i], Path + '/animals')
                    elif pred == 'architecture':
                        shutil.move(listOfFiles[i], Path + '/architecture') 
                    elif pred == 'nature':
                        shutil.move(listOfFiles[i], Path + '/nature') 
                    elif pred == 'food':
                        shutil.move(listOfFiles[i], Path + '/food') 
                    elif pred == 'car':
                        shutil.move(listOfFiles[i], Path + '/car') 
                    elif pred == 'people':
                        shutil.move(listOfFiles[i], Path + '/people') 
                    elif pred == 'other':
                        shutil.move(listOfFiles[i], Path + '/other') 
                    self.progressBar.setValue(i)
                self.label.setText('Готово!')
                self.listFolder.clear()


def predict(photo): 
    img = image.load_img(photo, target_size=(160, 160))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = MODEL.predict(x)
    if max(max(prediction)) <= 0.6:
        return 'other'
    else: 
        return CLASSES[np.argmax(prediction)]
    

def create_folders(Path):
    for c in CLASSES:        
        folder = Path + '/' + c
        if not os.path.exists(folder): os.mkdir(folder)
    if not os.path.exists(Path + '/other'): os.mkdir(Path + '/other')


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Application() 
    window.show() 
    app.exec_() 
    
if __name__ == '__main__': 
    main()