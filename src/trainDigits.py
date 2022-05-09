import numpy as np
import cv2 
from matplotlib import pyplot as plt

import os

class TrainDigits:
    def __init__(self):
        self._knn = self.trainDigits(r'c:\Users\mcj\Projects\GasMeter\data\digits')        

    def trainDigits(self, digitsDir, digitList=[0, 1, 2, 3, 4, 5,6, 7, 8, 9]):

        data = []
        label = []
        for digit in digitList:
            dir =os.path.join(digitsDir, str(digit))
            if not os.path.isdir(dir):
                continue

            files = os.listdir(dir)        
            for f in files:	    
                if not os.path.isfile(os.path.join(dir,f)): 
                    continue
                img = cv2.imread(os.path.join(dir,f))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                data.append(gray)
                label.append(digit)

            
        length = len(data)
        data = np.array(data)
        train = data[:,:].reshape(-1, 50*50).astype(np.float32)
        #train_labels = np.repeat(0,length) [:,np.newaxis]
        train_labels = np.array(label)[:,np.newaxis]
        knn = cv2.ml.KNearest_create()
        ret = knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
        
        return knn    
        pass

    def testPicFile(self, path):
        data=[]
        img = cv2.imread(path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        data.append(gray)
        length = len(data)
        data = np.array(data)
        train = data[:,:].reshape(-1, 50*50).astype(np.float32)   
        return train 

    def testData(self, dataPichFile):
        data=[]        
        data.append(dataPichFile)
        length = len(data)
        data = np.array(data)
        test = data[:,:].reshape(-1, 50*50).astype(np.float32)   
        ret,result,neighbours,dist = self._knn.findNearest(test,k=1)
        return ret,result,neighbours,dist     





