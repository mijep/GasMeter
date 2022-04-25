from asyncore import read
import os, uuid, sys
import imutils
import numpy as np
import cv2
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from image_processor import ImageProcessor
from PIL import Image
from skimage.transform import resize, pyramid_reduce


def getSquare(image, square_size):
    height, width = image.shape    
    if(height > width):
      differ = height
    else:
      differ = width
    differ += 4

    # square filler
    mask = np.zeros((differ, differ), dtype = "uint8")

    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)

    # center image inside the square
    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

    # downscale if needed
    if differ / square_size > 1:
      mask = pyramid_reduce(mask, differ / square_size)
    else:
      mask = cv2.resize(mask, (square_size, square_size), interpolation = cv2.INTER_AREA)
    return mask

class GasMeterReader:
    def __init__(self, trainFolder):
        self._debug = True
        self._trainFolder = trainFolder

        self._train()
        pass

    def classify(self, file):
        imp = ImageProcessor(file)
        imp.process()

        rois = imp.rois.copy()
        
        
        digits = []
        for roi in rois:
            cv2.imshow("test", roi)        
            key = cv2.waitKey(0)
            if key >= ord('0') and key <= ord('9'):
                print(f'you pressed: {int(chr(key))}')
                digits.append(int(chr(key)))
            else:
                digits.append(-1)            
            cv2.destroyAllWindows()

        rois1 = []
        for img in rois:
            img1 = getSquare(img, 50)
            img2 = cv2.normalize(src=img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            rois1.append(img2)

        img = rois1[0]
        for roi in rois1[1:]:
            img = np.concatenate((img, roi), axis=1)
        cv2.imshow("full", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for index, digit in enumerate(digits):
            folder = os.path.join(self._trainFolder, f'{digit}')
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            image = Image.fromarray(rois[index].astype(np.uint8))
            image.crop_pad((40, 75))

            filename = f'{digit}_{uuid.uuid4().hex}.jpeg'
            cv2.imwrite(os.path.join(folder, filename), np.array(image))        



    

        pass



        

    def processFile(self, file):
        im = ImageProcessor(file)
        im.process()
        picdigits= []
        for idx, img in enumerate (im.rois):            
            self._showImage(img)

            svd = self._svd(img, idx)
            recognizeddigit = min(self._digits[:10], key=lambda d: sum((d['singular']-svd['singular'])**2))    
            picdigits.append(recognizeddigit["number"])
        pass
        return picdigits

    def _train(self):
        self._digits = []
        for i in range(10):
            filename = os.path.join(self._trainFolder, f'{i}.png')
            M = cv2.imread(filename, 0)
            svd = self._svd(M, i)            
            self._digits.append(svd)

    def _svd(self, M, key):
        U, s, V = np.linalg.svd(M, full_matrices=False)
        s[100:] = 0        # keep the 10 biggest singular values only, discard others
        S = np.diag(s)
        M_reduced = np.dot(U, np.dot(S, V))      # reconstitution of image with 10 biggest singular values
        return {'original': M, 'singular': s[:10], 'reduced': M_reduced, 'number': key}

    def _showImage(self, image, title = "image"):
        if self._debug:
            cv2.imshow(title, image)
            cv2.waitKey(1)


file = r'c:\Users\mcj\Projects\gas_data\image220417-075008.650788.jpg'
file = r'c:\Users\mcj\Projects\gas_data\image220417-110021.169992.jpg'

reader = GasMeterReader(r'c:\Users\mcj\Projects\GasMeter\water-meter-ocr-master\data')
reader.classify(file)


img = cv2.imread(r'c:\Users\mcj\Projects\GasMeter\water-meter-ocr-master\data\0.png',  cv2.IMREAD_GRAYSCALE)
img1 = getSquare(img, 50)

img_n = cv2.normalize(src=img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite('square.tiff', img1.astype(np.float32))
cv2.imwrite('square.png', img_n)

cv2.imshow("test", img1)
        
key = cv2.waitKey(0)




#numbers = reader.processFile(file)




pass




