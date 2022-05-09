import cv2
import numpy as np
import random as rng
from image_processor import ImageProcessor
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

class ProcessMeterImage:
    def __init__(self) -> None:
        
        self._canny_threshold_1=120
        self._canny_threshold_2=280        
        pass

    def cropRelevant(self, image):
        edges = cv2.Canny(image, self._canny_threshold_1, self._canny_threshold_2)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        crop_img = image[y:y+h, x:x+w]
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(0)
        return crop_img    

    def findDigits(self, image):
        imp = ImageProcessor(image)
        imp.process()
        rois = imp.rois.copy()
        return rois

    def scaleDigits(self, imageList):
        scaledImagList = []
        for img in imageList:
            img1 = getSquare(img, 50)
            img2 = cv2.normalize(src=img1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            scaledImagList.append(img2)

        return scaledImagList

    def process(self, filename):
        self._image = cv2.imread(filename)
        self._cropedImage = self.cropRelevant(self._image.copy())
        self._digits = self.findDigits(self._cropedImage)
        self._scaledDigits = self.scaleDigits(self._digits)                
        pass




#p = ProcessMeterImage()
#filename = r'c:\Users\mcj\Projects\gas_data\image220416-205426.070884.jpg'
#p.process(filename)

