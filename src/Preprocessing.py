from Image import Image
import cv2
import numpy as np


class Preprocessing(Image):
    def blur(self):
        kernel = np.ones((11, 11), np.float32) / 121
        self.setimage(cv2.filter2D(self.getimage(), -1, kernel))

    def bilateral(self):
        self.setimage(cv2.bilateralFilter(self.getimage(), 5, 75, 75))

    def grayscale(self):
        self.setimage(cv2.cvtColor(self.getimage(), cv2.COLOR_BGR2GRAY))

    def otsu(self):
        ret2, th2 = cv2.threshold(self.getimage(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.setimage(th2)

    def closing(self):
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(self.getimage(), kernel, iterations=1)
        self.setimage(cv2.dilate(img, kernel, iterations=1))

