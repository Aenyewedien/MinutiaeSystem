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
        ret2, th2 = cv2.threshold(
            self.getimage(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.setimage(th2)

    def closing(self):
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(self.getimage(), kernel, iterations=1)
        self.setimage(cv2.dilate(img, kernel, iterations=1))

    def flood(self):
        self.setimage(cv2.floodFill(self.getimage(), None, (0, 0), (255, 255, 255)))

    def delete_small(self, min_size):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            self.getimage(), connectivity=8
        )
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        img2 = np.zeros(self.image.shape)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        self.setimage(img2)
