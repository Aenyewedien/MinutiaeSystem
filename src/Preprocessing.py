from Image import Image
import cv2
import numpy as np
import copy


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

    def gabor(self):
        filters = []
        ksize = 7
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel(
                (ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            kern /= 1.5 * kern.sum()
            filters.append(kern)
        accum = np.zeros_like(self.getimage())
        for kern in filters:
            image = cv2.cvtColor(self.getimage(), cv2.COLOR_BGR2RGB)
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        self.setimage(accum)

    def dilate(self):
        kernel = np.ones((5, 5), np.uint8)
        self.setimage(cv2.dilate(self.getimage(), kernel, iterations=1))

    def resize(self, size):
        self.setimage(
            cv2.resize(
                self.getimage(), (size, size), fx=0, fy=0, interpolation=cv2.INTER_AREA
            )
        )

    def wframe(self):
        value = [255, 255, 255]
        self.setimage(
            cv2.copyMakeBorder(
                self.getimage(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value
            )
        )

    def normalize(self):
        rows = self.getimage().shape[0]
        cols = self.getimage().shape[1]
        up, down, left, right = 0, rows - 1, 0, cols - 1
        vertical = np.sum(self.getimage(), axis=1).tolist()
        for i in range(rows):
            if vertical[i] > 0:
                left = i - 1
                break
        for i in range(rows - 1, 0, -1):
            if vertical[i] > 0:
                right = i + 1
                break

        horizontal = np.sum(self.getimage(), axis=0).tolist()
        for i in range(cols):
            if horizontal[i] > 0:
                up = i - 1
                break
        for i in range(cols - 1, 0, -1):
            if horizontal[i] > 0:
                down = i + 1
                break
        self.setimage(copy.deepcopy(self.getimage()[left:right, up:down]))
