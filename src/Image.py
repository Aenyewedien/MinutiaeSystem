import cv2
import numpy as np
import os


class Image:
    def __init__(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        try:
            n = np.fromfile(filename, dtype)
            img = cv2.imdecode(n, flags)
            self.image = img
        except Exception as e:
            print(e)
            self.image = None

    def getimage(self):
        return self.image

    def setimage(self, image):
        self.image = image

    def save(self, filename, params=None):
        try:
            ext = os.path.splitext(filename)[1]
            result, n = cv2.imencode(ext, self.image, params)

            if result:
                with open(filename, mode='w+b') as f:
                    n.tofile(f)
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False
