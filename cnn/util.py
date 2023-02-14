import numpy as np
import cv2

class PerspectiveArea:
    def __init__(self, left_top, right_top, left_bottom, right_bottom):
        self.left_top = left_top
        self.right_top = right_top
        self.left_bottom = left_bottom
        self.right_bottom = right_bottom
        self.width = 224 # 224 is 16 * 14
        self.height = 224
        
        #self.width = int(np.linalg.norm(
        #    np.array(right_top) - np.array(left_top)))
        #self.height = int(np.linalg.norm(
        #    np.array(left_bottom) - np.array(left_top)))

default_perspective = PerspectiveArea([200, 260], [440, 260], [0, 380], [640, 380])

class JajuchaCamera:
    def __init__(self, mtx, h, trans, trans_inv, flip=False):
        self.f_u = f_u = mtx[0, 0]
        self.f_v = f_v = mtx[1, 1]
        if not flip:
            self.c_u = c_u = mtx[0, 2]
            self.c_v = c_v = mtx[1, 2]
        else:
            self.c_u = c_u = 639 - mtx[0, 2]
            self.c_v = c_v = 479 - mtx[1, 2]
        self.h = h
        self.M = trans @ np.array([[-h / f_u, 0., h * c_u / f_u],
                                   [0., 0., -h],
                                   [0., -1 / f_v, c_v / f_v]], dtype=np.float32)
        self.M_inv = np.array([[f_u, c_u, 0],
                               [0., c_v, h * f_v],
                               [0., 1, 0]], dtype=np.float32) @ trans_inv

    def warpImg(self, img):
        return cv2.warpPerspective(img, self.M, (500, 300))

    def unWarpPts(self, pts):
        return cv2.perspectiveTransform(np.array([pts], dtype=np.float32), self.M_inv)[0]


class JajuchaCV:
    @staticmethod
    def cv_canny(img, par1=200, par2=400):
        l = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 1]
        blur = cv2.bilateralFilter(l, 7, 10, 20)
        edge = cv2.Canny(blur, par1, par2)
        return edge

    @staticmethod
    def cv_perspective(img, left_top, right_top, left_bottom, right_bottom):
        w = int(np.linalg.norm(np.array(right_top) - np.array(left_top)))
        h = int(np.linalg.norm(np.array(left_bottom) - np.array(left_top)))
        pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (224, 224))

    @classmethod
    def cv_perspective_area(cls, img, area):
        return cls.cv_perspective(img, area.left_top, area.right_top, area.left_bottom, area.right_bottom)
