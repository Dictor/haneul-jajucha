from jajucha.planning import BasePlanning
from jajucha.graphics import Graphics
from jajucha.control import mtx
import cv2
import numpy as np
import math
from cnn import model, util
import torch
import torchvision.transforms as transforms

class lane:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.length = math.sqrt(
            (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        self.angle = math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
        self.center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

class Planning(BasePlanning):
    def __init__(self, graphics):
        super().__init__(graphics)
        # --------------------------- #
        self.vars.redCnt = 0  # 변수 설정
        self.vars.greenCnt = 0  # 변수 설정
        self.vars.stop = True
        self.vars.steer = 0
        self.vars.velocity = 0
        self.model = model.CNN(util.default_perspective).to("cpu")
        self.model.load_state_dict(torch.load("./cnn/model", map_location="cpu"))
        self.model.eval()

    def process(self, t, frontImage, rearImage, frontLidar, rearLidar):
        """
        자주차의 센서 정보를 바탕으로 조향과 속도를 결정하는 함수
        t: 주행 시점으로부터의 시간 (초)
                frontImage: 전면 카메라 이미지
                rearImage: 후면 카메라 이미지
                frontLidar: 전면 거리 센서 (mm), 0은 오류를 의미함
                rearLidar: 후면 거리 센서 (mm), 0은 오류를 의미함
        """

        steer = 0
        velocity = 0

        # 시점 변환
        perspective = util.JajuchaCV.cv_perspective_area(
            frontImage, util.default_perspective)

        for val in (util.default_perspective.left_bottom, util.default_perspective.left_top, util.default_perspective.right_bottom, util.default_perspective.right_top):
            cv2.circle(frontImage, (int(val[0]), int(
                val[1])), 5, (0, 0, 255), -1)
        cv2.imshow("front", frontImage)

        # 캐니 엣지
        canny = util.JajuchaCV.cv_canny(perspective)
        cv2.imshow("canny", canny)

        tinput = transforms.ToTensor()(canny)
        tinput = tinput.unsqueeze_(0)
        tinput = tinput.to("cpu")
        out = self.model(tinput)
        _, outidx = out.max(dim=1)

        classes = ['four', 'left', 'perpendicular', 'right', 'three_left', 'three_right']
        situation = outidx.item()
        cv2.putText(perspective, "{0} {1:0.2f}%".format(
            classes[situation], out[0][outidx].item()), (10, 170), cv2.FONT_ITALIC, 0.8, (255, 255, 0), 2)

        cv2.imshow("perspective", perspective)
        self.vars.steer = steer
        self.vars.velocity = velocity
        return self.vars.steer, self.vars.velocity


if __name__ == "__main__":
    g = Graphics(Planning)  # 자주차 컨트롤러 실행
    g.root.mainloop()  # 클릭 이벤트 처리
    g.exit()  # 자주차 컨트롤러 종료
