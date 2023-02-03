from jajucha.planning import BasePlanning
from jajucha.graphics import Graphics
from jajucha.control import mtx
import cv2
import numpy as np
import math


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
        left_top = [200, 260]
        right_top = [440, 260]
        left_bottom = [0, 380]
        right_bottom = [640, 380]
        w = int(np.linalg.norm(np.array(right_top) - np.array(left_top)))
        h = int(np.linalg.norm(np.array(left_bottom) - np.array(left_top)))
        pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        perspective = cv2.warpPerspective(frontImage, M, (int(w), int(h)))

        for val in pts1:
            cv2.circle(frontImage, (int(val[0]), int(
                val[1])), 5, (0, 0, 255), -1)
        cv2.imshow("front", frontImage)

        # 캐니 엣지
        canny = self.canny(perspective)
        cv2.imshow("canny", canny)

        # 확률적 허프변환
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180,
                                30, minLineLength=30, maxLineGap=30)
        lanes = []
        if lines is not None:
            for i in range(lines.shape[0]):
                pt1 = (lines[i][0][0], lines[i][0][1])
                pt2 = (lines[i][0][2], lines[i][0][3])
                lanes.append(lane(pt1, pt2))
                cv2.line(perspective, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

        # 선분 중앙점을 통한 사분면 판별
        # 2 | 1
        # ------
        # 3 | 4
        q1 = []
        q2 = []
        q3 = []
        q4 = []
        for l in lanes:
            if l.center[0] < (w/2) and l.center[1] < (h/2):
                q2.append(l)
            elif l.center[0] < (w/2) and l.center[1] >= (h/2):
                q3.append(l)
            elif l.center[0] > (w/2) and l.center[1] < (h/2):
                q1.append(l)
            else:
                q4.append(l)

        cv2.putText(perspective, "q2: {0}, q1: {1}".format(
            len(q2), len(q1)), (10, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.putText(perspective, "q3: {0}, q4: {1}".format(
            len(q3), len(q4)), (10, 200), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        # 좌평면 우평면의 직선 차선 판별
        lhscnt = 0
        rhscnt = 0
        for l in q2 + q3:
            if l.length > 70:
                lhscnt += 1
        for l in q1 + q4:
            if l.length > 70:
                rhscnt += 1

        tiltrcnt = 0
        tiltlcnt = 0
        for l in lanes:
            if l.angle >= 0:
                tiltrcnt += 1
            else:
                tiltlcnt += 1

        cv2.putText(perspective, "lhs: {0}, rhs: {1}".format(
            lhscnt, rhscnt), (10, 100), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv2.putText(perspective, "tL: {0}, tR: {1}".format(
            tiltlcnt, tiltrcnt), (10, 150), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        cv2.imshow("hough", perspective)
        self.vars.steer = steer
        self.vars.velocity = velocity
        return self.vars.steer, self.vars.velocity


if __name__ == "__main__":
    g = Graphics(Planning)  # 자주차 컨트롤러 실행
    g.root.mainloop()  # 클릭 이벤트 처리
    g.exit()  # 자주차 컨트롤러 종료
