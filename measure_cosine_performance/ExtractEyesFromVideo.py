from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
from utils.MeasureUtils import MeasureUtils
from xml_helper.XMLHelper import XMLHelper

YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)

class ExtractEyesFromVideo:
    def __init__(self, shape_predictor):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)
        self.xml_helper = XMLHelper()
        self.right_eye_file = 'measure_shape_predictor_performance/right_eye_file.xml'
        self.left_eye_file = 'measure_shape_predictor_performance/left_eye_file.xml'
        self.right_eyes = []
        self.left_eyes = []

    def extract_eyes_from_video(self, frames):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        vid = cv2.VideoCapture(0)
        cam_w = 640
        cam_h = 480

        while True:
            frames -= 1
            _, frame = vid.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=cam_w, height=cam_h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)

            if len(rects) > 0:
                rect = rects[0]
            else:

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                continue

            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[mStart:mEnd]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            nose = shape[nStart:nEnd]

            temp = leftEye
            leftEye = rightEye
            rightEye = temp

            self.right_eyes.append(rightEye)
            self.left_eyes.append(leftEye)

            mouthHull = cv2.convexHull(mouth)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            noseHull = cv2.convexHull(nose)

            cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)
            cv2.drawContours(frame, [noseHull], -1, YELLOW_COLOR, 1)

            for (x, y) in np.concatenate((mouth, leftEye, rightEye, nose), axis=0):
                cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if frames == 0:
                self.xml_helper.write_eye_to_file(self.right_eyes, self.right_eye_file)
                self.xml_helper.write_eye_to_file(self.left_eyes, self.left_eye_file)
                break

        cv2.destroyAllWindows()
        vid.release()

# testVideo = TestSPVideo("model/shape_predictor_68_face_landmarks.dat")
# result = testVideo.test_predictor_video()
# cos = result[0]
# ear = result[1]
# eye = result[2]

# print(cos)
# print(ear)
# print(len(cos))
# print(len(ear))

# start = time.time()
# for e in eye:
#     cosine = eye_aspect_ratio(e)
# end = time.time()
# print("ear time", ((end - start) * 1000.0))
#
# start1 = time.time()
# for e in eye:
#     cosine = measureCosine(e)
# end1 = time.time()
# print("cos time", ((end1 - start1) * 1000.0))
