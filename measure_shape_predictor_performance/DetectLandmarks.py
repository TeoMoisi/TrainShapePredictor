import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
from utils.Constants import Constants


class DetectLandmarks:
    def __init__(self):
        self.__constants = Constants()

    def detect_landmarks(self, shape_predictor, predicted_file, image_list, show_landmarks, save_image_path = None):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)

        out = open(predicted_file, "w")
        out.write("<dataset>\n")
        out.write("<images>\n")

        for image_path in image_list:
            if len(image_list) > 1:
                print(image_path)
                image = imutils.resize(cv2.imread("ibug_300W_large_face_landmark_dataset/" + image_path), width=500)
            else:
                print(image_path)
                image = cv2.imread(image_list[0])
                image = imutils.resize(image, width=500)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            out.write(f" <image file='{image_path}'>\n")

            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                if show_landmarks:
                    mouth = shape[self.__constants.mStart:self.__constants.mEnd]
                    leftEye = shape[self.__constants.lStart:self.__constants.lEnd]
                    rightEye = shape[self.__constants.rStart:self.__constants.rEnd]
                    rightEyebrow = shape[self.__constants.reStart:self.__constants.reEnd]
                    leftEyebrow = shape[self.__constants.leStart:self.__constants.leEnd]
                    nose = shape[self.__constants.nStart:self.__constants.nEnd]
                    jawline = shape[self.__constants.jawStart:self.__constants.jawEnd]

                    temp = leftEye
                    leftEye = rightEye
                    rightEye = temp

                    mouthHull = cv2.convexHull(mouth)
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    noseHUll = cv2.convexHull(nose)

                    cv2.drawContours(image, [mouthHull], -1, self.__constants.YELLOW_COLOR, 1)
                    cv2.drawContours(image, [leftEyeHull], -1, self.__constants.YELLOW_COLOR, 1)
                    cv2.drawContours(image, [rightEyeHull], -1, self.__constants.YELLOW_COLOR, 1)

                    cv2.drawContours(image, [noseHUll], -1, self.__constants.YELLOW_COLOR, 1)

                    for (x, y) in np.concatenate((mouth, leftEye, rightEye, nose, jawline, leftEyebrow, rightEyebrow),
                                                 axis=0):
                        cv2.circle(image, (x, y), 2, self.__constants.RED_COLOR, -1)

                if save_image_path != None:
                    cv2.imwrite(save_image_path, image)
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                out.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")

                for (j, (x, y)) in enumerate(shape):
                    index = "{0:0=2d}".format(j)
                    out.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")

                out.write(f"</box>\n")

            out.write(f"</image>\n")

        out.write(f"</images>\n")
        out.write(f"</dataset>\n")
        out.close()
