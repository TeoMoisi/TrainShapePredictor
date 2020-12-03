from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from errors_computation.ErrorComputation import ErrorComputation
from xml_helper.XMLHelper import XMLHelper

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(jawStart, jawEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

def detectLandmarks(predictor_path, file, image_path, save_name):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    out = open(file, "w")
    out.write("<dataset>\n")
    out.write("<images>\n")

    print(image_path)
    image = cv2.imread(image_path, 0)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    out.write(f" <image file='{image_path}'>\n")

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[mStart:mEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        rightEyebrow = shape[reStart:reEnd]
        leftEyebrow = shape[leStart:leEnd]
        nose = shape[nStart:nEnd]
        jawline = shape[jawStart:jawEnd]

        temp = leftEye
        leftEye = rightEye
        rightEye = temp

        mouthHull = cv2.convexHull(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        noseHUll = cv2.convexHull(nose)

        cv2.drawContours(image, [mouthHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(image, [leftEyeHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(image, [rightEyeHull], -1, YELLOW_COLOR, 1)

        cv2.drawContours(image, [noseHUll], -1, YELLOW_COLOR, 1)

        for (x, y) in np.concatenate((mouth, leftEye, rightEye, nose, jawline, leftEyebrow, rightEyebrow), axis=0):
            cv2.circle(image, (x, y), 2, RED_COLOR, -1)



        (x, y, w, h) = face_utils.rect_to_bb(rect)

        out.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")

        for (j, (x, y)) in enumerate(shape):
            index = "{0:0=2d}".format(j)
            out.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")
        cv2.imwrite(save_name, image)
        out.write(f"</box>\n")

    out.write(f"</image>\n")


    out.write(f"</images>\n")
    out.write(f"</dataset>\n")
    out.close()



detectLandmarks("model/predictor6666.dat", "frontalLandmarks.xml", "frontal.JPG", "frontalPredicted.jpg")
detectLandmarks("model/shape_predictor_68_face_landmarks.dat", "frontalLandmarksGroundTruth.xml", "frontal.JPG", "frontalGroundTruth.jpg")
xmlhelper = XMLHelper()
ground_truth = xmlhelper.read_small_xml("frontalLandmarksGroundTruth.xml")
predicted = xmlhelper.read_small_xml("frontalLandmarks.xml")

error = ErrorComputation()
# print(ground_truth)
# print(predicted)
error = error.get_mean_error(predicted, ground_truth)
print(error)
print(sum(error) / 68)

#error = dlib.test_shape_predictor("ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml", "shape_predictor_68_face_landmarks.dat")
#print("[INFO] error: {}".format(error))