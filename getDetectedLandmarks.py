from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import re
import random


def detectLandmarks(predictor_path, file, imageList):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    out = open(file, "w")
    out.write("<dataset>\n")
    out.write("<images>\n")

    for image_path in imageList:
        image_path = "ibug_300W_large_face_landmark_dataset/" + image_path
        print(image_path)
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)


        out.write(f" <image file='{image_path}'>\n")


        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            out.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")

            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (j, (x, y)) in enumerate(shape):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                index = "{0:0=2d}".format(j)
                out.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")

            out.write(f"</box>\n")

        out.write(f"</image>\n")


    out.write(f"</images>\n")
    out.write(f"</dataset>\n")
    out.close()
