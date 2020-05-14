import os

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import re
import random



predictor_path = "shape_predictor_68_face_landmarks.dat"
#predictor_path = "detector.svm"
image_path = "images/imagine.jpg"

pathHelen = "ibug_300W_large_face_landmark_dataset/helen/trainset/*.jpg"

pathIBug = "ibug_300W_large_face_landmark_dataset/ibug/*.jpg"

pathLFPWJpg = "ibug_300W_large_face_landmark_dataset/lfpw/trainset/*.jpg"
pathLFPWPng = "ibug_300W_large_face_landmark_dataset/lfpw/trainset/*.png"

pathAFWJ = "ibug_300W_large_face_landmark_dataset/afw/*.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#predictor = dlib.simple_object_detector(predictor_path)

out = open("trainings100.xml", "w")
out.write("<dataset>\n")
out.write("<images>\n")
images_no = 0

for image_path in glob.glob(pathHelen):
    print(image_path)
    if images_no < 20:
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images_no += 1
        rects = detector(gray, 1)
        #rects = detector(image)

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

images_no = 0

for image_path in glob.glob(pathIBug):

    if images_no < 20:
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images_no += 1
        rects = detector(gray, 1)
        #rects = detector(image)

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
images_no = 0
for image_path in glob.glob(pathAFWJ):

    if images_no < 20:
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images_no += 1
        rects = detector(gray, 1)
        #rects = detector(image)

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
images_no = 0
for image_path in glob.glob(pathLFPWJpg):

    if images_no < 20:
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images_no += 1
        rects = detector(gray, 1)
        #rects = detector(image)

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
images_no = 0
for image_path in glob.glob(pathLFPWPng):

    if images_no < 20:
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images_no += 1
        rects = detector(gray, 1)
        #rects = detector(image)

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
