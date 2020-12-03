import time
from measure_cosine_performance.ExtractEyesFromVideo import ExtractEyesFromVideo
from xml_helper.XMLHelper import XMLHelper
from utils.Constants import Constants
from utils.MeasureUtils import MeasureUtils
import numpy as np


class EarVSCosine:
    def __init__(self):
        self.constants = Constants()
        self.video_capture = ExtractEyesFromVideo(self.constants.predictor_6666)
        self.xml_helper = XMLHelper()
        self.measure_utils = MeasureUtils()
        self.left_eyes = []
        self.right_eyes = []
        self.right_ear = []
        self.right_cosine = []
        self.left_ear = []
        self.left_cosine = []

    def get_eyes_landmarks(self, capture, frames):
        if capture:
            self.video_capture.extract_eyes_from_video(frames)
        self.left_eyes = self.xml_helper.read_eye_from_file(self.constants.left_eye_file)
        self.right_eyes = self.xml_helper.read_eye_from_file(self.constants.right_eye_file)

    def get_ear_and_cosine(self, eye):
        if eye == 'left':
            for left_eye in self.left_eyes:
                left_eye = np.array(left_eye).astype(int)
                self.left_cosine.append(self.measure_utils.measure_cosine(np.array(left_eye)))
                self.left_ear.append(self.measure_utils.eye_aspect_ratio(np.array(left_eye)))
            return [self.left_cosine, self.left_ear]
        else:
            for right_eye in self.right_eyes:
                right_eye = np.array(right_eye).astype(int)
                self.right_cosine.append(self.measure_utils.measure_cosine(right_eye))
                self.right_ear.append(self.measure_utils.eye_aspect_ratio(right_eye))
            return [self.right_cosine, self.right_ear]

    def measure_ear_performance(self, eye):
        if eye == "left":
            start = time.time()
            for left_eye in self.left_eyes:
                left_eye = np.array(left_eye).astype(int)
                self.measure_utils.eye_aspect_ratio(left_eye)
            end = time.time()
            return (end - start) * 1000.0
        else:
            start1 = time.time()
            for right_eye in self.right_eyes:
                right_eye = np.array(right_eye).astype(int)
                self.measure_utils.eye_aspect_ratio(right_eye)
            end1 = time.time()
            return (end1 - start1) * 1000.0

    def measure_cosine_performance(self, eye):
        if eye == "left":
            start = time.time()
            for left_eye in self.left_eyes:
                left_eye = np.array(left_eye).astype(int)
                self.measure_utils.measure_cosine(left_eye)
            end = time.time()
            return (end - start) * 1000.0
        else:
            start1 = time.time()
            for right_eye in self.right_eyes:
                right_eye = np.array(right_eye).astype(int)
                self.measure_utils.measure_cosine(right_eye)
            end1 = time.time()
            return (end1 - start1) * 1000.0

    def measure_ear_vs_cosine(self, eye, frames):
        self.get_eyes_landmarks(False, frames)
        self.get_ear_and_cosine(eye)
        ear_right = self.measure_ear_performance(eye)
        cosine_right = self.measure_cosine_performance(eye)
        return [ear_right, cosine_right]


earVScosine = EarVSCosine()
earVScosine.get_eyes_landmarks(True, 100)
[cosine, _] = earVScosine.get_ear_and_cosine("right")
[cosine_left, _] = earVScosine.get_ear_and_cosine("left")
print(cosine)
print(cosine_left)
print(len(cosine))
print(len(cosine_left))


# result = earVScosine.measure_ear_vs_cosine_time("right", 50)

#earVScosine.get_eyes_landmarks(False, 50)
# [cosine, ear] = earVScosine.get_ear_and_cosine("right")
# print(cosine)
# print(len(cosine))
#result = earVScosine.measure_ear_vs_cosine_time("right", 50)

#
# total_ear = 0
# total_cosine = 0
# earVScosine = EarVSCosine()
# for i in range(50):
#     #earVScosine = EarVSCosine()
#     result = earVScosine.measure_ear_vs_cosine_time("right", 50)
#     total_ear += result[0]
#     total_cosine += result[1]
#
# print("EAR", total_ear / 50)
# print("COSINE", total_cosine / 50)

# [cosine, ear] = earVScosine.get_ear_and_cosine("right")
# print(cosine)
# print(len(cosine))
# print(ear)
# print(len(ear))
#
# ear_right = earVScosine.measure_ear_performance("right")
# cosine_right = earVScosine.measure_cosine_performance("right")
# print(ear_right)
# print(cosine_right)
