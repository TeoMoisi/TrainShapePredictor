import numpy as np


class MeasureUtils:
    def __init__(self):
        pass

    def measure_cosine(self, eye):
        a = np.linalg.norm(eye[2] - eye[4])
        b = np.linalg.norm(eye[3] - eye[4])
        c = np.linalg.norm(eye[2] - eye[3])

        cosinA = (c * c + b * b - a * a) / (2 * b * c)
        return cosinA

    def eye_aspect_ratio(self, eye):
        firstDistance = np.linalg.norm(eye[1] - eye[5])
        secondDistance = np.linalg.norm(eye[2] - eye[4])
        thirdDistance = np.linalg.norm(eye[0] - eye[3])

        ear = (firstDistance + secondDistance) / (2.0 * thirdDistance)
        return ear
