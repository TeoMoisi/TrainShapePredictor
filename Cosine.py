import numpy as np


def measureCosine(eye):
    a = np.linalg.norm(eye[2] - eye[4])
    b = np.linalg.norm(eye[3] - eye[4])
    c = np.linalg.norm(eye[2] - eye[3])

    cosinA = (c * c + b * b - a * a) / (2 * c * b)
    return cosinA

def eye_aspect_ratio(eye):
    firstDistance = np.linalg.norm(eye[1] - eye[5])
    secondDistance = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    thirdDistance = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (firstDistance + secondDistance) / (2.0 * thirdDistance)

    # Return the eye aspect ratio
    return ear
