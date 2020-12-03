from math import sqrt
import numpy as np


class ErrorComputation:
    def __init__(self):
        pass

    def compute_mean_error(self, predicted, ground_truth):
        means = []
        variances = []
        distances = []
        sum_sqr = 0

        for i in range(0, 68):
            sum_err = 0
            for photo in range(0, len(ground_truth)):
                norm = sqrt((int(ground_truth[photo][36][0]) - int(ground_truth[photo][45][0])) ** 2 + (int(ground_truth[photo][36][1]) - int(ground_truth[photo][45][1])) ** 2)
                a = np.array([predicted[photo][i][0], predicted[photo][i][1]]).astype(int)
                b = np.array([ground_truth[photo][i][0], ground_truth[photo][i][1]]).astype(int)
                dist = np.linalg.norm(a - b) / norm

                distances.append(dist)
                sum_err += dist
                sum_sqr += dist**2

            mean = sum_err/(len(ground_truth))
            means.append(mean)

        print(variances)
        return means
