from math import sqrt
import numpy as np


def mean_error(pred, grtr):
    #no_pt: list of the points
    #pred: list of lists of tuples: [[(x1,y1), ..., (xn, yn)], [],...[]] for every photo
    #grtr: list of lists of tuples

    #print(pred['faces/7285955@N06/coarse_tilt_aligned_face.2050.9486768763_e52727c632_o.jpg'])
    err = 0
    means = []
    variances = []
    N = len(pred)
    sum_err = 0
    sum_sqr = 0
    for i in range(0, 68):
        sum_err = 0
        for photo in range(0, len(grtr)):
            print(int(pred[photo][i][0]), int(pred[photo][i][1]))
            a = np.array([pred[photo][i][0], pred[photo][i][1]]).astype(int)
            b = np.array([grtr[photo][i][0], grtr[photo][i][1]]).astype(int)
            #b = np.array(int(grtr[photo][i][0]), int(grtr[photo][i][1]))
            dist = np.linalg.norm(a - b)
            #dist = np.linalg.norm((pred[photo][i][0], pred[photo][i][1]) - (grtr[photo][i][0], grtr[photo][i][1]))
            err = sqrt((int(pred[photo][i][0])-int(grtr[photo][i][0]))**2 + (int(pred[photo][i][1])-int(grtr[photo][i][1]))**2)
            print("err", err)
            sum_err += dist
            sum_sqr += err**2


        mean = sum_err/len(grtr)
        #variances.append((sum_sqr - N * mean ** 2) / (N - 1))
        means.append(mean)

    print("Mean", sum(means)/68)
    #variances.append((sum_sqr - N*mean**2)/(N-1))
    print(min(means))
    print(variances)
    return means


# pred = [[[12, 13], [78, 78]],[[24, 24], [67, 56]]]
# for i in range(0, 2):
#     for photo in range(0, len(pred)):
#         print(pred[photo][i][0])