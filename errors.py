from math import sqrt
import numpy as np
import PIL
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
    distances = []
    sum_sqr = 0
    for i in range(0, 68):
        sum_err = 0
        for photo in range(0, len(grtr)):
            norm = sqrt((int(grtr[photo][36][0])-int(grtr[photo][45][0])) ** 2 + (int(grtr[photo][36][1])-int(grtr[photo][45][1])) ** 2)
            print(int(pred[photo][i][0]), int(pred[photo][i][1]))
            a = np.array([pred[photo][i][0], pred[photo][i][1]]).astype(int)
            b = np.array([grtr[photo][i][0], grtr[photo][i][1]]).astype(int)
            dist = np.linalg.norm(a - b) / norm
            err = sqrt((int(pred[photo][i][0])-int(grtr[photo][i][0]))**2 + (int(pred[photo][i][1])-int(grtr[photo][i][1]))**2)
            print("err", err)
            distances.append(dist)
            sum_err += dist
            sum_sqr += err**2

        mean = sum_err/(len(grtr))
        means.append(mean)

    print("Mean", sum(means)/68)
    print("Minimum mean", min(means))
    print(variances)
    return means
