from math import *
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import xml.etree.ElementTree as ET
import xml.etree.ElementTree

from errors import mean_error
from getDetectedLandmarks import detectLandmarks


def read_xml(file):

    l = []
    grtr = []
    i = 0
    imgs = []
    xmldoc = minidom.parse(file)
    #boxes = xmldoc.getElementsByTagName('box')
    # for im in img:
    #     #print(im.attributes['file'].value)
    #     imgs.append(im.attributes['file'].value)
    itemlist = xmldoc.getElementsByTagName('part')
    j = 0
    for s in itemlist:
        if (i < 68):
            l.append([s.attributes['x'].value, s.attributes['y'].value])
            i += 1

        else:
            grtr.append(l)
            j+=1
            i = 0
            l = []

    return grtr


def read_small_xml(file):

    l = []
    grtr = []
    xmldoc = minidom.parse(file)

    itemlist = xmldoc.getElementsByTagName('part')
    print(len(itemlist))
    for s in itemlist:
        print([s.attributes['x'].value, s.attributes['y'].value])
        l.append([s.attributes['x'].value, s.attributes['y'].value])

    grtr.append(l)
    return grtr

def getTestingImages(file):
    xmldoc = minidom.parse(file)
    img = xmldoc.getElementsByTagName('image')
    imgs = []
    for im in img:
        imgs.append(im.attributes['file'].value)

    return imgs



images = getTestingImages("ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml")
#
# predictor_path = "shape_predictor_68_face_landmarks.dat"
# detectLandmarks(predictor_path, "groundTruth.xml", images)
# #
# # predictor_path = "predictorMax.dat"
# # detectLandmarks(predictor_path, "predictedMax.xml", images)
# #
# # #detectLandmarks("predictor1000Tuned.dat", "predicted1000Tuned.xml", images)
#
detectLandmarks("predictor2000.dat", "predicte2000.xml", images)

ground_truth2 = read_xml("groundTruth.xml")
predicted2 = read_xml("predicted2000.xml")
error = mean_error(predicted2, ground_truth2)
print(error)