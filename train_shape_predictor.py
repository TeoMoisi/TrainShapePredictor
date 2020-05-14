#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014
#
#   In particular, we will train a face landmarking model based on a small
#   dataset and then evaluate it.  If you want to visualize the output of the
#   trained model on some images then you can run the
#   face_landmark_detection.py example program with predictor.dat as the input
#   model.
#
#   It should also be noted that this kind of model, while often used for face
#   landmarking, is quite general and can be used for a variety of shape
#   prediction tasks.  But here we demonstrate it only on a simple face
#   landmarking task.
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy
import multiprocessing
import os
import sys
import glob

import cv2
import dlib


# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.
# if len(sys.argv) != 2:
#     print(
#         "Give the path to the examples/faces directory as the argument to this "
#         "program. For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./train_shape_predictor.py ../examples/faces")
#     exit()
# faces_folder = sys.argv[1]
from imutils import face_utils

options = dlib.shape_predictor_training_options()
# Now make the object responsible for training the model.
# This algorithm has a bunch of parameters you can mess with.  The
# documentation for the shape_predictor_trainer explains all of them.
# You should also read Kazemi's paper which explains all the parameters
# in great detail.  However, here I'm just setting three of them
# differently than their default values.  I'm doing this because we
# have a very small dataset.  In particular, setting the oversampling
# to a high amount (300) effectively boosts the training set size, so
# that helps this example.
# options.oversampling_amount = 3
# # I'm also reducing the capacity of the model by explicitly increasing
# # the regularization (making nu smaller) and by using trees with
# # smaller depths.
options.nu = 0.1
options.tree_depth = 4
options.cascade_depth = 15
options.feature_pool_size = 400
options.num_test_splits = 50
options.oversampling_translation_jitter = 0.1
# options.num_threads = multiprocessing.cpu_count()

#Optimised parameters
# options.tree_depth = 4
# options.nu = 0.1033
# options.cascade_depth = 20
# options.feature_pool_size = 677
# options.num_test_splits = 295
# options.oversampling_amount = 29
# options.oversampling_translation_jitter = 0
# options.feature_pool_region_padding = 0.0975
# options.lambda_param = 0.0251

options.be_verbose = True
options.num_threads = multiprocessing.cpu_count()

# dlib.train_shape_predictor() does the actual training.  It will save the
# final predictor to predictor.dat.  The input is an XML file that lists the
# images in the training dataset and also contains the positions of the face
# parts.
print("Here")
training_xml_path = os.path.join("trainings100.xml")
dlib.train_shape_predictor(training_xml_path, "predictor100.dat", options)