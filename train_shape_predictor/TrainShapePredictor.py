import multiprocessing
import os
import dlib


class TrainShapePredictor:
    def __init__(self, data_size):
        self.__training_data_file = ""
        if data_size == 6666:
            self.__training_data_file = "ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml"
        else:
            self.__training_data_file = "xml_training/trainings" + str(data_size) + ".xml"
        self.__new_predictor_name = "model/predictor" + str(data_size) + ".xml"

    def train_shape_predictor(self):
        options = dlib.shape_predictor_training_options()
        options.nu = 0.1
        options.tree_depth = 4
        options.cascade_depth = 15
        options.feature_pool_size = 400
        options.num_test_splits = 50
        options.oversampling_translation_jitter = 0.1
        options.be_verbose = True
        options.num_threads = multiprocessing.cpu_count()

        training_xml_path = os.path.join(self.__training_data_file)
        dlib.train_shape_predictor(training_xml_path, self.__new_predictor_name, options)
