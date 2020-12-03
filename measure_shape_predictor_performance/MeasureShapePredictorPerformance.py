from measure_shape_predictor_performance.DetectLandmarks import DetectLandmarks
from xml_helper.XMLHelper import XMLHelper
from errors_computation.ErrorComputation import ErrorComputation
from utils.Constants import Constants


class MeasureShapePredictorPerformance:

    def __init__(self):
        self.xml_helper = XMLHelper()
        self.constants = Constants()
        self.image_list = self.xml_helper.get_testing_images(self.constants.testing_set)
        self.landmarks_detector = DetectLandmarks()
        self.error_computation = ErrorComputation()

    def measure_shape_predictor_performance(self, data_size):
        shape_predictor = "model/predictor" + str(data_size) + ".dat"
        predicted_file = "xml_predicted/predicted" + str(data_size) + ".xml"

        self.landmarks_detector.detect_landmarks(shape_predictor, predicted_file, self.image_list, False)

        ground_truth = self.xml_helper.read_xml("model/groundTruth.xml")
        predicted = self.xml_helper.read_xml(predicted_file)

        return self.error_computation.get_mean_error(predicted, ground_truth)

    def measure_sp_performance_frontal_image(self, image):
        image_list = [image]
        shape_predictor = "model/predictor6666.dat"
        predicted_file = "measure_shape_predictor_performance/predicted_frontal_landmarks.xml"
        self.landmarks_detector.detect_landmarks(shape_predictor, predicted_file, image_list, True,
                                                             save_image_path="measure_shape_predictor_performance/predicted_frontal.jpg")

        shape_predictor_68 = "model/shape_predictor_68_face_landmarks.dat"
        ground_truth_file = "measure_shape_predictor_performance/ground_truth_landmarks.xml"
        self.landmarks_detector.detect_landmarks(shape_predictor_68, ground_truth_file, image_list, True,
                                                             save_image_path="measure_shape_predictor_performance/ground_truth_frontal.jpg")

        predicted = self.xml_helper.read_small_xml(predicted_file)
        ground_truth = self.xml_helper.read_small_xml(ground_truth_file)
        return self.error_computation.compute_mean_error(predicted, ground_truth)


measure_performance = MeasureShapePredictorPerformance()
#measure_performance.measure_shape_predictor_performance(6666)
error = measure_performance.measure_sp_performance_frontal_image("measure_shape_predictor_performance/frontal.jpg")
print(error)
print(sum(error) / 68)