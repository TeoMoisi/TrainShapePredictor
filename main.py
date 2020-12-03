from train_shape_predictor.TrainShapePredictor import TrainShapePredictor
from xml_helper.XMLHelper import XMLHelper
from xml_helper.CreateXMLTrainingSet import CreateXMLTrainingSet

class Main:
    def __init__(self):
        pass

    def train_shape_predictor(self, data_size):
        trainShape_predictor = TrainShapePredictor(data_size)
        trainShape_predictor.train_shape_predictor()

    def create_training_set(self, data_size):
        createTrainingSet = CreateXMLTrainingSet(data_size)
        createTrainingSet.create_training_set()