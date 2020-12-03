from imutils import face_utils
import imutils
import dlib
import cv2
import glob
from utils.Constants import Constants


class CreateXMLTrainingSet:
    def __init__(self, dataset_size):
        self.constants = Constants()
        self.predictor_path = "../model/shape_predictor_68_face_landmarks.dat"
        self.image_path = "images/imagine.jpg"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.dataset_size = dataset_size
        self.training_file = "xml_training/trainings" + str(dataset_size) + ".xml"

    def create_training_set(self):

        output_file = open(self.training_file, "w")
        output_file.write("<dataset>\n")
        output_file.write("<images>\n")

        for train_set_path in self.constants.training_sets_paths:
            self.__exctract_images(train_set_path, output_file)

        output_file.write(f"</images>\n")
        output_file.write(f"</dataset>\n")
        output_file.close()

    def __exctract_images(self, train_set_path, output_file):
        images_limit = 0

        for image_path in glob.glob(train_set_path):
            print(image_path)
            if images_limit < self.dataset_size / 5:
                image = cv2.imread(image_path)
                image = imutils.resize(image, width=500)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                images_limit += 1
                rects = self.detector(gray, 1)

                output_file.write(f" <image file='{image_path}'>\n")

                for (i, rect) in enumerate(rects):
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    output_file.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")

                    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    for (j, (x, y)) in enumerate(shape):
                        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                        index = "{0:0=2d}".format(j)
                        output_file.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")


                    output_file.write(f"</box>\n")

                output_file.write(f"</image>\n")


createTrainig = CreateXMLTrainingSet(100)
createTrainig.create_training_set()