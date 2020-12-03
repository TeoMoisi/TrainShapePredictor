from imutils import face_utils


class Constants:
    def __init__(self):
        # files
        self.right_eye_file = 'xml_cosine_performance/right_eye_file.xml'
        self.left_eye_file = 'xml_cosine_performance/left_eye_file.xml'

        # predictors
        self.predictor_100 = "model/predictor100.dat"
        self.predictor_300 = "model/predictor300.dat"
        self.predictor_500 = "model/predictor500.dat"
        self.predictor_1000 = "model/predictor1000.dat"
        self.predictor_2000 = "model/predictor2000.dat"
        self.predictor_6666 = "model/predictor6666.dat"
        self.shape_predictor_68 = "model/shape_predictor_68_face_landmarks.dat"

        # training sets paths
        pathHelen = "ibug_300W_large_face_landmark_dataset/helen/trainset/*.jpg"
        pathIBug = "ibug_300W_large_face_landmark_dataset/ibug/*.jpg"
        pathLFPWJpg = "ibug_300W_large_face_landmark_dataset/lfpw/trainset/*.jpg"
        pathLFPWPng = "ibug_300W_large_face_landmark_dataset/lfpw/trainset/*.png"
        pathAFWJ = "ibug_300W_large_face_landmark_dataset/afw/*.jpg"
        self.training_sets_paths = [pathHelen, pathIBug, pathLFPWJpg, pathLFPWPng, pathAFWJ]

        self.testing_set = "ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml"

        # colours
        self.YELLOW_COLOR = (0, 255, 255)
        self.RED_COLOR = (0, 0, 255)

        # landmarks indexes
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        (jawStart, jawEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        self.lStart = lStart
        self.lEnd = lEnd
        self.rStart = rStart
        self.rEnd = rEnd
        self.nStart = nStart
        self.nEnd = nEnd
        self.mStart = mStart
        self.mEnd = mEnd
        self.reStart = reStart
        self.reEnd = reEnd
        self.leEnd = leEnd
        self.leStart = leStart
        self.jawStart = jawStart
        self.jawEnd = jawEnd
