from xml.dom import minidom


class XMLHelper:

    def __init__(self):
        pass

    def read_xml(self, file):
        l = []
        grtr = []
        i = 0
        xmldoc = minidom.parse(file)
        itemlist = xmldoc.getElementsByTagName('part')

        for j in range(len(itemlist)):
            if (i < 68):
                l.append([itemlist[j].attributes['x'].value, itemlist[j].attributes['y'].value])
                i += 1

            else:
                grtr.append(l)
                l = []
                l.append([itemlist[j].attributes['x'].value, itemlist[j].attributes['y'].value])
                i = 1

        grtr.append(l)
        return grtr

    def read_small_xml(self, file):
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

    def get_testing_images(self, file):
        xmldoc = minidom.parse(file)
        img = xmldoc.getElementsByTagName('image')
        imgs = []
        for im in img:
            imgs.append(im.attributes['file'].value)

        return imgs


    def write_eye_to_file(self, eyes, file):
        out = open(file, 'w')
        out.write("<dataset>\n")
        out.write("<eyes>\n")

        for eye in eyes:
            out.write(f" <eye>\n")
            for landmark in eye:
                out.write(f"    <part x='{landmark[0]}' y='{landmark[1]}'/>\n")
            out.write(f" </eye>\n")

        out.write(f"</eyes>\n")
        out.write(f"</dataset>\n")
        out.close()

    def read_eye_from_file(self, file):
        xmldoc = minidom.parse(file)
        eye_list = xmldoc.getElementsByTagName('part')
        landmarks = 0
        eye_landmarks = []
        all_eyes = []

        for j in range(len(eye_list)):
            if (landmarks < 6):
                eye_landmarks.append([eye_list[j].attributes['x'].value, eye_list[j].attributes['y'].value])
                landmarks += 1
            else:
                all_eyes.append(eye_landmarks)
                eye_landmarks = [[eye_list[j].attributes['x'].value, eye_list[j].attributes['y'].value]]
                landmarks = 1
        all_eyes.append(eye_landmarks)
        return all_eyes


#xmlhelper = XMLHelper()
#print(xmlhelper.read_eye_from_file('measure_shape_predictor_performance/right_eye_file.xml'))
#print(xmlhelper.read_xml("model/groundTruth.xml"))

# images = xmlhelper.getTestingImages("ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml")
# #
# # predictor_path = "shape_predictor_68_face_landmarks.dat"
# # detectLandmarks(predictor_path, "groundTruth.xml", images)
# # #
# # # predictor_path = "predictor6666.dat"
# # # detectLandmarks(predictor_path, "predictedMax.xml", images)
# # #
# # # #detectLandmarks("predictor1000Tuned.dat", "predicted1000Tuned.xml", images)
# #
# detectLandmarks("../model/predictor2000.dat", "predicte2000.xml", images)
# error = ErrorComputation()
# ground_truth = xmlhelper.read_xml("model/groundTruth.xml")
# predicted2 = xmlhelper.read_xml("xml_predicted/predicted2000.xml")
# error = error.mean_error(predicted2, ground_truth)
# print(error)