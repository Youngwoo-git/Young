from typing import List, Any

from sklearn import svm, datasets

#
# cancer = datasets.load_breast_cancer()
# print(cancer)
# # classifier = svm.SVC(kernel='linear')
#
# classifier = svm.SVC(gamma=0.001)
# classifier.fit(X_train, y_train)

import pickle
import cv2
import os
import numpy as np
import glob
from skimage.feature import hog
from imutils import paths

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", required=True,
#                 help = "path to input directory of faces+images")
# ap.add_argument("-e", "--encodings", required=True,
#                 help="path to serialized db of facial encodings")
# ap.add_argument("-d", "--detection-method", type=str, default="cnn",
#                 help="face detection model to use: either 'hog' or 'cnn'")
#
# args=vars(ap.parse_args())
#
# imagePaths = list(paths.list_images(args["dataset"]))
# knownEncodings=[]
# knownNames=[]
#
# for (i, imagePath) in enumerate(imagePaths):
#     print("processing image {}/{}".format(i+1,len(imagePaths)))
#     name = imagePath.split(os.path.sep)[-2]
#
#     image = cv2.imread(imagePath)
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
#
#     encodings = face_recognition.face_encodings(rgb, boxes)
#
#     for encoding in encodings:
#         knownEncodings.append(encoding)
#         knownNames.append(name)
#
# data ={"encodings": knownEncodings, "names": knownNames}
# f = open(args["encodings"], "wb")
# f.write(pickle.dumps(data))
# f.close()
directory_list = []
for root, dirs, files in os.walk("archive/train", topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name).replace("\\", "/"))
print(directory_list)

samples=[]
labels=[]

for positive_path in directory_list[:1]:
#     image = cv2.imread(positive_path)
    image_path_list = list(paths.list_images(positive_path))
    for image_path in image_path_list:
        image = cv2.imread(image_path,1)
        cv2.resize(image, (224, 224))
        hist = hog(image)
        samples.append(hist)
        labels.append(1)

for negative_path in directory_list[1:]:
    image_path_list = list(paths.list_images(negative_path))
    for image_path in image_path_list:
        image = cv2.imread(image_path,1)
        cv2.resize(image, (224, 224))
        hist = hog(image)
        samples.append(hist)
        labels.append(0)

test = []
play = [1,2,3]
test.append(play)
print(type(hist))
# print(hist.type())

# samples = np.array([np.float32(j) for j in samples])
samples = np.float32(test)
labels = np.array(labels)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(5.383)
svm.setC(2.67)
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
# print(samples)
# [l.tolist() for l in samples]

# print(samples)
# samples = np.float32(samples)


#

# print("samples", samples)
# [print("labels", labels)]
# positive_path = "archive/train/ben_afflek"
# # img = cv2.imread(positive_path)
# img = cv2.imread(glob.glob(os.path.join(positive_path, '*jpg'))[0])
# # print(img)
# print("anser: ",glob.glob(os.path.join(positive_path, '*jpg'))[0])
#
# print(glob.glob(os.path.join(positive_path, '*jpg'))[0].split(os.path.sep)[-2])
# # negative_path = "archive/train/madonna"
#
# for filename in glob.glob(os.path.join(positive_path, '*.jpg')):
#     img = cv.imread(filename,1)
#     hist = hog(img)
#     samples.append(hist)
#     labels.append(1)