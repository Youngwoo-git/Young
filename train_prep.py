import os, sys, random, shutil
from glob import glob
import pandas as pd
from shutil import copyfile
from sklearn import preprocessing, model_selection
# import matplotlib.pyplot as plt
import cv2
# %matplotlib inline
# from matplotlib import patches
import numpy as np
import ntpath


df = pd.read_csv("temp.csv")
df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=11, shuffle=True)

df_train.to_csv("df_train.csv", index = False)
df_valid.to_csv("df_valid.csv", index = False)
# os.mkdir('content/chip/')
# os.mkdir('content/chip/images/')
# os.mkdir('content/chip/images/train/')
# os.mkdir('content/chip/images/valid/')
#
# os.mkdir('content/chip/labels/')
# os.mkdir('content/chip/labels/train/')
# os.mkdir('content/chip/labels/valid/')

def segregate_data(df, train_img_path, train_label_path):
    filenames = []
    for filename in df.filename:
        filenames.append(filename)
    # test_temp = pd.DataFrame(filenames)
    # test_temp.to_csv("test.csv",mode='a', index=False)
    # print("length of file is:", len(filenames))
    filenames = set(filenames)

    for filename in filenames:
        yolo_list = []

        for _, row in df[df.filename == filename].iterrows():
            yolo_list.append([row.labels, row.x_norm, row.y_norm, row.w_norm, row.h_norm])

        yolo_list = np.array(yolo_list)

        # ("a/b/c")
        txt_filename = os.path.join(train_label_path, str(ntpath.basename(row.filename).replace(".tif", ".txt")))
        txt_filename = txt_filename.replace("\\", "/")
        print(txt_filename)
        # Save the .img & .txt files to the corresponding train and validation folders
        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
        if row.labels == 1:
            img_path = "OK"
        elif row.labels == 0:
            img_path = "NG"
        shutil.copyfile(os.path.join(img_path, ntpath.basename(row.filename)), os.path.join(train_img_path, ntpath.basename(row.filename)))

# src_img_path = "OK"
# src_label_path = "OK"

train_img_path = "content/chip/images/train"
train_label_path = "content/chip/labels/train"

valid_img_path = "content/chip/images/valid"
valid_label_path = "content/chip/labels/valid"

segregate_data(df_train, train_img_path, train_label_path)
segregate_data(df_valid, valid_img_path, valid_label_path)

print("No. of Training images", len(os.listdir('content/chip/images/train')))
print("No. of Training labels", len(os.listdir('content/chip/labels/train')))

print("No. of valid images", len(os.listdir('content/chip/images/valid')))
print("No. of valid labels", len(os.listdir('content/chip/labels/valid')))