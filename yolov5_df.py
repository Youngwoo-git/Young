import os
import pandas as pd
from sklearn import preprocessing, model_selection
import cv2
import numpy as np
from pathlib import Path

paths = []
condition = []
for f_path in os.listdir("OK"):
    if f_path.endswith(".tif"):
        # condition.append(Path(f_path).parts[0])
        condition.append("OK")
        paths.append(os.path.join("OK", f_path))
for f_path in os.listdir("NG"):
    if f_path.endswith(".tif"):
        # condition.append(Path(f_path).parts[0])
        condition.append("NG")
        paths.append(os.path.join("NG", f_path))
def get_img(paths):
    for path in paths:
        print(path)
        img = cv2.imread(path)
        yield img

# print(paths[0])
# img = cv2.imread(paths[0])
# cv2.imshow("test", img)


df = []
for i_path, i_cond in zip(paths, condition):
    img = cv2.imread(i_path)
    original = img.copy()
    thresh = 108
    canny_thres1, canny_thres2 = 3000, 3000
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    convex = original.copy()

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    dst = original.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, canny_thres1, canny_thres2, apertureSize=5, L2gradient=True)

    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=10, maxLineGap=100)


    #found largest square
    largcont = convex.copy()
    # str_list = []
    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(largcont, contours, -1, 255, 3)

        cnt = sorted(contours, key=cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(cnt[-1])
        row = [i_path, i_cond, x1, y1, w1, h1]
        df.append(row)
        # cv2.rectangle(largcont, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        if len(cnt) >= 2:
            x2, y2, w2, h2 = cv2.boundingRect(cnt[-2])
            if w2 * h2 > 1500:
                # cv2.rectangle(largcont, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                row = [i_path, i_cond, x2, y2, w2, h2]
                df.append(row)


        # find the biggest countour (c) by the area
        # c = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(c)
        # # draw the biggest contour (c) in green
        # cv2.rectangle(largcont, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # # str_list.append(str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
        # row = [i_path, "OK", x, y, w, h]
        # df.append(row)
    # cv2.imshow("???", largcont)

df = pd.DataFrame(df, columns=['filename', 'condition', 'x', 'y', 'width', 'height'])
print(df.head(10))

    # if lines is not None:
    #     str_list = []
    #     for i in lines:
    #         cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
    #         str_list.append(str((i[0][0] + i[0][2]) / 2) + ' ' + str((i[0][1] + i[0][3]) / 2) + ' ' +
    #                         str(abs(i[0][0] - i[0][2]) / 2) + ' ' + str(abs(i[0][1] - i[0][3]) / 2) + '\n')
        # with open(i_path.replace(".jpg", ".txt"), "w") as file:
        #     file.writelines(str_list)
        #     file.close()


    # img = np.vstack([thresh_img, original, dst])
    #
    # cv2.imshow('test', img)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break



img_width = 640
img_height = 240

def w_norm(df):
  return df/img_width
def h_norm(df):
  return df/img_height

le = preprocessing.LabelEncoder()
le.fit(df['condition'])
labels = le.transform(df['condition'])
df['labels'] = labels

df["x_norm"] = df['x'].apply(w_norm)
df["w_norm"] = df["width"].apply(w_norm)
df["y_norm"] = df['y'].apply(h_norm)
df["h_norm"] = df["height"].apply(h_norm)

print(df.head(10))

df.to_csv("temp.csv", index=False)
