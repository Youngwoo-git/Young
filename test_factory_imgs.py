import cv2
import os
import numpy as np
paths = []
for f_path in os.listdir("OK"):
    if f_path.endswith(".tif"):
        paths.append(os.path.join("OK", f_path))
def get_img(paths):
    for path in paths:
        print(path)
        img = cv2.imread(path)
        yield img

def onChange(pos):
    pass
cv2.namedWindow("test")

cv2.createTrackbar("threshold", "test", 0, 255, onChange)
cv2.setTrackbarPos("threshold", "test", 127)
cv2.createTrackbar("canny_thres1", "test", 0, 10000, onChange)
cv2.createTrackbar("canny_thres2", "test", 0, 10000, onChange)

cv2.createTrackbar("img_num", "test", 0, len(paths)-1, lambda x : x)
cv2.setTrackbarPos("img_num", "test", 0)


coordinate = []

# for path in paths:

gen = get_img(paths)
while True:

    thresh = cv2.getTrackbarPos("threshold", "test")
    img_num = cv2.getTrackbarPos("img_num", "test")
    canny_thres1 = cv2.getTrackbarPos("canny_thres1", "test")
    canny_thres2 = cv2.getTrackbarPos("canny_thres2", "test")

    img = cv2.imread(paths[img_num])
    original = img.copy()
    # img = next(gen)
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    convex = original.copy()


    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        hull = cv2.convexHull(i, clockwise=True)
        cv2.drawContours(convex, [hull], 0, (0, 0, 255), 2)
    cv2.imshow("convex", convex)

    dst = original.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, canny_thres1, canny_thres2, apertureSize=5, L2gradient=True)
    cv2.imshow('22', canny)
    largcont = convex.copy()
    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(largcont, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        # c = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(largcont, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cnt = sorted(contours, key=cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(cnt[-1])

        cv2.rectangle(largcont, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        if len(cnt) >= 2:
            x2, y2, w2, h2 = cv2.boundingRect(cnt[-2])
            if w2*h2 > 1500:
                cv2.rectangle(largcont, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

        # draw the biggest contour (c) in green

    cv2.imshow("???", largcont)
    # lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=10, maxLineGap=100)
    # if lines is not None:
    #     str_list = []
    #     for i in lines:
    #         cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
        #     str_list.append(str((i[0][0] + i[0][2]) / 2) + ' ' + str((i[0][1] + i[0][3]) / 2) + ' ' +
        #                 str(abs(i[0][0] - i[0][2]) / 2) + ' ' + str(abs(i[0][1] - i[0][3]) / 2) + '\n')
        # with open(paths[img_num].replace(".jpg", ".txt"), "w") as file:
        #     file.writelines(str_list)
        #     file.close()

            # need x,y,width,height
            # (i[0][0]+i[0][2])/2, (i[0][1]+i[0][3])/2, abs(i[0][0]-i[0][2])/2, abs(i[0][1]-i[0][3])/2




    img = np.vstack([thresh_img, original, dst])

    cv2.imshow('test', img)
    key = cv2.waitKey(1)
    if key == 27:
        break