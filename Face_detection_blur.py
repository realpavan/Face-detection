import cv2
import numpy as np


def blur_simple(image1, factor=3):
    ih, iw = image1.shape[:2]
    kh = int(iw / factor)
    kw = int(ih / factor)
    if kw % 2 == 0:
        kw -= 1
    if kh % 2 == 0:
        kh -= 1
    return cv2.GaussianBlur(image1, (kw, kh), 0)


def blur_pixel(image, blocks=3):
    ih, iw = image.shape[:2]
    bh = np.linspace(0, ih, blocks + 1, dtype="int32")
    bw = np.linspace(0, iw, blocks + 1, dtype="int32")
    for i in range(len(bh) - 1):
        for j in range(len(bw) - 1):
            sy = bh[i]
            sx = bw[j]
            ey = bh[i + 1]
            ex = bw[j + 1]

            roi = image[sy:ey, sx:ex]
            color = [c for c in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (sx, sy), (ex, ey), color, -1)
    return image


image = cv2.imread("test1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.04, 5)

for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    face = image[y:y + h, x:x + w]
    image[y:y + h, x:x + w] = blur_pixel(face)


cv2.imshow('blur', image)

cv2.waitKey(0)
