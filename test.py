#!/usr/bin/env python


import cv2

image = cv2.imread('template_8.jpg', 0)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()