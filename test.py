import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

threshold = 60000 # 200 * 300

new_image = cv.imread('test_draw_2.png', cv.IMREAD_UNCHANGED)

img_grey = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

thresh = 251

ret,thresh_img = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)

thresh_img = 255 - thresh_img

image = cv.imread('test_draw_2.png', cv.IMREAD_UNCHANGED)

img_grey = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

ret,thresh_img = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

img_contours = np.zeros(thresh_img.shape)

cv.drawContours(img_contours, contours, -1, (255,255,255), 1)

cv.imwrite('imagescontours.png',img_contours) 

contours_list = list(contours)

fst_cont_x = []
fst_cont_y = []
snd_cont_x = []
snd_cont_y = []
trd_cont_x = []
trd_cont_y = []
fth_cont_x = []
fth_cont_y = []
fith_cont_x = []
fith_cont_y = []
f_cont_x = []
f_cont_y = []

for i in range(len(contours_list[0])):
    fst_cont_x.append(contours_list[0][i][0][0])
    fst_cont_y.append(contours_list[0][i][0][1])

for i in range(len(contours_list[1])):
    snd_cont_x.append(contours_list[1][i][0][0])
    snd_cont_y.append(contours_list[1][i][0][1])
    
for i in range(len(contours_list[2])):
    trd_cont_x.append(contours_list[2][i][0][0])
    trd_cont_y.append(contours_list[2][i][0][1])

plt.scatter(fst_cont_x, fst_cont_y,  color = "blue")
plt.scatter(snd_cont_x, snd_cont_y,  color = "red")
plt.scatter(trd_cont_x, trd_cont_y,  color = "green")

plt.axis('scaled')