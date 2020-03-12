'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
@file       LaneDetection.py
@date       2020/03/08
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('sample_data1/data/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('video1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()