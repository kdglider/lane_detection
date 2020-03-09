'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
@file       LaneDetection.py
@date       2020/03/08
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2


'''
@brief      Undistort an image, crop to ROI and remove noise
@param      
@return     
'''
def prepareImage():
    pass


'''
@brief      Warp image to top view of the road and perform edge detection
@param      
@return     
'''
def detectEdges():
    pass


'''
@brief      Identify line candidates using histogram peak detection
@param      
@return     
'''
def getLineCandidates():
    pass


'''
@brief      Fit polynomials to the lane candidate pixels, calculate centerline and  
            radius of curvature
@param      
@return     
'''
def getLanePolynomials():
    pass


'''
@brief      Display the lane lines, lane mesh and radius of curvature on the original frame
@param      
@return     
'''
def visualization():
    pass


'''
@brief      Demonstration function to run the entire lane detection application with video
@param      
@return     
'''
def runApplication():
    pass

