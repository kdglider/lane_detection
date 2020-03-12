'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Girish Ethirajan, Anshuman Singh
@file       LaneDetection.py
@date       2020/03/08
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2


class LaneDetection:
    K = np.array([])
    distortionCoeffs = np.array([])
    cropFactor = 0

    yellowHSVLowBound = np.array([10, 700, 162])
    yellowHSVUpperBound = np.array([65, 180, 255])

    whiteHSVLowBound = np.array([0, 0, 200])
    whiteHSVUpperBound = np.array([170, 25, 255])

    srcCorners = np.array([[480, 350],
                           [810, 350],
                           [230, 512],
                           [935, 512]])
    
    dstCorners = np.array([[0, 0],
                           [700, 0],
                           [0, 700],
                           [700, 700]])
    
    H, _ = cv2.findHomography(srcCorners, dstCorners)


    def __init__(self, K, distortionCoeffs, cropFactor):
        self.K = K
        self.distortionCoeffs = distortionCoeffs
        self.cropFactor = cropFactor


    '''
    @brief      Undistort an image, crop to ROI and remove noise
    @param      
    @return     
    '''
    def prepareImage(self, frame):
        h = frame.shape[0]
        w = frame.shape[1]
        
        srcCorners = np.array([[int(w/2)-1, 0],
                           [int(w/2)+1, 0],
                           [0, h],
                           [w, h]])
    
        dstCorners = np.array([[0, 0],
                           [w, 0],
                           [0, h],
                           [w, h]])

        H, _ = cv2.findHomography(srcCorners, dstCorners)

        newK, roi = cv2.getOptimalNewCameraMatrix(self.K, self.distortionCoeffs, \
                                                 (w, h), 0, (w, h))
        undistortedFrame = cv2.undistort(frame, self.K, self.distortionCoeffs, None, newK)
        
        cropLineY = int(self.cropFactor*h)
        croppedFrame = undistortedFrame[cropLineY:h, :, :]

        #warppedFrame = cv2.warpPerspective(croppedFrame, H, (croppedFrame.shape[1], croppedFrame.shape[0]))
        
        hsvFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2HSV)

        yellowMask = cv2.inRange(hsvFrame, self.yellowHSVLowBound, self.yellowHSVUpperBound) 
        whiteMask = cv2.inRange(hsvFrame, self.whiteHSVLowBound, self.whiteHSVUpperBound)

        laneMask = yellowMask | whiteMask

        #edges = cv2.Canny(laneMask,50,150)

        lines = cv2.HoughLinesP(laneMask, 1, np.pi / 180, 50)
        print(lines)

        #print(lines)

        lineParams = []

        if (lines.all() != None):
            for i in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[i]:
                    
                    #lineParams.append([rho, theta])


                    
                    cv2.line(undistortedFrame,(x1,y1+cropLineY),(x2,y2+cropLineY),(0,0,255),2)

                    lineLength = np.sqrt((x1-x2)**2 + (y1-y2)**2)

                    if (y2 != y1 and x2 != x1):
                        m = (y2-y1) / (x2-x1)
                        b = y1 - m*x1

                        if (m == 0):
                            print(x1)
                            print(x2)
                            print(y1)
                            print(y2)

                        lineParams.append([m,b])
                    else:
                        continue
                    
                    if (y2 > y1):
                        m = (y2-y1) / (x2-x1)
                    elif (y2 < y1):
                        m = (y1-y2) / (x1-x2)
                    else:
                        continue
                    

        lineParams = np.array(lineParams)
        print(lineParams)           
        
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # Apply KMeans
        lineParams = np.float32(lineParams)
        compactness,labels,centers = cv2.kmeans(lineParams, 2, None, criteria, 10, flags)
        print(centers)

        m1 = centers[0,0]
        m2 = centers[1,0]
        b1 = centers[0,1]
        b2 = centers[1,1]

        print(m1)
        print(m2)
        print(b1)
        print(b2)

        x1 = int((h - b1) / m1)
        x2 = int((cropLineY - b1) / m1)
        x3 = int((h - b2) / m2)
        x4 = int((cropLineY - b2) / m2)

        cv2.line(undistortedFrame,(x1,h),(x2,cropLineY), color=(0,255,0), thickness=3)
        cv2.line(undistortedFrame,(x3,h),(x4,cropLineY), color=(0,255,0), thickness=3)

        scale = 1000 / w  # percent of original size
        dim = (int(w * scale), int(h * scale))
        dimCropped = (int(laneMask.shape[1] * scale), int(laneMask.shape[0] * scale))

        cv2.imshow("Frame", cv2.resize(undistortedFrame, dim))
        #cv2.imshow("warp", cv2.resize(warppedFrame, dim))
        cv2.imshow("Frame Undist", cv2.resize(laneMask, dimCropped))
        cv2.waitKey(0)

        return


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



if __name__ == '__main__':
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                   [0.000000e+00, 9.019653e+02, 2.242509e+02],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    distortionCoeffs = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

    '''
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                  [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    distortionCoeffs = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])
    '''

    cropFactor = 0.5
    laneDetector = LaneDetection(K, distortionCoeffs, cropFactor)

    frame = cv2.imread('sampleLane.png')

    
    laneDetector.prepareImage(frame)



