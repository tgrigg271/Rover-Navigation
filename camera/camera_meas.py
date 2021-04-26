#   This module is meant to mirror camera_meas_pi, which uses a PiCamera and
#   libraries to capture images and perform apriltag detection and pose 
#   extraction. In the absence of a PiCamera, this module uses a pre-recorded
#   image for producing sample outputs.

import numpy as np
import cv2
import apriltag as ap
import time

#   camera calibration
K = [[2.48891613e+03, 0.00000000e+00, 1.64425025e+03],
    [0.00000000e+00, 2.45741844e+03, 1.32156550e+03] ,
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
CAM_CAL_PARAMS = [K[0][0],K[1][1],K[2][0],K[2][1]]
TAG_SIZE = 74 # mm? These units need to be verified
TAG_FAMILY = "tag36h11"

# i/o
IN_IMG_FILE = "camera/img/tmp_image.jpg"
PRINT_OUTPUTS = 0   # print flag
EXPORT_IMAGE = 0

#   tuning parameters
THRESH_LEVEL = 150  # black/white threshold out of 255

def get_camera_measure():
    #   Call will request a measurement from camera

    #
    #   FETCH IMAGE
    #

    #   This module uses a pre-recorded image as a replacement for
    #   PiCamera raw data. The PiCamera libraries cannot apparently be loaded
    #   (at least with ease) onto non-Pi devices. 
    tov = time.time()

    #
    #   PROCESS IMAGE
    #

    #   apply pre-dectection filters on image to make BW
    image = cv2.imread(IN_IMG_FILE)
    greyimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, bwImg = cv2.threshold(greyimage, THRESH_LEVEL, 255, 
        cv2.THRESH_BINARY)

    #   run detection
    options = ap.DetectorOptions(families=TAG_FAMILY)
    detector = ap.Detector(options)
    if EXPORT_IMAGE == 1 :
        results, dimg = detector.detect(bwImg,return_image=True)
    else :
        results = detector.detect(bwImg)

    #   check if results are present
    if len(results) < 1:
        if PRINT_OUTPUTS > 0:
            print("No detection found.")
        return -1  # We may want to modify this if we choose to have multiple outputs
    else:
        if PRINT_OUTPUTS > 0:
            print(len(results)," detections found. ")
    
    #   extract values
    measurements= []  # List will be populated with dicts defining measurement
    for r in results:
        #   pose
        #   The pose matrix is the transpose of T matrix described in 
        #   https://www.mathworks.com/help/images/ref/rigid3d.html
        pose, e0, e1 = detector.detection_pose(r, CAM_CAL_PARAMS, TAG_SIZE)      
        rang = np.sqrt(np.power(pose[0][3],2) + np.power(pose[1][3],2)
            + np.power(pose[2][3],2))
        rang = rang / 1000
        #   Tag ID
        tagid = r.tag_id
        meas = {'pose': pose, 'id': tagid}  # TODO: Add TOV calculation/output
        measurements.append(meas)

        if EXPORT_IMAGE > 0:
            # draw the bounding box of apriltag detection
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)

            # draw the center (x,y) coordinates of Apriltag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

            #   Calculate range and displaty
            rngTxt = "Rng %6.3f" % rang
            cv2.putText(image, rngTxt, (ptA[0], ptA[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if PRINT_OUTPUTS > 0:
            print("Tag ID : ", tagid)


    if EXPORT_IMAGE > 0:
        cv2.imshow("Image",image)
        cv2.waitKey(2)

    #   report pose, tagID, time of validity
    return meas 

