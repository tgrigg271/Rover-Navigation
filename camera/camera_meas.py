import numpy as np
import cv2
import apriltag as ap

#   camera calibration
K = [[2.48891613e+03, 0.00000000e+00, 1.64425025e+03],
    [0.00000000e+00, 2.45741844e+03, 1.32156550e+03] ,
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
CAM_CAL_PARAMS = [K[0][0],K[1][1],K[2][0],K[2][1]]
TAG_SIZE = 74 # mm? These units need to be verified
TAG_FAMILY = "tag36h11"

# i/o
IN_IMG_FILE = "img/tmp_image.jpg"
PRINT_OUTPUTS = 1   # print flag
EXPORT_IMAGE = 0

#   tuning parameters
THRESH_LEVEL = 150  # black/white threshold out of 255

def get_camera_measure():
    #   Call will request a measurement from camera

    #   FETCH IMAGE


    #   PROCESS IMAGE
    #   apply pre-dectection filters on image to make BW
    image = cv2.imread(IN_IMG_FILE)
    print(IN_IMG_FILE)
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
        return -1
    
    #   extract values
    for r in results:
        #   pose
        #   The pose matrix is the transpose of T matrix described in 
        #   https://www.mathworks.com/help/images/ref/rigid3d.html
        pose, e0, e1 = detector.detection_pose(r, CAM_CAL_PARAMS, TAG_SIZE)      
        #   Tag ID
        tagid = r.tag_id

    #   report pose, tagID, time of validity
    return 0  

