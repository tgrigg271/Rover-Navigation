import camera_meas_pi as cm 
import time
from picamera import PiCamera


cam = PiCamera()

i = 0
while i < 30:
    i = i + 1
    print(" i = ", i)
    a = cm.get_camera_measure(cam)
    time.sleep(0.5)

cam.close()