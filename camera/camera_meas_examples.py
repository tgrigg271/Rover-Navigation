def camera_measurement_wo_pi_example():
#   This shows a use case of get_camera_measure from camera_meas module
    import camera_meas as cm 
    import time

    i = 0
    while i < 3:
        i = i + 1
        print(" i = ", i)
        cm.get_camera_measure()
        time.sleep(0.5)

def camera_measurement_example():
#   This shows a use of get_camera_measure from camera_meas_pi module
    import camera_meas_pi as cm 
    import time
    from picamera import PiCamera

    #   Open camera instance
    cam = PiCamera()

    i = 0
    while i < 30:
        i = i + 1
        print(" i = ", i)
        #   Pass PiCamera pointer into function. get_camera measure will
        #   capture an image and perform Apriltag detection on it.
        a = cm.get_camera_measure(cam)
        time.sleep(0.5)

    #   Close camera instance
    cam.close()


