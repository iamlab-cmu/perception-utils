import numpy as np
import matplotlib.pyplot as plt
import cv2
from apriltag import Detector, DetectorOptions

from autolab_core import Point, RigidTransform


class BoxDetector:

    def __init__(self, cfg):
        self._hue_low = cfg["hue_low"]
        self._hue_high = cfg["hue_high"]
        self._saturation_low = cfg["saturation_low"] #200
        self._saturation_high = cfg["saturation_high"] #400
        self._value_low = cfg["value_low"] #200
        self._value_high = cfg["value_high"] #400
        self._min_area = cfg["min_area"] # 1000

    def detect_from_thresh(self,  thresh, image,vis=False, verbose=True):
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img =  image
        detections = []
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            if w*h < self._min_area:
                continue
            print("Area", w*h)
            # draw a green rectangle to visualize the bounding rect
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            # get the min area rect
            rect = cv2.minAreaRect(c)
            print("Center", rect[0])
            print("Angle", rect[-1])
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (0, 0, 255), thickness=20)
            detections.append((rect[0], rect[-1]))

        print(f"Detected {len(detections)} over the minimum area of {self._min_area}")
        if vis:
            plt.imshow(thresh, cmap="gray")
            plt.show()
            cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            plt.imshow(img)
            plt.show()
        return detections


    def detect_from_frames(self, color_im, depth_im, intr, vis=False):
        camera_params = [getattr(intr, v) for v in ['fx', 'fy', 'cx', 'cy']]
        #convert to HSV here
        gray_frame = cv2.cvtColor(color_im.data, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_im.data, cv2.COLOR_BGR2HSV)
        image = color_im.data.copy() #TODO make color?


        hue_only = hsv[:,:,0]
        saturation_only = hsv[:,:,1]
        value_only = hsv[:,:,2]

        hue_mask = np.logical_and(hue_only > self._hue_low, hue_only < self._hue_high)
        saturation_mask = np.logical_and(saturation_only > self._saturation_low, saturation_only < self._saturation_high)
        value_mask = np.logical_and(value_only > self._value_low, value_only < self._value_high)
        thresh = hue_only * hue_mask * saturation_mask*value_mask
        vis=0
        detections = self.detect_from_thresh(thresh, image, vis=vis)
        import ipdb; ipdb.set_trace()
        T_tag_cameras = []
        for detection in detections:
            center, yaw = detection
            det_px_center = np.round(center).astype('int')
            det_px_center_pt = Point(det_px_center, frame=intr.frame)
            det_px_depth = depth_im[det_px_center[1], det_px_center[0]]

            det_px_center_pt_3d = intr.deproject_pixel(det_px_depth, det_px_center_pt)
            rotation = RigidTransform.z_axis_rotation(-yaw) #only works with overhead camera really then

            T_tag_cameras.append(RigidTransform(
                rotation=rotation, translation=det_px_center_pt_3d.data,
                from_frame='pencil/0'))

        return T_tag_cameras

    def detect(self, sensor, intr, vis=False):
        color_im, depth_im, _ = sensor.frames()
        return self.detect_from_frames(color_im, depth_im, intr, vis=vis)
