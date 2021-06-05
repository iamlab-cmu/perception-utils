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
        max_area = -np.inf 
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            if w*h < self._min_area:
                continue
            print("Area", w*h)
            if w*h > max_area:
                max_area  = w*h
            else:
                print("Skipping, found larger box")
                continue
            # draw a green rectangle to visualize the bounding rect
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            # get the min area rect
            rect = cv2.minAreaRect(c)
            print("Center", rect[0])
            print("Angle in degrees", rect[-1])
            orn = np.deg2rad(rect[-1])
            if h > w:
                #Convention: long end goes along the vector
                orn -= np.pi/2
            box = cv2.boxPoints(rect)
            endpoints = _points_to_endpoints(box)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (0, 0, 255), thickness=3)
            #detections.append((rect[0],orn))
            detections = endpoints

        print(f"Detected {len(detections)} over the minimum area of {self._min_area}")
        if vis:
            plt.imshow(thresh, cmap="gray")
            plt.show()
            cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            plt.imshow(img)
            plt.show()
        return detections

    def get_thresh(self, color_im, vis=False):
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
        return thresh, image


    def detect_from_frames(self, color_im, depth_im, intr, vis=False):
        camera_params = [getattr(intr, v) for v in ['fx', 'fy', 'cx', 'cy']]
        #convert to HSV here
        thresh, image = self.get_thresh(color_im, vis=vis)
        endpoints = self.detect_from_thresh(thresh, image, vis=vis)
        T_tag_cameras = []
        for i, endpoint in enumerate(endpoints):
            det_px_center = np.round(endpoint).astype('int')
            det_px_center_pt = Point(det_px_center, frame=intr.frame)
            det_px_depth = depth_im[det_px_center[1], det_px_center[0]]

            det_px_center_pt_3d = intr.deproject_pixel(det_px_depth, det_px_center_pt)
            
            #rotation = RigidTransform.z_axis_rotation(yaw + np.pi/2) #only works with overhead camera really then
            #c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
            #rotation_alt = np.array([[c_yaw, s_yaw, 0],[s_yaw, -c_yaw, 0],[0, 0, -1]])

            T_tag_cameras.append(RigidTransform(
                rotation=np.eye(3), translation=det_px_center_pt_3d.data,
                from_frame=f'pencil_endpoint_{i}_/0'))


        return T_tag_cameras

    def detect(self, sensor, intr, vis=False):
        color_im, depth_im, _ = sensor.frames()
        return self.detect_from_frames(color_im, depth_im, intr, vis=vis)

class InHandBoxDetector(BoxDetector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._min_area = 10

    def detect_from_thresh(self,  thresh, image,vis=False, verbose=True):
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img =  image
        detections = []
        max_area = -np.inf 
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            if w*h < self._min_area:
                continue
            print("Area", w*h)
            if w*h > max_area:
                max_area  = w*h
            else:
                print("Skipping, found larger box")
                continue
            # draw a green rectangle to visualize the bounding rect
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            # get the min area rect
            rect = cv2.minAreaRect(c)
            print("Center", rect[0])
            print("Angle in degrees", rect[-1])
            #check if orn is close to 90
            orn = np.deg2rad(rect[-1])
            if h > w:
                #Convention: long end goes along the vector
                orn -= np.pi/2
            if (np.abs(rect[-1])- 90) > 10:
                print("Not aligned with gripper")
                continue
            box = cv2.boxPoints(rect)
            endpoints = _points_to_endpoints(box)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (0, 0, 255), thickness=3)
            #detections.append((rect[0],orn))
            detections = rect[0], w*h

        print(f"Detected {len(detections)} over the minimum area of {self._min_area}")
        import ipdb; ipdb.set_trace()
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
        import ipdb; ipdb.set_trace()
        thresh, image = self.get_thresh(color_im, vis=vis)
        center, area = self.detect_from_thresh(thresh, image, vis=vis)
        T_tag_cameras = []
        det_px_center = np.round(center).astype('int')
        det_px_center_pt = Point(det_px_center, frame=intr.frame)
        det_px_depth = depth_im[det_px_center[1], det_px_center[0]]
        if det_px_depth > 0.2: #probably on the ground, just visible from the camera
            print("Detection is not in hand")
            print(det_px_depth)
            return []

        det_px_center_pt_3d = intr.deproject_pixel(det_px_depth, det_px_center_pt)
        if area < 50000: #this is where it might not be centered
            #check if it can see the pink, and check if it can see the blue
            #if it can see the pink, call it 8cm. If it can see the blue, call it 10cm. 
            #that's the best we can do with the realsense
            pass
            
        #Very hacky and manual method...
        slope = 0.00012215998849647464 #from other script
        intercept = 0.3821936490386042
        x_offset = slope*area + intercept
        translation = np.array([area,0,x_offset])
        
        #rotation = RigidTransform.z_axis_rotation(yaw + np.pi/2) #only works with overhead camera really then
        #c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        #rotation_alt = np.array([[c_yaw, s_yaw, 0],[s_yaw, -c_yaw, 0],[0, 0, -1]])

        T_tag_cameras.append(RigidTransform(
            rotation=np.eye(3), translation=translation,
            from_frame=f'pencil_inhand/0'))


        return T_tag_cameras


def _points_to_endpoints(pts):
    """
    Returns 2 endpoints 2D as 2d pts by taking the average of the 2 closest
    The rotation component is undefined as the points have no orientation
    """
    if isinstance(pts, list):
        pts = np.array(pts)
    assert(len(pts) == 4)
    anchor_pt = pts[0]
    rest = pts[1:]
    dists = np.linalg.norm(anchor_pt - rest, axis=1)
    min_dist_idx = np.argmin(dists)
    anchor_pair = rest[min_dist_idx]
    rest = np.delete(rest, min_dist_idx, axis=0)
    anchor_endpt = 0.5*(anchor_pt + anchor_pair)
    other_endpt = 0.5*(np.sum(rest, axis=0))
    end_pts = [anchor_endpt, other_endpt]
    return end_pts

print(_points_to_endpoints([[136,186], [173,258], [585, 83], [536,23]]))



