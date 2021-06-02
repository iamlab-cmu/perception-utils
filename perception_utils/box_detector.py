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

    def detect_from_frames(self, color_im, depth_im, intr, vis=False):
        camera_params = [getattr(intr, v) for v in ['fx', 'fy', 'cx', 'cy']]
        #convert to HSV here
        gray_frame = cv2.cvtColor(color_im.data, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(color_im.data, cv2.COLOR_BGR2HSV)
        hue_only = hsv[:,:,0]
        saturation_only = hsv[:,:,1]
        value_only = hsv[:,:,2]
        plt.imshow(hue_only)
        plt.show()
        import ipdb; ipdb.set_trace()
        hue_mask = np.logical_and(hue_only > self._hue_low, hue_only < self._hue_high)
        plt.imshow(hue_only * hue_mask, cmap="gray")
        plt.show()
        saturation_mask = np.logical_and(saturation_only > self._saturation_low, saturation_only < self._saturation_high)
        value_mask = np.logical_and(value_only > self._value_low, value_only < self._value_high)
        plt.imshow(hue_only * hue_mask * saturation_mask*value_mask, cmap="gray")
        plt.show()

        if vis:
            plt.figure()
            mask_idx = np.argwhere(dimg)
            overlay = color_im.data.copy()
            overlay[mask_idx[:,0], mask_idx[:,1]] = np.array([0, 1, 0]) * 255
            plt.imshow(overlay)
            plt.show()

        T_tag_cameras = []
        for detection in detections:
            M, _, _ = self._detector.detection_pose(detection, camera_params, tag_size=self._cfg['tag_size'])

            det_px_center = np.round(detection.center).astype('int')
            det_px_center_pt = Point(det_px_center, frame=intr.frame)
            det_px_depth = depth_im[det_px_center[1], det_px_center[0]]

            det_px_center_pt_3d = intr.deproject_pixel(det_px_depth, det_px_center_pt)

            T_tag_cameras.append(RigidTransform(
                rotation=M[:3, :3], translation=det_px_center_pt_3d.data,
                from_frame='{}/{}'.format(
                    detection.tag_family.decode(), detection.tag_id
                ), to_frame=intr.frame
            ))

        return T_tag_cameras

    def detect(self, sensor, intr, vis=False):
        color_im, depth_im, _ = sensor.frames()
        return self.detect_from_frames(color_im, depth_im, intr, vis=vis)
