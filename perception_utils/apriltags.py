import numpy as np
import matplotlib.pyplot as plt
import cv2
from apriltag import Detector, DetectorOptions

from autolab_core import Point, RigidTransform


class AprilTagDetector:

    def __init__(self, apriltag_cfg):
        detector_options = DetectorOptions(**apriltag_cfg['detector'])
        self._cfg = apriltag_cfg
        self._detector = Detector(detector_options)

    def detect_from_frames(self, color_im, depth_im, intr, vis=False):
        camera_params = [getattr(intr, v) for v in ['fx', 'fy', 'cx', 'cy']]
        gray_frame = cv2.cvtColor(color_im.data, cv2.COLOR_BGR2GRAY)
        detections, dimg = self._detector.detect(gray_frame, return_image=True)

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
        color_im, depth_im = sensor.frames()[:2]
        return self.detect_from_frames(color_im, depth_im, intr, vis=vis)
