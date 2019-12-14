import logging
import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform
from visualization import Visualizer3D as vis3d

from perception_utils.realsense import get_first_realsense_sensor
from perception_utils.apriltags import AprilTagDetector


def subsample(data, rate=0.1):
    idx = np.random.choice(np.arange(len(data)), size=int(rate * len(data)))
    return data[idx]


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/examples/get_april_tag_poses.yaml')
    args = parser.parse_args()

    cfg = YamlConfig(args.cfg)
    
    logging.info('Creating realsense sensor')
    sensor = get_first_realsense_sensor(cfg['rs'])
    sensor.start()
    
    logging.info('Creating detector')
    april = AprilTagDetector(cfg['april_tag'])
    
    logging.info('Detecting tags and transforms')
    T_tag_cameras = april.detect(sensor, sensor.color_intrinsics, vis=True)

    for T_tag_camera in T_tag_cameras:
        logging.info('Found: {}'.format(T_tag_camera.from_frame))

    logging.info('Visualizing poses')
    _, depth_im, _ = sensor.frames()
    points = sensor.color_intrinsics.deproject(depth_im)
    T_camera_origin = RigidTransform(from_frame=sensor.frame, to_frame='origin')

    vis3d.figure()
    vis3d.points(subsample(points.data.T), color=(0,1,0), scale=0.002)
    vis3d.pose(T_camera_origin)
    for T_tag_camera in T_tag_cameras:
        vis3d.pose(T_tag_camera, length=0.05, tube_radius=0.002, center_scale=0.005)
    vis3d.show()
    