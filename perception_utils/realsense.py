import pyrealsense2 as rs
from perception import RgbdSensorFactory


def discover_cams():
    """Returns a list of the ids of all cameras connected via USB."""
    ctx = rs.context()
    ctx_devs = list(ctx.query_devices())
    ids = []
    for i in range(ctx.devices.size()):
        ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
    return ids


def get_first_realsense_sensor(rs_cfg):
    rs_cfg['cam_id'] = discover_cams()[0]
    return RgbdSensorFactory.sensor('realsense', rs_cfg)


class RealSenseD405Sensor(RealSenseSensor):

    COLOR_IM_HEIGHT = 720
    COLOR_IM_WIDTH = 1280
    DEPTH_IM_HEIGHT = 720
    DEPTH_IM_WIDTH = 1280
    FPS = 30

    def _config_pipe(self):
        """Configures the pipeline to stream color and depth."""
        self._cfg.enable_device(self.id)

        # configure the color stream
        self._cfg.enable_stream(
            rs.stream.color,
            RealSenseD405Sensor.COLOR_IM_WIDTH,
            RealSenseD405Sensor.COLOR_IM_HEIGHT,
            rs.format.bgr8,
            RealSenseD405Sensor.FPS,
        )

        # configure the depth stream
        self._cfg.enable_stream(
            rs.stream.depth,
            RealSenseD405Sensor.DEPTH_IM_WIDTH,
            RealSenseD405Sensor.DEPTH_IM_HEIGHT,
            rs.format.z16,
            RealSenseD405Sensor.FPS,
        )

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : RealSense color camera intrinsics."""
        return CameraIntrinsics(
            self._frame,
            self._intrinsics[0, 0],
            self._intrinsics[1, 1],
            self._intrinsics[0, 2],
            self._intrinsics[1, 2],
            height=RealSenseD405Sensor.COLOR_IM_HEIGHT,
            width=RealSenseD405Sensor.COLOR_IM_WIDTH,
        )
