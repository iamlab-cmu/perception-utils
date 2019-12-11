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
