import glob
import sys
from dataclasses import dataclass, field

try:
    from lerobot.robots import RobotConfig
except ImportError:
    class RobotConfig:
        @classmethod
        def register_subclass(cls, _name):
            def decorator(subclass):
                return subclass

            return decorator


def default_gripper_port() -> str:
    if sys.platform.startswith("win"):
        return "COM5"

    candidates = sorted(glob.glob("/dev/serial/by-id/*"))
    candidates.extend(sorted(glob.glob("/dev/ttyACM*")))
    candidates.extend(sorted(glob.glob("/dev/ttyUSB*")))
    return candidates[0] if candidates else "/dev/ttyACM0"


@RobotConfig.register_subclass("dobot_cr3")
@dataclass
class DobotCR3Config(RobotConfig):
    robot_ip: str = "192.168.5.1"
    dashboard_port: int = 29999
    move_port: int = 30003

    gripper_port: str = field(default_factory=default_gripper_port)
    gripper_baudrate: int = 115200
    gripper_slave_address: int = 1
    gripper_modbus_timeout: float = 0.25
    gripper_modbus_retries: int = 0
    use_gripper: bool = True

    speed_factor: int = 50
    enable_robot_on_connect: bool = True
    disable_robot_on_disconnect: bool = False
    default_rx: float = 179.18
    default_ry: float = -0.10
    default_rz: float = -91.03
    default_motion: str = "MovL"

    use_opencv_camera: bool = False
    opencv_camera_name: str = "front"
    opencv_camera_index: int = 0
    opencv_camera_backend: str = "any"
    opencv_camera_width: int = 640
    opencv_camera_height: int = 480
    opencv_camera_fps: int = 30
