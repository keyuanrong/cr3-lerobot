from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .dobot_api import DobotApiDashboard, DobotApiMove

from .config_dobot_cr3 import DobotCR3Config

try:
    from lerobot.robots import Robot
except ImportError:
    class Robot:
        def __init__(self, config):
            self.config = config

if TYPE_CHECKING:
    from .lebai_api import lebai


class DobotCR3(Robot):
    config_class = DobotCR3Config
    name = "dobot_cr3"

    def __init__(self, config: DobotCR3Config):
        super().__init__(config)
        self.config = config
        self.dashboard: DobotApiDashboard | None = None
        self.move: DobotApiMove | None = None
        self.gripper: lebai | None = None
        self.opencv_camera: Any | None = None

    @property
    def observation_features(self) -> dict:
        features = {
            "x.pos": float,
            "y.pos": float,
            "z.pos": float,
            "rx.pos": float,
            "ry.pos": float,
            "rz.pos": float,
            "q1.pos": float,
            "q2.pos": float,
            "q3.pos": float,
            "q4.pos": float,
            "q5.pos": float,
            "q6.pos": float,
            "running_status": float,
            "error_status": float,
            "gripper.pos": float,
            "gripper.torque": float,
        }
        if self.config.use_opencv_camera:
            features[self.config.opencv_camera_name] = (
                self.config.opencv_camera_height,
                self.config.opencv_camera_width,
                3,
            )
        return features

    @property
    def action_features(self) -> dict:
        return {
            "dq1": float,
            "dq2": float,
            "dq3": float,
            "dq4": float,
            "dq5": float,
            "dq6": float,
            "gripper.open": float,
        }

    @property
    def is_connected(self) -> bool:
        robot_connected = self.dashboard is not None and self.move is not None
        if not robot_connected:
            return False
        if self.config.use_gripper:
            return self.gripper is not None and self.gripper.is_connected()
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")
        self._dashboard_checked("SpeedFactor", self.dashboard.SpeedFactor(int(self.config.speed_factor)))

    def connect(self, calibrate: bool = True) -> None:
        self.move = DobotApiMove(self.config.robot_ip, self.config.move_port)
        self.dashboard = DobotApiDashboard(self.config.robot_ip, self.config.dashboard_port)

        self._wait_dashboard_ready()
        if self.config.enable_robot_on_connect:
            self._dashboard_checked("EnableRobot", self.dashboard.EnableRobot())
            self.configure()

        if self.config.use_gripper:
            from .lebai_api import lebai

            self.gripper = lebai()
            self.gripper.configure(
                port=self.config.gripper_port,
                baudrate=self.config.gripper_baudrate,
                timeout=getattr(self.config, "gripper_modbus_timeout", None),
                retries=getattr(self.config, "gripper_modbus_retries", None),
            )
            self.gripper.slave_address = self.config.gripper_slave_address
            self.gripper.connect()

        if self.config.use_opencv_camera:
            self.opencv_camera = self._open_opencv_camera()

    def disconnect(self) -> None:
        if self.opencv_camera is not None:
            self.opencv_camera.release()
            self.opencv_camera = None

        if self.gripper is not None:
            self.gripper.disconnect()
            self.gripper = None

        if self.dashboard is not None:
            try:
                if self.config.disable_robot_on_disconnect:
                    self.dashboard.DisableRobot()
            finally:
                self.dashboard.close()
                self.dashboard = None

        if self.move is not None:
            self.move.close()
            self.move = None

    def get_observation(self) -> dict[str, Any]:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")

        tcp_pose = self.get_pose()
        joints = self.get_joints()
        obs: dict[str, Any] = {
            "x.pos": float(tcp_pose[0]),
            "y.pos": float(tcp_pose[1]),
            "z.pos": float(tcp_pose[2]),
            "rx.pos": float(tcp_pose[3]),
            "ry.pos": float(tcp_pose[4]),
            "rz.pos": float(tcp_pose[5]),
            "q1.pos": float(joints[0]),
            "q2.pos": float(joints[1]),
            "q3.pos": float(joints[2]),
            "q4.pos": float(joints[3]),
            "q5.pos": float(joints[4]),
            "q6.pos": float(joints[5]),
            "running_status": self.get_robot_mode(),
            "error_status": 0.0,
            "gripper.pos": -1.0,
            "gripper.torque": -1.0,
        }

        if self.gripper is not None and self.gripper.is_connected():
            try:
                obs["gripper.pos"] = float(self.gripper.get_position())
                obs["gripper.torque"] = float(self.gripper.get_torque())
            except Exception:
                obs["gripper.pos"] = -1.0
                obs["gripper.torque"] = -1.0

        if self.config.use_opencv_camera:
            obs[self.config.opencv_camera_name] = self.read_opencv_camera()

        return obs

    def read_opencv_camera(self) -> Any:
        if self.opencv_camera is None:
            raise ConnectionError("OpenCV camera is not connected.")

        ok, frame = self.opencv_camera.read()
        if not ok:
            raise RuntimeError("OpenCV camera failed to read a frame.")
        return frame

    def get_pose(self) -> list[float]:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")

        reply = self._dashboard_checked("GetPose", self.dashboard.GetPose())
        return self._parse_dashboard_numbers(reply, expected=6, command="GetPose")

    def get_joints(self) -> list[float]:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")

        reply = self._dashboard_checked("GetAngle", self.dashboard.GetAngle())
        return self._parse_dashboard_numbers(reply, expected=6, command="GetAngle")

    def get_robot_mode(self) -> float:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")

        try:
            values = self._parse_dashboard_numbers(
                self._dashboard_checked("RobotMode", self.dashboard.RobotMode()),
                expected=1,
                command="RobotMode",
            )
            return float(values[0])
        except Exception:
            return -1.0

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.move is None:
            raise ConnectionError("Dobot move interface is not connected.")

        if any(key in action for key in ["dq1", "dq2", "dq3", "dq4", "dq5", "dq6"]):
            return self.send_joint_delta_action(action)

        sent = {
            "x.pos": self._to_float(action["x.pos"]),
            "y.pos": self._to_float(action["y.pos"]),
            "z.pos": self._to_float(action["z.pos"]),
            "rx.pos": self._to_float(action.get("rx.pos", self.config.default_rx)),
            "ry.pos": self._to_float(action.get("ry.pos", self.config.default_ry)),
            "rz.pos": self._to_float(action.get("rz.pos", self.config.default_rz)),
        }

        if self.config.default_motion.lower() == "movj":
            self.move.MovJ(
                sent["x.pos"],
                sent["y.pos"],
                sent["z.pos"],
                sent["rx.pos"],
                sent["ry.pos"],
                sent["rz.pos"],
            )
        else:
            self.move.MovL(
                sent["x.pos"],
                sent["y.pos"],
                sent["z.pos"],
                sent["rx.pos"],
                sent["ry.pos"],
                sent["rz.pos"],
            )

        if "gripper.pos" in action and self.gripper is not None:
            width = int(np.clip(self._to_float(action["gripper.pos"]), 0, 100))
            self.gripper.set_width(width, wait=False)
            sent["gripper.pos"] = float(width)

        return sent

    def send_joint_delta_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.move is None:
            raise ConnectionError("Dobot move interface is not connected.")

        current = self.get_joints()
        deltas = [self._to_float(action.get(f"dq{i}", 0.0)) for i in range(1, 7)]
        target = [joint + delta for joint, delta in zip(current, deltas)]
        reply = self.move.JointMovJ(*target)
        sent = {f"dq{i}": deltas[i - 1] for i in range(1, 7)}
        sent.update({f"q{i}.target": target[i - 1] for i in range(1, 7)})
        sent["joint_reply"] = reply

        if "gripper.open" in action and self.gripper is not None:
            open_value = self._to_float(action["gripper.open"])
            width = 100 if open_value >= 0.5 else 0
            self.gripper.set_width(width, wait=False)
            sent["gripper.open"] = 1.0 if width == 100 else 0.0
            sent["gripper.pos"] = float(width)

        return sent

    def start_jog(self, axis_id: str) -> None:
        if self.move is None:
            raise ConnectionError("Dobot move interface is not connected.")
        self.move.MoveJog(axis_id)

    def stop_jog(self) -> None:
        if self.move is None:
            raise ConnectionError("Dobot move interface is not connected.")
        self.move.MoveJog("")

    def set_api_verbose(self, enabled: bool) -> None:
        for api in [self.dashboard, self.move]:
            if api is not None:
                api.verbose = enabled
        if self.gripper is not None:
            self.gripper.verbose = enabled

    def start_drag(self) -> None:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")
        self._dashboard_checked("StartDrag", self.dashboard.StartDrag())

    def stop_drag(self) -> None:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")
        self._dashboard_checked("StopDrag", self.dashboard.StopDrag())

    def wait_until_stopped(self, timeout_s: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            obs = self.get_observation()
            if obs["running_status"] != 7:
                return True
            time.sleep(0.1)
        return False

    @staticmethod
    def _parse_dashboard_numbers(reply: str, expected: int, command: str) -> list[float]:
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError(f"{command} returned an empty reply: {reply!r}")

        error_code = reply.split(",", 1)[0].strip()
        if error_code and error_code not in {"0", "0.0"}:
            raise RuntimeError(f"{command} failed: {reply}")

        match = re.search(r"\{([^{}]*)\}", reply)
        payload = match.group(1) if match else reply
        numbers = [
            float(item)
            for item in re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", payload
            )
        ]
        if len(numbers) < expected:
            raise ValueError(f"Could not parse {expected} values from {command} reply: {reply}")
        if match is None and len(numbers) > expected:
            numbers = numbers[1:]
        return numbers[:expected]

    def _wait_dashboard_ready(self, timeout_s: float = 20.0) -> None:
        if self.dashboard is None:
            raise ConnectionError("Dobot dashboard is not connected.")

        start = time.time()
        last_reply = ""
        while time.time() - start < timeout_s:
            last_reply = self.dashboard.GetPose()
            error_code = self._dashboard_error_code(last_reply)
            if error_code == 0:
                return
            if error_code != -2:
                time.sleep(1.0)
                continue
            time.sleep(1.0)

        raise RuntimeError(
            "Dobot dashboard is not ready after waiting "
            f"{timeout_s:.0f}s: {last_reply}\n"
            "Expected a normal GetPose reply. If the reply is empty, the controller "
            "closed the TCP connection or is not ready. Check TCP/IP remote mode, "
            "controller state, and then run the script again."
        )

    def _dashboard_checked(self, command: str, reply: str) -> str:
        error_code = self._dashboard_error_code(reply)
        if error_code == 0:
            return reply
        if error_code == -2:
            raise RuntimeError(
                f"{command} failed: {reply}\n"
                "Robot returned -2, usually meaning it is not in TCP/IP remote mode "
                "or the controller is still starting. Check the teach pendant/controller "
                "mode, then run the script again."
            )
        raise RuntimeError(f"{command} failed: {reply}")

    @staticmethod
    def _dashboard_error_code(reply: str) -> int | None:
        if not isinstance(reply, str) or not reply.strip():
            return None
        try:
            return int(float(reply.split(",", 1)[0].strip()))
        except ValueError:
            return None

    def _open_opencv_camera(self) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python") from exc

        backend = self._opencv_backend(cv2, self.config.opencv_camera_backend)
        cap = cv2.VideoCapture(self.config.opencv_camera_index, backend)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open OpenCV camera "
                f"index {self.config.opencv_camera_index} with backend "
                f"{self.config.opencv_camera_backend!r}."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.opencv_camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.opencv_camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.opencv_camera_fps)

        ok = False
        for _ in range(30):
            ok, _frame = cap.read()
            if ok:
                break
            time.sleep(0.05)

        if not ok:
            cap.release()
            raise RuntimeError(
                "OpenCV camera opened, but failed to read a frame. "
                "Close other camera preview windows, or try another backend such as "
                "--backend any or --backend dshow."
            )
        return cap

    @staticmethod
    def _opencv_backend(cv2: Any, name: str) -> int:
        backends = {
            "any": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
        }
        if name not in backends:
            raise ValueError(f"Unsupported OpenCV backend {name!r}. Use one of: any, dshow, msmf.")
        return backends[name]

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)
