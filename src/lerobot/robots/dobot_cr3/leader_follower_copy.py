# -*- coding: utf-8 -*-
import re
import time

from .dobot_api import DobotApiDashboard, DobotApiMove


LEADER_IP = "192.168.5.1"
FOLLOWER_IP = "192.168.6.1"

DASHBOARD_PORT = 29999
MOVE_PORT = 30003

SERVO_TIME = 0.05
LOOKAHEAD_TIME = 30
GAIN = 300

LEADER_SPEED = 10
FOLLOWER_SPEED = 30

ALIGN_FOLLOWER_TO_LEADER_ON_START = True
ALIGN_SPEED = 20
ALIGN_THRESHOLD_DEG = 1.0

MAX_STEP_DEG = 2.0
SMOOTH_ALPHA = 1.0
DELTA_DEADBAND = 0.02

USE_JOINT_LIMITS = True
JOINT_LIMITS = [
    (-350.0, 350.0),
    (-107.0, 92.0),
    (-140.0, 140.0),
    (-178.0, 127.0),
    (-178.0, 178.0),
    (-350.0, 350.0),
]

JOINT_SCALE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

ROBOT_ERROR_HINTS = {
    116: "emergency stop button is pressed",
    121: "failed to initialize controller",
}


def parse_robot_values(reply):
    match = re.search(r"\{([^}]*)\}", reply)
    if not match:
        raise ValueError(f"Can not parse robot reply: {reply!r}")

    values = [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", match.group(1))]
    if len(values) < 6:
        raise ValueError(f"Reply has fewer than 6 joint values: {reply!r}")
    return values[:6]


def reply_ok(reply):
    return reply.strip().startswith("0,")


def extract_error_ids(reply):
    return [
        int(value)
        for value in re.findall(r"-?\d+", reply.split("GetErrorID", 1)[0])
        if int(value) != 0
    ]


def format_error_hints(error_ids):
    hints = [
        f"{error_id}: {ROBOT_ERROR_HINTS.get(error_id, 'unknown error')}"
        for error_id in error_ids
    ]
    return "; ".join(hints) if hints else "none"


def normalize_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def normalize_joints(joints):
    return [normalize_angle(value) for value in joints]


def shortest_angle_delta(current, previous):
    return normalize_angle(current - previous)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_joint_limits(joints):
    if not USE_JOINT_LIMITS:
        return joints
    return [
        clamp(value, lower, upper)
        for value, (lower, upper) in zip(joints, JOINT_LIMITS)
    ]


def vector_sub_delta(current, previous):
    return [
        shortest_angle_delta(current_value, previous_value)
        for current_value, previous_value in zip(current, previous)
    ]


def max_abs(values):
    return max(abs(value) for value in values)


def nearest_equivalent_angle(angle, reference):
    return angle + round((reference - angle) / 360.0) * 360.0


def make_continuous_target(target, reference):
    return [
        nearest_equivalent_angle(target_value, reference_value)
        for target_value, reference_value in zip(target, reference)
    ]


def scaled_leader_target(leader_start, follower_start, leader_current):
    target = []
    for index, (start, current, follower_value) in enumerate(
        zip(leader_start, leader_current, follower_start)
    ):
        delta = shortest_angle_delta(current, start) * JOINT_SCALE[index]
        target.append(follower_value + delta)
    return target


def limit_step(current, target):
    limited = []
    for current_value, target_value in zip(current, target):
        delta = target_value - current_value
        delta = clamp(delta, -MAX_STEP_DEG, MAX_STEP_DEG)
        limited.append(current_value + delta)
    return limited


def smooth_target(previous_target, raw_target):
    return [
        previous + (current - previous) * SMOOTH_ALPHA
        for previous, current in zip(previous_target, raw_target)
    ]


def connect_robot(ip):
    dashboard = DobotApiDashboard(ip, DASHBOARD_PORT)
    move = DobotApiMove(ip, MOVE_PORT)
    dashboard.log = lambda _text: None
    move.log = lambda _text: None
    return dashboard, move


class LeaderFollowerCopyController:
    """Reusable leader-follower controller for dataset recording scripts."""

    def __init__(
        self,
        auto_enable=True,
        start_drag=True,
        leader_ip=LEADER_IP,
        follower_ip=FOLLOWER_IP,
        align_on_start=ALIGN_FOLLOWER_TO_LEADER_ON_START,
        leader_dashboard=None,
        leader_move=None,
    ):
        self.auto_enable = auto_enable
        self.start_drag = start_drag
        self.leader_ip = leader_ip
        self.follower_ip = follower_ip
        self.align_on_start = align_on_start
        self.leader_dashboard = leader_dashboard
        self.leader_move = leader_move
        self.owns_leader_connection = leader_dashboard is None or leader_move is None
        self.follower_dashboard = None
        self.follower_move = None
        self.leader_drag_started = False
        self.leader_start = None
        self.follower_start = None
        self.follower_target = None
        self.follower_smoothed = None

    def connect(self):
        if self.owns_leader_connection:
            self.leader_dashboard, self.leader_move = connect_robot(self.leader_ip)
        self.follower_dashboard, self.follower_move = connect_robot(self.follower_ip)

        if self.auto_enable:
            self.follower_dashboard.ClearError()
            if self.owns_leader_connection:
                self.leader_dashboard.ClearError()
                leader_enable = self.leader_dashboard.EnableRobot()
                if not reply_ok(leader_enable):
                    raise RuntimeError(f"Leader EnableRobot failed: {leader_enable.strip()}")

            follower_enable = self.follower_dashboard.EnableRobot()
            if not reply_ok(follower_enable):
                mode = self.follower_dashboard.RobotMode().strip()
                errors_reply = self.follower_dashboard.GetErrorID().strip()
                error_ids = extract_error_ids(errors_reply)
                raise RuntimeError(
                    "Follower EnableRobot failed: "
                    f"{follower_enable.strip()} | mode={mode} | errors={errors_reply} "
                    f"| hints={format_error_hints(error_ids)}"
                )

        if self.owns_leader_connection:
            self.leader_dashboard.SpeedFactor(LEADER_SPEED)
        self.follower_dashboard.SpeedFactor(FOLLOWER_SPEED)

        self.leader_start = self.get_leader_joints()
        follower_joints = self.get_follower_joints()

        if self.align_on_start:
            follower_joints = self.align_follower_to_leader(self.leader_start, follower_joints)

        self.leader_start = self.get_leader_joints()
        self.follower_start = self.get_follower_joints()
        self.follower_target = self.follower_start[:]
        self.follower_smoothed = self.follower_target[:]

        if self.start_drag:
            reply = self.leader_dashboard.StartDrag()
            self.leader_drag_started = reply.strip().startswith("0,")
            if not self.leader_drag_started:
                raise RuntimeError(f"StartDrag failed: {reply.strip()}")

    def get_leader_joints(self):
        return normalize_joints(parse_robot_values(self.leader_dashboard.GetAngle()))

    def get_follower_joints(self):
        return normalize_joints(parse_robot_values(self.follower_dashboard.GetAngle()))

    def align_follower_to_leader(self, leader_joints, follower_joints):
        max_delta = max_abs(vector_sub_delta(leader_joints, follower_joints))
        if max_delta <= ALIGN_THRESHOLD_DEG:
            return follower_joints

        self.follower_dashboard.SpeedFactor(ALIGN_SPEED)
        self.follower_move.JointMovJ(
            leader_joints[0],
            leader_joints[1],
            leader_joints[2],
            leader_joints[3],
            leader_joints[4],
            leader_joints[5],
        )
        self.follower_move.Sync()
        self.follower_dashboard.SpeedFactor(FOLLOWER_SPEED)
        return self.get_follower_joints()

    def step(self):
        leader_current = self.get_leader_joints()
        leader_delta = vector_sub_delta(leader_current, self.leader_start)

        raw_target = scaled_leader_target(
            self.leader_start,
            self.follower_start,
            leader_current,
        )
        raw_target = make_continuous_target(raw_target, self.follower_smoothed)
        raw_target = clamp_joint_limits(raw_target)
        follower_limited = limit_step(self.follower_smoothed, raw_target)
        self.follower_smoothed = smooth_target(self.follower_smoothed, follower_limited)

        moved = max_abs([
            current - previous
            for current, previous in zip(self.follower_smoothed, self.follower_target)
        ]) >= DELTA_DEADBAND

        if moved:
            reply = self.follower_move.ServoJ(
                self.follower_smoothed[0],
                self.follower_smoothed[1],
                self.follower_smoothed[2],
                self.follower_smoothed[3],
                self.follower_smoothed[4],
                self.follower_smoothed[5],
                t=SERVO_TIME,
                lookahead_time=LOOKAHEAD_TIME,
                gain=GAIN,
            )
            if not reply.strip().startswith("0,"):
                raise RuntimeError(f"ServoJ failed: {reply.strip()}")
            self.follower_target = self.follower_smoothed[:]

        follower_actual = self.get_follower_joints()
        return {
            "leader_joints": leader_current,
            "leader_delta": leader_delta,
            "action_target": self.follower_target[:],
            "follower_actual": follower_actual,
            "timestamp": time.time(),
        }

    def close(self):
        if self.leader_drag_started:
            try:
                self.leader_dashboard.StopDrag()
            except Exception:
                pass

        if self.follower_move is not None:
            try:
                self.follower_move.Sync()
            except Exception:
                pass

        clients = [self.follower_dashboard, self.follower_move]
        if self.owns_leader_connection:
            clients.extend([self.leader_dashboard, self.leader_move])

        for client in clients:
            if client is not None:
                client.close()
