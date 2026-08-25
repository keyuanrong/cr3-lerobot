#!/usr/bin/env python

"""Build a MuJoCo scene around the CR3 URDF.

This creates:
- sim/cr3_mujoco/generated/cr3_control.urdf with usable joint limits
- sim/cr3_mujoco/cr3_scene.xml with table, cube, target marker, cameras, and actuators

Run from the repository root:

    python sim/cr3_mujoco/build_cr3_scene.py
"""

from __future__ import annotations

from pathlib import Path
import math
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE_URDF = ROOT / "urdf" / "更新的夹爪（一半旋转）v1.0.SLDASM.urdf"
GENERATED_DIR = ROOT / "generated"
CONTROL_URDF = GENERATED_DIR / "cr3_control.urdf"
SCENE_XML = ROOT / "cr3_scene.xml"

TABLE_SIZE = (1.80, 0.70)
TABLE_THICKNESS = 0.03
TABLE_TOP_Z = 0.31
TABLE_WORLD_OFFSET = (-TABLE_SIZE[0] / 2, -TABLE_SIZE[1] / 2)
TABLE_CENTER = (0.0, 0.0, TABLE_TOP_Z - TABLE_THICKNESS / 2)
TABLE_HALF_SIZE = (TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_THICKNESS / 2)
# Desktop coordinates use the table's lower-left corner as (0, 0):
# x goes from left to right along the 180 cm side, y goes from bottom to top along the 70 cm side.
ROBOT_BASE_POS = (
    0.20 + TABLE_WORLD_OFFSET[0],
    0.29 + TABLE_WORLD_OFFSET[1],
    TABLE_TOP_Z,
)
TARGET_OUTER_SIZE = (0.24, 0.20)
TARGET_BORDER_WIDTH = 0.015
TARGET_LOWER_LEFT = (0.71, 0.38)
TARGET_CENTER = (
    TARGET_LOWER_LEFT[0] + TARGET_OUTER_SIZE[0] / 2 + TABLE_WORLD_OFFSET[0],
    TARGET_LOWER_LEFT[1] + TARGET_OUTER_SIZE[1] / 2 + TABLE_WORLD_OFFSET[1],
)
CUBE_SIZE = 0.025
CUBE_TABLE_Z = TABLE_TOP_Z + CUBE_SIZE / 2
CUBE_START_POS = (0.58 + TABLE_WORLD_OFFSET[0], 0.29 + TABLE_WORLD_OFFSET[1], CUBE_TABLE_Z)
YELLOW_CUBE_START_POS = (0.62 + TABLE_WORLD_OFFSET[0], 0.42 + TABLE_WORLD_OFFSET[1], CUBE_TABLE_Z)
GREEN_CUBE_START_POS = (0.76 + TABLE_WORLD_OFFSET[0], 0.28 + TABLE_WORLD_OFFSET[1], CUBE_TABLE_Z)
FIXED_CAMERA_X = 0.81 + TABLE_WORLD_OFFSET[0]
FIXED_CAMERA_Y = TABLE_SIZE[1] + TABLE_WORLD_OFFSET[1]
FIXED_CAMERA_HEIGHT_ABOVE_TABLE = 0.53
FIXED_CAMERA_PITCH_DEG = 25.0
FIXED_CAMERA_SIZE = (0.09, 0.02, 0.03)
WRIST_CAMERA_LINK = "Link6"
# The visible wrist-camera housing now comes from the updated URDF's Link6 mesh.
# Add only an invisible MuJoCo camera node on Link6. The Link6 origin is the
# joint center, so place the optical center forward on the camera/gripper side
# instead of leaving it inside the joint.
WRIST_CAMERA_POS_IN_LINK = (-0.045, -0.09, 0.0)
WRIST_CAMERA_XYAXES = "0 1 0 0 0 1"
GRIPPER_LOOP_SITES = {
    "left": {
        "link2_body": "Left_link2",
        "static_body": "Left_static1",
        "link2_site": "left_link2_pin_site",
        "static_site": "left_static_pin_site",
        "link2_axis_site": "left_link2_axis_site",
        "static_axis_site": "left_static_axis_site",
        "link2_pos": (-0.02693, -0.01376, 0.04404),
        "static_pos": (-0.02000, 0.00211, -0.01333),
        "link2_axis": (0.0, -0.987688340595141, -0.156434465040208),
        "static_axis": (0.0, -0.987688340595141, -0.156434465040210),
        "connect_name": "left_link2_static_connect",
        "axis_connect_name": "left_link2_static_axis_connect",
    },
    "right": {
        "link2_body": "Right_Link2",
        "static_body": "Right_static1",
        "link2_site": "right_link2_pin_site",
        "static_site": "right_static_pin_site",
        "link2_axis_site": "right_link2_axis_site",
        "static_axis_site": "right_static_axis_site",
        "link2_pos": (-0.02693, 0.00052, -0.04613),
        "static_pos": (-0.02000, -0.00211, 0.01333),
        "link2_axis": (0.0, -0.987688340595141, -0.156434465040211),
        "static_axis": (0.0, -0.987688340595141, -0.156434465040210),
        "connect_name": "right_link2_static_connect",
        "axis_connect_name": "right_link2_static_axis_connect",
    },
}
GRIPPER_ACTUATED_JOINTS = {
    "Left_joint1",
    "Right_joint",
}
GRIPPER_USE_CONNECT_CONSTRAINTS = True
GRIPPER_USE_HINGE_AXIS_CONNECTS = True
GRIPPER_USE_WELD_CONSTRAINTS = False
GRIPPER_JOINT_COUPLINGS = {}
GRIPPER_HINGE_AXIS_SITE_OFFSET = 0.008
GRIPPER_LOOP_SOLREF = "0.002 1"
GRIPPER_LOOP_SOLIMP = "0.99 0.999 0.0001"
GRIPPER_WELD_SOLREF = "0.006 1"
GRIPPER_WELD_SOLIMP = "0.95 0.99 0.001"
GRIPPER_JOINT_SOLREF = "0.0015 1"
GRIPPER_JOINT_SOLIMP = "0.995 0.999 0.0005"
SHOW_COLLISION_GEOMS = False
TASK_COLLISION_RGBA = "0 1 0 0.18" if SHOW_COLLISION_GEOMS else "0 1 0 0"
TASK_COLLISION_FRICTION = "1.0 0.05 0.005"
TASK_GRASP_FRICTION = "2.0 0.08 0.008"
TASK_COLLISION_SOLREF = "0.004 1"
TASK_COLLISION_SOLIMP = "0.95 0.99 0.001"
TABLE_CUBE_COLLISION_SOLREF = "0.002 1"
TABLE_CUBE_COLLISION_SOLIMP = "0.98 0.995 0.0005"
TASK_COLLISION_MARGIN = "0.001"

JOINT_LIMITS = {
    "J1": (-2 * math.pi, 2 * math.pi, 1800.0, 2.5),
    "J2": (math.radians(-184.545), math.radians(175.455), 1800.0, 2.5),
    "J3": (math.radians(-277), math.radians(83), 1400.0, 2.5),
    "J4": (math.radians(-172), math.radians(188), 900.0, 3.0),
    "J5": (math.radians(-271), math.radians(89), 600.0, 3.0),
    "J6": (-2 * math.pi, 2 * math.pi, 400.0, 3.0),
    "Left_joint1": (-1.03, 1.03, 12.0, 2.0),
    "Left_static1_joint": (-1.03, 1.03, 12.0, 2.0),
    "Left_joint2": (-1.03, 1.03, 12.0, 2.0),
    "Right_joint": (-1.03, 1.03, 12.0, 2.0),
    "Right_static1_joint": (-1.03, 1.03, 12.0, 2.0),
    "Right_joint2": (-1.03, 1.03, 12.0, 2.0),
}

JOINT_AXES = {}

INITIAL_JOINT_POS_RAD = {
    "J1": -1.5707963267948966,
    "J2": -0.07933394114940226,
    "J3": -1.6929693744344996,
    "J4": 0.13962634015954636,
    "J5": -1.5882496193148399,
    "J6": -4.834562028024293,
}

ACTUATOR_KP = {
    "J1": 1400,
    "J2": 1800,
    "J3": 1500,
    "J4": 800,
    "J5": 550,
    "J6": 400,
    "Left_joint1": 120,
    "Left_static1_joint": 120,
    "Left_joint2": 120,
    "Right_joint": 120,
    "Right_static1_joint": 120,
    "Right_joint2": 120,
}

ACTUATOR_KV = {
    "J1": 140,
    "J2": 180,
    "J3": 150,
    "J4": 80,
    "J5": 55,
    "J6": 40,
    "Left_joint1": 12,
    "Left_static1_joint": 12,
    "Left_joint2": 12,
    "Right_joint": 12,
    "Right_static1_joint": 12,
    "Right_joint2": 12,
}

ACTUATOR_CTRL_RANGES = {
    "Left_joint1": (0.0, 1.03),
    "Right_joint": (-1.03, 0.0),
}

JOINT_DAMPING = {
    "J1": 12.0,
    "J2": 16.0,
    "J3": 14.0,
    "J4": 8.0,
    "J5": 5.0,
    "J6": 4.0,
    "Left_joint1": 0.9,
    "Left_static1_joint": 0.9,
    "Left_joint2": 0.9,
    "Right_joint": 0.9,
    "Right_static1_joint": 0.9,
    "Right_joint2": 0.9,
}

JOINT_FRICTION = {
    "J1": 1.0,
    "J2": 1.2,
    "J3": 1.0,
    "J4": 0.6,
    "J5": 0.4,
    "J6": 0.3,
    "Left_joint1": 0.035,
    "Left_static1_joint": 0.035,
    "Left_joint2": 0.035,
    "Right_joint": 0.035,
    "Right_static1_joint": 0.035,
    "Right_joint2": 0.035,
}


def patch_urdf_limits() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()

    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in JOINT_LIMITS:
            continue
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        lower, upper, effort, velocity = JOINT_LIMITS[name]
        limit.set("lower", str(lower))
        limit.set("upper", str(upper))
        limit.set("effort", str(effort))
        limit.set("velocity", str(velocity))

        if name in JOINT_AXES:
            axis = joint.find("axis")
            if axis is None:
                axis = ET.SubElement(joint, "axis")
            axis.set("xyz", format_vec(JOINT_AXES[name], precision=15))

        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", str(JOINT_DAMPING.get(name, 0.8)))
        dynamics.set("friction", str(JOINT_FRICTION.get(name, 0.05)))

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and filename.startswith("package://"):
            mesh.set("filename", f"../meshes/{Path(filename).name}")

    tree.write(CONTROL_URDF, encoding="utf-8", xml_declaration=True)


def convert_to_mjcf() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(CONTROL_URDF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_saveLastXML(str(SCENE_XML), model)


def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def configure_geom_defaults(root: ET.Element) -> None:
    for default in list(root.findall("default")):
        classes = {child.attrib.get("class") for child in default.findall("default")}
        if default.attrib.get("class") in {"visual_only", "no_visual_collision", "visual_collision"} or classes.intersection(
            {"visual_only", "no_visual_collision", "visual_collision"}
        ):
            root.remove(default)

    default_root = ET.Element("default")

    visual_only = ET.SubElement(default_root, "default", {"class": "visual_only"})
    ET.SubElement(visual_only, "geom", {"contype": "0", "conaffinity": "0", "group": "2"})

    no_visual_collision = ET.SubElement(default_root, "default", {"class": "no_visual_collision"})
    ET.SubElement(
        no_visual_collision,
        "geom",
        {
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
            "rgba": TASK_COLLISION_RGBA,
            "friction": TASK_COLLISION_FRICTION,
            "condim": "4",
            "margin": TASK_COLLISION_MARGIN,
            "solref": TASK_COLLISION_SOLREF,
            "solimp": TASK_COLLISION_SOLIMP,
        },
    )

    visual_collision = ET.SubElement(default_root, "default", {"class": "visual_collision"})
    ET.SubElement(visual_collision, "geom", {"contype": "1", "conaffinity": "1", "group": "2"})

    compiler = root.find("compiler")
    insert_at = list(root).index(compiler) + 1 if compiler is not None else 0
    root.insert(insert_at, default_root)


def format_vec(values: tuple[float, float, float], precision: int = 5) -> str:
    return " ".join(f"{v:.{precision}g}" for v in values)


def offset_vec(
    pos: tuple[float, float, float],
    axis: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float]:
    norm = math.sqrt(sum(v * v for v in axis))
    if norm <= 0.0:
        raise ValueError(f"Invalid zero-length hinge axis: {axis}")
    return tuple(pos[i] + axis[i] / norm * distance for i in range(3))


def camera_xyaxes_from_position_and_target(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> str:
    forward = tuple(target[i] - pos[i] for i in range(3))
    norm = math.sqrt(sum(v * v for v in forward))
    forward = tuple(v / norm for v in forward)
    world_up = (0.0, 0.0, 1.0)
    right = (
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    )
    right_norm = math.sqrt(sum(v * v for v in right))
    right = tuple(v / right_norm for v in right)
    camera_up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    return f"{format_vec(right)} {format_vec(camera_up)}"


def camera_xyaxes_from_forward(forward: tuple[float, float, float]) -> str:
    norm = math.sqrt(sum(v * v for v in forward))
    forward = tuple(v / norm for v in forward)
    world_up = (0.0, 0.0, 1.0)
    right = (
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    )
    right_norm = math.sqrt(sum(v * v for v in right))
    right = tuple(v / right_norm for v in right)
    camera_up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    return f"{format_vec(right)} {format_vec(camera_up)}"


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


def disable_robot_mesh_collisions(worldbody: ET.Element) -> None:
    robot_mount = find_body(worldbody, "cr3_base_mount")
    if robot_mount is None:
        return

    for geom in robot_mount.iter("geom"):
        if geom.attrib.get("type") == "mesh":
            geom.set("class", "visual_only")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "2")


def task_collision_attrs(
    name: str,
    geom_type: str,
    *,
    rgba: str = TASK_COLLISION_RGBA,
    friction: str = TASK_COLLISION_FRICTION,
) -> dict[str, str]:
    return {
        "name": name,
        "type": geom_type,
        "class": "no_visual_collision",
        "rgba": rgba,
        "group": "3",
        "contype": "1",
        "conaffinity": "1",
        "friction": friction,
        "condim": "4",
        "margin": TASK_COLLISION_MARGIN,
        "solref": TASK_COLLISION_SOLREF,
        "solimp": TASK_COLLISION_SOLIMP,
    }


def add_named_geom(body: ET.Element, attrs: dict[str, str]) -> None:
    name = attrs.get("name")
    if name:
        for geom in list(body.findall("geom")):
            if geom.attrib.get("name") == name:
                body.remove(geom)
    ET.SubElement(body, "geom", attrs)


def add_colored_cube(
    worldbody: ET.Element,
    *,
    body_name: str,
    joint_name: str,
    visual_geom_name: str,
    collision_geom_name: str,
    pos: tuple[float, float, float],
    rgba: str,
) -> None:
    cube = ET.SubElement(
        worldbody,
        "body",
        {"name": body_name, "pos": f"{pos[0]} {pos[1]} {pos[2]}"},
    )
    ET.SubElement(cube, "joint", {"name": joint_name, "type": "free", "limited": "false"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": visual_geom_name,
            "class": "visual_only",
            "type": "box",
            "size": f"{CUBE_SIZE / 2} {CUBE_SIZE / 2} {CUBE_SIZE / 2}",
            "rgba": rgba,
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        cube,
        "geom",
        {
            "name": collision_geom_name,
            "class": "no_visual_collision",
            "type": "box",
            "size": f"{CUBE_SIZE / 2} {CUBE_SIZE / 2} {CUBE_SIZE / 2}",
            "mass": "0.008",
            "rgba": TASK_COLLISION_RGBA,
            "group": "3",
            "contype": "1",
            "conaffinity": "1",
            "friction": TASK_GRASP_FRICTION,
            "condim": "4",
            "margin": TASK_COLLISION_MARGIN,
            "solref": TABLE_CUBE_COLLISION_SOLREF,
            "solimp": TABLE_CUBE_COLLISION_SOLIMP,
        },
    )


def add_simple_robot_collisions(worldbody: ET.Element) -> None:
    robot_mount = find_body(worldbody, "cr3_base_mount")
    if robot_mount is None:
        return

    ET.SubElement(
        robot_mount,
        "geom",
        {
            "name": "base_collision",
            "class": "visual_only",
            "type": "cylinder",
            "pos": "0 0 0.035",
            "size": "0.09 0.04",
            "rgba": "0.2 0.2 0.2 0.15",
            "group": "3",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    gripper_proxy_collisions = {
        "Link6": [
            {
                "name": "Link6_or_gripper_base_collision",
                "type": "box",
                "pos": "-0.110 0.004 0",
                "size": "0.038 0.035 0.038",
            }
        ],
        "Left_Link1": [
            {
                "name": "Left_Link1_collision",
                "type": "capsule",
                "fromto": "0 0 0 -0.023 -0.010 0.036",
                "size": "0.007",
            }
        ],
        "Right_Link1": [
            {
                "name": "Right_Link1_collision",
                "type": "capsule",
                "fromto": "0 0 0 -0.023 0.001 -0.036",
                "size": "0.007",
            }
        ],
    }
    for body_name, configs in gripper_proxy_collisions.items():
        body = find_body(robot_mount, body_name)
        if body is None:
            continue
        for config in configs:
            attrs = task_collision_attrs(config["name"], config["type"])
            attrs.update({key: value for key, value in config.items() if key not in {"name", "type"}})
            add_named_geom(body, attrs)

    finger_collision_pos = {
        "Left_static1": "-0.035 0 -0.045",
        "Right_static1": "-0.035 0 0.045",
    }
    for finger_name in ("Left_static1", "Right_static1"):
        finger_body = find_body(robot_mount, finger_name)
        if finger_body is None:
            continue
        attrs = task_collision_attrs(
            f"{finger_name.lower()}_finger_collision",
            "box",
            friction=TASK_GRASP_FRICTION,
        )
        attrs.update({"pos": finger_collision_pos[finger_name], "size": "0.03 0.018 0.006"})
        add_named_geom(finger_body, attrs)


def remove_existing_equality(root: ET.Element) -> None:
    for equality in list(root.findall("equality")):
        root.remove(equality)


def add_loop_site(body: ET.Element, name: str, pos: tuple[float, float, float], rgba: str) -> None:
    for site in list(body.findall("site")):
        if site.attrib.get("name") == name:
            body.remove(site)
    ET.SubElement(
        body,
        "site",
        {
            "name": name,
            "type": "sphere",
            "pos": format_vec(pos, precision=8),
            "size": "0.006",
            "rgba": rgba,
            "group": "2",
        },
    )


def add_gripper_loop_constraints(root: ET.Element) -> None:
    remove_existing_equality(root)
    equality = ET.Element("equality")
    for name, config in GRIPPER_LOOP_SITES.items():
        link2_body_name = config["link2_body"]
        carrier_body_name = config["static_body"]
        link2_body = find_body(root, link2_body_name)
        carrier_body = find_body(root, carrier_body_name)
        if link2_body is None:
            raise ValueError(f"Body not found while building gripper loop constraint: {link2_body_name}")
        if carrier_body is None:
            raise ValueError(f"Body not found while building gripper loop constraint: {carrier_body_name}")
        link2_site_name = config["link2_site"]
        carrier_site_name = config["static_site"]
        add_loop_site(link2_body, link2_site_name, config["link2_pos"], "1 0.15 0.05 1")
        add_loop_site(carrier_body, carrier_site_name, config["static_pos"], "0.05 0.4 1 1")

        link2_axis_site_name = config["link2_axis_site"]
        carrier_axis_site_name = config["static_axis_site"]
        add_loop_site(
            link2_body,
            link2_axis_site_name,
            offset_vec(config["link2_pos"], config["link2_axis"], GRIPPER_HINGE_AXIS_SITE_OFFSET),
            "1 0.55 0.05 1",
        )
        add_loop_site(
            carrier_body,
            carrier_axis_site_name,
            offset_vec(config["static_pos"], config["static_axis"], GRIPPER_HINGE_AXIS_SITE_OFFSET),
            "0.05 0.75 1 1",
        )
        if GRIPPER_USE_WELD_CONSTRAINTS:
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"{name}_site_weld",
                    "site1": link2_site_name,
                    "site2": carrier_site_name,
                    "solref": GRIPPER_WELD_SOLREF,
                    "solimp": GRIPPER_WELD_SOLIMP,
                    "torquescale": "0.01",
                },
            )
        if not GRIPPER_USE_CONNECT_CONSTRAINTS:
            continue
        ET.SubElement(
            equality,
            "connect",
            {
                "name": config["connect_name"],
                "site1": link2_site_name,
                "site2": carrier_site_name,
                "solref": GRIPPER_LOOP_SOLREF,
                "solimp": GRIPPER_LOOP_SOLIMP,
            },
        )
        if GRIPPER_USE_HINGE_AXIS_CONNECTS:
            ET.SubElement(
                equality,
                "connect",
                {
                    "name": config["axis_connect_name"],
                    "site1": link2_axis_site_name,
                    "site2": carrier_axis_site_name,
                    "solref": GRIPPER_LOOP_SOLREF,
                    "solimp": GRIPPER_LOOP_SOLIMP,
                },
            )

    for name, (joint1, joint2, scale) in GRIPPER_JOINT_COUPLINGS.items():
        ET.SubElement(
            equality,
            "joint",
            {
                "name": name,
                "joint1": joint1,
                "joint2": joint2,
                "polycoef": f"0 {scale} 0 0 0",
                "solref": GRIPPER_JOINT_SOLREF,
                "solimp": GRIPPER_JOINT_SOLIMP,
            },
        )

    if len(equality) > 0:
        actuator = root.find("actuator")
        insert_at = list(root).index(actuator) if actuator is not None else len(root)
        root.insert(insert_at, equality)


def place_robot_on_table_left(worldbody: ET.Element) -> None:
    existing_children = list(worldbody)
    if len(existing_children) == 1 and existing_children[0].tag == "body":
        if existing_children[0].attrib.get("name") == "cr3_base_mount":
            existing_children[0].set("pos", f"{ROBOT_BASE_POS[0]} {ROBOT_BASE_POS[1]} {ROBOT_BASE_POS[2]}")
            return

    mount = ET.Element(
        "body",
        {
            "name": "cr3_base_mount",
            "pos": f"{ROBOT_BASE_POS[0]} {ROBOT_BASE_POS[1]} {ROBOT_BASE_POS[2]}",
        },
    )
    for child in existing_children:
        worldbody.remove(child)
        mount.append(child)
    worldbody.append(mount)


def add_scene_objects() -> None:
    tree = ET.parse(SCENE_XML)
    root = tree.getroot()

    asset = ensure_child(root, "asset")
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.attrib.get("file")
        if mesh_file and mesh_file.startswith("../meshes/"):
            mesh.set("file", mesh_file.replace("../meshes/", "meshes/", 1))
        elif mesh_file and mesh_file.startswith("package://"):
            mesh.set("file", f"meshes/{Path(mesh_file).name}")
    for elem in list(asset):
        if elem.tag in {"texture", "material"} and elem.attrib.get("name") in {
            "blue_checker_floor",
            "blue_checker_floor_mat",
        }:
            asset.remove(elem)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "blue_checker_floor",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.18 0.28 0.38",
            "rgb2": "0.08 0.12 0.18",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "blue_checker_floor_mat",
            "texture": "blue_checker_floor",
            "texrepeat": "40 40",
            "reflectance": "0.1",
        },
    )

    compiler = ensure_child(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    configure_geom_defaults(root)

    option = ensure_child(root, "option")
    option.set("timestep", "0.001")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "150")
    option.set("tolerance", "1e-10")

    visual = ensure_child(root, "visual")
    global_visual = ensure_child(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "720")
    headlight = ensure_child(visual, "headlight")
    headlight.set("ambient", "0.45 0.45 0.45")
    headlight.set("diffuse", "0.85 0.85 0.85")
    headlight.set("specular", "0.2 0.2 0.2")

    size = ensure_child(root, "size")
    size.set("njmax", "2000")
    size.set("nconmax", "500")

    worldbody = ensure_child(root, "worldbody")
    place_robot_on_table_left(worldbody)
    disable_robot_mesh_collisions(worldbody)
    add_simple_robot_collisions(worldbody)
    for joint in worldbody.findall(".//joint"):
        joint_name = joint.attrib.get("name")
        if joint_name in INITIAL_JOINT_POS_RAD:
            joint_ref = INITIAL_JOINT_POS_RAD[joint_name]
            joint.set("ref", str(joint_ref))

    ET.SubElement(worldbody, "light", {"name": "top_light", "pos": "0 -0.8 1.8", "dir": "0 0.4 -1"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "20 20 0.02",
            "material": "blue_checker_floor_mat",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.05 0.005",
        },
    )

    table = ET.SubElement(
        worldbody,
        "body",
        {"name": "table", "pos": f"{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_CENTER[2]}"},
    )
    ET.SubElement(
        table,
        "geom",
        {
            "name": "table_top",
            "class": "visual_only",
            "type": "box",
            "size": f"{TABLE_HALF_SIZE[0]} {TABLE_HALF_SIZE[1]} {TABLE_HALF_SIZE[2]}",
            "rgba": "0.45 0.42 0.38 1",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    table_collision = task_collision_attrs("table_top_collision", "box")
    table_collision.update(
        {
            "size": f"{TABLE_HALF_SIZE[0]} {TABLE_HALF_SIZE[1]} {TABLE_HALF_SIZE[2]}",
            "solref": TABLE_CUBE_COLLISION_SOLREF,
            "solimp": TABLE_CUBE_COLLISION_SOLIMP,
        }
    )
    add_named_geom(table, table_collision)
    table_edge_height = 0.055
    table_edge_thickness = 0.012
    table_edge_z = TABLE_HALF_SIZE[2] + table_edge_height
    table_edges = [
        (
            "table_edge_collision_xmin",
            f"{-TABLE_HALF_SIZE[0] - table_edge_thickness} 0 {table_edge_z}",
            f"{table_edge_thickness} {TABLE_HALF_SIZE[1]} {table_edge_height}",
        ),
        (
            "table_edge_collision_xmax",
            f"{TABLE_HALF_SIZE[0] + table_edge_thickness} 0 {table_edge_z}",
            f"{table_edge_thickness} {TABLE_HALF_SIZE[1]} {table_edge_height}",
        ),
        (
            "table_edge_collision_ymin",
            f"0 {-TABLE_HALF_SIZE[1] - table_edge_thickness} {table_edge_z}",
            f"{TABLE_HALF_SIZE[0]} {table_edge_thickness} {table_edge_height}",
        ),
        (
            "table_edge_collision_ymax",
            f"0 {TABLE_HALF_SIZE[1] + table_edge_thickness} {table_edge_z}",
            f"{TABLE_HALF_SIZE[0]} {table_edge_thickness} {table_edge_height}",
        ),
    ]
    for name, pos, size in table_edges:
        attrs = task_collision_attrs(name, "box")
        attrs.update(
            {
                "pos": pos,
                "size": size,
                "solref": TABLE_CUBE_COLLISION_SOLREF,
                "solimp": TABLE_CUBE_COLLISION_SOLIMP,
            }
        )
        add_named_geom(table, attrs)

    target_z = TABLE_TOP_Z + 0.004
    target = ET.SubElement(
        worldbody,
        "body",
        {"name": "target", "pos": f"{TARGET_CENTER[0]} {TARGET_CENTER[1]} {target_z}"},
    )
    target_half_x = TARGET_OUTER_SIZE[0] / 2
    target_half_y = TARGET_OUTER_SIZE[1] / 2
    line_half_width = TARGET_BORDER_WIDTH / 2
    line_half_height = 0.002
    target_lines = [
        (
            "target_bottom",
            f"0 {-target_half_y + line_half_width} 0",
            f"{target_half_x} {line_half_width} {line_half_height}",
        ),
        (
            "target_top",
            f"0 {target_half_y - line_half_width} 0",
            f"{target_half_x} {line_half_width} {line_half_height}",
        ),
        (
            "target_left",
            f"{-target_half_x + line_half_width} 0 0",
            f"{line_half_width} {target_half_y - TARGET_BORDER_WIDTH} {line_half_height}",
        ),
        (
            "target_right",
            f"{target_half_x - line_half_width} 0 0",
            f"{line_half_width} {target_half_y - TARGET_BORDER_WIDTH} {line_half_height}",
        ),
    ]
    for name, pos, size in target_lines:
        ET.SubElement(
            target,
            "geom",
            {
                "name": name,
                "class": "visual_only",
                "type": "box",
                "pos": pos,
                "size": size,
                "rgba": "0.01 0.01 0.01 1",
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    add_colored_cube(
        worldbody,
        body_name="cube",
        joint_name="cube_free",
        visual_geom_name="cube",
        collision_geom_name="cube_collision",
        pos=CUBE_START_POS,
        rgba="0.9 0.15 0.12 1",
    )
    add_colored_cube(
        worldbody,
        body_name="yellow_cube",
        joint_name="yellow_cube_free",
        visual_geom_name="yellow_cube",
        collision_geom_name="yellow_cube_collision",
        pos=YELLOW_CUBE_START_POS,
        rgba="0.95 0.78 0.05 1",
    )
    add_colored_cube(
        worldbody,
        body_name="green_cube",
        joint_name="green_cube_free",
        visual_geom_name="green_cube",
        collision_geom_name="green_cube_collision",
        pos=GREEN_CUBE_START_POS,
        rgba="0.1 0.75 0.25 1",
    )

    fixed_camera_z = TABLE_TOP_Z + FIXED_CAMERA_HEIGHT_ABOVE_TABLE
    fixed_camera_pos = (FIXED_CAMERA_X, FIXED_CAMERA_Y, fixed_camera_z)
    cube_group_center = (
        (CUBE_START_POS[0] + YELLOW_CUBE_START_POS[0] + GREEN_CUBE_START_POS[0]) / 3,
        (CUBE_START_POS[1] + YELLOW_CUBE_START_POS[1] + GREEN_CUBE_START_POS[1]) / 3,
    )
    fixed_camera_target = (
        TARGET_CENTER[0] * 7/8 + cube_group_center[0] * 1/8,
        TARGET_CENTER[1] * 7/8 + cube_group_center[1] * 1/8,
        TABLE_TOP_Z,
    )
    fixed_camera_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "front_camera_body", "pos": format_vec(fixed_camera_pos)},
    )
    ET.SubElement(
        fixed_camera_body,
        "geom",
        {
            "name": "front_camera_body",
            "class": "visual_only",
            "type": "box",
            "size": format_vec(
                (FIXED_CAMERA_SIZE[0] / 2, FIXED_CAMERA_SIZE[1] / 2, FIXED_CAMERA_SIZE[2] / 2)
            ),
            "rgba": "0.01 0.01 0.01 1",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        fixed_camera_body,
        "camera",
        {
            "name": "front",
            "pos": "0 0 0",
            "xyaxes": camera_xyaxes_from_position_and_target(fixed_camera_pos, fixed_camera_target),
            "fovy": "52",
        },
    )

    wrist_body = find_body(worldbody, WRIST_CAMERA_LINK)
    if wrist_body is not None:
        ET.SubElement(
            wrist_body,
            "camera",
            {
                "name": "wrist",
                "pos": format_vec(WRIST_CAMERA_POS_IN_LINK),
                "xyaxes": WRIST_CAMERA_XYAXES,
                "fovy": "70",
            },
        )

    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for joint_name, (lower, upper, _effort, _velocity) in JOINT_LIMITS.items():
        if not (joint_name in INITIAL_JOINT_POS_RAD or joint_name in GRIPPER_ACTUATED_JOINTS):
            continue
        ctrl_lower, ctrl_upper = ACTUATOR_CTRL_RANGES.get(joint_name, (lower, upper))
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_pos",
                "joint": joint_name,
                "kp": str(ACTUATOR_KP.get(joint_name, 80)),
                "kv": str(ACTUATOR_KV.get(joint_name, 0)),
                "ctrlrange": f"{ctrl_lower} {ctrl_upper}",
            },
        )

    add_gripper_loop_constraints(root)

    ET.indent(tree, space="  ")
    tree.write(SCENE_XML, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not SOURCE_URDF.exists():
        raise FileNotFoundError(f"Missing CR3 URDF: {SOURCE_URDF}")

    patch_urdf_limits()
    convert_to_mjcf()
    add_scene_objects()
    print(f"Wrote controllable URDF: {CONTROL_URDF}")
    print(f"Wrote MuJoCo scene: {SCENE_XML}")
    print("Try: python examples/view_my_mujoco.py --mode sine")


if __name__ == "__main__":
    main()
