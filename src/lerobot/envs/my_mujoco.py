#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.types import RobotObservation


DEFAULT_SCENE_XML = """
<mujoco model="lerobot_my_mujoco">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <joint damping="0.8" armature="0.02" limited="false"/>
    <geom friction="1.0 0.1 0.01"/>
    <default class="arm">
      <geom type="capsule" size="0.025" rgba="0.25 0.35 0.85 1"/>
      <joint limited="true" range="-2.6 2.6"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.82 0.84" rgb2="0.58 0.6 0.62"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 -0.6 1.7" dir="0 0.4 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.02" material="grid"/>
    <body name="table" pos="0.45 0 0.28">
      <geom name="table_top" type="box" size="0.45 0.35 0.03" rgba="0.45 0.42 0.38 1"/>
    </body>
    <body name="target" pos="0.58 0.12 0.335">
      <geom name="target_marker" type="cylinder" size="0.045 0.003" rgba="0.1 0.7 0.35 0.45" contype="0" conaffinity="0"/>
    </body>
    <body name="cube" pos="0.45 -0.08 0.34">
      <joint name="cube_free" type="free" limited="false"/>
      <geom name="cube" type="box" size="0.03 0.03 0.03" mass="0.06" rgba="0.9 0.15 0.12 1"/>
    </body>

    <body name="base" pos="0 0 0.34">
      <geom name="base" type="cylinder" size="0.06 0.04" rgba="0.1 0.1 0.12 1"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="joint1" type="hinge" axis="0 0 1"/>
        <geom class="arm" fromto="0 0 0 0.08 0 0.08"/>
        <body name="link2" pos="0.08 0 0.08">
          <joint name="joint2" type="hinge" axis="0 1 0"/>
          <geom class="arm" fromto="0 0 0 0.12 0 0"/>
          <body name="link3" pos="0.12 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0"/>
            <geom class="arm" fromto="0 0 0 0.11 0 0"/>
            <body name="link4" pos="0.11 0 0">
              <joint name="joint4" type="hinge" axis="1 0 0"/>
              <geom class="arm" fromto="0 0 0 0.1 0 0"/>
              <body name="link5" pos="0.1 0 0">
                <joint name="joint5" type="hinge" axis="0 1 0"/>
                <geom class="arm" fromto="0 0 0 0.08 0 0"/>
                <body name="link6" pos="0.08 0 0">
                  <joint name="joint6" type="hinge" axis="1 0 0"/>
                  <geom class="arm" fromto="0 0 0 0.06 0 0"/>
                  <body name="eef" pos="0.06 0 0">
                    <joint name="joint7" type="hinge" axis="0 1 0"/>
                    <geom name="eef" type="sphere" size="0.035" rgba="0.05 0.05 0.06 1"/>
                    <site name="eef_site" pos="0.035 0 0" size="0.012" rgba="0.1 0.8 0.95 1"/>
                    <camera name="wrist" pos="-0.03 0 0.05" xyaxes="0 -1 0 0.55 0 0.83" fovy="70"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="front" pos="0.95 -0.65 0.75" xyaxes="0.7 0.7 0 -0.35 0.35 0.87" fovy="45"/>
  </worldbody>

  <actuator>
    <position name="joint1_pos" joint="joint1" kp="80" ctrlrange="-2.6 2.6"/>
    <position name="joint2_pos" joint="joint2" kp="80" ctrlrange="-2.0 2.0"/>
    <position name="joint3_pos" joint="joint3" kp="80" ctrlrange="-2.0 2.0"/>
    <position name="joint4_pos" joint="joint4" kp="60" ctrlrange="-2.6 2.6"/>
    <position name="joint5_pos" joint="joint5" kp="60" ctrlrange="-2.0 2.0"/>
    <position name="joint6_pos" joint="joint6" kp="40" ctrlrange="-2.6 2.6"/>
    <position name="joint7_pos" joint="joint7" kp="40" ctrlrange="-2.0 2.0"/>
  </actuator>
</mujoco>
"""


class MyMujocoEnv(gym.Env):
    """Small MuJoCo manipulation environment for wiring custom robot scenes into LeRobot."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str | None = None,
        task: str = "pick_cube",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        camera_name: str = "front",
        observation_height: int = 480,
        observation_width: int = 640,
        action_dim: int = 7,
        state_dim: int = 7,
        episode_length: int = 300,
        frame_skip: int = 10,
        action_scale: float = 1.0,
        target_body: str = "target",
        object_body: str = "cube",
        success_distance: float = 0.06,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.episode_length = episode_length
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.target_body = target_body
        self.object_body = object_body
        self.success_distance = success_distance

        self._mujoco = None
        self._model = None
        self._data = None
        self._renderer = None
        self._step_count = 0
        self._home_ctrl: np.ndarray | None = None
        self.task_description = "Move the cube to the target marker."

        spaces_dict: dict[str, spaces.Space] = {}
        if self.obs_type in ("pixels", "pixels_agent_pos"):
            spaces_dict["pixels"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
        if self.obs_type in ("state", "pixels_agent_pos"):
            spaces_dict["agent_pos"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32,
            )
        if not spaces_dict:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self.observation_space = spaces.Dict(spaces_dict)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def _ensure_env(self) -> None:
        if self._model is not None:
            return

        try:
            import mujoco
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "The 'mujoco' package is required for --env.type=my_mujoco. "
                "Install it with `uv pip install mujoco`."
            ) from e

        self._mujoco = mujoco
        if self.model_path:
            path = Path(self.model_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"MuJoCo model_path does not exist: {path}")
            self._model = mujoco.MjModel.from_xml_path(str(path))
        else:
            self._model = mujoco.MjModel.from_xml_string(DEFAULT_SCENE_XML)

        self._data = mujoco.MjData(self._model)
        if self._model.nu != self.action_dim:
            raise ValueError(
                f"Configured action_dim={self.action_dim}, but MuJoCo model has {self._model.nu} actuators. "
                "Set --env.action_dim to match your actuator count or update the model actuators."
            )

        self._home_ctrl = np.zeros(self.action_dim, dtype=np.float32)
        if self._model.nu:
            self._home_ctrl[:] = self._joint_position_ctrl_from_qpos()

    def _render_pixels(self) -> np.ndarray:
        self._ensure_env()
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self._model,
                height=self.observation_height,
                width=self.observation_width,
            )
        self._renderer.update_scene(self._data, camera=self.camera_name)
        return self._renderer.render().copy()

    def _agent_pos(self) -> np.ndarray:
        self._ensure_env()
        qpos_values = []
        for actuator_id in range(self._model.nu):
            joint_id = int(self._model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_type = int(self._model.jnt_type[joint_id])
            if joint_type not in (self._mujoco.mjtJoint.mjJNT_HINGE, self._mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            qpos_addr = int(self._model.jnt_qposadr[joint_id])
            qpos_values.append(float(self._data.qpos[qpos_addr]))

        if not qpos_values:
            qpos_values = np.asarray(self._data.qpos, dtype=np.float32).tolist()

        qpos = np.asarray(qpos_values, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        n = min(self.state_dim, qpos.shape[0])
        state[:n] = qpos[:n]
        return state

    def _joint_position_ctrl_from_qpos(self) -> np.ndarray:
        ctrl = np.zeros(self.action_dim, dtype=np.float32)
        for actuator_id in range(self._model.nu):
            joint_id = int(self._model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_type = int(self._model.jnt_type[joint_id])
            if joint_type not in (self._mujoco.mjtJoint.mjJNT_HINGE, self._mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            qpos_addr = int(self._model.jnt_qposadr[joint_id])
            ctrl[actuator_id] = float(self._data.qpos[qpos_addr])

        if self._model.nu:
            limited = np.asarray(self._model.actuator_ctrllimited, dtype=bool)
            if limited.any():
                low = self._model.actuator_ctrlrange[:, 0]
                high = self._model.actuator_ctrlrange[:, 1]
                ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])
        return ctrl

    def _format_obs(self) -> RobotObservation:
        obs: RobotObservation = {}
        if self.obs_type in ("pixels", "pixels_agent_pos"):
            obs["pixels"] = self._render_pixels()
        if self.obs_type in ("state", "pixels_agent_pos"):
            obs["agent_pos"] = self._agent_pos()
        return obs

    def _body_pos(self, name: str) -> np.ndarray | None:
        self._ensure_env()
        try:
            body_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            return None
        if body_id < 0:
            return None
        return np.asarray(self._data.xpos[body_id], dtype=np.float32)

    def _success_and_reward(self) -> tuple[bool, float]:
        obj_pos = self._body_pos(self.object_body)
        target_pos = self._body_pos(self.target_body)
        if obj_pos is None or target_pos is None:
            return False, 0.0
        distance = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        is_success = distance < self.success_distance
        reward = float(is_success) - distance
        return is_success, reward

    def reset(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[RobotObservation, dict[str, Any]]:
        self._ensure_env()
        super().reset(seed=seed)
        self._mujoco.mj_resetData(self._model, self._data)
        self._step_count = 0
        if self._home_ctrl is not None and self._model.nu:
            self._home_ctrl[:] = self._joint_position_ctrl_from_qpos()
            self._data.ctrl[:] = self._home_ctrl
        self._mujoco.mj_forward(self._model, self._data)
        return self._format_obs(), {"is_success": False, "task": self.task}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {action.shape}.")

        ctrl = self._home_ctrl + np.clip(action, -1.0, 1.0) * self.action_scale
        if self._model.nu:
            limited = np.asarray(self._model.actuator_ctrllimited, dtype=bool)
            if limited.any():
                low = self._model.actuator_ctrlrange[:, 0]
                high = self._model.actuator_ctrlrange[:, 1]
                ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])
            self._data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            self._mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        is_success, reward = self._success_and_reward()
        terminated = bool(is_success)
        truncated = self._step_count >= self.episode_length
        info = {
            "is_success": is_success,
            "task": self.task,
            "step": self._step_count,
        }
        if terminated or truncated:
            info["final_info"] = dict(info)

        return self._format_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._render_pixels()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._data = None
        self._model = None


def create_my_mujoco_envs(
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    fns = [(lambda kwargs=gym_kwargs: MyMujocoEnv(**kwargs)) for _ in range(n_envs)]
    return {"my_mujoco": {0: env_cls(fns)}}
