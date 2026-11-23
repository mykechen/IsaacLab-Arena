# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote client for an OpenWAM policy server driving the G1 WBC embodiment.

The server keeps one stateful OpenWAM rollout per env and returns one model control per request. A model control is
a pair of consecutive 50 Hz actions (the model runs at 25 Hz), so each env queries the server every second step and
executes the buffered second half in between. Inputs and outputs match GR00T M-wide: head-camera RGB plus the 31
upper-body joints (waist, arms, hands) in, the same 31 joints plus base-height and navigate commands out.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import time
import torch
import yaml
from dataclasses import dataclass
from typing import Any

import zmq

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.openwam_wire import pack, unpack
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

UPPER_BODY_GROUPS = ("waist", "left_arm", "left_hand", "right_arm", "right_hand")
"""Policy joint groups in model order; matches observation.state[12:43] of the LeRobot dataset."""

NUM_NAVIGATE_CMD = 3
NUM_BASE_HEIGHT_CMD = 1
NUM_TORSO_ORIENTATION_RPY_CMD = 3


@dataclass
class OpenWamRemotePolicyCfg(PolicyCfg):
    """Configuration for OpenWamRemotePolicy."""

    num_envs: int = 1
    """Number of parallel environments served by the policy."""

    remote_host: str = "localhost"
    """OpenWAM policy server hostname."""

    remote_port: int = 5560
    """OpenWAM policy server port."""

    request_timeout_s: float = 600.0
    """Seconds to wait for one server reply before failing."""

    policy_device: str = "cuda"
    """Device of the returned action tensor."""

    pov_cam_name_sim: str = "robot_head_cam_rgb"
    """Camera observation sent to the server."""

    server_camera_name: str = "observation.images.ego_view"
    """Camera name the server's model was trained with."""

    policy_joints_config_path: str = "isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml"
    """Policy joint groups (group -> joint names) in dataset order."""

    sim_joints_config_path: str = "isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml"
    """Sim joint name -> index, for both the joint-position observation and the action vector."""

    steps_per_control: int = 2
    """Arena steps covered by one model control (50 Hz sim / 25 Hz model)."""


def _load_joints(path: str) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)["joints"]


@register_policy
class OpenWamRemotePolicy(PolicyBase[OpenWamRemotePolicyCfg]):
    """Closed-loop policy whose actions come from a remote OpenWAM server, one stateful rollout per env."""

    name = "openwam_remote"

    def __init__(self, config: OpenWamRemotePolicyCfg):
        super().__init__(config)
        self.num_envs = config.num_envs
        self.device = torch.device(config.policy_device)
        policy_groups = _load_joints(config.policy_joints_config_path)
        self.sim_joints: dict[str, int] = _load_joints(config.sim_joints_config_path)
        self.joint_names = [name for group in UPPER_BODY_GROUPS for name in policy_groups[group]]
        self.sim_index = np.asarray([self.sim_joints[name] for name in self.joint_names], dtype=np.int64)
        self.state_dim = len(self.joint_names)
        self.step_action_dim = self.state_dim + NUM_BASE_HEIGHT_CMD + NUM_NAVIGATE_CMD
        self.sim_action_dim = (
            len(self.sim_joints) + NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
        )

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(config.request_timeout_s * 1000))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{config.remote_host}:{config.remote_port}")
        info = self._request({"cmd": "ping"})
        assert info.get("ok"), f"OpenWAM server at {config.remote_host}:{config.remote_port} did not answer ping"
        assert info["state_dim"] == self.state_dim, f"server state_dim {info['state_dim']} != {self.state_dim}"
        assert (
            info["action_dim"] == config.steps_per_control * self.step_action_dim
        ), f"server action_dim {info['action_dim']} != {config.steps_per_control} x {self.step_action_dim}"
        assert info["num_envs"] >= self.num_envs, f"server serves {info['num_envs']} envs, need {self.num_envs}"
        assert config.server_camera_name in info["camera_names"], info["camera_names"]

        self._buffered: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        """Remaining per-step actions of each env's current model control."""
        self._episode_start = [True] * self.num_envs
        """Whether the next observation sent for an env is the first of a new episode."""
        self.request_wall_s: list[float] = []
        self.task_description: str | None = None

    def _request(self, message: dict) -> dict:
        self._socket.send(pack(message))
        reply = unpack(self._socket.recv())
        assert "error" not in reply, f"OpenWAM server error: {reply['error']}"
        return reply

    def set_task_description(self, task_description: str | None) -> str:
        self.task_description = task_description  # informational; the server holds the task text it was configured with
        return self.task_description

    def _step_to_sim(self, step_action: np.ndarray) -> np.ndarray:
        """Map one 35-D model step action to the 50-D G1 WBC action; legs and torso RPY stay zero as with GR00T."""
        sim = np.zeros(self.sim_action_dim, dtype=np.float32)
        sim[self.sim_index] = step_action[: self.state_dim]
        base_height = step_action[self.state_dim : self.state_dim + NUM_BASE_HEIGHT_CMD]
        navigate = step_action[self.state_dim + NUM_BASE_HEIGHT_CMD :]
        n = len(self.sim_joints)
        sim[n : n + NUM_NAVIGATE_CMD] = navigate
        sim[n + NUM_NAVIGATE_CMD : n + NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD] = base_height
        return sim

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        need = [e for e in range(self.num_envs) if not self._buffered[e]]
        if need:
            rgb = observation["camera_obs"][self.config.pov_cam_name_sim]
            joint_pos = observation["policy"]["robot_joint_pos"]
            rgb_np = rgb[need].detach().cpu().numpy()
            assert rgb_np.dtype == np.uint8, f"expected uint8 camera images, got {rgb_np.dtype}"
            state_np = joint_pos[need].detach().cpu().numpy().astype(np.float32)[:, self.sim_index]
            items = [
                {
                    "env_id": e,
                    "episode_start": self._episode_start[e],
                    "views": {self.config.server_camera_name: np.ascontiguousarray(rgb_np[i, ..., :3])},
                    "state": state_np[i],
                }
                for i, e in enumerate(need)
            ]
            began = time.perf_counter()
            reply = self._request({"cmd": "step", "envs": items})
            self.request_wall_s.append(time.perf_counter() - began)
            for e in need:
                control = np.asarray(reply["actions"][e], dtype=np.float32)
                self._buffered[e] = list(control.reshape(self.config.steps_per_control, self.step_action_dim))
                self._episode_start[e] = False
        actions = np.stack([self._step_to_sim(self._buffered[e].pop(0)) for e in range(self.num_envs)])
        return torch.from_numpy(actions).to(self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = range(self.num_envs) if env_ids is None else [int(e) for e in torch.as_tensor(env_ids).flatten()]
        for e in ids:
            self._buffered[e] = []
            self._episode_start[e] = True

    def server_stats(self) -> dict:
        """Return the server's latency/memory statistics (also written to its --stats-out file)."""
        return self._request({"cmd": "stats"})

    def close(self) -> None:
        if self.request_wall_s:
            ms = np.asarray(self.request_wall_s) * 1e3
            print(
                f"[OpenWamRemotePolicy] {ms.size} requests: median {np.median(ms):.1f} ms, "
                f"p95 {np.percentile(ms, 95):.1f} ms"
            )
        if self._socket is None:
            return
        try:
            self._socket.close(linger=0)
        finally:
            self._context.term()
            self._socket = None
