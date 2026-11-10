# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    """Convert a torch tensor or warp array to a CPU numpy array."""
    if not isinstance(value, torch.Tensor):
        import warp as wp

        value = wp.to_torch(value)
    return value.detach().cpu().numpy()


class StateTrajectoryLogger:
    """Logs per-step scene state and policy actions of every env during a policy rollout.

    Call ``record(actions)`` once per control step, before ``env.step(actions)``, so each row pairs the
    state the policy acted on with the action it chose; rows are never overwritten by auto-resets.
    ``save(path)`` writes one ``.npz`` with arrays shaped ``(num_steps, num_envs, ...)``: per-asset root
    poses in the env-local frame (world minus env origin), robot body positions (env-local), robot joint
    positions, actions, and each env's ``episode_in_env`` and ``step_in_episode`` indices.
    """

    def __init__(self, env):
        self._base = env.unwrapped
        scene = self._base.scene
        self._rigid_objects = dict(scene.rigid_objects)
        self._articulations = dict(scene.articulations)
        self._robot = self._articulations.get("robot")
        self._rows: dict[str, list[np.ndarray]] = {}

    def _append(self, key: str, value) -> None:
        self._rows.setdefault(key, []).append(_to_numpy(value))

    def record(self, actions: torch.Tensor) -> None:
        """Record the current (pre-step) state of all envs and the actions about to be applied.

        Args:
            actions: Actions about to be passed to ``env.step``, shaped (num_envs, action_dim).
        """
        base = self._base
        origins = _to_numpy(base.scene.env_origins)
        for name, asset in {**self._rigid_objects, **self._articulations}.items():
            pose = _to_numpy(asset.data.root_pose_w).copy()
            pose[:, :3] -= origins
            self._rows.setdefault(f"root_pose/{name}", []).append(pose)
        if self._robot is not None:
            body_pos = _to_numpy(self._robot.data.body_pos_w) - origins[:, None, :]
            self._rows.setdefault("robot/body_pos", []).append(body_pos)
            self._append("robot/joint_pos", self._robot.data.joint_pos)
        self._append("actions", actions)
        self._append("step_in_episode", base.episode_length_buf)
        self._rows.setdefault("episode_in_env", []).append(
            np.array([base.get_episode_index(env_id) for env_id in range(base.num_envs)], dtype=np.int64)
        )

    def save(self, path: str) -> None:
        """Write all recorded rows plus robot body/joint names to ``path`` as a compressed ``.npz``.

        Args:
            path: Output file path.
        """
        arrays = {key.replace("/", "__"): np.stack(rows) for key, rows in self._rows.items()}
        if self._robot is not None:
            arrays["robot__body_names"] = np.array(self._robot.body_names)
            arrays["robot__joint_names"] = np.array(self._robot.joint_names)
        np.savez_compressed(path, **arrays)
        print(f"Saved state trajectory ({len(self._rows.get('actions', []))} steps) to: {path}")
