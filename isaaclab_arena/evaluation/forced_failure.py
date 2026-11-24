# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import torch


def _to_torch(value) -> torch.Tensor:
    """Return a torch view of a torch tensor or warp array."""
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp

    return wp.to_torch(value)


class ForcedFailurePerturber:
    """Teleports an object by a fixed XY offset the first time the robot's hand closes in each episode.

    Used to evaluate recovery: the first grasp is made to fail, so every episode tests the re-grasp.
    The hand counts as closing when the mean absolute position of the joints named ``joint_prefix*`` first
    exceeds ``close_threshold`` within an episode. Call ``after_step(terminated, truncated)`` once per
    control step, right after ``env.step``.
    """

    def __init__(
        self,
        env,
        object_name: str,
        offset_xy: tuple[float, float],
        joint_prefix: str = "left_hand",
        close_threshold: float = 0.30,
    ):
        self._base = env.unwrapped
        self._object = self._base.scene[object_name]
        robot = self._base.scene["robot"]
        self._robot = robot
        self._hand_ids = [i for i, n in enumerate(robot.joint_names) if n.startswith(joint_prefix)]
        assert self._hand_ids, f"No robot joints start with {joint_prefix!r}"
        self._offset = torch.tensor(offset_xy, device=self._base.device, dtype=torch.float32)
        self._threshold = close_threshold
        self._fired = torch.zeros(self._base.num_envs, dtype=torch.bool, device=self._base.device)
        self.events: list[dict] = []
        """One dict per teleport: env, episode index, step in episode, apple XY before/after (env-local)."""

    def after_step(self, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Re-arm envs that just reset, then teleport the object in envs whose hand closes for the first time.

        Args:
            terminated: Per-env terminated flags returned by ``env.step``.
            truncated: Per-env truncated flags returned by ``env.step``.
        """
        base = self._base
        self._fired[terminated | truncated] = False
        closure = _to_torch(self._robot.data.joint_pos)[:, self._hand_ids].abs().mean(dim=1)
        started = base.episode_length_buf > 0
        trigger = (~self._fired) & started & (closure > self._threshold)
        env_ids = trigger.nonzero().flatten()
        if env_ids.numel() == 0:
            return
        pose = _to_torch(self._object.data.root_pose_w)[env_ids].clone()
        before = pose[:, :2] - base.scene.env_origins[env_ids, :2]
        pose[:, :2] += self._offset
        self._object.write_root_pose_to_sim(pose, env_ids=env_ids)
        self._object.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=base.device), env_ids=env_ids)
        self._fired[env_ids] = True
        for i, env_id in enumerate(env_ids.tolist()):
            self.events.append(
                dict(
                    env_id=env_id,
                    episode_in_env=int(base.get_episode_index(env_id)),
                    step_in_episode=int(base.episode_length_buf[env_id]),
                    xy_before=[round(float(v), 4) for v in before[i]],
                    xy_after=[round(float(v), 4) for v in before[i] + self._offset],
                )
            )

    def save(self, path: str) -> None:
        """Write the teleport events to ``path`` as JSON lines.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        print(f"Saved {len(self.events)} forced-failure events to: {path}")
