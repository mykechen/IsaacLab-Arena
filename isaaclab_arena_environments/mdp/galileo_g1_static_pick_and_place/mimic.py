# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic support for the G1 static apple-to-plate task (left-arm pick, no navigation)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

import warp as wp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.g1.g1 import G1MimicEnv
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS

GRASP_SIGNAL_NAME = "grasp_left"
FAILED_GRASP_SIGNAL_NAME = "failed_grasp_left"
HAND_CLOSED_MEAN_ABS_RAD = 0.30
"""Mean |left_hand_* joint| above which the left hand counts as closed (same threshold as the failure classifier)."""
HAND_OPEN_MEAN_ABS_RAD = 0.15
"""Mean |left_hand_* joint| below which a closed left hand counts as re-opened (classifier re-arm threshold)."""
LIFT_HEIGHT_M = 0.03
"""Apple rise above its height at episode start that counts as lifted."""


class G1StaticPickPlaceMimicEnv(G1MimicEnv):
    """G1 Mimic env with an automatic grasp signal, so ``annotate_demos.py --auto`` works headless."""

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return ``{"grasp_left": bool[num_envs], "failed_grasp_left": bool[num_envs]}``.

        ``grasp_left``: left hand closed while the apple is lifted. ``failed_grasp_left``: the left hand has closed
        at least once this episode and is now open again while the apple is not lifted (used by recovery demos).

        Args:
            env_ids: Environment indices to get the signals for. If None, all envs are considered.
        """
        if env_ids is None:
            env_ids = slice(None)
        robot = self.scene["robot"]
        if not hasattr(self, "_left_hand_joint_ids"):
            self._left_hand_joint_ids = [i for i, n in enumerate(robot.data.joint_names) if n.startswith("left_hand")]
            assert self._left_hand_joint_ids, "no left_hand_* joints found on the robot"
        joint_pos = wp.to_torch(robot.data.joint_pos)
        closure = joint_pos[:, self._left_hand_joint_ids].abs().mean(dim=-1)
        hand_closed = closure > HAND_CLOSED_MEAN_ABS_RAD
        if not hasattr(self, "_closed_once"):
            self._closed_once = torch.zeros_like(hand_closed)
        self._closed_once = torch.where(self.episode_length_buf == 0, torch.zeros_like(hand_closed), self._closed_once)
        self._closed_once |= hand_closed

        apple_name = self.cfg.subtask_configs["left"][0].object_ref
        apple_z = wp.to_torch(self.scene[apple_name].data.root_pos_w)[:, 2]
        if not hasattr(self, "_apple_start_z"):
            self._apple_start_z = apple_z.clone()
        self._apple_start_z = torch.where(self.episode_length_buf == 0, apple_z, self._apple_start_z)
        lifted = apple_z > self._apple_start_z + LIFT_HEIGHT_M

        failed_grasp = self._closed_once & (closure < HAND_OPEN_MEAN_ABS_RAD) & ~lifted
        return {GRASP_SIGNAL_NAME: (hand_closed & lifted)[env_ids], FAILED_GRASP_SIGNAL_NAME: failed_grasp[env_ids]}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Same as ``G1MimicEnv.target_eef_pose_to_action`` but tolerant of interpolated gripper actions.

        Mimic's interpolation waypoints carry hand actions of shape (1,) instead of (), which the parent
        would turn into (1, 1); flatten them to scalars and the body command to (7,) first.
        """
        gripper_action_dict = {
            name: action.reshape(-1) if name == "body" else action.reshape(-1)[0]
            for name, action in gripper_action_dict.items()
        }
        return super().target_eef_pose_to_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id)


def _subtask(object_ref: str | None, term_signal: str | None, selection: str, interpolation_steps: int = 0):
    return SubTaskConfig(
        object_ref=object_ref,
        subtask_term_signal=term_signal,
        first_subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        selection_strategy=selection,
        selection_strategy_kwargs={"nn_k": 3} if selection == "nearest_neighbor_object" else {},
        action_noise=0.003 if object_ref is not None else 0.0,
        num_interpolation_steps=interpolation_steps,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )


@configclass
class G1StaticPickPlaceMimicEnvCfg(MimicEnvCfg):
    """Mimic cfg for the G1 static apple task.

    Left arm: reach-and-grasp segment transformed with the apple, then carry-and-place segment transformed with the
    plate (bridged by interpolation). Right arm and body: one untransformed segment each (replayed as recorded).
    """

    pick_up_object_name: str = MISSING
    destination_location_name: str = MISSING
    arm_mode: ArmMode = ArmMode.DUAL_ARM
    place_interpolation_steps: int = 20
    """Steps to bridge the apple-relative grasp segment to the plate-relative place segment (0.4 s at 50 Hz)."""
    datagen_seed: int | None = None
    """Seed for Mimic data generation; None keeps the shared default."""
    recovery: bool = False
    """Split the left arm into failed grasp -> re-grasp -> place, each grasp segment transformed with the apple pose at
    its own start, so generated re-grasps go to where the apple actually is after the failed attempt."""

    def __post_init__(self):
        from isaaclab_arena_g1.g1_env.mdp.recorders.g1_locomanip_recorder_cfg import G1LocomanipRecorderManagerCfg

        super().__post_init__()
        assert self.arm_mode == ArmMode.DUAL_ARM, f"G1 static Mimic expects ArmMode.DUAL_ARM, got {self.arm_mode}"
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)
        if self.datagen_seed is not None:
            self.datagen_config.seed = self.datagen_seed
        self.datagen_config.name = (
            f"g1_static_pick_and_place_{self.pick_up_object_name}_to_{self.destination_location_name}_D0"
        )
        self.datagen_config.use_navigation_controller = False
        # Same recorder set as the G1 locomanip Mimic path, so generated files carry the keys the LeRobot converter reads.
        self.mimic_recorder_config = G1LocomanipRecorderManagerCfg()

        grasp = [_subtask(self.pick_up_object_name, GRASP_SIGNAL_NAME, "nearest_neighbor_object")]
        if self.recovery:
            grasp.insert(0, _subtask(self.pick_up_object_name, FAILED_GRASP_SIGNAL_NAME, "nearest_neighbor_object"))
        self.subtask_configs["left"] = grasp + [
            _subtask(self.destination_location_name, None, "nearest_neighbor_object", self.place_interpolation_steps),
        ]
        self.subtask_configs["right"] = [_subtask(None, None, "random")]
        self.subtask_configs["body"] = [_subtask(None, None, "random")]
