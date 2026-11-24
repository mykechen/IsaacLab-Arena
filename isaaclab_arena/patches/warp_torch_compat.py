# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Restore ``warp.torch`` for cuRobo on Warp >= 1.16.

Warp 1.16 (shipped in the Isaac Sim 6.1.0 image) moved its torch interop to the private
``warp._src.torch`` and dropped the public ``warp.torch`` module, but the pinned cuRobo still calls
``wp.torch.device_from_torch`` -- see ``curobo/geom/sdf/world_mesh.py``. Without this alias, building
any cuRobo collision checker raises ``AttributeError: module 'warp' has no attribute 'torch'``, which
takes down every placement path that uses the IK-reachability validator.

Import this module before cuRobo touches Warp. It is a no-op when ``warp.torch`` already resolves.
"""

from __future__ import annotations

import sys

import warp as wp


def apply() -> None:
    """Alias ``warp._src.torch`` onto ``warp.torch`` when the public module is gone."""
    try:
        import warp.torch  # noqa: F401

        return
    except ImportError:
        pass

    try:
        from warp._src import torch as _warp_torch
    except ImportError:
        # Neither layout is present; leave Warp untouched so the original error surfaces.
        return

    wp.torch = _warp_torch
    sys.modules["warp.torch"] = _warp_torch


apply()
