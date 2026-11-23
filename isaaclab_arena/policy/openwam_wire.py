# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""msgpack codec with numpy arrays for the OpenWAM policy server protocol.

Must stay byte-compatible with the server-side copy (openwam-g1 g1_server/wire.py).
"""

from __future__ import annotations

import numpy as np
from typing import Any

import msgpack

_ND = "__nd__"


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {_ND: True, "dtype": arr.dtype.str, "shape": list(arr.shape), "data": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialise {type(obj)!r}")


def _hook(obj: dict) -> Any:
    if obj.get(_ND):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def pack(message: Any) -> bytes:
    """Serialise a message that may contain numpy arrays."""
    return msgpack.packb(message, default=_default, use_bin_type=True)


def unpack(payload: bytes) -> Any:
    """Deserialise a message produced by pack; integer map keys are preserved."""
    return msgpack.unpackb(payload, object_hook=_hook, raw=False, strict_map_key=False)
