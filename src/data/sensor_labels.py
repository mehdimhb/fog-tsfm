"""What the p*/s* combination codes actually mean.

The s0/s1 codes were documented the wrong way round until the channel order was
checked against the data: the npz is ordered accelerometer x/y/z then gyroscope
x/y/z, so indices 0-2 are the accelerometer and 3-5 the gyroscope, while
``sensor_groups`` mapped s0 to [3,4,5] under the name "accelerometer".

That is easy to get wrong twice, so the mapping lives here and nowhere else.
Every producer and consumer imports it from this module rather than restating it,
and ``save_features`` writes the resolved name into each LMDB record so a result
carries its own meaning instead of relying on a lookup someone has to remember.

This module deliberately imports nothing, so reporting tools can use it without
pulling in torch, lmdb or the model stack.
"""

from __future__ import annotations

SENSOR_GROUP_LABELS = {
    "s0": "gyroscope",
    "s1": "accelerometer",
    "s2": "accelerometer+gyroscope",
}

SENSOR_GROUP_SHORT = {
    "s0": "gyr",
    "s1": "acc",
    "s2": "acc+gyr",
}

PLACEMENT_LABELS = ("ankleL", "ankleR", "back", "wrist")

# Channel indices within a placement's six, as stored in the npz.
SENSOR_GROUP_CHANNELS = {
    "s0": [3, 4, 5],
    "s1": [0, 1, 2],
    "s2": [0, 1, 2, 3, 4, 5],
}


def placement_names(slots: str) -> list[str]:
    """'03' -> ['ankleL', 'wrist']."""
    return [PLACEMENT_LABELS[int(slot)] for slot in slots.lstrip("p")]


def describe_combo(combo: str, short: bool = True) -> str:
    """'combo_p03_s0' or 'p03_s0' -> 'ankleL+wrist x gyr'."""
    name = combo[len("combo_"):] if combo.startswith("combo_") else combo
    if "_" not in name:
        return name
    slots, sensor = name.rsplit("_", 1)
    table = SENSOR_GROUP_SHORT if short else SENSOR_GROUP_LABELS
    return f"{'+'.join(placement_names(slots))} x {table.get(sensor, sensor)}"


__all__ = [
    "SENSOR_GROUP_LABELS",
    "SENSOR_GROUP_SHORT",
    "SENSOR_GROUP_CHANNELS",
    "PLACEMENT_LABELS",
    "placement_names",
    "describe_combo",
]
