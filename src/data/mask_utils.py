"""Make ``valid_mask`` visible to the feature extractors instead of dropping it.

Why this exists
---------------
``processed_windows.npz`` ships a ``valid_mask`` alongside ``signals``, and
``_Dataset`` can already return it (``return_masks=True``). Nothing downstream
asked for it, so the mask was never read.

That is not a harmless omission. Preprocessing holds every invalid sample at
exactly zero, so a channel with no data is byte-for-byte identical to a channel
that was perfectly still. Measured on FoG-STAR:

    channel-windows                 264,480
    fully valid                     242,942   (91.86 %)
    entirely gap (all zeros)         11,890
    part real / part gap              9,648
    contexts with no fully valid channel  421 (20 of them labelled FoG)

Two distinct failure modes hide in there. A fully absent channel reaches the
backbone as a flat zero signal. A partially gapped channel reaches it as real
data with a flat segment spliced into the middle.

Read that last figure carefully: it counts contexts where every channel fails the
all-samples-valid rule, which includes contexts that are only partially gapped and
still carry real signal in between. The genuinely all-zero contexts are a subset of
it, and those are the ones asked to predict FoG from nothing and then scored as if
the model had a fair chance. The two should be reported
separately, because conflating them overstates the problem.

Two eligibility rules
---------------------
``entirely_gap`` (the default)
    A channel is ineligible only when *every* sample is invalid. This targets the
    actual defect - a wholly absent channel arriving as a flat zero signal,
    indistinguishable from a channel that was perfectly still - and discards
    nothing that previously reached the backbone. Partially gapped channels are
    left exactly as they were, keeping their real stretches.

``all_valid``
    A channel is ineligible if *any* sample is invalid. Cleaner in principle: no
    window with a spliced-in flat segment ever reaches the model. But on
    FoG-STAR it also discards 9,648 channel-windows that still carry genuine
    signal, which is a real cost for an artifact the model may well tolerate.

The default is ``entirely_gap`` because it makes this change a strict improvement:
every channel it zeroes was already carrying nothing. Nothing is repaired by
interpolation under either rule - inventing samples inside a gap is exactly what
the preprocessing deliberately avoids.

Ineligible channels are zeroed explicitly, and eligibility is carried alongside
the features as its own vector so it can be used in analysis. Note that the model
does not currently consume it: ``_FeatureDataset`` reads only the ``*_features``
keys, so wiring it into the classifier is still to be done.
"""

from __future__ import annotations

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


ELIGIBILITY_RULES = ("entirely_gap", "all_valid")
DEFAULT_ELIGIBILITY_RULE = "entirely_gap"


def channel_eligibility(
    valid_mask: ArrayLike, rule: str = DEFAULT_ELIGIBILITY_RULE
) -> ArrayLike:
    """``(B, P, C, T)`` sample mask -> ``(B, P, C)`` eligibility.

    ``entirely_gap`` (default)
        Eligible if the channel has at least one valid sample. Only wholly absent
        channels are masked, so nothing that previously reached the backbone is
        thrown away.

    ``all_valid``
        Eligible only if every sample is valid. Stricter, and on FoG-STAR it
        discards 9,648 channel-windows that still hold real signal.
    """
    if valid_mask.ndim != 4:
        raise ValueError(f"expected (B,P,C,T); got {tuple(valid_mask.shape)}")
    if rule not in ELIGIBILITY_RULES:
        raise ValueError(f"rule must be one of {ELIGIBILITY_RULES}; got {rule!r}")

    if isinstance(valid_mask, torch.Tensor):
        mask = valid_mask.bool()
        return mask.any(dim=-1) if rule == "entirely_gap" else mask.all(dim=-1)
    mask = np.asarray(valid_mask, dtype=bool)
    return mask.any(axis=-1) if rule == "entirely_gap" else mask.all(axis=-1)


def select_eligibility(
    eligibility: ArrayLike,
    placements: tuple[int, ...],
    channels: list[int],
) -> ArrayLike:
    """Apply the same placement/sensor selection ``save_features`` applies.

    Mirrors ``frames[:, p_combo, :, :][:, :, s_idx, :].reshape(B, -1, T)`` so the
    eligibility vector stays aligned with the flattened channel axis.
    """
    # list(), not the tuple save_features passes around: a tuple in an index
    # position is advanced-indexing in some versions and multi-dimensional
    # indexing in others, and the two disagree about the result shape.
    placements = list(placements)
    channels = list(channels)
    if isinstance(eligibility, torch.Tensor):
        selected = eligibility[:, placements, :][:, :, channels]
        return selected.reshape(selected.shape[0], -1)
    selected = np.asarray(eligibility)[:, placements, :][:, :, channels]
    return selected.reshape(selected.shape[0], -1)


def zero_ineligible(signals: ArrayLike, eligibility: ArrayLike) -> ArrayLike:
    """Force ineligible channels to exactly zero before feature extraction.

    Preprocessing already does this today, but re-applying it here makes the
    contract explicit and survives any upstream change that fills gaps
    differently.
    """
    if tuple(signals.shape[:2]) != tuple(eligibility.shape[:2]):
        raise ValueError(
            f"signals {tuple(signals.shape)} and eligibility "
            f"{tuple(eligibility.shape)} disagree on (B, C)"
        )
    if isinstance(signals, torch.Tensor):
        return signals * eligibility.unsqueeze(-1).to(signals.dtype)
    return signals * np.asarray(eligibility)[..., None].astype(signals.dtype)


def mask_features(features: ArrayLike, eligibility: ArrayLike) -> ArrayLike:
    """Zero the feature vector of every ineligible channel.

    ``features`` is ``(B, C, D)`` as the backbones emit it.

    This step is easy to skip and doing so silently undoes the whole fix: a
    frozen backbone fed an all-zero window still returns a *non-zero* embedding,
    because biases and positional terms survive. Zeroing the input alone
    achieves nothing.
    """
    if tuple(features.shape[:2]) != tuple(eligibility.shape[:2]):
        raise ValueError(
            f"features {tuple(features.shape)} and eligibility "
            f"{tuple(eligibility.shape)} disagree on (B, C)"
        )
    if isinstance(features, torch.Tensor):
        return features * eligibility.unsqueeze(-1).to(features.dtype)
    return features * np.asarray(eligibility)[..., None].astype(features.dtype)


def usable_contexts(eligibility: ArrayLike) -> ArrayLike:
    """Contexts holding at least one eligible channel.

    Contexts failing this carry a label but no signal. Excluding them from
    *training* is clearly right. Excluding them from *evaluation* changes the
    denominator, so if you drop them, report how many alongside the metrics.
    """
    if isinstance(eligibility, torch.Tensor):
        return eligibility.flatten(1).any(dim=1)
    flat = np.asarray(eligibility).reshape(len(eligibility), -1)
    return flat.any(axis=1)


def summarise(
    valid_mask: ArrayLike, rule: str = DEFAULT_ELIGIBILITY_RULE
) -> dict[str, float]:
    """Numbers worth printing once, so the scale of missingness is on the record.

    Both rules are reported, because the difference between them is the whole
    question: it is the count of channel-windows that are part real and part gap.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    eligibility = channel_eligibility(mask, rule)
    usable = usable_contexts(eligibility)
    complete = eligibility.reshape(len(eligibility), -1).all(axis=1)

    any_valid = mask.any(axis=-1)
    all_valid = mask.all(axis=-1)
    return {
        "rule": rule,
        "contexts": int(eligibility.shape[0]),
        "channel_windows": int(eligibility.size),
        "eligible_channel_windows": int(eligibility.sum()),
        "eligible_fraction": float(eligibility.mean()),
        "contexts_fully_valid": int(complete.sum()),
        "contexts_partly_valid": int((usable & ~complete).sum()),
        "contexts_with_no_eligible_channel": int((~usable).sum()),
        # The two rules differ by exactly this many channel-windows.
        "channel_windows_entirely_gap": int((~any_valid).sum()),
        "channel_windows_part_real_part_gap": int((any_valid & ~all_valid).sum()),
    }


__all__ = [
    "ELIGIBILITY_RULES",
    "DEFAULT_ELIGIBILITY_RULE",
    "channel_eligibility",
    "select_eligibility",
    "zero_ineligible",
    "mask_features",
    "usable_contexts",
    "summarise",
]
