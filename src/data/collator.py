import torch

from .base import BaseCollator

class Collator(BaseCollator):
    def __init__(self):
        super().__init__()

    def __call__(self, batch: list) -> dict[str, torch.Tensor]:
        frames, labels = zip(*[item[:2] for item in batch])

        frames = torch.stack(frames)
        labels = torch.stack(labels)

        # If the labels are sequential (B, t), compute the mode along the time dimension
        if labels.dim() == 2:
            # torch.mode returns a namedtuple (values, indices); we only need the values
            labels = torch.mode(labels, dim=1).values

        collated = {
            'frames': frames,
            'labels': labels
        }

        # A dataset built with return_masks=True also yields (valid_mask, label_mask).
        # Pass them through rather than discarding them: valid_mask is what tells a
        # zeroed channel apart from a channel that was genuinely still. Datasets
        # without masks are unaffected - the keys simply do not appear.
        if len(batch[0]) >= 4:
            valid_mask, label_mask = zip(*[item[2:4] for item in batch])
            collated['valid_mask'] = torch.stack(valid_mask)
            collated['label_mask'] = torch.stack(label_mask)

        return collated
