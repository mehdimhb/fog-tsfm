import os
import pickle
import lmdb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import combinations
from torch.utils.data import DataLoader

from src import data, model
from src.data.mask_utils import (
    DEFAULT_ELIGIBILITY_RULE,
    channel_eligibility,
    select_eligibility,
    zero_ineligible,
    mask_features,
)
from src.data.sensor_labels import (
    SENSOR_GROUP_CHANNELS,
    SENSOR_GROUP_LABELS,
    placement_names,
)
from mantis.architecture import MantisV2
from src.model import get_loaded_nutime_channel_encoder

def extract_features(batch_frames: torch.Tensor, unishape_model, mantis_model, nutime_model, device: torch.device):
    x_raw = batch_frames.to(device)  # Expected shape: (B, C, L)
    if not x_raw.is_floating_point():
        x_raw = x_raw.float()

    # Interpolate time dimension (L) to 512 regardless of input length or channel count
    if x_raw.shape[-1] != 512:
        x_interpolated = F.interpolate(x_raw, size=512, mode='linear', align_corners=False)
    else:
        x_interpolated = x_raw

    B, C, L = x_interpolated.shape
    x_univariate = x_interpolated.view(B * C, 1, L)

    uni_features, mantis_features, nutime_features = None, None, None

    with torch.no_grad():
        # --- Unishape ---
        if unishape_model is not None:
            uni = unishape_model(x_univariate)
            uni_features = uni.reshape(B, C, -1)

        # --- Mantis ---
        if mantis_model is not None:
            mantis = mantis_model(x_univariate)
            mantis_features = mantis.reshape(B, C, -1)

        # --- NuTime (Expects (B, C, L)) ---
        if nutime_model is not None:
            nutime_features = nutime_model(x_univariate)
            nutime_features = nutime_features.reshape(B, C, -1)

    return uni_features, mantis_features, nutime_features


def save_split_to_lmdb(loader, unishape_model, mantis_model, nutime_model, txn, split_name, device, group_ids=None, eligibility_rule=DEFAULT_ELIGIBILITY_RULE):
    print(f"Processing {split_name} split...")
    keys = []
    global_idx = 0

    # Generate all 15 possible combinations of the 4 placements (0, 1, 2, 3)
    placements = [0, 1, 2, 3]
    placement_combos = []
    for r in range(1, 5):
        placement_combos.extend(list(combinations(placements, r)))

    # The 3 sensor configurations, from the single mapping in
    # src/data/sensor_labels.py. s0 is the GYROSCOPE and s1 the ACCELEROMETER:
    # the npz is ordered accelerometer x/y/z then gyroscope x/y/z, so 0-2 is the
    # accelerometer and 3-5 the gyroscope, and the old comments said the reverse.
    # Verified from the data rather than the column names - channels 0-2 have p95
    # magnitude ~1 (g) against ~67-81 (deg/s) for 3-5, an 87x ratio.
    #
    # The indices are unchanged, so results already computed stay valid; only the
    # names attached to them were wrong. Each record now carries its own resolved
    # label, so nothing downstream has to decode the shorthand from memory.
    sensor_groups = SENSOR_GROUP_CHANNELS

    for batch in tqdm(loader, desc=f"Extracting {split_name}"):
        if isinstance(batch, dict):
            frames, labels = batch['frames'], batch['labels']
            valid_mask = batch.get('valid_mask')
        else:
            frames, labels = batch[0], batch[1]
            valid_mask = batch[2] if len(batch) >= 3 else None

        # frames expected shape: (B, 4, 6, 120)
        B = frames.shape[0]
        labels_np = labels.cpu().numpy()

        # (B, 4, 6): which channel-windows carry data. Under the default
        # 'entirely_gap' rule only wholly absent channels are masked - the ones
        # that arrive as a flat zero signal and are indistinguishable from a
        # channel that was perfectly still. Partially gapped channels keep their
        # real stretches, so nothing that previously reached the backbone is
        # discarded. Pass eligibility_rule='all_valid' for the stricter rule.
        eligibility = (
            channel_eligibility(valid_mask, eligibility_rule)
            if valid_mask is not None else None
        )

        # Initialize dictionary to hold all 45 configurations for the current batch
        batch_data = [{"label": labels_np[i]} for i in range(B)]

        # Process all 45 combinations per batch
        for p_combo in placement_combos:
            for s_name, s_idx in sensor_groups.items():

                # Create the concise naming convention (e.g., "p03_s0")
                p_str = "".join([str(p) for p in p_combo])
                combo_name = f"p{p_str}_{s_name}"

                # 1. Select specific placements: (B, len(p_combo), 6, 120)
                sub_frames = frames[:, p_combo, :, :]

                # 2. Select specific sensors: (B, len(p_combo), len(s_idx), 120)
                sub_frames = sub_frames[:, :, s_idx, :]

                # 3. Combine placement and sensor dimensions: (B, len(p_combo) * len(s_idx), 120)
                sub_frames = sub_frames.reshape(B, -1, sub_frames.shape[-1])

                # 3b. Apply the same selection to the eligibility vector, then
                #     zero the channels that carry no data.
                sub_eligibility = None
                if eligibility is not None:
                    sub_eligibility = select_eligibility(eligibility, p_combo, s_idx)
                    sub_frames = zero_ineligible(sub_frames, sub_eligibility)

                # 4. Extract features
                uni, mantis, nutime = extract_features(
                    sub_frames, unishape_model, mantis_model, nutime_model, device
                )

                # 4b. Zero the features of ineligible channels as well. A frozen
                #     backbone fed an all-zero window still emits a non-zero
                #     embedding - biases and positional terms survive - so masking
                #     the input alone achieves nothing.
                if sub_eligibility is not None:
                    mask_device = sub_eligibility.to(device)
                    if uni is not None: uni = mask_features(uni, mask_device)
                    if mantis is not None: mantis = mask_features(mantis, mask_device)
                    if nutime is not None: nutime = mask_features(nutime, mask_device)

                # 5. Populate dictionaries for each item in the batch
                for i in range(B):
                    batch_data[i][combo_name] = {}
                    if uni is not None: batch_data[i][combo_name]["unishape_features"] = uni[i].cpu().numpy()
                    if mantis is not None: batch_data[i][combo_name]["mantis_features"] = mantis[i].cpu().numpy()
                    if nutime is not None: batch_data[i][combo_name]["nutime_features"] = nutime[i].cpu().numpy()
                    # Stored so a classifier can tell "this channel read zero"
                    # from "this channel is absent".
                    if sub_eligibility is not None:
                        batch_data[i][combo_name]["channel_eligibility"] = (
                            sub_eligibility[i].cpu().numpy()
                        )
                    # The resolved meaning of the p*/s* shorthand, written into
                    # the record itself so a result is self-describing.
                    batch_data[i][combo_name]["sensor"] = SENSOR_GROUP_LABELS[s_name]
                    batch_data[i][combo_name]["placements"] = placement_names(p_str)

        # Save the populated dictionaries to LMDB
        for i in range(B):
            key_str = f"{split_name}_{global_idx}"
            key = key_str.encode('ascii')

            # Subject id, so results can be analysed per participant later.
            # The loader is unshuffled and global_idx advances in dataset order,
            # so this indexes the same window the features came from.
            if group_ids is not None:
                batch_data[i]["group_id"] = str(group_ids[global_idx])

            txn.put(key, pickle.dumps(batch_data[i]))
            keys.append(key_str)
            global_idx += 1

    txn.put(f"{split_name}_keys".encode('ascii'), pickle.dumps(keys))
    print(f"Finished {split_name}. Saved {len(keys)} samples.\n")


def main(
    npz_path="datasets/processed_output2/processed_windows.npz",
    models_to_use=("unishape", "mantis", "nutime"),
    seed=42,
    eligibility_rule=DEFAULT_ELIGIBILITY_RULE,
):
    models_to_use = set(models_to_use)
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}\nUsing models: {models_to_use}")

    # -----------------------------
    # 1. Load Raw Data Separately
    # -----------------------------
    print("Initializing DataLoaders...")
    # return_masks=True so valid_mask reaches the extractor. The dataset has always
    # been able to supply it; nothing asked for it until now.
    ds_train = data._Dataset(data_path=npz_path, split_csv='split_11_5_6.csv', mode='train', return_masks=True)
    ds_val = data._Dataset(data_path=npz_path, split_csv='split_11_5_6.csv', mode='val', return_masks=True)
    ds_test = data._Dataset(data_path=npz_path, split_csv='split_11_5_6.csv', mode='test', return_masks=True)

    # Use Collator to guarantee classification-ready labels
    collator = data.Collator()

    loaders = {
        "train": DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=4, collate_fn=collator),
        "val": DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=4, collate_fn=collator),
        "test": DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4, collate_fn=collator)
    }

    # -----------------------------
    # 2. Load Models Conditionally
    # -----------------------------
    uni, mantis, nutime_model = None, None, None

    if "unishape" in models_to_use:
        uni = model.get_loaded_uni_shape('weights/unishape_checkpoint_zeroshot.pth', return_all_cls_tokens=False).to(device).eval()
    if "mantis" in models_to_use:
        mantis = MantisV2(device=str(device)).from_pretrained("paris-noah/MantisV2").to(device).eval()
    if "nutime" in models_to_use:
        nutime_model = get_loaded_nutime_channel_encoder(
            weights_path='weights/checkpoint_bias9.pth',
            device=device
        )

    # -----------------------------
    # 3. Feature Extraction to LMDB
    # -----------------------------
    target_lmdb_path = 'datasets/new_lmdb_features'
    os.makedirs(target_lmdb_path, exist_ok=True)
    env = lmdb.open(target_lmdb_path, map_size=1099511627776)

    with env.begin(write=True) as txn:
        for split_name in ["train", "val", "test"]:
            save_split_to_lmdb(
                loader=loaders[split_name],
                unishape_model=uni,
                mantis_model=mantis,
                nutime_model=nutime_model,
                txn=txn,
                split_name=split_name,
                device=device,
                group_ids={"train": ds_train, "val": ds_val, "test": ds_test}[split_name].group_id,
                eligibility_rule=eligibility_rule
            )

    env.close()
    print(f"Feature extraction complete! LMDB saved to {target_lmdb_path}")

if __name__ == "__main__":
    main()
