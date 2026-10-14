"""Smoke test (run in the NDT3 WSL env): register one exported session through NDT3's own preprocessed-split path and
inspect the batches its SpikingDataset produces.

  wsl -d Ubuntu-24.04 -u root -- bash -c "cd /opt/ndt3 && /opt/ndt3-venv/bin/python '/mnt/g/codes/invasive bci/scripts/ndt3/smoke_dataset.py'"
"""
import os
import sys

os.environ["NDT_SUPPRESS_DEFAULT_REGISTRY"] = "1"
os.environ["WANDB_MODE"] = "disabled"
import numpy as np
import torch

ROOT = "/mnt/h/ndt3/data"            # NDT3 root_dir: raw splits and its preprocessing cache live under it
WORK = f"{ROOT}/ours_smoke"
SRC = "/mnt/h/ibci-data/ndt3_pilot/N/2020-01-30_CO.npz"


def write_pth(path, x, y):
    torch.save({"binned_spikes": torch.as_tensor(np.clip(x, 0, 255).astype(np.uint8))[..., None],
                "covariate": torch.as_tensor(y, dtype=torch.float32)}, path)


def main():
    os.makedirs(WORK, exist_ok=True)
    d = np.load(SRC)
    print("x_tr", d["x_tr"].shape, "y_tr", d["y_tr"].shape, "x_te", d["x_te"].shape)
    # file stem: <subject>_<date>_<tail>; build_preproc maps subject 'indy' -> 96-channel M1 array registration
    write_pth(f"{WORK}/indy_N20200130_cal.pth", d["x_tr"], d["y_tr"])

    from context_general_bci.contexts.context_info import RTTContextInfo
    from context_general_bci.contexts.context_registry import context_registry
    context_registry.register([*RTTContextInfo.build_preproc(WORK, alias_prefix="ours", arrays=["M1"])])
    from context_general_bci.config import RootConfig, propagate_config, DataKey
    from hydra import compose, initialize_config_module
    initialize_config_module(config_module="context_general_bci.config", job_name="smoke", version_base="1.3")
    cfg: RootConfig = compose(config_name="config", overrides=["+exp/v5/tune/rtt=smoketest"])
    propagate_config(cfg)
    from pathlib import Path
    cfg.dataset.root_dir = Path(ROOT)
    cfg.dataset.datasets = ["ours.*"]
    cfg.dataset.eval_datasets = []
    cfg.dataset.explicit_norm = ""
    cfg.dataset.behavior_dim = d["y_tr"].shape[1] + 1
    print("chop_size_ms", cfg.dataset.odoherty_rtt.chop_size_ms, "minmax", cfg.dataset.odoherty_rtt.minmax,
          "bin", cfg.dataset.bin_size_ms, "max_channels", cfg.dataset.max_channels)
    from context_general_bci.dataset import SpikingDataset
    ds = SpikingDataset(cfg.dataset)
    ds.build_context_index()
    print("n items", len(ds), "data_attrs", ds.get_data_attrs())
    item = ds[0]
    for k, v in item.items():
        print(" ", k, tuple(v.shape) if hasattr(v, "shape") else v)
    b = ds.collater_factory()([ds[i] for i in range(4)])
    for k, v in b.items():
        print(" batch", k, tuple(v.shape) if hasattr(v, "shape") else type(v))


if __name__ == "__main__":
    main()
