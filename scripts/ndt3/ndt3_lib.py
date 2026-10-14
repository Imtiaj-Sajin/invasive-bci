"""NDT3 fine-tuning and evaluation on our exported sessions, through NDT3's own data path (run in the WSL env).

Data path: each split is written as a preprocessed RTT-style file (binned_spikes (T, C, 1) uint8 at 20 ms, covariate
(T, D)), registered with RTTContextInfo.build_preproc under a subject whose array layout matches ours (Indy: 'M1' =
channels 0-95, 'S1' = 96-191; NDT3 v5 has no subject or session embeddings, so the label only sets the tokenization).
NDT3's loader then chops 1 s windows, normalizes behaviour and builds the payloads; its SpikingDataset, collater,
Lightning training step and streaming_eval are used unchanged.

Behaviour normalization is fixed from the fine-tuning source split (99.9th percentile of |y| per dimension, the
paper's default) and applied to every split via explicit_norm, so no labels of an evaluation day set the scale.
Predictions are rescaled and R2 is computed on our own unclamped labels, as for the linear methods.
"""
import os
import shutil
from pathlib import Path

os.environ.setdefault("NDT_SUPPRESS_DEFAULT_REGISTRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np
import torch

ROOT = Path("/mnt/h/ndt3/data")
CKPT_2KH = "/mnt/h/ndt3/ckpt/l2ulo75e/checkpoints/val-epoch=254-val_loss=0.4149.ckpt"
_CFG = None


def r2(pred, y):
    ss = ((y - pred) ** 2).sum(0)
    st = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss / np.maximum(st, 1e-12)))


def arrays_for(n_channels):
    return ["M1"] if n_channels <= 96 else ["M1", "S1"]


def write_split(job, name, x, y, chop_ms=1000, pad_to=None):
    """Write one split as <ROOT>/<job>/indy_<name>_split.pth (name without underscores). Optionally pad the end with
    zero bins to a multiple of `pad_to` bins (padding is after every real bin, so causal predictions are unaffected)."""
    d = ROOT / job
    d.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x), 0, 255).astype(np.uint8)
    y = np.asarray(y, dtype=np.float32)
    if pad_to:
        extra = (-len(x)) % pad_to
        x = np.concatenate([x, np.zeros((extra, x.shape[1]), np.uint8)])
        y = np.concatenate([y, np.zeros((extra, y.shape[1]), np.float32)])
    # NDT3's registry identifies a context by the file name, so the job name is part of it (no underscores allowed)
    torch.save({"binned_spikes": torch.as_tensor(x)[..., None], "covariate": torch.as_tensor(y)},
               d / f"indy_{job.replace('_', '')}{name.replace('_', '')}_split.pth")
    return len(x)


def write_norm(job, y_source):
    """Fixed behaviour scale from the source split: 99.9th percentile of |y| per dimension (NDT3 default)."""
    (ROOT / job).mkdir(parents=True, exist_ok=True)
    cov_max = torch.quantile(torch.as_tensor(np.abs(y_source), dtype=torch.float32), 0.999, dim=0)
    cov_max[torch.isclose(cov_max, torch.tensor(0.))] = 1
    p = ROOT / job / "norm.pt"
    torch.save({"cov_mean": None, "cov_min": None, "cov_max": cov_max}, p)
    return str(p), cov_max.numpy()


def base_cfg():
    global _CFG
    if _CFG is None:
        from hydra import compose, initialize_config_module
        from context_general_bci.config import propagate_config
        initialize_config_module(config_module="context_general_bci.config", job_name="ibci", version_base="1.3")
        _CFG = compose(config_name="config", overrides=["+exp/v5/tune/rtt=smoketest"])
        propagate_config(_CFG)
    import copy
    return copy.deepcopy(_CFG)


def register(job, n_channels):
    """Register the split(s) in <ROOT>/<job>, first dropping any earlier registry entries for that folder. NDT3's
    registry keeps entries for the life of the process and keys them by file name and array layout; re-registering a
    job folder with a different layout (96 vs 192 channels) left both entries matching the alias, so the files were
    loaded twice (caught by the item-count assertion, 2026-10-15). With the drop there is exactly one entry per job."""
    from context_general_bci.contexts.context_info import RTTContextInfo
    from context_general_bci.contexts.context_registry import context_registry as reg
    folder = str((ROOT / job).resolve())
    if len(reg.search_index):
        stale = reg.search_index[reg.search_index.datapath.astype(str).str.startswith(folder + "/")]
        for i in stale.id:
            reg._registry.pop(i, None)
        reg.search_index = reg.search_index[~reg.search_index.id.isin(stale.id)]
    reg.register([*RTTContextInfo.build_preproc(str(ROOT / job), alias_prefix=f"ours{job}",
                                                arrays=arrays_for(n_channels))])
    return f"ours{job}-.*"


def set_labels(n_dims):
    """The RTT loader labels every split with REACH_DEFAULT_KIN_LABELS (2 dims); our monkey N has 4 outputs (two
    finger positions and velocities). Labels only name the dimensions (used to group them in NDT3's R2 metric)."""
    import context_general_bci.tasks.rtt as rtt
    rtt.REACH_DEFAULT_KIN_LABELS = [f"dim{i}" for i in range(n_dims)]


def make_cfg(alias, behavior_dim, norm_path, lr=1e-4, epochs=150, patience=50, batch_size=32, kin_weight=1.0,
             seed=0, chop_ms=1000):
    set_labels(behavior_dim)
    cfg = base_cfg()
    cfg.seed = seed
    cfg.dataset.root_dir = ROOT
    cfg.dataset.datasets = [alias]
    cfg.dataset.eval_datasets = []
    cfg.dataset.explicit_norm = norm_path
    cfg.dataset.behavior_dim = behavior_dim + 1          # NDT3 fine-tuning convention (D + 1)
    cfg.dataset.odoherty_rtt.chop_size_ms = chop_ms
    cfg.model.lr_init = lr
    cfg.model.task.task_weights = [1.0, kin_weight]      # [spike_infill, kinematic_linear]
    cfg.train.epochs = epochs
    cfg.train.patience = patience
    cfg.train.batch_size = batch_size
    cfg.train.autoscale_batch_size = False
    return cfg


def dataset(cfg):
    from context_general_bci.dataset import SpikingDataset
    ds = SpikingDataset(cfg.dataset, load_workers=0)   # NDT3's process pool deadlocks here (as its own code warns)
    ds.build_context_index()
    return ds


def load_pretrained(path):
    """Load a public NDT3 checkpoint as a model. The 2 kh checkpoint lists Pitt subjects (e.g. CRS02b) missing from
    the public SubjectName enum, so Lightning's loader fails; unknown enum values are kept as placeholders. v5 models
    have no subject or session embeddings, so these lists do not affect the architecture; strict loading checks that
    every weight is restored."""
    import pickle
    import types
    from context_general_bci.model import BrainBertInterface

    class Tolerant(pickle.Unpickler):
        def find_class(self, module, name):
            cls = super().find_class(module, name)
            if name in ("SubjectName", "ExperimentalTask"):
                def make(value, _cls=cls):
                    try:
                        return _cls(value)
                    except ValueError:
                        return next(iter(_cls))          # placeholder member; identity lists are unused in v5
                return make
            return cls

    mod = types.ModuleType("tolerant_pickle")
    mod.__dict__.update(pickle.__dict__)
    mod.Unpickler = Tolerant
    ck = torch.load(path, map_location="cpu", pickle_module=mod)
    hp = ck["hyper_parameters"]
    model = BrainBertInterface(hp["cfg"], hp["data_attrs"])
    model.load_state_dict(ck["state_dict"], strict=True)
    return model


def fix_r2_metrics(model, n_dims):
    """NDT3 builds its validation R2 metric without num_outputs; with torchmetrics 1.4 that means one output, which
    fails for multi-dimensional behaviour. Rebuild it with the right size (used only for early stopping)."""
    import torchmetrics
    for pipe in model.task_pipelines.values():
        for name in ("val_r2_score", "eval_r2_score"):
            m = getattr(pipe, name, None)
            if isinstance(m, torchmetrics.R2Score):
                setattr(pipe, name, torchmetrics.R2Score(num_outputs=n_dims, multioutput=m.multioutput))


def finetune(cfg, init, workdir, monitor="val_kinematic_r2", mode="max"):
    """Fine-tune `init` (checkpoint path or BrainBertInterface) on cfg.dataset; returns the best model (by `monitor`
    on NDT3's random 1 s-window validation split of the training data)."""
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from context_general_bci.dataset import SpikingDataModule
    from context_general_bci.model import BrainBertInterface, transfer_model
    pl.seed_everything(cfg.seed)
    ds = dataset(cfg)
    ds.subset_split(keep_index=True)
    attrs = ds.get_data_attrs()
    train, val = ds.create_tv_datasets(train_ratio=cfg.dataset.tv_ratio)
    if isinstance(init, str):
        init = load_pretrained(init)
    model = transfer_model(init, cfg.model, attrs)
    fix_r2_metrics(model, cfg.dataset.behavior_dim - 1)
    ck = ModelCheckpoint(dirpath=workdir, monitor=monitor, mode=mode, save_top_k=1, filename="best")
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=cfg.train.epochs,
                         logger=False, enable_progress_bar=False, gradient_clip_val=1.0,
                         callbacks=[ck, EarlyStopping(monitor=monitor, mode=mode, patience=cfg.train.patience)],
                         num_sanity_val_steps=0)
    dm = SpikingDataModule(cfg.train.batch_size, num_workers=2, train=train, val=val)
    trainer.fit(model, datamodule=dm)
    best = BrainBertInterface.load_from_checkpoint(ck.best_model_path)
    info = {"epochs": trainer.current_epoch, "best_score": float(ck.best_model_score) if ck.best_model_score is not None
            else None, "n_train_windows": len(train), "n_val_windows": len(val)}
    return best, info


def predict(model, cfg, n_real_bins, cov_max, y_true_norm_check=None):
    """Causal streaming predictions (1 s context, NDT3's streaming_eval) for the single registered split in cfg;
    returns (n_real_bins, D) predictions in original units."""
    from context_general_bci.analyze_utils import streaming_eval
    from context_general_bci.config import Output
    ds = dataset(cfg)
    model = model.to("cuda").eval()
    model.cfg.task.outputs = [Output.behavior_pred, Output.behavior]   # returned only if requested
    outputs, _, _, _ = streaming_eval(model, ds, stream_buffer_s=1.0, use_kv_cache=True, skip_cache_reset=True,
                                      autoregress_cue=False)
    D = len(cov_max)
    pred = outputs[Output.behavior_pred].float().cpu().numpy().reshape(-1, D)
    truth = outputs[Output.behavior].float().cpu().numpy().reshape(-1, D)
    if y_true_norm_check is not None:   # alignment check: NDT3's served targets must equal our normalized labels
        n = min(len(truth), len(y_true_norm_check))
        err = float(np.abs(truth[:n] - y_true_norm_check[:n]).max())
        assert err < 1e-3, f"streamed targets do not match our labels (max abs diff {err})"
    assert len(pred) >= n_real_bins, f"{len(pred)} predictions for {n_real_bins} bins"
    return pred[:n_real_bins] * cov_max


def predict_chunked(model, job, x, y, n_channels, behavior_dim, norm_path, cov_max, win=100, keep=50):
    """Batch-mode causal predictions (as NDT3's offline evaluation): overlapping windows of `win` bins advancing by
    `keep`; each window is one item of NDT3's dataset and is predicted with behaviour inputs masked; only the last
    `keep` bins of each window are kept (the first window keeps all), so every kept prediction sees
    win-keep .. win-1 bins of past context. Returns (len(x), D) predictions in original units."""
    from context_general_bci.analyze_utils import get_dataloader
    from context_general_bci.config import Output
    from context_general_bci.utils import to_device
    T = len(x)
    starts = list(range(0, max(T - win, 0) + 1, keep))
    if starts[-1] + win < T:
        starts.append(starts[-1] + keep)
    pad = starts[-1] + win - T
    xp = np.concatenate([np.asarray(x), np.zeros((max(pad, 0), x.shape[1]), dtype=np.asarray(x).dtype)])
    yp = np.concatenate([np.asarray(y, np.float32), np.zeros((max(pad, 0), y.shape[1]), np.float32)])
    xw = np.concatenate([xp[s:s + win] for s in starts])
    yw = np.concatenate([yp[s:s + win] for s in starts])
    write_split(job, "chunks", xw, yw)
    cfg = make_cfg(register(job, n_channels), behavior_dim, norm_path, chop_ms=win * 20)
    ds = dataset(cfg)
    assert len(ds) == len(starts), f"{len(ds)} items for {len(starts)} windows"
    model = model.to("cuda").eval()
    model.cfg.task.outputs = [Output.behavior_pred, Output.behavior]
    ds.cfg.max_length_ms = max(ds.cfg.max_length_ms, win * 20)
    ds.set_no_crop(True)
    D = len(cov_max)
    kin_mask = torch.ones(win, dtype=torch.bool, device="cuda")      # behaviour inputs masked at every step
    preds, truths = [], []
    for batch in get_dataloader(ds, batch_size=1, num_workers=0):   # same calls as streaming_eval's batch branch
        batch = to_device(batch, "cuda")
        out = model.predict_simple_batch(batch, kin_mask_timesteps=kin_mask, last_step_only=False)
        preds.append(out[Output.behavior_pred].float().cpu().numpy().reshape(win, D))
        truths.append(out[Output.behavior].float().cpu().numpy().reshape(win, D))
    pred, truth = np.stack(preds), np.stack(truths)
    ref = np.clip(yw.reshape(len(starts), win, D) / cov_max, -1, 1)
    err = float(np.abs(truth - ref).max())
    assert err < 1e-3, f"served targets do not match our windows (max abs diff {err}); item order differs?"
    out = np.empty((T, D), np.float32)
    for i, s in enumerate(starts):
        lo = 0 if i == 0 else win - keep
        a, b = s + lo, min(s + win, T)
        if b > a:
            out[a:b] = pred[i, lo:lo + (b - a)]
    return out * cov_max


def cleanup(job):
    shutil.rmtree(ROOT / job, ignore_errors=True)
    shutil.rmtree(ROOT / "preprocessed" / job, ignore_errors=True)
