"""
EB-JEPA Prescribed Axes Experiment v4 — extends v3 with prescribed_4.

prescribed_4 = (x_agent, y_agent, wall_x, door_y).
wall_x and door_y are constant along T (per-trajectory) but vary per sample.
fix_wall=False in data_config — wall and door locations are randomized.

Dual save: local (fast) + drive (backup).
- Mid-epoch (every 200 batches): save to LOCAL only.
- End of epoch: save to BOTH local and drive.
- Resume: prefer local if newer; fallback to drive.

Conditions: free, prescribed (=prescribed_2), prescribed_4, hybrid, hybrid_4.
"""

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

# EB-JEPA imports
from eb_jepa.architectures import (
    ImpalaEncoder, InverseDynamicsModel, Projector, RNNPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import (
    load_config, save_checkpoint, setup_device, setup_seed,
)
from eb_jepa.planning import main_eval

logger = get_logger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# Prescribed and Hybrid encoders
# ================================================================

class PrescribedEncoder(nn.Module):
    """Uses prescribed coordinates (e.g., agent x,y or x,y,wall_x,door_y) instead of pixels.

    Output: [B, D, T, 1, 1].

    For prescribed_dim=2: input is agent (x, y) per timestep — `locations` of shape [B, 2, T].
    For prescribed_dim=4: input is (x_a, y_a, wall_x, door_y) per timestep —
      caller must construct loc4 of shape [B, 4, T] before passing in.
    """

    def __init__(self, prescribed_dim=2, output_dim=512, hidden_dim=256, final_ln=True):
        super().__init__()
        self.mlp_output_dim = output_dim
        self.prescribed_dim = prescribed_dim
        self.projection = nn.Sequential(
            nn.Linear(prescribed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.final_ln = nn.LayerNorm(output_dim) if final_ln else nn.Identity()

    def forward(self, observations, locations=None):
        if locations is None:
            raise ValueError("PrescribedEncoder requires locations")
        # locations: [B, prescribed_dim, T]
        loc = locations.permute(0, 2, 1)  # -> [B, T, prescribed_dim]
        B, T, D_in = loc.shape
        if D_in != self.prescribed_dim:
            raise ValueError(
                f"locations has {D_in} channels but encoder expects {self.prescribed_dim}"
            )
        features = self.projection(loc.reshape(B * T, -1))
        features = self.final_ln(features)
        features = features.reshape(B, T, -1)
        return features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)


class HybridEncoder(nn.Module):
    """Combines prescribed coordinates with free pixel features. Output: [B, D, T, 1, 1]."""

    def __init__(self, pixel_encoder, prescribed_dim=2, prescribed_output_dim=128, final_ln=True):
        super().__init__()
        self.pixel_encoder = pixel_encoder
        self.prescribed_dim = prescribed_dim
        total_dim = pixel_encoder.mlp_output_dim
        free_dim = total_dim - prescribed_output_dim
        self.prescribed_output_dim = prescribed_output_dim
        self.free_dim = free_dim
        self.mlp_output_dim = total_dim
        self.prescribed_projection = nn.Sequential(
            nn.Linear(prescribed_dim, prescribed_output_dim),
            nn.ReLU(),
            nn.Linear(prescribed_output_dim, prescribed_output_dim),
        )
        self.pixel_reduction = nn.Linear(pixel_encoder.mlp_output_dim, free_dim)
        self.final_ln = nn.LayerNorm(total_dim) if final_ln else nn.Identity()

    def forward(self, observations, locations=None):
        if locations is None:
            raise ValueError("HybridEncoder requires locations")
        pixel_features = self.pixel_encoder(observations)
        B, D, T, _, _ = pixel_features.shape
        pixel_feat = pixel_features.squeeze(-1).squeeze(-1).transpose(1, 2)
        pixel_feat = self.pixel_reduction(pixel_feat)
        loc = locations.permute(0, 2, 1)
        loc_flat = loc.reshape(B * T, -1)
        prescribed_feat = self.prescribed_projection(loc_flat).reshape(B, T, -1)
        combined = torch.cat([prescribed_feat, pixel_feat], dim=-1)
        combined = self.final_ln(combined)
        return combined.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)


# ================================================================
# PrescribedJEPA — passes locations to encoder
# ================================================================

class PrescribedJEPA(JEPA):
    """JEPA that passes locations to encoder during training and planning."""

    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        super().__init__(encoder, aencoder, predictor, regularizer, predcost)
        self._current_locations = None

    def set_locations_for_planning(self, loc):
        self._current_locations = loc

    def clear_planning_locations(self):
        self._current_locations = None

    @torch.no_grad()
    def encode(self, observations):
        if self._current_locations is not None and hasattr(self.encoder, 'prescribed_dim'):
            return self.encoder(observations, locations=self._current_locations)
        if hasattr(self.encoder, 'prescribed_dim'):
            raise ValueError("PrescribedEncoder.encode() needs locations. Call set_locations_for_planning().")
        return self.encoder(observations)

    def unroll(self, observations, actions, nsteps=1, unroll_mode="parallel",
               ctxt_window_time=1, compute_loss=True, return_all_steps=False,
               locations=None):
        if locations is not None and hasattr(self.encoder, 'prescribed_dim'):
            state = self.encoder(observations, locations=locations)
        elif self._current_locations is not None and hasattr(self.encoder, 'prescribed_dim'):
            state = self.encoder(observations, locations=self._current_locations)
        else:
            state = self.encoder(observations)

        context_length = getattr(self.predictor, "context_length", 0)

        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        actions_encoded = self.action_encoder(actions) if actions is not None else None
        all_steps = [] if return_all_steps else None

        if unroll_mode == "parallel":
            predicted_states = state
            for _ in range(nsteps):
                predicted_states = self.predictor(predicted_states, actions_encoded)[:, :, :-1]
                if return_all_steps:
                    all_steps.append(predicted_states)
                predicted_states = torch.cat(
                    (state[:, :, :context_length], predicted_states), dim=2
                )
                if compute_loss:
                    ploss += self.predcost(state, predicted_states) / nsteps

        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(f"nsteps ({nsteps}) > actions ({actions.size(2)})")
            effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time
            predicted_states = state[:, :, :effective_ctxt_window]
            for i in range(nsteps):
                context_states = predicted_states[:, :, -effective_ctxt_window:]
                if actions_encoded is not None:
                    context_actions = actions_encoded[
                        :, :, max(0, i + 1 - effective_ctxt_window): i + 1
                    ]
                else:
                    context_actions = None
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += torch.nn.functional.mse_loss(
                        pred_step, state[:, :, i + 1: i + 2]
                    ) / nsteps
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        if compute_loss:
            losses = (ploss + rloss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        return (all_steps if return_all_steps else predicted_states), losses


# ================================================================
# Condition configs
# ================================================================

CONDITIONS = {
    'free': {},
    'prescribed': {'encoder_type': 'prescribed', 'prescribed_dim': 2},
    'prescribed_4': {'encoder_type': 'prescribed', 'prescribed_dim': 4},
    'hybrid': {'encoder_type': 'hybrid', 'prescribed_dim': 2},
    'hybrid_4': {'encoder_type': 'hybrid', 'prescribed_dim': 4},
    'prescribed_no_idm': {'encoder_type': 'prescribed', 'prescribed_dim': 2, 'idm_coeff': 0},
    'prescribed_no_vicreg': {'encoder_type': 'prescribed', 'prescribed_dim': 2, 'std_coeff': 0, 'cov_coeff': 0},
    'prescribed_no_sim': {'encoder_type': 'prescribed', 'prescribed_dim': 2, 'sim_coeff_t': 0},
    'prescribed_4_no_sim': {'encoder_type': 'prescribed', 'prescribed_dim': 4, 'sim_coeff_t': 0},
}


# ================================================================
# Save helpers — dual save (local + drive)
# ================================================================

def _local_dir(mode, local_base):
    d = os.path.join(local_base, mode)
    os.makedirs(d, exist_ok=True)
    return d


def _drive_dir(mode, drive_base):
    if drive_base is None:
        return None
    d = os.path.join(drive_base, mode)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        logger.warning(f"Cannot create drive dir {d}: {e}. Drive save skipped.")
        return None
    return d


def _safe_copy(src, dst):
    """Copy with try/except — if drive is offline, skip silently."""
    if dst is None:
        return False
    if os.path.abspath(src) == os.path.abspath(dst):
        return False
    try:
        shutil.copy2(src, dst)
        return True
    except (OSError, IOError, shutil.Error) as e:
        logger.warning(f"Copy to drive failed ({src} -> {dst}): {e}")
        return False


def _save_endepoch(mode, local_base, drive_base, results_log, epoch_stats):
    """End-of-epoch save: writes to local, then mirrors to drive (best-effort).

    Local is source of truth. Drive is backup.
    """
    local_dst = _local_dir(mode, local_base)
    local_ckpt = os.path.join(local_dst, "latest.pth.tar")  # already saved by caller

    # Local: results.json (full history)
    with open(os.path.join(local_dst, "results.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    # Local: encoder_stats.json (append)
    stats_path = os.path.join(local_dst, "encoder_stats.json")
    existing = []
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not read existing {stats_path}, starting fresh.")
            existing = []
    existing.append(epoch_stats)
    with open(stats_path, "w") as f:
        json.dump(existing, f, indent=2)

    # Drive mirror (best-effort)
    drive_dst = _drive_dir(mode, drive_base)
    if drive_dst is not None:
        n_copied = 0
        for fname in ("latest.pth.tar", "results.json", "encoder_stats.json"):
            src = os.path.join(local_dst, fname)
            if os.path.exists(src):
                if _safe_copy(src, os.path.join(drive_dst, fname)):
                    n_copied += 1
        logger.info(f"  [SAVED] local: {local_dst} | drive: {drive_dst} ({n_copied}/3 files mirrored) "
                    f"| results: {len(results_log)} epochs")
    else:
        logger.info(f"  [SAVED] local only: {local_dst} | results: {len(results_log)} epochs")


def _save_midepoch_local_only(mode, local_base):
    """Mid-epoch checkpoint already written to local by caller. Nothing to do here.

    We DO NOT mirror to drive mid-epoch — that's slow and unreliable.
    Drive sync only at end-of-epoch.
    """
    pass


def _load_resume(mode, local_base, drive_base, device, jepa, jepa_opt, jepa_sched, xy_head,
                 probe_opt=None, probe_sched=None):
    """Resume: prefer local; fall back to drive if local missing or corrupt."""
    local_path = os.path.join(_local_dir(mode, local_base), "latest.pth.tar")
    drive_path = None
    drive_dst = _drive_dir(mode, drive_base)
    if drive_dst is not None:
        drive_path = os.path.join(drive_dst, "latest.pth.tar")

    candidates = []
    for path, label in ((local_path, 'local'), (drive_path, 'drive')):
        if path and os.path.exists(path):
            candidates.append((path, label, os.path.getmtime(path)))

    if not candidates:
        logger.info(f"No checkpoint found (local: {local_path}, drive: {drive_path}). Starting from scratch.")
        return 0, 0, []

    # Sort by mtime, newest first
    candidates.sort(key=lambda x: -x[2])
    ckpt_path, label, _ = candidates[0]
    logger.info(f"Resuming from {label}: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load {label} checkpoint {ckpt_path}: {e}")
        # Try fallback if available
        if len(candidates) > 1:
            ckpt_path, label, _ = candidates[1]
            logger.info(f"Fallback to {label}: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        else:
            raise

    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa_opt.load_state_dict(ckpt["optimizer_state_dict"])
    jepa_sched.load_state_dict(ckpt["scheduler_state_dict"])
    if "xy_head_state_dict" in ckpt:
        xy_head.load_state_dict(ckpt["xy_head_state_dict"])
    if probe_opt and "probe_optimizer_state_dict" in ckpt:
        probe_opt.load_state_dict(ckpt["probe_optimizer_state_dict"])
    if probe_sched and "probe_scheduler_state_dict" in ckpt:
        probe_sched.load_state_dict(ckpt["probe_scheduler_state_dict"])

    saved_epoch = ckpt["epoch"]
    saved_step = ckpt.get("batch_idx", -1)
    if saved_step == -1:
        start_epoch = saved_epoch + 1
        start_step = 0
    else:
        start_epoch = saved_epoch
        start_step = saved_step + 1
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Read results.json — prefer local, fall back to drive
    results_log = []
    for path in (os.path.join(_local_dir(mode, local_base), "results.json"),
                 os.path.join(drive_dst, "results.json") if drive_dst else None):
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    results_log = json.load(f)
                break
            except (json.JSONDecodeError, IOError):
                continue

    logger.info(f"Resumed: epoch {start_epoch}, batch {start_step}, {len(results_log)} results loaded")
    return start_epoch, start_step, results_log


# ================================================================
# Encoder stats for drift analysis
# ================================================================

@torch.no_grad()
def compute_encoder_stats(jepa, loader, device, locations_available, prescribed_dim=2):
    all_features = []
    n_batches = min(10, len(loader))
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x, a, loc, wall_x, door_y = batch
        x = x.to(device); loc = loc.to(device)
        wall_x = wall_x.to(device); door_y = door_y.to(device)

        if locations_available and hasattr(jepa.encoder, 'prescribed_dim'):
            loc_in = build_loc_input(loc, wall_x, door_y, prescribed_dim)
            feat = jepa.encoder(x, locations=loc_in)
        else:
            feat = jepa.encoder(x)
        B, D, T, _, _ = feat.shape
        feat_flat = feat.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)
        all_features.append(feat_flat.cpu())
    all_features = torch.cat(all_features, dim=0)
    return {
        'mean_per_dim': all_features.mean(dim=0).tolist()[:20],
        'std_per_dim': all_features.std(dim=0).tolist()[:20],
        'global_mean': all_features.mean().item(),
        'global_std': all_features.std().item(),
        'n_samples': all_features.shape[0],
    }


# Z-score normalization constants for locations (from EB-JEPA Normalizer.normalize_location).
# Same statistics that are applied to agent (x, y) by dataset.normalizer when normalize=True
# in data_config. We apply the same to wall_x and door_y so that the four prescribed
# coordinates live on a single consistent scale.
LOC_MEAN_X = 31.5863
LOC_MEAN_Y = 32.0618
LOC_STD_X = 16.1025
LOC_STD_Y = 16.1353


def build_loc_input(loc, wall_x, door_y, prescribed_dim):
    """Build location tensor for prescribed encoder.

    Agent loc is already z-score normalized by dataset.normalizer.normalize_location
    when normalize=True in data_config (mean ~ [31.59, 32.06], std ~ [16.10, 16.14]).
    For prescribed_4, we apply the SAME z-score to wall_x and door_y so that all four
    coordinates are in a single normalized space — this matches LeCun's normalization
    methodology and avoids mixing scales (which would force the encoder to compensate).

    Decision rationale (work_plan_2026_04_30.md): z-score chosen over min-max [0,1]
    to keep one variable changing relative to E32 (encoder input), not two
    (encoder input + normalization scheme).

    Inputs:
        loc:     [B, 2, T] — agent (x, y) trajectory, ALREADY z-score normalized.
        wall_x:  [B] or [B, 1] — wall x-coordinate, RAW pixel coords (0..65).
        door_y:  [B] or [B, 1] — door y-coordinate, RAW pixel coords (0..65).
        prescribed_dim: 2 or 4.

    Output:
        For dim=2: returns loc unchanged.
        For dim=4: [B, 4, T] = (x_a, y_a, wall_x_z, door_y_z) — all z-score normalized.
    """
    if prescribed_dim == 2:
        return loc

    if prescribed_dim == 4:
        B, _, T = loc.shape
        # Reshape wall_x, door_y from [B] or [B, 1] to [B, 1, T] (broadcast over time).
        wx = wall_x.view(B, 1, 1).float().expand(B, 1, T)
        dy = door_y.view(B, 1, 1).float().expand(B, 1, T)
        # Apply z-score with X-statistics for wall_x (it's an x-coordinate),
        # Y-statistics for door_y (it's a y-coordinate).
        wx = (wx - LOC_MEAN_X) / LOC_STD_X
        dy = (dy - LOC_MEAN_Y) / LOC_STD_Y
        loc4 = torch.cat([loc, wx, dy], dim=1)  # [B, 4, T]
        return loc4

    raise ValueError(f"Unsupported prescribed_dim={prescribed_dim}")


# ================================================================
# Build encoder
# ================================================================

def build_encoder(encoder_type, cfg, data_config, prescribed_dim=2):
    if encoder_type == 'free':
        return ImpalaEncoder(
            width=1,
            stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
            num_blocks=2, dropout_rate=None, layer_norm=False,
            input_channels=cfg.model.dobs, final_ln=True,
            mlp_output_dim=512,
            input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
        )
    elif encoder_type == 'prescribed':
        return PrescribedEncoder(prescribed_dim=prescribed_dim, output_dim=512)
    elif encoder_type == 'hybrid':
        pixel_enc = ImpalaEncoder(
            width=1,
            stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
            num_blocks=2, dropout_rate=None, layer_norm=False,
            input_channels=cfg.model.dobs, final_ln=True,
            mlp_output_dim=512,
            input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
        )
        return HybridEncoder(pixel_enc, prescribed_dim=prescribed_dim, prescribed_output_dim=128)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


# ================================================================
# Main training
# ================================================================

def run_condition(mode, local_base, drive_base=None):
    r"""Run a training condition with dual save (local + drive).

    local_base: path on D:\ — fast, source of truth, where mid-epoch checkpoints go.
    drive_base: optional path on Google Drive — best-effort backup, mirrored only
                at end of epoch. Pass None to disable drive sync.
    """
    cond = CONDITIONS[mode]
    encoder_type = cond.get('encoder_type', 'free')
    prescribed_dim = cond.get('prescribed_dim', 2)
    locations_available = encoder_type in ('prescribed', 'hybrid')

    logger.info(f"=== {mode} === encoder={encoder_type}, prescribed_dim={prescribed_dim}, "
                f"locations_available={locations_available}")
    logger.info(f"Save targets: local={local_base}, drive={drive_base}")

    cfg = load_config("examples/ac_video_jepa/cfgs/train.yaml")
    # Colab overrides
    cfg.logging.log_wandb = False
    cfg.data.num_workers = 0
    cfg.data.pin_mem = False
    cfg.data.persistent_workers = False
    cfg.data.batch_size = 64
    cfg.training.dtype = "float16"
    cfg.model.compile = False
    cfg.meta.load_model = False
    cfg.meta.enable_plan_eval = False

    # Ablation overrides
    for key in ('idm_coeff', 'std_coeff', 'cov_coeff', 'sim_coeff_t'):
        if key in cond:
            setattr(cfg.model.regularizer, key, cond[key])
            logger.info(f"  Ablation: {key}={cond[key]}")

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )
    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device(DEVICE)

    use_amp = (DEVICE == 'cuda')
    scaler = GradScaler(device.type, enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    local_folder = Path(_local_dir(mode, local_base))
    steps_per_epoch = len(loader)
    total_steps = cfg.optim.epochs * steps_per_epoch

    # Build model
    encoder = build_encoder(encoder_type, cfg, data_config, prescribed_dim=prescribed_dim)
    mlp_dim = encoder.mlp_output_dim

    # Get final_ln from encoder
    enc_final_ln = getattr(encoder, 'final_ln', None)
    if enc_final_ln is None or isinstance(enc_final_ln, bool):
        enc_final_ln = nn.LayerNorm(mlp_dim)

    predictor = RNNPredictor(hidden_size=mlp_dim, final_ln=enc_final_ln)
    aencoder = nn.Identity()

    idm = InverseDynamicsModel(state_dim=mlp_dim, hidden_dim=256, action_dim=2).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.get("idm_coeff", 0.1),
        idm=idm,
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj,
    )
    ploss_fn = SquareLossSeq()

    if locations_available:
        jepa = PrescribedJEPA(encoder, aencoder, predictor, regularizer, ploss_fn).to(device)
    else:
        jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss_fn).to(device)

    xy_head = MLPXYHead(
        input_shape=mlp_dim,
        normalizer=loader.dataset.normalizer,
    ).to(device)
    xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())

    jepa_opt = AdamW(jepa.parameters(), lr=cfg.optim.lr,
                     weight_decay=cfg.optim.get("weight_decay", 1e-6))
    jepa_sched = CosineWithWarmup(jepa_opt, total_steps, warmup_ratio=0.1)
    probe_opt = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_sched = CosineWithWarmup(probe_opt, total_steps, warmup_ratio=0.1)

    start_epoch, start_step, results_log = _load_resume(
        mode, local_base, drive_base, device, jepa, jepa_opt, jepa_sched, xy_head,
        probe_opt, probe_sched
    )

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    logger.info(f"=== {mode} === encoder={enc_params:,} predictor={pred_params:,}")
    logger.info(f"Epochs: {start_epoch} -> {cfg.optim.epochs}, batches/epoch: {steps_per_epoch}")
    logger.info(f"Config: cov={cfg.model.regularizer.cov_coeff} std={cfg.model.regularizer.std_coeff} "
                f"sim={cfg.model.regularizer.sim_coeff_t} idm={cfg.model.regularizer.get('idm_coeff', 0.1)}")

    # ---- Training loop ----
    SAVE_EVERY_N_BATCHES = 200  # mid-epoch checkpoint (LOCAL only — drive sync only at end of epoch)

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time.time()
        ep_losses = {'pred': [], 'reg': [], 'probe': [], 'total': []}
        ep_reg = {'std': [], 'cov': [], 'idm': [], 'sim': []}

        # Determine which batch to start from (for mid-epoch resume)
        skip_to = start_step if epoch == start_epoch else 0

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"[{mode}] Ep {epoch}/{cfg.optim.epochs-1}")

        for idx, (x, a, loc, wall_x, door_y) in pbar:
            # Skip batches already completed (mid-epoch resume)
            if idx < skip_to:
                if idx == 0:
                    logger.info(f"Skipping {skip_to} batches (mid-epoch resume)...")
                continue
            x = x.to(device); a = a.to(device); loc = loc.to(device)
            wall_x = wall_x.to(device); door_y = door_y.to(device)

            # Build location input for encoder.
            # For prescribed_dim=2: loc_in == loc.
            # For prescribed_dim=4: loc_in == [loc; wall_x_z; door_y_z], all z-score normalized.
            # For free encoder: locations_available is False, loc_in is unused.
            if locations_available:
                loc_in = build_loc_input(loc, wall_x, door_y, prescribed_dim)
            else:
                loc_in = None

            jepa_opt.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                if locations_available:
                    _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                        x, a, nsteps=cfg.model.nsteps,
                        unroll_mode="autoregressive", ctxt_window_time=1,
                        compute_loss=True, locations=loc_in,
                    )
                else:
                    _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                        x, a, nsteps=cfg.model.nsteps,
                        unroll_mode="autoregressive", ctxt_window_time=1,
                        compute_loss=True,
                    )

            scaler.scale(jepa_loss).backward()
            if cfg.optim.get("grad_clip_enc") and cfg.optim.get("grad_clip_pred"):
                scaler.unscale_(jepa_opt)
                torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), cfg.optim.grad_clip_enc)
                torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), cfg.optim.grad_clip_pred)
            scaler.step(jepa_opt)
            scaler.update()
            jepa_sched.step()

            probe_opt.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                with torch.no_grad():
                    if locations_available:
                        # Probe sees encoder output (always 512-D), not raw locations.
                        # We pass first-timestep loc_in to the encoder so it can produce
                        # the latent state; the probe head then maps latent → (x_a, y_a).
                        loc_probe_in = loc_in[:, :, :1]
                        probe_state = jepa.encoder(x[:, :, :1], locations=loc_probe_in)
                    else:
                        probe_state = jepa.encoder(x[:, :, :1])
                probe_pred = xy_head(probe_state.detach())
                # Probe target is always 2D (x_a, y_a) — same as in E32, comparable across
                # prescribed_2, prescribed_4, free.
                xy_loss = nn.functional.mse_loss(probe_pred, loc[:, :, :1])
                xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
            scaler.scale(xy_loss).backward()
            scaler.step(probe_opt)
            scaler.update()
            probe_sched.step()

            ep_losses['pred'].append(pl.item())
            ep_losses['reg'].append(regl.item())
            ep_losses['probe'].append(xy_loss.item())
            ep_losses['total'].append(jepa_loss.item())
            ep_reg['std'].append(regldict.get('std_loss', 0))
            ep_reg['cov'].append(regldict.get('cov_loss', 0))
            ep_reg['idm'].append(regldict.get('idm_loss', 0))
            ep_reg['sim'].append(regldict.get('sim_loss_t', 0))

            pbar.set_postfix({
                'pred': f"{pl.item():.4f}",
                'reg': f"{regl.item():.4f}",
                'probe': f"{xy_loss.item():.4f}",
            })

            # Mid-epoch checkpoint every N batches — LOCAL ONLY.
            # Drive sync happens at end of epoch only (slow + unreliable mid-epoch).
            if (idx + 1) % SAVE_EVERY_N_BATCHES == 0 and idx + 1 < len(loader):
                save_checkpoint(
                    local_folder / "latest.pth.tar",
                    model=jepa, optimizer=jepa_opt, scheduler=jepa_sched,
                    epoch=epoch, step=epoch * steps_per_epoch + idx,
                    xy_head_state_dict=xy_head.state_dict(),
                    probe_optimizer_state_dict=probe_opt.state_dict(),
                    probe_scheduler_state_dict=probe_sched.state_dict(),
                    batch_idx=idx,
                )
                # Mid-epoch: LOCAL ONLY. Drive sync only at end of epoch.
                logger.info(f"Mid-epoch save (local): ep {epoch} batch {idx+1}/{len(loader)}")

        epoch_time = time.time() - epoch_start
        avg = {k: float(np.mean(v)) for k, v in ep_losses.items()}
        avg_reg = {k: float(np.mean(v)) for k, v in ep_reg.items()}

        # Planning eval disabled — too slow on CPU. Will run separately after training.
        sr, md = -1.0, -1.0

        enc_stats = compute_encoder_stats(jepa, loader, device, locations_available, prescribed_dim=prescribed_dim)
        enc_stats['epoch'] = epoch

        logger.info(f"[{mode}] Ep {epoch}: pred={avg['pred']:.4f} reg={avg['reg']:.4f} "
                    f"probe={avg['probe']:.4f} time={epoch_time:.0f}s "
                    f"(prescribed_dim={prescribed_dim})")

        epoch_result = {
            'epoch': epoch, 'mode': mode,
            'prescribed_dim': prescribed_dim,  # audit trail
            'pred_loss': avg['pred'], 'reg_loss': avg['reg'],
            'probe_loss': avg['probe'], 'total_loss': avg['total'],
            'std_loss': avg_reg['std'], 'cov_loss': avg_reg['cov'],
            'idm_loss': avg_reg['idm'], 'sim_loss': avg_reg['sim'],
            'success_rate': sr, 'mean_dist': md, 'epoch_time': epoch_time,
            'encoder_params': enc_params, 'predictor_params': pred_params,
        }
        results_log.append(epoch_result)

        # End-of-epoch save: write checkpoint locally, then dual-mirror via _save_endepoch.
        save_checkpoint(
            local_folder / "latest.pth.tar",
            model=jepa, optimizer=jepa_opt, scheduler=jepa_sched,
            epoch=epoch, step=(epoch + 1) * steps_per_epoch,
            xy_head_state_dict=xy_head.state_dict(),
            probe_optimizer_state_dict=probe_opt.state_dict(),
            probe_scheduler_state_dict=probe_sched.state_dict(),
            batch_idx=-1,
        )
        _save_endepoch(mode, local_base, drive_base, results_log, enc_stats)

    logger.info(f"=== {mode} COMPLETE === {len(results_log)} epochs (prescribed_dim={prescribed_dim})")
    return results_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="EB-JEPA training v4 — prescribed_4 + dual save (local + drive)"
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=list(CONDITIONS.keys()),
        help="Training condition (see CONDITIONS dict)."
    )
    parser.add_argument(
        "--local_base", type=str, required=True,
        help=r"Local results dir, e.g. D:\experiments\E33_prescribed_4\results. "
             "Source of truth. Mid-epoch checkpoints write here."
    )
    parser.add_argument(
        "--drive_base", type=str, default=None,
        help="Optional Drive results dir for backup mirror at end of each epoch. "
             "If unreachable, training continues on local only."
    )
    args = parser.parse_args()
    run_condition(args.mode, args.local_base, args.drive_base)
