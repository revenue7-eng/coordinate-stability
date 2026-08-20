"""
EB-JEPA Prescribed Axes Experiment v3.

Reproduces LeCun et al. (2602.03604) Two Rooms experiment with prescribed axes.
Clean implementation: no patches, no monkey-patching.

Per-epoch: checkpoint + results.json + planning eval saved to Google Drive.
Resume: automatic from Drive checkpoint.
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
    """Uses agent coordinates (x,y) instead of pixels. Output: [B, D, T, 1, 1]."""

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
        loc = locations.permute(0, 2, 1)  # [B, 2, T] -> [B, T, 2]
        B, T, _ = loc.shape
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
    'prescribed': {'encoder_type': 'prescribed'},
    'hybrid': {'encoder_type': 'hybrid'},
    'prescribed_no_idm': {'encoder_type': 'prescribed', 'idm_coeff': 0},
    'prescribed_no_vicreg': {'encoder_type': 'prescribed', 'std_coeff': 0, 'cov_coeff': 0},
    'prescribed_no_sim': {'encoder_type': 'prescribed', 'sim_coeff_t': 0},
}


# ================================================================
# Drive helpers
# ================================================================

def _drive_dir(mode, drive_base):
    d = os.path.join(drive_base, mode)
    os.makedirs(d, exist_ok=True)
    return d


def _save_to_drive(mode, drive_base, local_folder, results_log, epoch_stats):
    dst = _drive_dir(mode, drive_base)
    src_ckpt = os.path.join(local_folder, "latest.pth.tar")
    dst_ckpt = os.path.join(dst, "latest.pth.tar")
    if os.path.exists(src_ckpt) and os.path.abspath(src_ckpt) != os.path.abspath(dst_ckpt):
        shutil.copy2(src_ckpt, dst_ckpt)
    with open(os.path.join(dst, "results.json"), "w") as f:
        json.dump(results_log, f, indent=2)
    stats_path = os.path.join(dst, "encoder_stats.json")
    existing = []
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            existing = json.load(f)
    existing.append(epoch_stats)
    with open(stats_path, "w") as f:
        json.dump(existing, f, indent=2)
    # Also print to confirm save
    print(f"  [SAVED] {dst} — results: {len(results_log)} epochs")


def _load_resume(mode, drive_base, device, jepa, jepa_opt, jepa_sched, xy_head,
                 probe_opt=None, probe_sched=None):
    ckpt_path = os.path.join(_drive_dir(mode, drive_base), "latest.pth.tar")
    if not os.path.exists(ckpt_path):
        return 0, 0, []
    logger.info(f"Resuming from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    saved_step = ckpt.get("batch_idx", -1)  # -1 means end of epoch
    if saved_step == -1:
        # Full epoch completed — start next epoch from batch 0
        start_epoch = saved_epoch + 1
        start_step = 0
    else:
        # Mid-epoch checkpoint — resume same epoch from next batch
        start_epoch = saved_epoch
        start_step = saved_step + 1
    del ckpt
    torch.cuda.empty_cache()
    results_path = os.path.join(_drive_dir(mode, drive_base), "results.json")
    results_log = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results_log = json.load(f)
    logger.info(f"Resumed: epoch {start_epoch}, batch {start_step}, {len(results_log)} results loaded")
    return start_epoch, start_step, results_log


# ================================================================
# Encoder stats for drift analysis
# ================================================================

@torch.no_grad()
def compute_encoder_stats(jepa, loader, device, locations_available):
    all_features = []
    n_batches = min(10, len(loader))
    for i, (x, a, loc, _, _) in enumerate(loader):
        if i >= n_batches:
            break
        x, loc = x.to(device), loc.to(device)
        if locations_available and hasattr(jepa.encoder, 'prescribed_dim'):
            feat = jepa.encoder(x, locations=loc)
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


# ================================================================
# Build encoder
# ================================================================

def build_encoder(encoder_type, cfg, data_config):
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
        return PrescribedEncoder(prescribed_dim=2, output_dim=512)
    elif encoder_type == 'hybrid':
        pixel_enc = ImpalaEncoder(
            width=1,
            stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
            num_blocks=2, dropout_rate=None, layer_norm=False,
            input_channels=cfg.model.dobs, final_ln=True,
            mlp_output_dim=512,
            input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
        )
        return HybridEncoder(pixel_enc, prescribed_dim=2, prescribed_output_dim=128)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


# ================================================================
# Main training
# ================================================================

def run_condition(mode, drive_base):
    cond = CONDITIONS[mode]
    encoder_type = cond.get('encoder_type', 'free')
    locations_available = encoder_type in ('prescribed', 'hybrid')

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

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )
    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device(DEVICE)

    use_amp = (DEVICE == 'cuda')
    scaler = GradScaler(device.type, enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    local_folder = Path(os.path.join(drive_base, mode))
    os.makedirs(local_folder, exist_ok=True)
    steps_per_epoch = len(loader)
    total_steps = cfg.optim.epochs * steps_per_epoch

    # Build model
    encoder = build_encoder(encoder_type, cfg, data_config)
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
        mode, drive_base, device, jepa, jepa_opt, jepa_sched, xy_head,
        probe_opt, probe_sched
    )

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    logger.info(f"=== {mode} === encoder={enc_params:,} predictor={pred_params:,}")
    logger.info(f"Epochs: {start_epoch} -> {cfg.optim.epochs}, batches/epoch: {steps_per_epoch}")
    logger.info(f"Config: cov={cfg.model.regularizer.cov_coeff} std={cfg.model.regularizer.std_coeff} "
                f"sim={cfg.model.regularizer.sim_coeff_t} idm={cfg.model.regularizer.get('idm_coeff', 0.1)}")

    # ---- Training loop ----
    SAVE_EVERY_N_BATCHES = 200  # mid-epoch checkpoint

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time.time()
        ep_losses = {'pred': [], 'reg': [], 'probe': [], 'total': []}
        ep_reg = {'std': [], 'cov': [], 'idm': [], 'sim': []}

        # Determine which batch to start from (for mid-epoch resume)
        skip_to = start_step if epoch == start_epoch else 0

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"[{mode}] Ep {epoch}/{cfg.optim.epochs-1}")

        for idx, (x, a, loc, _, _) in pbar:
            # Skip batches already completed (mid-epoch resume)
            if idx < skip_to:
                if idx == 0:
                    logger.info(f"Skipping {skip_to} batches (mid-epoch resume)...")
                continue
            x, a, loc = x.to(device), a.to(device), loc.to(device)

            jepa_opt.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                if locations_available:
                    _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                        x, a, nsteps=cfg.model.nsteps,
                        unroll_mode="autoregressive", ctxt_window_time=1,
                        compute_loss=True, locations=loc,
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
                        probe_state = jepa.encoder(x[:, :, :1], locations=loc[:, :, :1])
                    else:
                        probe_state = jepa.encoder(x[:, :, :1])
                probe_pred = xy_head(probe_state.detach())
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

            # Mid-epoch checkpoint every N batches
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
                dst = _drive_dir(mode, drive_base)
                dst_file = os.path.join(dst, "latest.pth.tar")
                src_file = str(local_folder / "latest.pth.tar")
                if os.path.abspath(src_file) != os.path.abspath(dst_file):
                    shutil.copy2(src_file, dst_file)
                logger.info(f"Mid-epoch save: ep {epoch} batch {idx+1}/{len(loader)}")

        epoch_time = time.time() - epoch_start
        avg = {k: float(np.mean(v)) for k, v in ep_losses.items()}
        avg_reg = {k: float(np.mean(v)) for k, v in ep_reg.items()}

        # Planning eval disabled — too slow on CPU. Will run separately after training.
        sr, md = -1.0, -1.0
        logger.info(f"Planning eval disabled (will run separately after training)")

        enc_stats = compute_encoder_stats(jepa, loader, device, locations_available)
        enc_stats['epoch'] = epoch

        logger.info(f"[{mode}] Ep {epoch}: pred={avg['pred']:.4f} reg={avg['reg']:.4f} "
                    f"probe={avg['probe']:.4f} SR={sr} time={epoch_time:.0f}s")

        epoch_result = {
            'epoch': epoch, 'mode': mode,
            'pred_loss': avg['pred'], 'reg_loss': avg['reg'],
            'probe_loss': avg['probe'], 'total_loss': avg['total'],
            'std_loss': avg_reg['std'], 'cov_loss': avg_reg['cov'],
            'idm_loss': avg_reg['idm'], 'sim_loss': avg_reg['sim'],
            'success_rate': sr, 'mean_dist': md, 'epoch_time': epoch_time,
            'encoder_params': enc_params, 'predictor_params': pred_params,
        }
        results_log.append(epoch_result)

        save_checkpoint(
            local_folder / "latest.pth.tar",
            model=jepa, optimizer=jepa_opt, scheduler=jepa_sched,
            epoch=epoch, step=(epoch + 1) * steps_per_epoch,
            xy_head_state_dict=xy_head.state_dict(),
            probe_optimizer_state_dict=probe_opt.state_dict(),
            probe_scheduler_state_dict=probe_sched.state_dict(),
            batch_idx=-1,
        )
        _save_to_drive(mode, drive_base, str(local_folder), results_log, enc_stats)
        logger.info(f"Saved to Drive: {_drive_dir(mode, drive_base)}")

    logger.info(f"=== {mode} COMPLETE === {len(results_log)} epochs")
    return results_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=list(CONDITIONS.keys()))
    parser.add_argument("--drive_base", type=str, default="results")
    args = parser.parse_args()
    run_condition(args.mode, args.drive_base)
