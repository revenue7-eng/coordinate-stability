"""
Planning eval for EB-JEPA v3 — runs on Colab GPU.

Loads checkpoints from Drive, runs planning eval (success rate),
saves results back to Drive.

Handles both free (ImpalaEncoder) and prescribed (PrescribedEncoder) models.
For prescribed: patches GCAgent to pass locations from env.info to encoder.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

# EB-JEPA imports
from eb_jepa.architectures import InverseDynamicsModel, RNNPredictor
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from eb_jepa.planning import GCAgent

# Import the training-side definitions instead of keeping a second copy.
# Divergence between duplicated encoder code has cost debugging time before.
# PrescribedJEPA is NOT imported: the planning variant below adds an expand()
# over the MPPI candidate batch, which the training variant does not need.
from run_experiment_v4_windows import (
    CONDITIONS, PrescribedEncoder, HybridEncoder, build_encoder, build_loc_input,
)

logger = get_logger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# Encoders (same as training script)
# ================================================================

class PrescribedJEPA(JEPA):
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
            loc = self._current_locations
            # Expand locations to match observation batch size
            B = observations.shape[0]
            if loc.shape[0] == 1 and B > 1:
                loc = loc.expand(B, -1, -1)
            return self.encoder(observations, locations=loc)
        if hasattr(self.encoder, 'prescribed_dim'):
            raise ValueError("PrescribedEncoder.encode() needs locations.")
        return self.encoder(observations)

    def unroll(self, observations, actions, nsteps=1, unroll_mode="parallel",
               ctxt_window_time=1, compute_loss=True, return_all_steps=False,
               locations=None):
        if locations is not None and hasattr(self.encoder, 'prescribed_dim'):
            B = observations.shape[0]
            if locations.shape[0] == 1 and B > 1:
                locations = locations.expand(B, -1, -1)
            state = self.encoder(observations, locations=locations)
        elif self._current_locations is not None and hasattr(self.encoder, 'prescribed_dim'):
            loc = self._current_locations
            B = observations.shape[0]
            if loc.shape[0] == 1 and B > 1:
                loc = loc.expand(B, -1, -1)
            state = self.encoder(observations, locations=loc)
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
# Patched planning eval for prescribed encoder
# ================================================================

def planning_eval_prescribed(plan_cfg, model, env_creator, eval_folder,
                              num_episodes=10, loader=None, prober=None,
                              prescribed_dim=2):
    """
    Planning eval that passes locations from env.info to prescribed encoder.
    
    Key changes from main_eval:
    1. set_goal: sets locations from target_position before model.encode
    2. act loop: sets locations from dot_position before each planning step
    """
    plan_cfg = OmegaConf.create(plan_cfg)
    env = env_creator()
    env.reset()

    agent = GCAgent(
        model, action_dim=2, plan_cfg=plan_cfg,
        normalizer=env.normalizer, loc_prober=prober, env=env,
    )
    logger.info(f"Prescribed planning eval with {agent.planner.__class__.__name__}")

    successes = []
    distances = []
    episode_times = []
    # Episode geometry. Not recoverable after the run: needed to normalize the
    # final distance against the scale of the task (start-goal separation,
    # wall/door layout) without committing to a threshold before the run.
    geometry = []

    for ep in range(num_episodes):
        ep_start = time.time()
        ep_folder = Path(eval_folder) / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)

        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(np.zeros(env.action_space.shape[0]))
        goal_img = info["target_obs"]
        goal_position = info["target_position"]
        dot_position = info["dot_position"]
        start_position = dot_position.detach().clone()

        # --- set_goal with locations ---
        # For prescribed encoder: set locations before encode
        goal_loc_tensor = goal_position.detach().clone().to(dtype=torch.float32)
        # Normalize on shape [2] BEFORE unsqueeze: normalize_location broadcasts
        # over the last axis, so a [1, 2, 1] input would silently become [1, 2, 2].
        goal_loc_tensor = env.normalizer.normalize_location(goal_loc_tensor).to(agent.device)
        # Shape: [2] -> [1, 2, 1] (batch=1, dim=2, time=1)
        goal_loc_for_enc = goal_loc_tensor.unsqueeze(0).unsqueeze(-1)
        goal_loc_for_enc = build_loc_input(
            goal_loc_for_enc, env.wall_x, env.hole_y, prescribed_dim
        )
        model.set_locations_for_planning(goal_loc_for_enc)
        agent.set_goal(goal_img.detach().clone().to(dtype=torch.float32), goal_position)
        model.clear_planning_locations()

        done = False
        steps_left = env.n_allowed_steps
        pbar = tqdm(desc=f"ep {ep}", total=steps_left, leave=True,
                    disable=plan_cfg.logging.tqdm_silent)
        t0 = True
        observations = [obs]

        while steps_left > 0:
            # Get current dot position from env info
            dot_position = info["dot_position"]
            dot_loc_tensor = dot_position.detach().clone().to(dtype=torch.float32)
            # Same normalization as training (wall_dataset applies normalize_location
            # when config.normalize is true). Must run on shape [2], see set_goal above.
            dot_loc_tensor = env.normalizer.normalize_location(dot_loc_tensor).to(agent.device)
            dot_loc_for_enc = dot_loc_tensor.unsqueeze(0).unsqueeze(-1)
            # wall_x / hole_y are set in env.reset() and constant within an episode.
            # build_loc_input applies the SAME z-score constants as training.
            dot_loc_for_enc = build_loc_input(
                dot_loc_for_enc, env.wall_x, env.hole_y, prescribed_dim
            )

            # Set locations for all encode() calls during planning
            model.set_locations_for_planning(dot_loc_for_enc)

            obs_tensor = (
                env.normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                ).unsqueeze(0).unsqueeze(2)
            )
            with torch.no_grad():
                action = agent.act(obs_tensor, steps_left=steps_left, t0=t0).cpu().numpy()

            model.clear_planning_locations()

            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                observations.append(obs)
                steps_left -= 1
                pbar.update(1)
                eval_results = env.eval_state(info["target_position"], info["dot_position"])
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            pbar.set_postfix({"success": success, "dist": f"{state_dist:.3f}"})
        pbar.close()

        successes.append(success)
        distances.append(state_dist)
        geometry.append({
            "wall_x": float(env.wall_x),
            "hole_y": float(env.hole_y),
            "start_position": [float(v) for v in start_position],
            "goal_position": [float(v) for v in goal_position],
            "final_position": [float(v) for v in info["dot_position"]],
        })
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        logger.info(f"  ep {ep}: {'SUCCESS' if success else 'FAIL'} dist={state_dist:.4f} time={ep_time:.0f}s")

    results = {
        "success_rate": float(np.mean(successes)),
        "mean_state_dist": float(np.mean(distances)),
        "avg_episode_time": float(np.mean(episode_times)),
        "successes": [bool(s) for s in successes],
        "distances": [float(d) for d in distances],
        "geometry": geometry,
    }
    with open(os.path.join(eval_folder, "planning_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"SR={results['success_rate']:.2f} mean_dist={results['mean_state_dist']:.4f}")
    return results


# ================================================================
# Build model (same as training)
# ================================================================

def planning_eval_free(plan_cfg, model, env_creator, eval_folder,
                       num_episodes=10, loader=None, prober=None):
    """
    Planning eval for the free (pixel) encoder.

    Structurally identical to planning_eval_prescribed minus the location
    plumbing, so both branches are measured by the same code path with the
    same accounting. Not main_eval: that one returns aggregates only (no
    per-episode successes/distances), writes a gif and a state.pdf on every
    episode regardless of plan_cfg.logging.optional_plots, and binds
    save_path inside the optional_plots branch while calling save_gif
    outside it (NameError when plots are off).
    """
    plan_cfg = OmegaConf.create(plan_cfg)
    env = env_creator()
    env.reset()

    agent = GCAgent(
        model, action_dim=2, plan_cfg=plan_cfg,
        normalizer=env.normalizer, loc_prober=prober, env=env,
    )
    logger.info(f"Free planning eval with {agent.planner.__class__.__name__}")

    successes = []
    distances = []
    episode_times = []
    geometry = []

    for ep in range(num_episodes):
        ep_start = time.time()
        ep_folder = Path(eval_folder) / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)

        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(np.zeros(env.action_space.shape[0]))
        goal_img = info["target_obs"]
        goal_position = info["target_position"]
        start_position = info["dot_position"].detach().clone()

        agent.set_goal(goal_img.detach().clone().to(dtype=torch.float32), goal_position)

        done = False
        steps_left = env.n_allowed_steps
        pbar = tqdm(desc=f"ep {ep}", total=steps_left, leave=True,
                    disable=plan_cfg.logging.tqdm_silent)
        t0 = True

        while steps_left > 0:
            obs_tensor = (
                env.normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                ).unsqueeze(0).unsqueeze(2)
            )
            with torch.no_grad():
                action = agent.act(obs_tensor, steps_left=steps_left, t0=t0).cpu().numpy()

            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                steps_left -= 1
                pbar.update(1)
                eval_results = env.eval_state(info["target_position"], info["dot_position"])
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            pbar.set_postfix({"success": success, "dist": f"{state_dist:.3f}"})
        pbar.close()

        successes.append(success)
        distances.append(state_dist)
        geometry.append({
            "wall_x": float(env.wall_x),
            "hole_y": float(env.hole_y),
            "start_position": [float(v) for v in start_position],
            "goal_position": [float(v) for v in goal_position],
            "final_position": [float(v) for v in info["dot_position"]],
        })
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        logger.info(f"  ep {ep}: {'SUCCESS' if success else 'FAIL'} dist={state_dist:.4f} time={ep_time:.0f}s")

    results = {
        "success_rate": float(np.mean(successes)),
        "mean_state_dist": float(np.mean(distances)),
        "avg_episode_time": float(np.mean(episode_times)),
        "successes": [bool(s) for s in successes],
        "distances": [float(d) for d in distances],
        "geometry": geometry,
    }
    with open(os.path.join(eval_folder, "planning_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"SR={results['success_rate']:.2f} mean_dist={results['mean_state_dist']:.4f}")
    return results


def load_model(mode, checkpoint_path, device):
    """Build model and load checkpoint."""
    cond = CONDITIONS[mode]
    encoder_type = cond.get('encoder_type', 'free')
    prescribed_dim = cond.get('prescribed_dim', 2)
    locations_available = encoder_type in ('prescribed', 'hybrid')

    cfg = load_config("examples/ac_video_jepa/cfgs/train.yaml")
    cfg.data.num_workers = 0
    cfg.data.pin_mem = False
    cfg.data.persistent_workers = False

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )

    encoder = build_encoder(encoder_type, cfg, data_config, prescribed_dim=prescribed_dim)
    mlp_dim = encoder.mlp_output_dim

    enc_final_ln = getattr(encoder, 'final_ln', None)
    if enc_final_ln is None or isinstance(enc_final_ln, bool):
        enc_final_ln = nn.LayerNorm(mlp_dim)

    predictor = RNNPredictor(hidden_size=mlp_dim, final_ln=enc_final_ln)
    aencoder = nn.Identity()

    idm = InverseDynamicsModel(state_dim=mlp_dim, hidden_dim=256, action_dim=2).to(device)

    # Apply ablation overrides
    reg_cfg = dict(cfg.model.regularizer)
    for key in ('idm_coeff', 'std_coeff', 'cov_coeff', 'sim_coeff_t'):
        if key in cond:
            reg_cfg[key] = cond[key]

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=reg_cfg.get('cov_coeff', cfg.model.regularizer.cov_coeff),
        std_coeff=reg_cfg.get('std_coeff', cfg.model.regularizer.std_coeff),
        sim_coeff_t=reg_cfg.get('sim_coeff_t', cfg.model.regularizer.sim_coeff_t),
        idm_coeff=reg_cfg.get('idm_coeff', cfg.model.regularizer.get("idm_coeff", 0.1)),
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

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    jepa.load_state_dict(ckpt["model_state_dict"])
    if "xy_head_state_dict" in ckpt:
        xy_head.load_state_dict(ckpt["xy_head_state_dict"])
    epoch = ckpt["epoch"]
    logger.info(f"Loaded epoch {epoch}")
    del ckpt
    torch.cuda.empty_cache()

    return (jepa, xy_prober, loader, val_loader, cfg, data_config, epoch,
            locations_available, prescribed_dim)


# ================================================================
# Main eval function
# ================================================================

def run_planning_eval(mode, drive_base, num_episodes=20):
    """Run planning eval for one condition."""
    device = torch.device(DEVICE)
    setup_device("auto")
    setup_seed(1)

    checkpoint_path = os.path.join(drive_base, mode, "latest.pth.tar")
    if not os.path.exists(checkpoint_path):
        logger.error(f"No checkpoint for {mode}: {checkpoint_path}")
        return None

    (jepa, xy_prober, loader, val_loader, cfg, data_config, epoch,
     locations_available, prescribed_dim) = load_model(mode, checkpoint_path, device)

    jepa.eval()

    # Load planning configs
    with open("examples/ac_video_jepa/cfgs/planning_mppi.yaml") as f:
        plan_cfg = yaml.safe_load(f)
    with open("examples/ac_video_jepa/cfgs/eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    plan_cfg["logging"] = {"tqdm_silent": False}

    _, _, env_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(eval_cfg.get("data", {}))
    )

    def env_creator():
        from eb_jepa.datasets.two_rooms.env import DotWall
        return DotWall(config=env_config, **eval_cfg.get("env", {}))

    eval_folder = Path(os.path.join(drive_base, mode, f"planning_eval_ep{epoch}"))
    os.makedirs(eval_folder, exist_ok=True)

    logger.info(f"=== Planning eval: {mode} (epoch {epoch}) ===")
    logger.info(f"  locations_available={locations_available}, "
                f"prescribed_dim={prescribed_dim}, device={device}")
    logger.info(f"  num_episodes={num_episodes}")

    if locations_available:
        # Use patched planning eval for prescribed/hybrid
        results = planning_eval_prescribed(
            plan_cfg=plan_cfg, model=jepa, env_creator=env_creator,
            eval_folder=eval_folder, num_episodes=num_episodes,
            loader=val_loader, prober=xy_prober,
            prescribed_dim=prescribed_dim,
        )
    else:
        # Use original planning eval for free
        results = planning_eval_free(
            plan_cfg=plan_cfg, model=jepa, env_creator=env_creator,
            eval_folder=eval_folder, num_episodes=num_episodes,
            loader=val_loader, prober=xy_prober,
        )

    # Save summary
    results['mode'] = mode
    results['epoch'] = epoch
    results['device'] = str(device)
    summary_path = os.path.join(drive_base, mode, "planning_eval_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {summary_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=list(CONDITIONS.keys()))
    parser.add_argument("--drive_base", type=str, default="/content/drive/MyDrive/eb_jepa_v3_clean")
    parser.add_argument("--num_episodes", type=int, default=20)
    args = parser.parse_args()
    run_planning_eval(args.mode, args.drive_base, args.num_episodes)
