# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 25
    experiment_name = "pick_place_frozen_net"
    # Kept False: the privileged obs group has near-constant dims (binary
    # contact_states flags, the static-cube quaternion). Empirical
    # normalization divides by per-dim std, so a zero-variance dim that
    # occasionally flips (e.g. is_grasped) produces a huge normalized spike
    # that blows up the policy output and the value loss.
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "isaac_so_arm101_pick_place"
    # The env now exposes two ObsGroups ("policy", "privileged"); be explicit
    # so this vision-PPO run keeps using the camera "policy" group only.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # log-parameterized action-noise std: exp() keeps it strictly positive,
        # so a policy-gradient update can never drive it negative and crash
        # `Normal.sample()` (the default "scalar" std is an unclamped raw param).
        noise_std_type="log",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        # "fixed", NOT "adaptive": the adaptive schedule multiplies the LR x1.5
        # per minibatch whenever KL is small and ramps it to rsl_rl's hard-coded
        # 1e-2 ceiling (100x nominal) within the first iteration. At 1e-2 the
        # value function inflates without bound (bootstrap feedback loop) and
        # eventually explodes the critic. "fixed" keeps the LR at 1e-4.
        schedule="fixed",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PickPlaceTeacherPPORunnerCfg(PickPlacePPORunnerCfg):
    """Phase 1 of the teacher-student workflow: train a PPO teacher purely on
    the ground-truth ``privileged`` ObsGroup (no camera / DINOv2 features).

    Both actor and critic read the privileged state, so RL converges far more
    reliably than on raw DINOv2 features. The resulting checkpoint is loaded as
    the frozen teacher by ``PickPlaceDistillationRunnerCfg`` — its actor MLP
    (``actor_hidden_dims`` / ``activation``) MUST match the distillation cfg's
    ``teacher_hidden_dims`` / ``activation`` or the weight load will fail.

    Logs under experiment ``pick_place_teacher`` with run-name suffix
    ``teacher`` so the distillation run's ``load_run`` regex can find it.
    """

    experiment_name = "pick_place_teacher"
    run_name = "teacher"
    obs_groups = {"policy": ["privileged"], "critic": ["privileged"]}
