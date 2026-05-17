# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 of the teacher-student workflow: distill the privileged PPO teacher
into a vision student.

The :class:`DistillationRunner` builds a ``StudentTeacher`` module: a frozen
teacher MLP (loaded from the phase-1 PPO checkpoint) and a trainable student
MLP. The student imitates the teacher's actions, but reads only the camera
``policy`` ObsGroup while the teacher reads the ``privileged`` group.

``obs_groups`` here uses RSL-RL's algorithm-side keys: ``"policy"`` is the
*student* input set and ``"teacher"`` is the teacher input set (see
``StudentTeacher.__init__`` in rsl_rl).

Teacher checkpoint loading: ``train.py`` auto-loads a checkpoint whenever the
algorithm is ``Distillation``. It resolves ``logs/rsl_rl/{experiment_name}/
{load_run}/{load_checkpoint}`` — so ``experiment_name`` is kept equal to the
teacher's and ``load_run`` is a regex matching only teacher runs.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class PickPlaceDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 50
    # Same experiment dir as the teacher so train.py can resolve its checkpoint.
    experiment_name = "pick_place_teacher"
    run_name = "distill"
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "isaac_so_arm101_pick_place"

    # Load the frozen teacher: latest run whose name ends in "teacher".
    load_run = ".*teacher"
    load_checkpoint = "model_.*.pt"

    # student reads the camera "policy" group; teacher reads "privileged".
    obs_groups = {"policy": ["policy"], "teacher": ["privileged"]}

    # teacher_hidden_dims / activation MUST match PickPlaceTeacherPPORunnerCfg's
    # actor_hidden_dims / activation, or loading the teacher weights will fail.
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="mse",
    )
