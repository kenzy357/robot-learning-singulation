# SO-ARM100 / SO-ARM101 — Pick-and-Place Task

Custom pick-and-place task built on top of the `isaac_so_arm101` repo. This README is a working cheat sheet for building, testing, training, and evaluating the task.

## Task IDs

The task is registered in 4 variants:

| ID | Robot | Mode | Envs | Use for |
|---|---|---|---|---|
| `Isaac-SO-ARM100-PickPlace-v0` | SO-ARM100 | Train | 4096 | Full training |
| `Isaac-SO-ARM100-PickPlace-Play-v0` | SO-ARM100 | Play | 50 | Visualization / eval |
| `Isaac-SO-ARM101-PickPlace-v0` | SO-ARM101 | Train | 4096 | Full training |
| `Isaac-SO-ARM101-PickPlace-Play-v0` | SO-ARM101 | Play | 50 | Visualization / eval |

`-Play-v0` variants disable observation noise and use 50 envs for clearer visuals.

## Workflow at a glance

```
zero_agent      → scene & cfg sanity check
random_agent    → action space sanity check
train (smoke)   → 100 iterations, verify pipeline
train (full)    → real training run, headless
play            → load checkpoint, watch policy
```

## Daily commands

### List all registered tasks

```bash
uv run list_envs
```

The four `PickPlace` IDs should appear alongside `Reach` and `Lift-Cube`.

### Sanity check 1 — zero agent

Sends all-zero actions. The arm holds its default pose. Use this to confirm the scene compiles, the cube spawns at the right size, and the goal markers appear.

```bash
uv run zero_agent --task Isaac-SO-ARM100-PickPlace-Play-v0
```

What to verify visually:

- Robot spawns upright on the table
- 50 cubes scatter at random positions on the table each reset
- 50 red marker spheres appear at random goal positions each reset
- Markers stay put within an episode (don't resample mid-rollout)

### Sanity check 2 — random agent

Sends uniform random actions in `[-1, 1]`. The arm jitters wildly. Use this to confirm actions actually flow into the joints.

```bash
uv run random_agent --task Isaac-SO-ARM100-PickPlace-Play-v0
```

### Smoke train — 100 iterations

Quick test to confirm the training pipeline runs without errors and reward moves in the right direction.

```bash
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless --max_iterations 100
```

### Full training run

Real training. Runs headless for speed. Expect 30 min – 2 h on a decent GPU depending on `max_iterations` in `agents/rsl_rl_ppo_cfg.py`.

```bash
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless
```

Optional flags:

```bash
# Override iteration count from CLI
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless --max_iterations 3000

# Resume from a previous run (set in PPO cfg or CLI)
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless --resume

# Record videos during training (slower)
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless --video --video_interval 2000

# Set a seed for reproducibility
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless --seed 42
```

### Watch training progress

In a second terminal:

```bash
tensorboard --logdir logs/rsl_rl/
```

Open `http://localhost:6006` in a browser. Key metrics: `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value_function`, `Policy/entropy`.

### Play a trained policy

Loads the latest checkpoint and runs the policy in the play env. No learning, just inference.

```bash
uv run play --task Isaac-SO-ARM100-PickPlace-Play-v0
```

Play a specific checkpoint:

```bash
uv run play --task Isaac-SO-ARM100-PickPlace-Play-v0 \
    --checkpoint logs/rsl_rl/pick_place/<run_dir>/model_500.pt
```

Record a video of the policy:

```bash
uv run play --task Isaac-SO-ARM100-PickPlace-Play-v0 --video --video_length 500
```

## Files you'll edit most

| File | What's in it |
|---|---|
| `tasks/pick_place/__init__.py` | Gym task registration |
| `tasks/pick_place/pick_place_env_cfg.py` | Scene, commands, rewards, terminations, events, curriculum |
| `tasks/pick_place/joint_pos_env_cfg.py` | SO-ARM100 / SO-ARM101 specializations, cube spawn |
| `tasks/pick_place/mdp/rewards.py` | Custom reward functions |
| `tasks/pick_place/mdp/observations.py` | Custom observation functions |
| `tasks/pick_place/mdp/terminations.py` | Custom termination functions |
| `tasks/pick_place/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |

## Key knobs

### Cube size and spawn

In `joint_pos_env_cfg.py` (both SoArm100 and SoArm101 blocks):

```python
init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.01], rot=[1, 0, 0, 0])
spawn=UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.25, 0.25, 0.25),  # 8 cm × 0.25 = 2 cm cube
    ...
)
```

### Cube position randomization at reset

In `pick_place_env_cfg.py`, `EventCfg.reset_object_position`:

```python
"pose_range": {"x": (-0.05, 0.05), "y": (-0.1, 0.1), "z": (0.0, 0.0)}
```

Offset added to the cube's `init_state.pos` on every env reset.

### Goal randomization

In `pick_place_env_cfg.py`, `CommandsCfg.object_pose.ranges`:

```python
ranges=mdp.UniformPoseCommandCfg.Ranges(
    pos_x=(-0.1, 0.1),
    pos_y=(-0.3, -0.1),
    pos_z=(0.0, 0.01),     # on the table for place
    roll=(0.0, 0.0),
    pitch=(0.0, 0.0),
    yaw=(0.0, 0.0),
)
```

`resampling_time_range=(10.0, 10.0)` keeps one goal per episode (set equal to `episode_length_s`).

### Episode length and PPO iterations

In `pick_place_env_cfg.py`:

```python
self.episode_length_s = 10.0   # ~10 s per episode for full pick-and-place
```

In `agents/rsl_rl_ppo_cfg.py`:

```python
max_iterations = 3000   # bump from lift's 1000
experiment_name = "pick_place"   # rename so logs go to logs/rsl_rl/pick_place/
```

## Coordinate frames quick reference

| Frame | Origin | Used for |
|---|---|---|
| World | Global `(0,0,0)`, table top at `z=0` | Internal physics |
| Env-local | Each env's grid cell | `init_state.pos` in cfg files |
| Robot base | Robot root link (after 90° z-rotation) | `CommandsCfg` ranges, object position observations |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `KeyError: 'gripper'` | Body name mismatch in rewards/commands. SO-ARM100 = `"gripper"`, SO-ARM101 = `"gripper_link"` |
| `ModuleNotFoundError` for env cfg class | Class name in `joint_pos_env_cfg.py` doesn't match the gym registration |
| Long hang at startup | Nucleus asset server fetching DexCube USD; usually resolves |
| Empty viewer / black screen | GPU driver issue, unrelated to the code |
| Reward goes to zero / NaN | Wrong body name in reward params, or term_name typo in curriculum |
| Cube intersects table on spawn | `init_state.pos[2]` < cube half-height; raise z |

## Reward design reminder

The cloned lift rewards reward **lifting and holding**, not placing. To get true pick-and-place, the reward stack needs to:

1. Drop or strongly reduce `lifting_object` (currently weight 15)
2. Add `is_grasped` indicator (gripper closed AND cube near fingertips)
3. Replace the goal-tracking term with one gated on `is_grasped`
4. Add a `is_placed` term (cube at goal AND on table AND low velocity)
5. Add a `is_released` term (gripper open AND cube at goal AND EE retracted)
6. Add a success termination

Until those are in, training will produce a "lift and hover near goal" policy, not pick-and-place.
