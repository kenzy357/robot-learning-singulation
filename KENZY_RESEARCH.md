# Kenzy — Research Notes (SO-ARM101 Pick & Place)

Working notebook for the SO-ARM100/101 pick-and-place RL task built on Isaac Lab. Captures what has been built, what's still open, and the pretrained-CNN options for the wrist-camera observation.

---

## 1. Project at a glance

- **Goal**: train a PPO policy that picks a small cube off a table and drops it in a bowl, on the SO-ARM100/101 (Lerobot-style 6-DoF arm).
- **Sim**: Isaac Lab (manager-based RL env). `tasks/pick_place/` was scaffolded by cloning the upstream `lift` task and adding place/release behavior.
- **Trainer**: `rsl_rl` PPO. Task IDs registered: `Isaac-SO-ARM100-PickPlace-v0` / `-Play-v0` and `Isaac-SO-ARM101-PickPlace-v0` / `-Play-v0`.
- **Status**: scene + camera + reward stack + terminations are wired up. Wrist camera feeds a frozen ResNet-18 encoder. Not yet trained to convergence; reward shaping + camera mounting are still being tuned.

---

## 2. What has been built so far

### 2.1 Scene (`pick_place_env_cfg.py`)

| Asset | Notes |
|---|---|
| `robot` | `SO_ARM101_CFG` (or `SO_ARM100_CFG`), populated by `joint_pos_env_cfg.py` |
| `block` | DexCube USD scaled to 0.5× (~4 cm). Drop-off target. |
| `bowl_floor` | Approximated as a flat **kinematic** cylinder (radius 5 cm, height 4 cm). Primitive instead of YCB USD because Isaac Lab's `RigidObjectCfg` requires `RigidBodyAPI`, and YCB meshes don't ship with it. |
| `table` | Cuboid 0.80 × 1.00 × 0.04 m, color `#B8ADA9`. |
| `wrist_camera` | `TiledCameraCfg`, 224×224 RGB, mounted on `wrist_link` via offset `CAMERA_POS / CAMERA_ROT`. Convention is `"ros"`. |
| `cam_marker` + `cam_marker_tip` | Red cylinder + black tip in the viewport so we can see where the camera actually points (still being aligned). |
| `ee_frame` | `FrameTransformer` with `gripper_link` as target, offset `(0.01, 0, -0.09)` from `base_link`. |

Camera offset (current best guess):
```python
CAMERA_ROT = euler_to_quat(90.0, 14.0, 90.0)
CAMERA_POS = (-0.066, 0.021, 0.012)
```
The custom `euler_to_quat` uses **intrinsic XYZ** so it matches Isaac Sim's Properties panel directly. Camera mount is still flagged as "not well mounted" in commit `2a03374`.

### 2.2 Observations (policy group)

```
joint_pos_rel
joint_vel_rel
target_object_position   (from object_pose command)
last_action
image_features           (wrist_cam → frozen ResNet-18 → 512-d feature vector)
```

`object_position` is currently commented out — relying on the camera + target command.

### 2.3 Rewards (`mdp/rewards.py` + cfg)

Staged shaping, weights summarized:

| Term | Weight | Purpose |
|---|---|---|
| `reaching_block` (tanh, std=0.05) | +1 | EE-to-cube approach |
| `lifting_block` (z > 0.025) | +15 | Bonus for lifting |
| `block_to_bowl_coarse` (std=0.30) | +16 | Transport, gated on lifted |
| `block_to_bowl_fine` (std=0.05) | +5 | Tight kernel near bowl |
| `success_bonus` (`block_in_bowl`) | +50 | Sparse success |
| `action_rate_l2` | -1e-4 → -1e-1 (curriculum @ 10k steps) | Smoothness |
| `joint_vel_l2` | -1e-4 → -1e-1 (curriculum @ 10k steps) | Smoothness |

**Known design issue (from `PICK_PLACE_README.md`)**: this stack rewards "lift and hover" because `lifting_block` (+15) keeps paying as long as the cube is in the air. To get true place behavior, plan calls for:
1. Drop or strongly reduce `lifting_block`.
2. Add `is_grasped` indicator (gripper closed AND cube near fingertips) to gate transport.
3. Add `is_placed` (cube at goal AND on/in bowl AND low velocity).
4. Add `is_released` (gripper open AND cube at goal AND EE retracted).

### 2.4 Terminations

- `time_out` — episode length 10 s.
- `block_dropped` — cube z < -0.05 m.
- `success_block_in_bowl` — cube xy-distance to bowl < 4 cm AND z within 5 cm above bowl.

### 2.5 Events

- `reset_scene_to_default`
- `reset_block_position` — uniform xy ∈ ±5 cm at reset.

### 2.6 Sim settings

- `decimation = 2`, `sim.dt = 0.01` (100 Hz physics, 50 Hz control).
- `episode_length_s = 10.0`.
- `num_envs = 1024` (train) / 50 (play).

---

## 3. Open questions (from `TODO.md`)

- Do we randomize the goal position every episode during training, or keep it fixed?
- Why does Rayan's reference code not use Isaac Lab `commands` at all? (We do.)
- `joint_pos_env_cfg.py` initialization differs from Rayan's — possible source of divergence.
- Gripper joint is currently binary (open/closed); may need finer control.
- Add a history buffer (last N states) to observations.
- Confirm `target_object_position` is actually showing up in the obs vector.

---

## 4. Pretrained CNN options for the wrist-camera obs

### 4.1 What Isaac Lab supports out-of-the-box

`mdp.image_features` (used at `pick_place_env_cfg.py:229`) supports two families natively — switch by changing the `model_name` parameter only, no other code:

**ResNet (ImageNet weights)** — `_prepare_resnet_model`:
- `resnet18` (current), `resnet34`, `resnet50`, `resnet101`

**Theia (robotics-distilled ViT, AI Institute, Shang et al. 2024)** — `_prepare_theia_transformer_model`:
- `theia-tiny-patch16-224-cddsv`
- `theia-tiny-patch16-224-cdiv`
- `theia-small-patch16-224-cdiv`
- `theia-small-patch16-224-cddsv`
- `theia-base-patch16-224-cdiv`
- `theia-base-patch16-224-cddsv`

Theia is distilled from CLIP, DINOv2, ViT, SAM, and Depth-Anything specifically for robot learning. **Recommended starting point** if you want to upgrade from ResNet-18 with zero code changes:
```python
"model_name": "theia-tiny-patch16-224-cddsv"
```
(`tiny` is fast; promote to `small` or `base` if features look weak.)

### 4.2 Alternative pretrained CNNs/ViTs (require a `model_zoo_cfg` wrapper)

Not in Isaac Lab's defaults, but they're proven manipulation backbones — pass a custom `model_zoo_cfg` dict containing `model`, `reset`, `inference` callables.

| Model | Backbone | Pretraining | Notes |
|---|---|---|---|
| **R3M** (Nair et al. 2022) | ResNet-18 / 34 / 50 | Ego4D human video, time-contrastive + video-language alignment | Built specifically for manipulation. Reports +20–40% success vs from-scratch. Best on Adroit / MetaWorld / DMControl in published benchmarks. Closest drop-in replacement for ResNet-18. |
| **VC-1** (Majumdar et al. 2023) | ViT-B / ViT-L | Ego4D + ImageNet + manipulation/navigation | Best **across-the-board** PVR in the Meta empirical study. Bigger, slower. |
| **VIP** (Ma et al. 2022) | ResNet-50 | Human video, value-implicit pretraining | Doubles as a dense reward signal — can score "visual progress toward goal". Useful if we want a learned reward in addition to the hand-shaped one. |
| **MVP** (Xiao et al. 2022) | ViT-S / B / L | Egocentric/manipulation video, MAE | Won on Trifinger, ImageNav, Mobile Pick in published benchmarks. |
| **DINOv2** (Oquab et al. 2023) | ViT-S / B / L / G | Web images, self-supervised | Not robotics-specific, but the CVPR 2025 "Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning" paper found DINO/iBOT objectives outperform MAE for robot learning. Strong general-purpose option. |

### 4.3 Recommendation for this project

Practical order of attempts for the wrist-cam pick-and-place:

1. **`theia-tiny-patch16-224-cddsv`** — zero-code change, same Isaac Lab path. Confirms whether a robotics-pretrained backbone helps before investing in custom integration.
2. **R3M (ResNet-50)** — same architecture family as the current ResNet-18, manipulation-pretrained. Worth a custom wrapper.
3. **VC-1 (ViT-B)** — if Theia and R3M plateau and we want the strongest single backbone. Bigger, slower, may need to drop `num_envs`.

### 4.4 How to add a non-default model (custom `model_zoo_cfg`)

The `image_features` term accepts a `model_zoo_cfg: dict` mapping a model name to `{"model", "reset", "inference"}`. Sketch for R3M:

```python
import torch
from r3m import load_r3m  # pip install r3m

def _prepare_r3m(model_device: str):
    model = load_r3m("resnet50").to(model_device).eval()
    for p in model.parameters():
        p.requires_grad = False

    def inference(model, images: torch.Tensor) -> torch.Tensor:
        # images: (N, H, W, 3) uint8 [0, 255] from TiledCamera
        x = images.permute(0, 3, 1, 2).float()  # NCHW
        with torch.no_grad():
            return model(x)  # (N, 2048) for resnet50

    def reset():
        pass

    return {"model": model, "reset": reset, "inference": inference}

# In ObservationsCfg.PolicyCfg:
image_features = ObsTerm(
    func=mdp.image_features,
    params={
        "sensor_cfg": SceneEntityCfg("wrist_camera"),
        "data_type": "rgb",
        "model_name": "r3m_resnet50",
        "model_zoo_cfg": {"r3m_resnet50": _prepare_r3m("cuda:0")},
    },
)
```

The same pattern works for VC-1 (`from vc_models.models.vit import model_utils`), VIP, MVP, or any HuggingFace ViT.

### 4.5 Things to watch when swapping the backbone

- **Feature dimensionality changes** (ResNet-18 → 512, ResNet-50 → 2048, ViT-B → 768). The downstream MLP gets a bigger input automatically because `concatenate_terms = True`, but check that PPO's actor/critic head sizes still make sense.
- **Image preprocessing**: each model expects different normalization (ImageNet mean/std for ResNet, CLIP norms for some Theia variants, R3M does internal normalization). Mismatched norm = garbage features.
- **Speed**: ViT-B at 1024 envs × 224×224 hits VRAM and step time hard. Drop `num_envs` if needed.
- **Frozen vs fine-tuned**: all of these are used **frozen** by default. Fine-tuning a vision encoder inside PPO is unstable; if needed, use a small adapter (LoRA / linear probe) instead of unfreezing.

---

## 5. Training our own CNN end-to-end with PPO

The current setup runs the encoder **outside** the policy — `mdp.image_features` extracts features *before* PPO sees them, and gradients can't flow back. To train a CNN jointly with PPO, the encoder has to live **inside** the actor-critic network.

### 5.1 Architectural shift

| Piece | Now (frozen pretrained) | After (end-to-end CNN) |
|---|---|---|
| Image goes through encoder | inside `ObsTerm` (no grad) | inside `ActorCritic` forward pass (grad ON) |
| Obs to policy | 512-d feature vec + proprio | raw image tensor + proprio |
| Optimizer updates | MLP head only | CNN + MLP heads |
| `model_zoo_cfg` / Theia / ResNet | unused | unused |

### 5.2 Change the observation to expose raw pixels

Replace the `image_features` ObsTerm in `pick_place_env_cfg.py:229` with a raw-image term, and split obs into two groups so the policy can route them differently:

```python
@configclass
class PolicyCfg(ObsGroup):
    """Low-dim state."""
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    target_object_position = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "object_pose"}
    )
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class CriticCfg(PolicyCfg):
    """(optional) privileged obs for asymmetric actor-critic."""
    pass

@configclass
class ImageCfg(ObsGroup):
    """Raw RGB image, NOT concatenated with state."""
    image = ObsTerm(
        func=mdp.image,                          # built-in: returns raw tensor
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "data_type": "rgb",
            "normalize": True,                   # → float in [0, 1]
        },
    )

    def __post_init__(self):
        self.concatenate_terms = False           # keep image as (N,H,W,3)

policy: PolicyCfg = PolicyCfg()
image: ImageCfg = ImageCfg()
```

`mdp.image` is the upstream Isaac Lab obs term that returns the raw camera tensor (look at `cartpole_camera_env_cfg.py` for the full pattern). The env now emits a dict `{"policy": (N, D_state), "image": (N, H, W, 3)}`.

### 5.3 Custom `ActorCritic` with a CNN trunk

`rsl_rl` provides `ActorCritic` (state-only). For images you write a subclass. Add `src/isaac_so_arm101/agents/visual_actor_critic.py`:

```python
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

class CNNEncoder(nn.Module):
    """Small Nature-DQN-style CNN. ~1M params, fast enough for 1024 envs."""
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # 224x224 → with the strides above → flat dim depends; compute dynamically:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            flat = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())

    def forward(self, img_nhwc):
        x = img_nhwc.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
        return self.fc(self.conv(x))


class VisualActorCritic(ActorCritic):
    """Concatenates CNN(img) with proprio state, then standard MLP heads."""
    is_recurrent = False

    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=(256, 128), critic_hidden_dims=(256, 128),
                 cnn_out_dim=128, image_shape=(224, 224, 3), **kwargs):
        # num_actor_obs here = state dim ONLY; CNN feature is added inside.
        super().__init__(
            num_actor_obs=num_actor_obs + cnn_out_dim,
            num_critic_obs=num_critic_obs + cnn_out_dim,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            **kwargs,
        )
        self.encoder = CNNEncoder(in_channels=image_shape[2], out_dim=cnn_out_dim)
        self._state_dim = num_actor_obs

    def _split(self, obs_dict):
        return obs_dict["policy"], obs_dict["image"]

    def act(self, obs, **kw):
        state, img = self._split(obs)
        feat = self.encoder(img)
        return super().act(torch.cat([state, feat], dim=-1), **kw)

    def evaluate(self, obs, **kw):
        state, img = self._split(obs)
        feat = self.encoder(img)
        return super().evaluate(torch.cat([state, feat], dim=-1), **kw)

    def act_inference(self, obs):
        state, img = self._split(obs)
        feat = self.encoder(img)
        return super().act_inference(torch.cat([state, feat], dim=-1))
```

Two design choices worth flagging:
- **Asymmetric AC**: critic can take privileged info (true cube pose) while actor only sees pixels. Cheap variance reduction. Add the cube pose to `CriticCfg` only.
- **Shared vs separate encoders**: above, actor and critic *share* a CNN. Sharing cuts memory ~2× but couples gradients — usually fine for PPO.

### 5.4 Wire it into the PPO config

In `tasks/pick_place/agents/rsl_rl_ppo_cfg.py`, point the `policy` field at the new class:

```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PickPlaceVisualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "pick_place_visual"
    empirical_normalization = False              # don't normalize images by running stats
    policy = RslRlPpoActorCriticCfg(
        class_name="VisualActorCritic",          # registered via custom import path
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,                      # was 1e-3 — vision is unstable at high LR
        num_learning_epochs=4,                   # fewer epochs to limit overfit per batch
        num_mini_batches=8,
        clip_param=0.2,
        gamma=0.99, lam=0.95,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
    )
```

You'll also need to register `VisualActorCritic` so `class_name=` resolves — add it to the `rsl_rl.modules` registry or import it in your task `__init__.py` before training starts.

### 5.5 Practical numbers / pitfalls

| Issue | What happens | Fix |
|---|---|---|
| **VRAM** | 1024 envs × 224×224×3 fp32 ≈ 600 MB just for storage; backward triples that. | Drop `num_envs` to 256 or 512; or downscale camera to 84×84. |
| **Throughput** | Tiled rendering + CNN forward each step is the new bottleneck (was the sim). | Use `decimation` to skip rendering some control steps; lower `update_period` carefully. |
| **Empirical normalization** | `RslRlOnPolicyRunnerCfg.empirical_normalization=True` builds running mean/std over the **whole** obs vector. With pixels in there it explodes. | Set `False` and normalize images explicitly (`/255.0` is in the obs term; that's enough). |
| **PPO LR** | 1e-3 (typical for state-only) blows up CNN weights. | Start at 3e-4; consider linear warmup. |
| **Init noise** | Same `init_noise_std=1.0` is fine, but the policy sees garbage features for the first ~100 iters. | Expect flat reward early; don't kill runs at iter 50. |
| **Frame stack** | Single frame loses motion cues (whether the cube is moving). | Stack last 2–4 frames in the obs, or add an LSTM (`ActorCriticRecurrent`). |
| **Domain randomization** | If you ever transfer to real, pixel policies overfit hard. | Randomize lighting, table color, camera offset, cube color via `EventCfg`. |
| **Log it** | Add a video recorder in `play.py` and a few image samples in TensorBoard so you can see what the CNN sees. | `--video --video_interval 500` flag is already supported. |

### 5.6 Minimal first run

1. Switch obs to raw image (5.2). Drop `num_envs` to 512.
2. Use the small CNN above (≈1M params) — don't start with ResNet-50 from scratch, it won't converge in PPO without huge data.
3. Train smoke test: `uv run train --task ...PickPlace-v0 --headless --max_iterations 200`. Goal: reward at least *moves*.
4. Once it's training, scale up: bigger CNN, frame stack, more iterations.

If end-to-end never converges, the standard fallback is **frozen pretrained CNN + small trainable adapter** (linear or 2-layer MLP after Theia/R3M features). Almost all the win, almost none of the instability — and it's a 5-line change from your current setup.

---

## 6. References

- [Visual Cortex (VC-1) — project page](https://eai-vc.github.io/)
- [R3M paper (arXiv 2203.12601)](https://ar5iv.labs.arxiv.org/html/2203.12601)
- [Theia (arXiv 2407.20179) — distilled vision FM for robotics](https://arxiv.org/abs/2407.20179)
- [On the use of Pre-trained Visual Representations in Visuo-Motor Robot Learning (WCVPR 2025)](https://tsagkas.github.io/pvrobo/assets/pdfs/WCVPR_2025.pdf)
- [A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Wen_A_Data-Centric_Revisit_of_Pre-Trained_Vision_Models_for_Robot_Learning_CVPR_2025_paper.pdf)
- [Visual Pretraining for Robotic Manipulation — overview](https://medium.com/@mjatkin/visual-pretraining-for-robotic-manipulation-4d1cab9ff642)
- Isaac Lab `image_features` source: `.venv/.../isaaclab/source/isaaclab/isaaclab/envs/mdp/observations.py:424`
- Reference camera/encoder pattern: `IsaacLab/source/isaaclab_tasks/.../classic/cartpole/cartpole_camera_env_cfg.py` (`ResNet18FeaturesCameraPolicyCfg`)
- Reference wrist-camera mount: `IsaacLab/source/isaaclab_tasks/.../manipulation/stack/config/franka/stack_ik_rel_blueprint_env_cfg.py`
