# Camera, observations, and sim-to-real notes

Everything image-related in this project, in one place. Use this as the source of truth when buying real hardware, writing the real-robot inference loop, or debugging a sim-to-real gap.

---

## 1. What the simulation camera is right now

Defined in [pick_place_env_cfg.py](src/isaac_so_arm101/tasks/pick_place/pick_place_env_cfg.py#L218-L236):

```python
wrist_camera: TiledCameraCfg = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/wrist_link/cam_marker/tip/wrist_cam",
    update_period=0.0,
    height=224,
    width=224,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 5.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
)
```

| Property        | Value               | Notes |
|-----------------|---------------------|-------|
| Resolution      | 224 × 224 RGB       | Sized to match DINOv2's input directly. |
| Frame rate      | ~50 Hz              | `update_period=0.0`, `decimation=2`, `sim.dt=0.01`. |
| Horizontal FOV  | ~47°                | `2·atan(20.955 / (2·24)) ≈ 47°`. |
| Vertical FOV    | ~47°                | Square sensor. |
| Near / far clip | 5 cm to 5 m         | OK for tabletop work. |
| Noise/distortion| None                | Perfect pinhole. No motion blur, AE drift, rolling shutter. |
| Channel order   | RGB                 | Critical — OpenCV defaults to BGR. |

**Physical mount** (in [pick_place_env_cfg.py](src/isaac_so_arm101/tasks/pick_place/pick_place_env_cfg.py#L70-L73)):

```python
CAMERA_ROT = euler_to_quat(90.0, 14.0, 90.0)
CAMERA_POS = (-0.066, 0.021, 0.012)   # relative to wrist_link
```

The camera sits at the tip of a marker cylinder, child of `wrist_link`. Whatever offset and rotation the real hardware mount has must match this — or update the sim and retrain.

---

## 2. Image preprocessing pipeline (sim side)

From [mdp/feature_extractors.py](src/isaac_so_arm101/tasks/pick_place/mdp/feature_extractors.py#L46-L56):

```python
# images: (N, H, W, C) uint8 from TiledCamera
x = images.permute(0, 3, 1, 2).float() / 255.0          # (N, 3, H, W) in [0,1]
mean = torch.tensor((0.485, 0.456, 0.406)).view(1,3,1,1)
std  = torch.tensor((0.229, 0.224, 0.225)).view(1,3,1,1)
x = (x - mean) / std                                     # ImageNet normalization
features = model(x)                                      # CLS token, (N, 384) for ViT-S/14
```

Key facts the real-robot code must reproduce **exactly**:
- Input layout: `(N, C, H, W)`, **not** `(N, H, W, C)`.
- Pixel scale: float32 in `[0, 1]` **before** mean/std.
- Mean/std: ImageNet `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`.
- Color order: **RGB** (sim is RGB native, OpenCV is BGR — swap with `cv2.cvtColor`).
- Model: `dinov2_vits14` from `torch.hub`, `.eval()`, no grad.
- Output: CLS token, shape `(N, 384)`.

---

## 3. Full policy observation vector

From `PolicyCfg` in [pick_place_env_cfg.py](src/isaac_so_arm101/tasks/pick_place/pick_place_env_cfg.py#L277-L300):

| Term            | Source                                          | Shape per env |
|-----------------|-------------------------------------------------|---------------|
| `joint_pos`     | `joint_pos_rel`                                 | `(n_joints,)` — relative to default. |
| `joint_vel`     | `joint_vel_rel`                                 | `(n_joints,)` |
| `ee_position`   | `ee_position_in_robot_root_frame`               | `(3,)` — EE in robot root frame. |
| `goal_position` | `goal_position_in_robot_root_frame`             | `(3,)` — bowl floor in robot root frame. |
| `actions`       | `last_action`, `history_length=4`               | `(4 × n_actions,)` |
| `image_features`| `image_features` → DINOv2 ViT-S/14 CLS          | `(384,)` |

`__post_init__` sets `concatenate_terms=True`, so the policy sees a single flat vector. **Order of concatenation matters** — the real-robot inference loop must concatenate in the same order.

---

## 4. Real-camera spec to target

When buying / mounting a real wrist camera, the things that actually matter (in priority order):

1. **Horizontal FOV ≈ 47°.** This is the *single most important* spec to match. Wrong FOV means the policy is looking at a different geometry than it trained on. Either pick a lens that gives ~47°, or update the sim's `focal_length` / `horizontal_aperture` to match your lens *before* training. Measure FOV by pointing at an object of known width at a known distance: `FOV = 2·atan(width / (2·distance))`.
2. **Global shutter** if possible (avoids rolling-shutter artifacts during arm motion). CMOS rolling shutter is acceptable if you train with motion blur augmentation.
3. **Manual exposure + manual white balance lock.** Auto-exposure drifts as the arm moves; the policy was trained on constant exposure.
4. **At least 30 fps** at full resolution. Higher is nice but the policy can run at 30 Hz.
5. **USB UVC compliance** on Linux. Plug-and-play. Avoid vendor SDKs unless necessary.
6. **Native resolution ≥ 224×224.** Anything ≥ that works since you downscale. Going from 2 MP → 224×224 is fine.

Physical mounting:
- Replicate `CAMERA_POS = (-0.066, 0.021, 0.012)` and `CAMERA_ROT = euler(90°, 14°, 90°)` relative to `wrist_link`. Use a 3D-printed mount or shim until it matches.
- Verify by manually moving the arm and confirming the sim-rendered view and real view of the same scene roughly overlap.

---

## 5. Sim-to-real gaps and mitigations

| Gap                       | Sim today  | Real                          | Mitigation |
|---------------------------|------------|-------------------------------|-----------|
| Lens distortion           | None       | Barrel / pincushion           | Calibrate intrinsics, undistort with `cv2.undistort` before resize. |
| Motion blur               | None       | Depends on shutter + speed    | Add motion-blur augmentation in sim, or use global shutter and slow arm motion. |
| Auto-exposure / WB drift  | Fixed      | Drifts with lighting          | Lock manual exposure and WB on the real cam. |
| Sensor noise              | None       | Gaussian + read noise         | Add gaussian noise event in sim, or denoise in preprocessing. |
| Color cast / WB           | Pure       | Tinted                        | Color-jitter augmentation in sim. |
| FOV                       | 47°        | Whatever lens you bought      | **Match in sim before training.** |
| Frame rate                | 50 Hz      | 30 Hz typical                 | Run policy at 30 Hz, or set sim `render_interval` to match. |
| Brightness / contrast     | Fixed      | Varies with room lighting     | Brightness/contrast jitter in sim. |
| Static scene assumption   | Pure       | Distractors, hands, etc.      | Train with cluttered backgrounds, or restrict deployment to a clean workspace. |

Currently the sim has **zero domain randomization on images**. Before deploying, add at least:
- Brightness / contrast / saturation jitter (uniform random per episode).
- Gaussian noise.
- Small random affine warps (sim crops, ±2-3% scale and translation).
- Optional: random color cast to simulate WB drift.

Isaac Lab supports these via events; the cleanest pattern is a `randomize_visual_*` event on the camera config.

---

## 6. Sim-to-real action plan (concrete steps)

### Phase 0 — before training the deploy-target policy

1. **Pick the real camera.** Measure its FOV.
2. **Update sim FOV** to match: edit `focal_length` and/or `horizontal_aperture` in [pick_place_env_cfg.py](src/isaac_so_arm101/tasks/pick_place/pick_place_env_cfg.py#L218-L236) so the computed FOV matches your hardware.
3. **Verify camera mount.** Confirm `CAMERA_POS` and `CAMERA_ROT` match the physical mount on the SO-ARM101 wrist.
4. **Add domain randomization** (brightness, contrast, noise, color jitter, affine) to the camera observations during training. This is the highest-leverage sim-to-real change.

### Phase 1 — train

5. Train PPO with the updated, randomized sim. Expect a small reward hit vs. the un-randomized run — that's the cost of robustness.

### Phase 2 — build the inference module

6. Create `src/isaac_so_arm101/deploy/obs_pipeline.py` that does the **exact** preprocessing chain (capture → BGR→RGB → undistort → square-crop → resize 224×224 → uint8 → normalize). Same module used in sim eval and real inference — never two implementations.
7. Calibrate camera intrinsics (`K`, `D`) with OpenCV `cv2.calibrateCamera` and a checkerboard.
8. Lock manual exposure + WB on the camera driver. Document the values.

### Phase 3 — sim eval with corrupted images

9. In sim, evaluate the trained policy with strong image corruptions (noise + blur + jitter) that exceed what randomization saw during training. If success rate stays acceptable, deploy. If it collapses, train with stronger randomization.

### Phase 4 — real-robot deploy

10. Run the policy at 30 Hz. Log every observation (image, joint state, action) to disk for offline debugging.
11. Compare logged real obs to sim obs side by side. Big visual mismatch ⇒ retrain with matched augmentation; small mismatch ⇒ tune mount / exposure / FOV.

---

## 7. Minimal real-robot inference loop (reference)

```python
# obs_pipeline.py — keep this as the single source of truth
import cv2, numpy as np, torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(frame_bgr, K, D):
    rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    und       = cv2.undistort(rgb, K, D)
    h, w      = und.shape[:2]
    s         = min(h, w)
    y0, x0    = (h - s) // 2, (w - s) // 2
    square    = und[y0:y0+s, x0:x0+s]
    img224    = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_norm  = (img224.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)   # (1, 3, 224, 224)
```

```python
# inference loop sketch — runs at the camera's frame rate (~30 Hz)
img_t       = preprocess_frame(cam.read(), K, D).to(device)
feat        = dinov2(img_t)                                            # (1, 384) — same model as sim
joint_pos   = robot.joint_pos_rel()
joint_vel   = robot.joint_vel_rel()
ee_pos      = robot.ee_pos_in_root()
goal_pos    = robot.goal_in_root()
act_hist    = action_history.flat()                                    # (4 * n_actions,)
obs         = torch.cat([joint_pos, joint_vel, ee_pos, goal_pos,
                         act_hist, feat.flatten()], dim=-1).unsqueeze(0)
action      = policy(obs)
robot.apply(action)
action_history.push(action)
```

The concat order must match `PolicyCfg`. Confirm by printing the obs vector size on both sides and asserting equality at startup.

---

## 8. Common sim-to-real failure modes (debug checklist)

If the policy works in sim but fails on the real robot, walk this list in order:

1. **Observation size mismatch** — print `obs.shape` on both sides at startup. Off by 3? Probably missed `ee_position` or `goal_position`. Off by 4 × n_actions? `actions` history. Off by 384? `image_features`.
2. **BGR vs RGB.** Visualize the input that goes into DINOv2 — does it look color-correct?
3. **Aspect ratio.** Are you stretching a 4:3 frame to 1:1? Center-crop first.
4. **Normalization stats wrong.** Print mean and std of the normalized tensor — should be ~0 mean, ~1 std per channel.
5. **FOV mismatch.** Side-by-side sim render vs real frame from the same arm pose. Same objects should occupy similar pixel fractions.
6. **Camera mount drift.** Physically check screws / 3D print tolerance. Even 5 mm of offset moves a 5 cm cube halfway across the frame.
7. **Joint convention.** Are sim and real joint orderings identical? Are velocities in rad/s vs deg/s? `joint_pos_rel` requires the same default pose on both sides.
8. **Action scale.** If the sim uses scaled joint position deltas, the real robot must apply the same scaling. Print the action range on both sides.
9. **Goal position frame.** `goal_position_in_robot_root_frame` requires the real bowl's pose in the robot root frame — make sure your real-world bowl placement matches what the policy expects.

If you make it through that list and the policy still fails, the most likely remaining culprit is missing **domain randomization** during training — go back to Phase 0 and add it.
