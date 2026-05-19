# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils

# from . import mdp
import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaac_so_arm101.tasks.pick_place.mdp.feature_extractors import DINOV2_MODEL_ZOO
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

## added for pick and place
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
import math
import torch


# from isaaclab.utils.offset import OffsetCfg
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


##
# Scene definition
##

'''Camera postion'''
def euler_to_quat(roll_deg, pitch_deg, yaw_deg, device="cpu"):
    """Intrinsic XYZ convention (R = Rx·Ry·Rz) — matches Isaac Sim Properties panel directly."""
    to_rad = lambda v: torch.tensor(v, dtype=torch.float32, device=device) * (torch.pi / 180)
    r, p, y = to_rad(roll_deg), to_rad(pitch_deg), to_rad(yaw_deg)
    cr, sr = torch.cos(r / 2), torch.sin(r / 2)
    cp, sp = torch.cos(p / 2), torch.sin(p / 2)
    cy, sy = torch.cos(y / 2), torch.sin(y / 2)
    return (
        (cr * cp * cy - sr * sp * sy).item(),
        (sr * cp * cy + cr * sp * sy).item(),
        (cr * sp * cy - sr * cp * sy).item(),
        (cr * cp * sy + sr * sp * cy).item(),
    )
# --- previous (hand-tuned) wrist-camera poses, kept for reference ---
# CAMERA_ROT = euler_to_quat(110.0, 0.0, 180.0); CAMERA_POS = (0.0, 0.035, 0.082)
# CAMERA_ROT = euler_to_quat(90.0, 14.0, 90.0);  CAMERA_POS = (-0.066, 0.021, 0.012)


def squint_euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Replicate squint's WristCameraEnv euler->quat exactly.

    squint composes the wrist-camera rotation as
    ``q = q_pitch(Y) . q_yaw(Z) . q_roll(X)`` (see ``_update_wrist_camera_pose``
    in singulation-project/squint/envs/base_random_env.py) — NOT the
    intrinsic-XYZ order used by ``euler_to_quat`` above, hence its own
    converter. Returns ``(w, x, y, z)``.
    """
    r = math.radians(roll_deg) / 2.0
    p = math.radians(pitch_deg) / 2.0
    y = math.radians(yaw_deg) / 2.0
    ci, si = math.cos(r), math.sin(r)
    cj, sj = math.cos(p), math.sin(p)
    ck, sk = math.cos(y), math.sin(y)
    q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk
    return (
        q_py_w * ci - q_py_x * si,
        q_py_w * si + q_py_x * ci,
        q_py_y * ci + q_py_z * si,
        q_py_z * ci - q_py_y * si,
    )


# Wrist-camera pose ported from the squint ``Place`` task — squint's
# WristCameraEnv.WRIST_CAMERA_BASE_POS / _ROT_RAD (re-verified 2026-05-19),
# expressed relative to ``gripper_link``. SAPIEN's camera convention
# (+X forward, +Z up) maps to IsaacLab's ``convention="world"``, so CAMERA_ROT
# is fed straight into the TiledCamera OffsetCfg below.
CAMERA_POS = (0.0001, 0.0498, -0.0691)
CAMERA_ROT = squint_euler_to_quat(-90.0, 91.0, -35.31)


'''Bowl geometry — real-scan mesh asset (``assets/bowl.usd``, converted from
   the squint ``Place`` env's ``bowl.obj`` by ``scripts/convert_bowl_to_usd.py``).
   The mesh origin is the bowl bottom-centre; its AABB is ~0.150 x 0.150 x
   0.053 m. ``BOWL_RADIUS`` is the success/reward footprint radius and
   ``BOWL_RIM_HEIGHT`` the wall height — both feed the staged Place reward and
   the success termination so the visual matches the success criterion.'''
BOWL_USD_PATH = str(Path(__file__).resolve().parent / "assets" / "bowl.usd")
BOWL_POS = (0.27, 0.09, 0.0)
BOWL_RADIUS = 0.07          # success footprint (mesh AABB half-extent ~0.075)
BOWL_RIM_HEIGHT = 0.053     # mesh wall height (AABB z-extent)

# ContactSensor filter target — the single mesh bowl prim. Keeps robot↔bowl,
# robot↔cube and robot↔table contacts isolated across separate sensors.
BOWL_PART_PATHS = ["{ENV_REGEX_NS}/Bowl"]
BLOCK_PART_PATHS = ["{ENV_REGEX_NS}/Block"]
TABLE_PART_PATHS = ["{ENV_REGEX_NS}/Table"]

# Gripper body prim names the contact sensors are mounted on.
_GRIPPER_BODY = "gripper_link"
_JAW_BODY = "moving_jaw_so101_v1_link"


def _contact_cfg(body: str, filter_paths: list[str]) -> ContactSensorCfg:
    """A ContactSensor on one gripper body, reporting PhysX *filtered* contact
    forces against only ``filter_paths`` (one filter-target set)."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
        update_period=0.0,
        filter_prim_paths_expr=filter_paths,
    )


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    block: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Bowl — real-scan mesh asset (converted from the squint Place env's
    # bowl.obj). Kinematic + gravity-disabled so it stays put, exactly like the
    # procedural octagonal bowl it replaces. The convex-decomposition collider
    # (baked into the USD) preserves the inner cavity so the cube physically
    # rests inside. Contact reporting on so the gripper-vs-bowl ContactSensors
    # work. The mesh origin is the bowl bottom-centre.
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=list(BOWL_POS), rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=BOWL_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Table — primitive cuboid with the exact spec color #B8ADA9.
    # Kinematic RigidObjectCfg (not a plain AssetBaseCfg collider) so it is a
    # valid contact-sensor target: activate_contact_sensors needs a rigid body
    # to attach to. Kinematic + gravity-disabled keeps it immovable, exactly
    # like the bowl floor/walls above.
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0, -0.02]),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 1.00, 0.04),
            # contact reporting on so the gripper-vs-table ContactSensor works
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.485, 0.426, 0.408),  # #B8ADA9
                roughness=1.0,
                metallic=0.0,
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )



    

    # thin red cylinder so the camera location is visible in the Isaac Sim viewport
    cam_marker: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/cam_marker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=CAMERA_POS,
                                                rot=CAMERA_ROT),
        spawn=sim_utils.CylinderCfg(
            radius=0.008,
            height=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )
    # black tip — 1 cm cap at the front (+Z of the body = optical axis direction)
    # child of cam_marker so it inherits rotation automatically; pos is in body-local frame
    cam_marker_tip: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/cam_marker/tip",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0175)),
        spawn=sim_utils.CylinderCfg(
            radius=0.008,
            height=0.010,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
    )

    # Wrist camera → frozen DINOv2 encoder (see ObservationsCfg.image_features).
    # Pose ported from the squint ``Place`` task: mounted on ``gripper_link`` at
    # the squint WRIST_CAMERA base pose. ``convention="world"`` (+X forward,
    # +Z up) matches SAPIEN's camera convention, so CAMERA_ROT is used directly.
    # focal_length 12.05 mm + 20.955 mm aperture → ~82deg FOV = squint's fovy.
    # Resolution stays 224x224 (square, multiple of 14) — required by DINOv2.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.05,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_POS,
            rot=CAMERA_ROT,
            convention="world",
        ),
    )

    # Gripper contact sensors — one per (gripper body) x (filter target).
    # Each ContactSensor reports PhysX *filtered* contact forces between its
    # gripper body and only the listed prims, so robot↔bowl, robot↔cube and
    # robot↔table contacts are isolated. Read by the staged ``place`` reward
    # and the ``place_success`` termination (see mdp/rewards.py).
    # Requires contact reporting on the gripper bodies (robot spawn, set in
    # joint_pos_env_cfg) and on every filter target (bowl/cube/table spawns).
    gripper_bowl_contact: ContactSensorCfg = _contact_cfg(_GRIPPER_BODY, BOWL_PART_PATHS)
    jaw_bowl_contact: ContactSensorCfg = _contact_cfg(_JAW_BODY, BOWL_PART_PATHS)
    gripper_item_contact: ContactSensorCfg = _contact_cfg(_GRIPPER_BODY, BLOCK_PART_PATHS)
    jaw_item_contact: ContactSensorCfg = _contact_cfg(_JAW_BODY, BLOCK_PART_PATHS)
    gripper_table_contact: ContactSensorCfg = _contact_cfg(_GRIPPER_BODY, TABLE_PART_PATHS)
    jaw_table_contact: ContactSensorCfg = _contact_cfg(_JAW_BODY, TABLE_PART_PATHS)


##
# MDP settings
##


# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""
# ## that's the goal 
#     object_pose = mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         body_name=MISSING,  # will be set by agent env cfg
#         resampling_time_range=(10.0, 10.0),
#         debug_vis=False,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(-0.1, 0.1),
#             pos_y=(-0.3, -0.1),
#             pos_z=(0.1, 0.1),
#             roll=(0.0, 0.0),
#             pitch=(0.0, 0.0),
#             yaw=(0.0, 0.0),
#         ),
#     )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        goal_position = ObsTerm(func=mdp.goal_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action, history_length=4)

        image_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "model_zoo_cfg": DINOV2_MODEL_ZOO,
                "model_name": "dinov2_vits14",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged teacher observations — ground-truth low-dim state.

        Consumed only by the teacher in the teacher-student workflow (the
        teacher PPO run and the ``teacher`` slot of the distillation run, see
        the ``obs_groups`` mappings in the agent cfgs). The vision student must
        infer all of this from the wrist camera instead. No DINOv2 image
        features here — that is what makes the teacher cheap to train.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        goal_position = ObsTerm(func=mdp.goal_position_in_robot_root_frame)
        cube_position = ObsTerm(func=mdp.cube_position_in_robot_root_frame)
        cube_orientation = ObsTerm(func=mdp.cube_orientation_in_robot_root_frame)
        cube_lin_vel = ObsTerm(func=mdp.cube_lin_vel)
        gripper_openness = ObsTerm(func=mdp.gripper_openness)
        contact_states = ObsTerm(func=mdp.privileged_contact_states)
        actions = ObsTerm(func=mdp.last_action, history_length=4)

        def __post_init__(self):
            # teacher gets clean ground-truth state — no observation noise
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

X_MAX=0.20
X_MIN=0.15
Y_MAX=0.08
Y_MIN=-0.08


@configclass
class EventCfg:
    """Reset behavior: scene defaults + (optionally) randomize block."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_block_and_bowl = EventTerm(
    #     func=mdp.reset_block_and_bowl_uniform,
    #     mode="reset",
    #     params={
    #         "block_pose_range": {"x": (X_MIN - 0.17, X_MAX - 0.17), "y": (Y_MIN, Y_MAX)},
    #         "bowl_pose_range": {
    #             "x": (0.13 - BOWL_POS[0], 0.36 - BOWL_POS[0]),
    #             "y": (-0.15, 0.15),
    #         },
    #         # bowl_radius (0.07) + cube half-extent (0.01) + margin
    #         "min_distance": 0.10,
    #         "bowl_asset_names": ["bowl"],
    #     },
    # )

    # Snapshot the block's spawn position each reset into env.block_spawn_pos_w
    # so reward/termination terms can measure displacement from it. Keep this
    # LAST among reset events so it captures whatever reset/randomization ran
    # above (default reset now, randomization if re-enabled).
    record_block_spawn = EventTerm(func=mdp.record_block_spawn_pose, mode="reset")


CYLINDER_RADIUS = 0.04
# Cube half-extent (m). The block is spawned as a 0.02 m cube (see
# joint_pos_env_cfg.py), so the half-size is 0.01.
CUBE_HALF = 0.01

# Geometry params shared by the four staged Place reward terms — kept in sync
# with the scene constants above. (Deactivated along with the staged reward.)
_PLACE_PARAMS = {
    "cube_half_size": CUBE_HALF,
    "bowl_radius": BOWL_RADIUS,
    "rim_height": BOWL_RIM_HEIGHT,
}

# above_bin fires within `approach_radius` of the bowl centre. In the ring
# between `bowl_radius` and `approach_radius` the cube must be lifted above
# `min_lift_height` (a cube shoved across the table cannot trigger it); once
# xy is inside `bowl_radius` the lift gate is dropped so the cube can be
# lowered/released into the bowl. success_bonus and the success termination
# still use the full BOWL_RADIUS.
_ABOVE_BIN_PARAMS = {
    "cube_half_size": CUBE_HALF,
    "bowl_radius": BOWL_RADIUS - 0.02,   # placement zone: 0.05 m
    "rim_height": BOWL_RIM_HEIGHT,
    "min_lift_height": 0.04,             # ring lift gate: cube >= 4 cm off table
    "approach_radius": 1,             # above_bin footprint: 0.15 m
}


@configclass
class RewardsCfg:
    """Squint ``Place`` reward, split into per-stage terms.

    The staged dense reward (later stages OVERRIDE earlier ones) produces four
    mutually-exclusive regions, so it is decomposed into four ``place_stage_*``
    terms plus three penalties. Exactly one stage term is nonzero per env, so
    the seven terms SUM to the original single staged reward bit-for-bit — the
    split exists only so each part is logged separately in wandb as
    ``Episode_Reward/<term>``. See ``mdp/rewards.py``.

    ``action_rate`` / ``joint_vel`` are small CAPS-style regularizers — they
    are NOT part of upstream's dense reward, kept as separate additive terms.
    """

    # --- lift portion: exact reward terms from the reference `lift` task ---
    # object_ee_distance (reach, tanh) + object_is_lifted (binary lift) —
    # ported verbatim from tasks/lift/mdp/rewards.py with the lift task's
    # own weights (1.0 / 15.0).
    reaching_object = RewTerm(
        func=mdp.object_ee_distance, weight=1.0, params={"std": 0.05}
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted, weight=10.0, params={"minimal_height": 0.025,"maximal_height": 0.06}
    )

    # --- place portion: staged Place reward — guide the cube to a position
    # over the bowl, then release it. Only the above-bowl and success stages
    # are used (the lift terms above replace the reach/grasp stages).
    above_bin = RewTerm(func=mdp.place_stage_above_bin, weight=10.0, params=_ABOVE_BIN_PARAMS)
    success_bonus = RewTerm(func=mdp.place_stage_success, weight=80.0, params=_PLACE_PARAMS)

    end_with_success  = RewTerm(
        func=mdp.is_terminated_term,
        weight=30000.0,
        params={"term_keys": "success"},
    )

    # --- penalties --------------------------------------------------------
    # #pen_touch_table = RewTerm(func=mdp.robot_touching_table, weight=-6.0)
    pen_touch_bin = RewTerm(func=mdp.robot_touching_bin, weight=-5.0)

    # Penalty when the gripper points above the horizontal plane (toward the
    # sky). Zero for any down/horizontal pose — only a skyward wrist is
    # penalized, with no dense pull toward the ground. See mdp/rewards.py.
    pen_ee_skyward = RewTerm(func=mdp.ee_pointing_up_penalty, weight=-4.0)

    # Penalty for arm motion while the cube sits in the bowl placement zone
    # (xy within BOWL_RADIUS, z in [0, rim_height + 0.01]). Pushes the arm to
    # settle into the static pose the success predicate requires.
    # pen_move_when_placed = RewTerm(
    #     func=mdp.robot_moving_when_placed,
    #     weight=-10.0,
    #     params={"bowl_radius": BOWL_RADIUS, "rim_height": BOWL_RIM_HEIGHT, "z_margin": 0},
    # )

    # Reward for opening the gripper while the cube xy is within
    # (BOWL_RADIUS - 0.005) of the bowl centre — encourages releasing the cube
    # into the bowl. See mdp/rewards.py.
    # release_over_bowl = RewTerm(
    #     func=mdp.open_gripper_over_bowl,
    #     weight=5.0,
    #     params={"bowl_radius": BOWL_RADIUS - 0.005},
    # )

    # Big reward for holding the arm STILL once the cube is in the bowl
    # placement zone (xy within BOWL_RADIUS, z in [0, rim_height]) — a still
    # arm with the cube placed is the success pose, so this rewards respecting
    # the success conditions. See mdp/rewards.py.
    static_when_placed = RewTerm(
        func=mdp.robot_static_when_placed,
        weight=50.0,
        params={"bowl_radius": BOWL_RADIUS, "rim_height": BOWL_RIM_HEIGHT, "z_margin": 0},
    )

    # Flat time penalty — a constant cost every step, so the policy is pushed
    # to reach an episode-ending state (ideally success) as fast as possible.
    pen_time = RewTerm(func=mdp.step_penalty, weight=-0.001)

    # Dense per-step penalty: cube still on the table but shoved >4 cm from its
    # spawn position — discourages dragging the cube instead of lifting it.
    # pen_cube_displaced = RewTerm(
    #     func=mdp.cube_displaced_on_table,
    #     weight=-1.0,
    #     params={"cube_half_size": CUBE_HALF, "max_radius": 0.07},
    # )

    # Big one-shot penalty fired on the step the ``block_dropped`` termination
    # triggers (block falls below ``minimum_height``). ``is_terminated_term``
    # excludes time-outs, so this only hits genuine drops, not episode timeout.
    pen_block_dropped = RewTerm(
        func=mdp.is_terminated_term,
        weight=-7.0,
        params={"term_keys": "block_dropped"},
    )

    # --- potential-based shaping (un-farmable: pays change, not level) -----
    # Bridges the discrete reach->grasp cliff and rewards lifting, without
    # creating a hover local optimum — a held state yields 0 reward.
    # grasp_bridge = RewTerm(
    #     func=mdp.grasp_progress_reward,
    #     weight=2.0,
    #     params={"proximity_scale": 10.0},
    # )
    # lift_progress = RewTerm(
    #     func=mdp.lift_progress_reward,
    #     weight=20.0,
    #     params={"max_lift_height": 0.15},
    # )

    # --- TEMPORARY DEBUG — prints extreme/non-finite values every step ------
    # Weight must be non-zero or RewardManager.compute() skips the func
    # entirely (zero-weight micro-optimization). The func returns all-zeros, so
    # any weight contributes exactly 0 to the reward — 1.0 is fine.
    # debug_extremes = RewTerm(func=mdp.debug_extreme_values, weight=1.0)

    # --- regularizers (not part of upstream's dense reward) ----------------
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-6)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )



@configclass
class TerminationsCfg:
    """Episode endings: timeout, block falling off the world, success."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    block_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.01, "asset_cfg": SceneEntityCfg("block")},
    )

    # Squint Place success — same predicate the staged reward uses for its
    # flat-9 success stage: cube over the bowl, robot static, and the robot
    # touching neither the cube nor the bowl.
    success = DoneTerm(
        func=mdp.place_success,
        params={
            "bowl_radius": BOWL_RADIUS,
            "cube_cfg": SceneEntityCfg("block"),
            "bowl_cfg": SceneEntityCfg("bowl"),
            "robot_cfg": SceneEntityCfg("robot"),
            "rim_height": BOWL_RIM_HEIGHT,
        },
    )

    # block_stalled = DoneTerm(
    #     func=mdp.block_stalled,
    #     params={"stall_time_s": 7.0, "move_threshold": 0.005},
    # )

    # block_in_target_radius = DoneTerm(
    #     func=mdp.block_in_target_radius,
    #     params={"radius": 0.05},
    # )


##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    #commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # GPU narrowphase contact buffer. The default (64 MB) overflows with
        # 1024 envs once the convex-decomposition bowl mesh is in the scene —
        # PhysX reported needing ~330 MB ("collisionStackSize buffer overflow,
        # Contacts have been dropped"). 512 MB gives comfortable headroom.
        self.sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024
