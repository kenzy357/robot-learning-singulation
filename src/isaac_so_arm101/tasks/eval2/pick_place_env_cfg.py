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

import isaaclab.sim as sim_utils

# from . import mdp
import isaac_so_arm101.tasks.eval2.mdp as mdp
from isaac_so_arm101.tasks.eval2.mdp.feature_extractors import DINOV2_MODEL_ZOO
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
from isaaclab.sensors import TiledCameraCfg
import torch
import math


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
# CAMERA_ROT = euler_to_quat(110.0, 0.0, 180.0)
# CAMERA_POS = (0.0, 0.035, 0.082)

CAMERA_ROT = euler_to_quat(90.0, 14.0, 90.0)
# CAMERA_POS = (-0.04, -0.02, 0.003)
# CAMERA_POS = (-0.057, -0.006, 0.012)
CAMERA_POS = (-0.066, 0.021, 0.012)


'''Bowl geometry — octagonal rim approximating a circle of radius BOWL_RADIUS.
   Radius matches the ``block_in_target_radius`` termination so the visual
   matches the success criterion.'''
BOWL_POS = (0.20, -0.15, 0.0)
BOWL_RADIUS = 0.05
BOWL_WALL_HEIGHT = 0.025
BOWL_WALL_THICK = 0.003
BOWL_FLOOR_THICK = 0.003
BOWL_N_WALLS = 8
BOWL_COLOR = (0.732, 0.482, 0.243)  # linear RGB for sRGB #DEB887 (burlywood)


def _bowl_wall_cfg(idx: int, n: int = BOWL_N_WALLS) -> RigidObjectCfg:
    """One wall segment of the octagonal bowl rim, kinematic + collidable."""
    theta = 2.0 * math.pi * idx / n
    width = 2.0 * BOWL_RADIUS * math.sin(math.pi / n) + BOWL_WALL_THICK
    cx = BOWL_POS[0] + (BOWL_RADIUS + BOWL_WALL_THICK / 2.0) * math.cos(theta)
    cy = BOWL_POS[1] + (BOWL_RADIUS + BOWL_WALL_THICK / 2.0) * math.sin(theta)
    cz = BOWL_POS[2] + BOWL_FLOOR_THICK + BOWL_WALL_HEIGHT / 2.0
    rot = (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/BowlWall{idx}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[cx, cy, cz], rot=list(rot)),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_WALL_THICK, width, BOWL_WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BOWL_COLOR, roughness=1.0, metallic=0.0
            ),
        ),
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
    # primary cube: will be populated by agent env cfg
    block: RigidObjectCfg | DeformableObjectCfg = MISSING
    # second cube (always spawned adjacent to ``block``)
    block_b: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Bowl floor: circular cylinder sized to match the reward radius. Collision
    # is enabled so the cube physically rests inside; the 8 wall segments below
    # form an octagonal rim that keeps the cube from sliding out.
    bowl_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BowlFloor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[BOWL_POS[0], BOWL_POS[1], BOWL_POS[2] + BOWL_FLOOR_THICK / 2.0],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=BOWL_RADIUS,
            height=BOWL_FLOOR_THICK,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BOWL_COLOR,
                roughness=1.0,
                metallic=0.0,
            ),
        ),
    )

    bowl_wall_0 = _bowl_wall_cfg(0)
    bowl_wall_1 = _bowl_wall_cfg(1)
    bowl_wall_2 = _bowl_wall_cfg(2)
    bowl_wall_3 = _bowl_wall_cfg(3)
    bowl_wall_4 = _bowl_wall_cfg(4)
    bowl_wall_5 = _bowl_wall_cfg(5)
    bowl_wall_6 = _bowl_wall_cfg(6)
    bowl_wall_7 = _bowl_wall_cfg(7)

    # Table — primitive cuboid with the exact spec color #B8ADA9.
    # See pick_in_clutter_env_cfg.py for the rationale (sized + positioned so
    # the cluster + cluster-randomization stays on the table).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.40, 0, -0.02]),
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 1.00, 0.04),
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
        prim_path="{ENV_REGEX_NS}/Robot/wrist_link/cam_marker",
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
        prim_path="{ENV_REGEX_NS}/Robot/wrist_link/cam_marker/tip",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0175)),
        spawn=sim_utils.CylinderCfg(
            radius=0.008,
            height=0.010,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
    )

    # wrist-mounted camera → feeds frozen ResNet18 encoder
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
        offset=TiledCameraCfg.OffsetCfg(
                                        rot=(0.0, 0.0, 0.0, 1.0),  ## may need to be arranged
                                        convention="ros"),
        # offset=TiledCameraCfg.OffsetCfg(pos=CAMERA_POS, 
        #                                 rot=CAMERA_ROT,  ## may need to be arranged
        #                                 convention="ros"),
    )


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
        target_color = ObsTerm(func=mdp.target_color)
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()


_CUBE_COLOR_PALETTE = [
    (1.0, 0.0, 0.0),    # red
    (1.0, 0.45, 0.0),   # orange
    (1.0, 1.0, 0.0),    # yellow
    (0.0, 0.8, 0.0),    # green
    (0.55, 0.0, 0.75),  # purple
    (0.0, 0.2, 1.0),    # blue
]


@configclass
class EventCfg:
    """Reset behavior: scene defaults, paired-block placement, bowl move, colors."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Place block_a + block_b adjacent at a randomized pair-center.
    reset_paired_blocks = EventTerm(
        func=mdp.reset_paired_blocks_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "cube_size": 0.02,
            "epsilon": 0.001,
            "block_a_cfg": SceneEntityCfg("block"),
            "block_b_cfg": SceneEntityCfg("block_b"),
        },
    )

    # Move bowl_floor and all 8 walls together by a sampled xy offset.
    reset_bowl_assembly = EventTerm(
        func=mdp.reset_bowl_assembly_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "asset_names": [
                "bowl_floor",
                "bowl_wall_0", "bowl_wall_1", "bowl_wall_2", "bowl_wall_3",
                "bowl_wall_4", "bowl_wall_5", "bowl_wall_6", "bowl_wall_7",
            ],
        },
    )

    # Pick one of two cubes as the target each episode. Samples palette colors
    # for both cubes (different indices) and the target_idx, then applies the
    # colors visually. Populates ``env.target_idx`` and ``env.target_color``
    # used by ``mdp.target_color`` obs and the ``target_*`` rewards.
    randomize_cubes_and_target = EventTerm(
        func=mdp.randomize_cube_colors_and_target,
        mode="reset",
        params={
            "palette": _CUBE_COLOR_PALETTE,
            "block_a_cfg": SceneEntityCfg("block"),
            "block_b_cfg": SceneEntityCfg("block_b"),
        },
    )


@configclass
class RewardsCfg:
    """Additive rewards — mirror of the upstream Isaac Lab Lift task pattern.

    All terms are always-on; ``goal_xy_*`` is multiplied by a lifted flag so it
    only contributes once the block is off the table. ``success`` is the sparse
    high-weight term that dominates once the block sits in the bowl.
    """

    reach = RewTerm(
        func=mdp.target_ee_distance_tanh,
        params={"std": 0.05},
        weight=1.0,
    )

    lift = RewTerm(
        func=mdp.target_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0,
    )

    goal_xy_coarse = RewTerm(
        func=mdp.target_to_goal_xy_distance_tanh,
        params={"std": 0.30, "minimal_height": 0.04},
        weight=10.0,
    )

    goal_xy_fine = RewTerm(
        func=mdp.target_to_goal_xy_distance_tanh,
        params={"std": 0.05, "minimal_height": 0.04},
        weight=5.0,
    )

    success = RewTerm(
        func=mdp.target_in_bowl,
        params={"xy_threshold": 0.04, "z_max_above_bowl": 0.05},
        weight=100.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
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

    success = DoneTerm(
        func=mdp.success_target_in_bowl,
        params={"xy_threshold": 0.04, "z_max_above_bowl": 0.05},
    )

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
    # NOTE: ``replicate_physics=False`` is required for per-env color
    # randomization via ``mdp.randomize_visual_color``.
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=False
    )
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
