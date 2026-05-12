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
import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

    # Bowl approximated as a flat kinematic cylinder. We use a primitive
    # (vs. an external USD) so the prim has built-in RigidBodyAPI — Isaac Lab
    # requires it for any RigidObjectCfg, and YCB meshes don't ship with it.
    # Asset name is ``bowl_floor`` for symmetry with v1 (which uses 5 prims).
    bowl_floor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BowlFloor",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, -0.15, 0], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.01,
            height=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # bowl stays put even if hit
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.732, 0.482, 0.243),  # linear RGB for sRGB #DEB887 (burlywood)
                roughness=1.0,
                metallic=0.0,
            ),
        ),
    )

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
        goal_position = ObsTerm(func=mdp.goal_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        image_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )   

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset behavior: scene to default + randomize block position on the table."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_block_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )

@configclass
class RewardsCfg:
    """Dense reward shaping. Weights tuned to match upstream Lift task."""

    # Approach the block (~1 when on it).
    reaching_block = RewTerm(
        func=mdp.block_ee_distance_tanh,
        params={"std": 0.05},
        weight=1.0,
    )

    # Big bonus for getting the block off the table.
    lifting_block = RewTerm(
        func=mdp.block_is_lifted,
        params={"minimal_height": 0.025},
        weight=15.0,
    )

    # Bring the lifted block close to the bowl (multiplied by lifted-condition).
    block_to_bowl_coarse = RewTerm(
        func=mdp.block_to_bowl_distance_tanh,
        params={"std": 0.30, "minimal_height": 0.025},
        weight=16.0,
    )

    # Same but with a much tighter kernel — bonus only when very close.
    block_to_bowl_fine = RewTerm(
        func=mdp.block_to_bowl_distance_tanh,
        params={"std": 0.05, "minimal_height": 0.025},
        weight=5.0,
    )

    # Sparse success bonus: block actually inside the bowl.
    success_bonus = RewTerm(
        func=mdp.block_in_bowl,
        params={"xy_threshold": 0.04, "z_max_above_bowl": 0.05},
        weight=50.0,
    )

    # Smooth-action penalties.
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
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("block")},
    )

    success = DoneTerm(
        func=mdp.success_block_in_bowl,
        params={"xy_threshold": 0.04, "z_max_above_bowl": 0.01},
    )

    # block_in_target_radius = DoneTerm(
    #     func=mdp.block_in_target_radius,
    #     params={"radius": 0.05},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
