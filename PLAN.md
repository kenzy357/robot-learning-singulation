## Plan

What you need to do to add a Pick-and-Place task
You already have lift (pick + lift to target) and reach (just move EE). Pick-and-place = lift + place at a destination + release. The cleanest path is to clone lift/ and extend it. Below is the minimal change list.
1. Folder/registration scaffolding
Create tasks/pick_place/ mirroring lift/:
tasks/pick_place/
├── __init__.py            # gym.register the new task IDs
├── pick_place_env_cfg.py  # base config (clone of lift_env_cfg.py)
├── joint_pos_env_cfg.py   # SO-ARM100/101 specializations
├── agents/rsl_rl_ppo_cfg.py
└── mdp/
    ├── rewards.py
    ├── observations.py
    └── terminations.py
In tasks/__init__.py add from . import pick_place. In pick_place/__init__.py register IDs like Isaac-SO-ARM100-PickPlace-v0 and -Play-v0 (copy the gym.register blocks from lift/__init__.py).
2. Scene — add a place target
The lift task already has robot, ee_frame, object (cube), table. You need a destination. Two options:

Visual-only target (simplest): a pose command sampled per episode, drawn as a debug marker. No new asset needed — you already have CommandsCfg.object_pose doing this for "where to lift to". You just need to ground it on the table (lower z range) and make the reward depend on the object reaching that pose after release, not just being held there.
Physical target (a tray/bin): add a second AssetBaseCfg (static USD) or RigidObjectCfg for the destination, plus randomize its pose in EventCfg.

Recommended: start with the visual-only target — it's just changing pos_z ranges on the existing object_pose command to (0.0, 0.05) so the goal is on the table, not in the air.
3. Actions — keep gripper, but it must learn to release
Lift uses BinaryJointPositionActionCfg (open/close). Keep this. The release behavior emerges from the reward, not the action space. No change needed here.
4. Observations — add what the policy needs to see
Current lift obs: joint_pos, joint_vel, object_position, target_object_position, last_action. For pick-and-place add:

Gripper state (open/closed) — the policy needs to know if it's currently holding the cube
Optionally: object velocity (helps detect "placed and stable")

That's it. target_object_position already encodes the destination.
5. Rewards — this is the real work
The lift reward stack is wrong for place because lifting_object (weight 15) rewards keeping the cube in the air forever. You need a staged reward that rewards putting it down at the goal, not holding it up.
Conceptual reward design (no code, just terms to add/modify in RewardsCfg):
StageTermBehaviorApproachreaching_object (tanh of EE-to-cube distance)unchanged from liftGraspnew is_grasped indicator (gripper closed AND cube near fingertips)gates the next stagesTransportobject_goal_distance conditioned on is_graspedreplaces lift's height rewardPlacenew term: small distance to goal AND cube on table (z ≈ table height) AND low cube velocitythe actual task signalReleasenew term: gripper open AND cube at goal AND EE retractedthe part that distinguishes place from lift
Critical change: drop or strongly reduce lifting_object. Otherwise the policy learns "lift forever" because that's a free +15 reward. Replace it with is_grasped (also binary, weight ~1–5) which only activates while the cube is in contact with both fingers.
6. Terminations
Add a success termination: cube within ε of goal, low velocity, gripper open. This both bounds episode length on success and lets you log success rate. Keep the existing object_dropping (cube falls off table) and time_out.
7. Curriculum (optional but useful)
Pick-and-place is much harder than lift — sparse rewards in the place/release stages mean PPO will struggle. Two useful curricula:

Start with goal close to the cube's spawn, then expand the goal range over training steps (modify commands.object_pose.ranges via a curriculum term).
Phase in the release reward only after success rate on "transport to goal while grasped" exceeds a threshold.

8. PPO config
Clone agents/rsl_rl_ppo_cfg.py. Two things probably need to change:

max_iterations: lift used 1000 — pick-and-place will need more (3000–5000) because of the longer horizon and harder credit assignment.
episode_length_s: lift uses 5 s. Pick-and-place needs ~8–10 s (approach + grasp + transport + place + release).
Network is fine at [64, 64] for proprioceptive obs; only go bigger if you add vision.

9. The hard parts you should expect

Releasing without flinging the cube: if release reward only checks gripper_open AND at_goal, the policy will learn to drop from height. Add a low-velocity term on the cube and require the cube's z to be near table height before opening counts.
Premature release: if is_grasped reward is too weak, the policy opens early. Make is_grasped's contribution while transporting strictly larger than what it gets from just touching the cube.
Goal in air vs on table: lift's goal range is pos_z ∈ (0.2, 0.35). For place, set pos_z ∈ (0.0, 0.02) so the goal sits on the table.

👉 Minimal change summary

Clone lift/ → pick_place/, register new gym IDs.
Lower the goal pos_z range to table height.
Remove lifting_object, replace with is_grasped indicator.
Add place reward (cube at goal + on table + low velocity) and release reward (gripper open at goal).
Add success termination + add gripper state to observations.
Bump episode_length_s to ~10 and max_iterations to ~3000–5000.
Optional: goal-distance curriculum for stable training.

The robot configs in robots/ and the URDFs do not change — they're task-agnostic.