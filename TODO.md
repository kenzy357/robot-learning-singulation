# train with this already 
# change the reward
# change the table so it's more simple, put the right color
# right now the joint of the gripper is either open or closed 
# add last states, history buffer in the observations
# add target pos in observation ( if not already done )


# do we also randomize the position of the goal every time during training ?
# do we need to include the camera for training or can we just process the data and give it to the policy ?
# whhy no commands in Rayan's code ?



##   ce qui pourrait ne pas marcher
- j'ai commands et pas lui
- initialisation de joint_pos_env pas la meme


-pretrained CNN       or    -CNN that we train
first try with pretrained



# add camera using 
github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/

For a wrist-camera pick-and-place with a frozen feature encoder:

Camera mounting + scene setup → copy from manipulation/stack/config/franka/stack_ik_rel_blueprint_env_cfg.py (wrist camera on a manipulator)
Frozen-encoder obs term → copy from classic/cartpole/cartpole_camera_env_cfg.py (the ResNet18FeaturesCameraPolicyCfg ObsGroup with mdp.image_features)
Adapt the prim_path to mount on a SO-ARM link (check your URDF; likely wrist_link or gripper)