# activate environment 
source /home/kenzy/isaac_so_arm101/.venv/bin/activate   


# position 0 agents just to see if everything spawns
uv run zero_agent --task Isaac-SO-ARM101-PickPlace-Play-v0 --enable_cameras

# random agents
uv run random_agent --task Isaac-SO-ARM101-PickPlace-Play-v0 --enable_cameras

# train 
uv run train --task Isaac-SO-ARM101-PickPlace-v0 --headless --enable_cameras
uv run train --task Isaac-SO-ARM101-PickPlace-v0 --headless --enable_cameras --num_envs 128 --max_iterations 10000

# continue training from last checkpoint
uv run train --task Isaac-SO-ARM101-PickPlace-v0 --headless --enable_cameras --num_envs 128 --max_iterations 10000 --resume

# continue training from a specific checkpoint
uv run train --task Isaac-SO-ARM101-PickPlace-v0 --headless --enable_cameras \
  --num_envs 128 --max_iterations 10000 --resume \
  --load_run 2026-05-14_23-04-36_simplest_config --checkpoint model_3900.pt

# test the trained policy (loads the latest checkpoint from logs/rsl_rl/<experiment_name>/<latest_run>/)
uv run play --task Isaac-SO-ARM101-PickPlace-Play-v0 --enable_cameras

# play specific checkpoint 
uv run play --task Isaac-SO-ARM101-PickPlace-Play-v0 --enable_cameras \ --checkpoint logs/rsl_rl/pick_place/<run>/model_500.pt

# eval 2
uv run train --task Isaac-SO-ARM101-Eval2-v0 --headless --enable_cameras --num_envs 64 --max_iterations 5000


#### privileged teacher
# teacher
uv run train --task Isaac-SO-ARM101-PickPlace-Teacher-v0 --headless --num_envs 4096 --max_iterations 10000
uv run play --task Isaac-SO-ARM101-PickPlace-Teacher-v0 --num_envs 4

# student
uv run train --task Isaac-SO-ARM101-PickPlace-Distill-v0 --headless --enable_cameras --num_envs 128 --max_iterations 10000






# Viewing the camera feed
                         
  Since the viewer is open, in the Isaac Sim GUI:
                                                                                                                                                     
  1. Window → Viewport → Viewport 2 — opens a second viewport panel                                                                                  
  2. In that viewport, click the camera icon (top-left of the viewport)                                                                              
  3. Select /World/envs/env_0/Robot/wrist_link/wrist_cam from the dropdown                                                                                
                                                                                                                                                     
  You'll see env_0's wrist camera live.                                                                                                              
                                                                                                                                                     
  Alternatively, to save a frame to disk from the running env, you can add this temporarily to your zero_agent script or a quick one-off:            
                  
  import torch                                                                                                                                       
  from PIL import Image                                                                                                                              
   
  -after env.step()                                                                                                                                 
  rgb = env.scene["wrist_camera"].data.output["rgb"][0]  # env 0, shape (224, 224, 4) RGBA
  img = Image.fromarray(rgb[:, :, :3].cpu().numpy())                                                                                                 
  img.save("/tmp/wrist_cam.png") 
