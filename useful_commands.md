# activate environment 
source /home/kenzy/isaac_so_arm101/.venv/bin/activate   


# position 0 agents just to see if everything spawns
uv run zero_agent --task Isaac-SO-ARM100-PickPlace-Play-v0

# random agents
uv run random_agent --task Isaac-SO-ARM100-PickPlace-Play-v0

# train 
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless

# test the trained policy (oads the latest checkpoint from logs/rsl_rl/<experiment_name>/<latest_run>/)
uv run play --task Isaac-SO-ARM100-PickPlace-Play-v0
    # play specific checckpoint 
        uv run play --task Isaac-SO-ARM100-PickPlace-Play-v0 \ --checkpoint logs/rsl_rl/pick_place/<run>/model_500.pt


##### run with camera 
uv run train --task Isaac-SO-ARM100-PickPlace-v0 --headless --enable_cameras
uv run play  --task Isaac-SO-ARM100-PickPlace-Play-v0 --enable_cameras
uv run zero_agent --task Isaac-SO-ARM100-PickPlace-Play-v0 --enable_cameras

# Viewing the camera feed
                         
  Since the viewer is open, in the Isaac Sim GUI:
                                                                                                                                                     
  1. Window → Viewport → Viewport 2 — opens a second viewport panel                                                                                  
  2. In that viewport, click the camera icon (top-left of the viewport)                                                                              
  3. Select /World/envs/env_0/Robot/wrist/wrist_cam from the dropdown                                                                                
                                                                                                                                                     
  You'll see env_0's wrist camera live.                                                                                                              
                                                                                                                                                     
  Alternatively, to save a frame to disk from the running env, you can add this temporarily to your zero_agent script or a quick one-off:            
                  
  import torch                                                                                                                                       
  from PIL import Image                                                                                                                              
   
  -after env.step()                                                                                                                                 
  rgb = env.scene["wrist_camera"].data.output["rgb"][0]  # env 0, shape (224, 224, 4) RGBA
  img = Image.fromarray(rgb[:, :, :3].cpu().numpy())                                                                                                 
  img.save("/tmp/wrist_cam.png") 