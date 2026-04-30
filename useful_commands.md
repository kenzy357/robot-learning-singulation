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
