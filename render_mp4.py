import os
import gymnasium as gym
import torch
import imageio
from sac import SAC
from reward_wrapper import ClassicMountainCarRewardWrapper

ENV_NAME    = "MountainCarContinuous-v0"
CHECKPOINT  = "/home/kapel/Downloads/Extras/hassan/soft_actor_critic_on_mountain_car/checkpoints/sac_checkpoint_MountainCarContinuous-v0_final"
OUTPUT_MP4  = "mountaincar_sac.mp4"
SEED        = 123456
NUM_EPISODES = 3
FPS = 30

class Args:
    gamma = 0.999
    tau = 0.005
    alpha = 0.01
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = True
    cuda = torch.cuda.is_available()
    hidden_size = 256
    lr = 0.0003

args = Args()

base_env = gym.make(ENV_NAME, render_mode="rgb_array")
env = ClassicMountainCarRewardWrapper(base_env)

state_dim = env.observation_space.shape[0]
agent = SAC(state_dim, env.action_space, args)

device = torch.device("cuda" if args.cuda else "cpu")
ckpt = torch.load(CHECKPOINT, map_location=device)
agent.policy.load_state_dict(ckpt["policy_state_dict"])
agent.policy.to(device).eval()

os.makedirs(os.path.dirname(OUTPUT_MP4) or ".", exist_ok=True)
writer = imageio.get_writer(OUTPUT_MP4, fps=FPS, codec='libx264', bitrate='16M')

for ep in range(NUM_EPISODES):
    state, _ = env.reset(seed=SEED + ep)
    done = False
    ep_reward = 0
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        writer.append_data(frame)
        ep_reward += reward
    print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")

writer.close()
env.close()
print(f"Saved video to {OUTPUT_MP4}")

