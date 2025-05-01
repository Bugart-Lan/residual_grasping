
import gymnasium as gym
import numpy as np
import torch

from pydrake.all import StartMeshcat
from stable_baselines3 import SAC

import envs.e2e_one


env_name = "EndToEndGraspOne-v0"
time_limit = 3
time_step = 0.05
zip = "data/e2e_grasp_one"


def main():

    meshcat = StartMeshcat()
    input("Press Enter to continue...")
    env = gym.make(
        env_name,
        meshcat=meshcat,
        time_limit=time_limit,
        obs_noise=True,
        debug=False,
    )
    model = SAC.load(zip, env)
    obs, _ = env.reset()
    meshcat.StartRecording()
    total_reward = 0
    n_success = 0
    for i in range(1000):
        action, state = model.predict(obs, deterministic=True)
        print(f"action = {action}")
        # action = np.array([0, 0, 0, 0, -0.05, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(terminated, truncated)
        print(f"reward = {reward}")
        total_reward += reward
        n_success += reward >= 1
        env.render()

        if terminated:
            # meshcat.PublishRecording()
            # input("Press Enter to continue...")
            obs, _ = env.reset()

    meshcat.PublishRecording()

    input("Press Enter to continue...")
    print(f"Total reward = {total_reward}")
    print(f"# of successful grasp = {n_success}")


if __name__ == "__main__":
    main()
