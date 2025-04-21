import gymnasium as gym
import numpy as np
import torch

from pydrake.all import StartMeshcat
from stable_baselines3 import PPO, TD3

# import manipulation.envs.box_flipup
import envs.floating_joint


env_name = "FloatingJoint-v0"
time_limit = 10
time_step = 0.05
zip = "data/floating_joint.zip"


def main():

    meshcat = StartMeshcat()
    input("Press Enter to continue...")
    env = gym.make(
        env_name,
        meshcat=meshcat,
        time_limit=10,
    )
    model = PPO.load(zip, env)
    obs, _ = env.reset()
    meshcat.StartRecording()
    for i in range(int(time_limit / time_step)):
        action, state = model.predict(obs, deterministic=True)
        print(f"action = {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward = {reward}")
        env.render()
        if terminated:
            obs, _ = env.reset()
    meshcat.PublishRecording()

    input("Press Enter to continue...")
    obs, _ = env.reset()
    Q, Qdot = np.meshgrid(np.arange(0, np.pi, 0.05), np.arange(-2, 2, 0.05))
    V = 0 * Q
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            obs[2] = Q[i, j]
            obs[7] = Qdot[i, j]
            with torch.no_grad():
                V[i, j] = (
                    model.policy.predict_values(model.policy.obs_to_tensor(obs)[0])[0]
                    .cpu()
                    .numpy()[0]
                )
    V = V - np.min(np.min(V))
    V = V / np.max(np.max(V))

    meshcat.Delete()
    meshcat.ResetRenderMode()
    meshcat.PlotSurface("Critic", Q, Qdot, V, wireframe=True)


if __name__ == "__main__":
    main()
