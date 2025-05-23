import argparse
import os
import sys


import gymnasium as gym
import wandb
import torch

from pydrake.all import StartMeshcat
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import envs.e2e_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_single_env", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log_path", help="path to the logs directory", default="/logs"
    )
    args = parser.parse_args()

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e4 if not args.test else 5,
        "env_name": "EndToEndGraspOne-v0",
        "env_time_limit": 3 if not args.test else 0.5,
        "observations": "state",
    }

    if args.wandb:
        run = wandb.init(
            project=config["env_name"],
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    else:
        run = wandb.init(mode="disabled")

    zip = "data/e2e_grasp_one_wo_noise.zip"

    num_cpu = 40
    if args.train_single_env:
        meshcat = StartMeshcat()
        env = gym.make(
            config["env_name"],
            meshcat=meshcat,
            time_limit=config["env_time_limit"],
            debug=True,
            obs_noise=False,
        )
        check_env(env)
        input("Open meshcat (optional). Press Enter to continue...")
    else:
        # Use a callback so that the forked process imports the environment.
        def make_env():
            return gym.make(
                config["env_name"],
                time_limit=config["env_time_limit"],
                obs_noise=False,
            )

        print(f"Number of CPU used for training = {num_cpu}")
        env = make_vec_env(
            make_env,
            n_envs=num_cpu,
            seed=1,
            vec_env_cls=SubprocVecEnv,
        )

    policy_kwargs = {"net_arch": [64, 64], "activation_fn": torch.nn.ReLU}
    if args.test:
        print("Testing mode")
        model = SAC(config["policy_type"], env, batch_size=4)
    elif os.path.exists(zip):
        print(f"Loading model @ {zip}")
        model = SAC.load(
            zip, env, verbose=1, tensorboard_log=args.log_path or f"runs/{run.id}"
        )
    else:
        print("Creating SAC model...")
        model = SAC(
            config["policy_type"],
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_path or f"runs/{run.id}",
        )

    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print("Total number of parameters:", total_params)

    new_log = True
    while True:
        model.learn(
            total_timesteps=config["total_timesteps"] if not args.test else 4,
            reset_num_timesteps=new_log,
            callback=WandbCallback(),
        )
        print("Finish!")
        if args.test:
            break
        model.save(zip)
        new_log = False


if __name__ == "__main__":
    sys.exit(main())
