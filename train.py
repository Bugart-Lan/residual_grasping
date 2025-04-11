import argparse
import os
import sys

import gymnasium as gym
import wandb

from psutil import cpu_count
from pydrake.all import StartMeshcat
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback


# import envs.floating_joint
# import envs.end_to_end_grasp
import envs.residual_grasp

# import manipulation.envs.box_flipup


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
        "policy_type": "MultiInputPolicy",
        "total_timesteps": 10 if not args.test else 5,
        "env_name": "ResidualGrasp-v0",
        # "env_name": "EndToEndGrasp-v0",
        # "env_name": "FloatingJoint-v0",
        "env_time_limit": 10 if not args.test else 0.5,
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

    # zip = "data/end_to_end_grasp.zip"
    zip = "data/residual_grasp"

    num_cpu = int(cpu_count() / 4)
    if args.train_single_env:
        meshcat = StartMeshcat()
        env = gym.make(
            config["env_name"],
            meshcat=meshcat,
            time_limit=config["env_time_limit"],
            debug=True,
        )
        check_env(env)
        input("Open meshcat (optional). Press Enter to continue...")
    else:
        # Use a callback so that the forked process imports the environment.
        def make_env():
            return gym.make(
                config["env_name"],
                time_limit=config["env_time_limit"],
            )

        print(f"Number of CPU used for training = {num_cpu}")
        env = make_vec_env(
            make_env,
            n_envs=num_cpu,
            seed=0,
            vec_env_cls=SubprocVecEnv,
        )

    if args.test:
        print("Testing mode")
        model = PPO(
            config["policy_type"],
            env,
            n_steps=4,
            n_epochs=2,
            batch_size=4,
            device="cpu",
        )
    elif os.path.exists(zip):
        print(f"Loading model @ {zip}")
        model = PPO.load(
            zip,
            env,
            verbose=1,
            tensorboard_log=args.log_path or f"runs/{run.id}",
            device="cpu",
        )
    else:
        print("Creating PPO model...")
        model = PPO(
            config["policy_type"],
            env,
            verbose=1,
            tensorboard_log=args.log_path or f"runs/{run.id}",
            device="cpu",
        )

    new_log = True
    while True:
        model.learn(
            total_timesteps=config["total_timesteps"] if not args.test else 4,
            reset_num_timesteps=new_log,
            callback=WandbCallback(),
        )
        if args.test:
            break
        model.save(zip)
        new_log = False


if __name__ == "__main__":
    sys.exit(main())
