from typing import Optional
from pydrake.gym import DrakeGymEnv

import gymnasium as gym
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

from PIL import Image


class CustomDrakeGymEnvResidual(DrakeGymEnv):
    def __init__(
        self,
        simulator,
        time_step,
        action_space,
        observation_space,
        reward,
        action_port_id=None,
        observation_port_id=None,
        render_rgb_port_id=None,
        render_mode="human",
        reset_handler=None,
        info_handler=None,
        hardware=False,
    ):
        super().__init__(
            simulator,
            time_step,
            action_space,
            observation_space,
            reward,
            action_port_id,
            observation_port_id,
            render_rgb_port_id,
            render_mode,
            reset_handler,
            info_handler,
            hardware,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": observation_space["state"],
                "image": gym.spaces.Box(
                    low=-np.inf * np.ones(512),
                    high=np.inf * np.ones(512),
                    dtype=np.float32,
                ),
            }
        )

        weights = ResNet18_Weights.DEFAULT
        self.feature_extractor = torch.nn.Sequential(
            *list(resnet18(weights=weights, progress=False).children())[:-1]
        ).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),  # Resize the smaller edge to 256
                transforms.CenterCrop(224),  # Crop to a 224x224 image
                transforms.ToTensor(),  # Convert image to PyTorch tensor (scaling [0, 1])
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],  # Normalize using ImageNet's mean and std
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        image = Image.fromarray(observation.pop("image0", None)[:, :, :3])

        image_tensor = self.preprocess(image).unsqueeze(0)
        observation["image"] = (
            self.feature_extractor(image_tensor).squeeze().detach().numpy()
        )
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = super().reset(seed=seed, options=options)
        image = Image.fromarray(observation.pop("image0", None)[:, :, :3])
        image_tensor = self.preprocess(image).unsqueeze(0)
        observation["image"] = (
            self.feature_extractor(image_tensor).squeeze().detach().numpy()
        )
        return observation, info
