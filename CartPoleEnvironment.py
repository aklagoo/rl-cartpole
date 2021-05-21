import gym
from typing import Union, Optional
import numpy as np
from torchvision import transforms
import const


class CartPoleEnvironment:
    """Acts as a Proxy to the gym CartPole-v1 environment with transforms.

    Attributes:
        _env: A gym.Env object representing the CartPole-v1 environment.
        _out_transforms (Optional): Image transforms applied to the cropped environment output.
        _pre_crop_transforms: Image transforms consisting of image resizing and conversion.
        _CROP_WIDTH: Constant indicating maximum width of crop.
    """
    def __init__(self, crop_width: int = const.CROP_WIDTH, out_transforms: Optional[transforms.Compose] = None):
        """Initializes environment and stores attributes."""
        self._env = gym.make('CartPole-v1')
        self._out_transforms = out_transforms
        self._CROP_WIDTH = crop_width

        self._pre_crop_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                const.RESIZE_SHAPE,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def reset(self):
        """Resets environment and returns starting screen after cropping and transforms."""
        # We reset the environment and calculate the position (observation[0]).
        observations = self._env.reset()
        position = self._conv_position(observations[0])

        # Capture the screen and crop image
        screen = self._env.render(mode='rgb_array')
        image = self._crop(screen, position)

        # Transform output and return
        if self._out_transforms is not None:
            image = self._out_transforms(image)

        return image

    def step(self, action):
        """Passes action to self.env.step and transforms rendered output."""
        # We perform an action and calculate the new position (observation[0]).
        observations, reward, done, _ = self._env.step(action)
        position = self._conv_position(observations[0])

        # Capture the screen and crop image
        screen = self._env.render(mode='rgb_array')
        image = self._crop(screen, position)

        # Transform output and return
        if self._out_transforms is not None:
            image = self._out_transforms(image)

        return image, reward, done, {}

    def close(self):
        self._env.close()

    def _crop(self, image: np.ndarray, position):
        """Crops image based on the output parameters.

        image: An image in the format (H, W, C).
        position: An integer representing the screen x-coordinate.
        """
        image = self._pre_crop_transforms(image)

        width_min = position - self._CROP_WIDTH
        width_max = position + self._CROP_WIDTH
        if width_min < 0:
            width_min = 0
            width_max = self._CROP_WIDTH
        if width_max > const.RESIZE_SHAPE[1]:
            width_min = const.RESIZE_SHAPE[1] - self._CROP_WIDTH
            width_max = const.RESIZE_SHAPE[1]

        image = image[:, const.CROP_HEIGHT_MIN:const.CROP_HEIGHT_MAX, width_min:width_max]

        return image

    def _conv_position(self, position):
        """Converts angular position to screen x-coordinate."""
        world_width = self._env.x_threshold * 2
        screen_width = 600
        scale = screen_width / world_width

        return int(position * scale + screen_width / 2)
