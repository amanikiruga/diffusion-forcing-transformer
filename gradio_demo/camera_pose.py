import torch
from utils.geometry_utils import CameraPose
from einops import rearrange, repeat
import math
import roma

class ControllableCameraPose(CameraPose):
    def to_vectors(self) -> torch.Tensor:
        """
        Returns the raw camera poses.
        Returns:
            torch.Tensor: The raw camera poses. Shape (B, T, 4 + 12).
        """
        RT = torch.cat([self._R, rearrange(self._T, "b t i -> b t i 1")], dim=-1)
        return torch.cat([self._K, rearrange(RT, "b t i j -> b t (i j)")], dim=-1)

    def extend(
        self,
        num_frames: int,
        x_angle: float = 0.0,
        y_angle: float = 0.0,
        distance: float = 100.0,
    ) -> None:
        """
        Extends the camera poses.
        Let's say 0 degree is the direction of the last camera pose.
        Smoothly Move & rotate the camera poses in the direction of the given angle (clockwise) in a 2D plane.
        Args:
            num_frames (int): The number of frames to extend.
            x_angle (float): The angle to extend. The angle is in degrees.
            y_angle (float): The angle to extend. The angle is in degrees.
        """
        MOVING_SCALE = 0.5 * distance / 100
        self._normalize_by(self._R[:, -1], self._T[:, -1])

        # first compute relative poses for the final n + num_frames th frame

        # compute the rotation matrix for the given angle
        R_final = roma.euler_to_rotmat(
            convention="xyz",
            angles=torch.tensor(
                [-x_angle, -y_angle, 0], device=self._R.device, dtype=torch.float32
            ),
            degrees=True,
            dtype=torch.float32,
            device=self._R.device,
        ).unsqueeze(0)

        # compute the translation vector for the given angle
        T_final = torch.tensor(
            [
                -MOVING_SCALE * num_frames * math.sin(math.radians(y_angle)),
                MOVING_SCALE * num_frames * math.sin(math.radians(x_angle)),
                -MOVING_SCALE * num_frames * math.cos(math.radians(y_angle)),
            ],
            device=self._T.device,
            dtype=self._T.dtype,
        ).unsqueeze(0)

        R = torch.cat(
            [self._R, repeat(R_final, "b i j -> b t i j", t=num_frames).clone()], dim=1
        )
        T = torch.cat(
            [self._T, repeat(T_final, "b i -> b t i", t=num_frames).clone()], dim=1
        )
        K = torch.cat(
            [self._K, repeat(self._K[:, -1], "b i -> b t i", t=num_frames).clone()],
            dim=1,
        )
        self._R = R
        self._T = T
        self._K = K
        # interpolate all frames btwn the last frame and the final frame
        self.replace_with_interpolation(
            torch.cat(
                [
                    torch.zeros_like(self._T[:, :-num_frames, 0]),
                    torch.ones_like(self._T[:, -num_frames:-1, 0]),
                    torch.zeros_like(self._T[:, -1:, 0]),
                ],
                dim=-1,
            ).bool()
        )

def extend_poses(
    conditions: torch.Tensor,
    n: int,
    x_angle: float = 0.0,
    y_angle: float = 0.0,
    distance: float = 0.0,
) -> torch.Tensor:
    poses = ControllableCameraPose.from_vectors(conditions)
    poses.extend(n, x_angle, y_angle, distance)
    return poses.to_vectors()