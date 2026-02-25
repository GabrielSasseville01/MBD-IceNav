"""
Physics-informed Sea Current RNN.

Ported from AUTO-IceNav (ship_ice_planner/controller/seanet.py).
Uses shallow water equations with learnable parameters for
spatially-varying ocean current evolution.

Architecture:
  CNN encoder -> Physics-informed RNN (shallow water) -> CNN decoder

Training data: ECCO4 monthly surface velocity fields, preprocessed
to (T, H, W, 2) tensors of (u, v) velocity components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_dataset(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Temporally upsample a (T, C, H, W) tensor by *factor* via trilinear interpolation."""
    aug_seq_len = factor * x.shape[0]
    x = x.permute(0, 2, 3, 1).unsqueeze(0)           # (1, T, H, W, C)
    x = x.permute(0, 4, 1, 2, 3)                       # (1, C, T, H, W)
    x = F.interpolate(x, size=(aug_seq_len, x.shape[3], x.shape[4]),
                       mode='trilinear', align_corners=False)
    return x.squeeze(0).permute(1, 0, 2, 3)            # (T, C, H, W)


class ShallowWaterPhysics(nn.Module):
    """2D shallow water momentum equations with learnable physical parameters.

    du/dt = -u du/dx - v du/dy + f v + nu nabla^2 u - r u
    dv/dt = -u dv/dx - v dv/dy - f u + nu nabla^2 v - r v

    A small CNN provides a learned correction blended with the physics.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        self.coriolis = nn.Parameter(torch.tensor(1e-4))
        self.viscosity = nn.Parameter(torch.tensor(1e-3))
        self.friction = nn.Parameter(torch.tensor(1e-4))
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.physics_weight = nn.Parameter(torch.tensor(0.5))
        self.correction_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )

    @staticmethod
    def _central_diff(field):
        dx = (torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)) / 2.0
        dy = (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / 2.0
        return dx, dy

    @staticmethod
    def _laplacian(field):
        return (torch.roll(field, 1, dims=2) + torch.roll(field, -1, dims=2)
                + torch.roll(field, 1, dims=1) + torch.roll(field, -1, dims=1)
                - 4 * field)

    def forward(self, velocity_field):
        """velocity_field: (2, H, W) -> (2, H, W)."""
        u, v = velocity_field[0:1], velocity_field[1:2]
        du_dx, du_dy = self._central_diff(u)
        dv_dx, dv_dy = self._central_diff(v)
        du_dt = (-(u * du_dx + v * du_dy)
                 + self.coriolis * v
                 + self.viscosity * self._laplacian(u)
                 - self.friction * u)
        dv_dt = (-(u * dv_dx + v * dv_dy)
                 - self.coriolis * u
                 + self.viscosity * self._laplacian(v)
                 - self.friction * v)
        vel_phys = torch.cat([u + self.dt * du_dt, v + self.dt * dv_dt], dim=0)
        correction = self.correction_net(vel_phys.unsqueeze(0)).squeeze(0)
        w = torch.sigmoid(self.physics_weight)
        return w * vel_phys + (1 - w) * (vel_phys + correction)


class SeaCurrentRNN(nn.Module):
    """CNN encoder-decoder + physics-informed RNN for ocean current evolution.

    Input  : (T, H, W, 2)  — sequence of velocity frames
    Output : (T, H, W, 2)  — evolved velocity frames (one-step forecast per input)
    """

    def __init__(self, hidden_size: int = 128,
                 input_height: int = 600, input_width: int = 600):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoded_h = input_height // 4
        self.encoded_w = input_width // 4
        self.encoded_channels = 8
        self.encoded_flat_size = self.encoded_channels * self.encoded_h * self.encoded_w

        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.physics_encoder = ShallowWaterPhysics(self.encoded_h, self.encoded_w)
        self.to_physics = nn.Conv2d(self.encoded_channels, 2, kernel_size=1)
        self.from_physics = nn.Conv2d(2, self.encoded_channels, kernel_size=1)

        self.W_in = nn.Linear(self.encoded_flat_size, hidden_size, bias=True)
        self.W_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_out = nn.Linear(hidden_size, self.encoded_flat_size, bias=True)

        self.deconv1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(2, 2, kernel_size=3, padding=1)

    def _encode(self, x):
        x = x.permute(0, 3, 1, 2)          # (T, 2, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x                             # (T, 8, H/4, W/4)

    def _decode(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return self.conv_final(x)            # (T, 2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self._encode(x)
        T = x_enc.shape[0]
        h = torch.zeros(self.hidden_size, device=x.device, dtype=x.dtype)
        states = []
        for t in range(T):
            x_t = x_enc[t]
            vel = self.to_physics(x_t.unsqueeze(0)).squeeze(0)
            vel_ev = self.physics_encoder(vel)
            feat = self.from_physics(vel_ev.unsqueeze(0)).squeeze(0)
            feat_flat = feat.reshape(-1)
            h = torch.tanh(self.W_in(feat_flat) + self.W_hidden(h))
            correction = self.W_out(h).reshape(
                self.encoded_channels, self.encoded_h, self.encoded_w)
            states.append(feat + 0.1 * correction)
        out = self._decode(torch.stack(states))
        return out.permute(0, 2, 3, 1)      # (T, H, W, 2)
