"""Loss functions."""

from torch import nn


def get_loss(name="mse"):
    """Chọn loss cho regression."""
    name = name.lower()

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    raise ValueError(
        f"Loss không hợp lệ: {name}. "
        "Chọn mse hoặc mae."
    )
