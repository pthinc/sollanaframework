"""Backend-aware training step loss utilities."""

from typing import Optional

from backends import ensure_backend


def compute_training_loss(out, behaviors, target, target_power: float, alpha: float = 1.0,
						  backend_name: Optional[str] = None):
	"""Compute task + behavioral regularization loss across backends.

	- Torch path uses F.mse_loss and torch.relu
	- TensorFlow path uses reduce_mean + tf.nn.relu
	- NumPy fallback for lightweight scenarios
	"""

	backend = ensure_backend(backend_name)

	if backend.name.startswith("torch"):
		import torch
		import torch.nn.functional as F

		task_loss = F.mse_loss(out, target)
		behavior_power = behaviors.abs().mean()
		behavior_reg = torch.relu(behavior_power - target_power)
		return task_loss + alpha * behavior_reg

	if backend.name.startswith("tensorflow"):
		import tensorflow as tf  # type: ignore

		task_loss = tf.reduce_mean(tf.square(out - target))
		behavior_power = tf.reduce_mean(tf.abs(behaviors))
		behavior_reg = tf.nn.relu(behavior_power - target_power)
		return task_loss + alpha * behavior_reg

	import numpy as np

	task_loss = np.mean((out - target) ** 2)
	behavior_power = np.mean(np.abs(behaviors))
	behavior_reg = np.maximum(behavior_power - target_power, 0)
	return task_loss + alpha * behavior_reg


__all__ = ["compute_training_loss"]
