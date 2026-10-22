"""
config.py

TIM-DPO

Central configuration for all TIM-DPO hyperparameters.

Keeping algorithmic settings here ensures that
experiments are reproducible and simplifies
future ablation studies.
"""

from dataclasses import dataclass


@dataclass
class TIMConfig:
    """
    Token Importance Memory configuration.
    """

    # -------------------------------------------------------
    # Memory Update
    # -------------------------------------------------------

    # Memory update strategy.
    #
    # Supported:
    #   "ema"             -> persistent exponential moving average
    #   "previous_batch"  -> use only the most recent batch
    memory_mode: str = "ema"

    # Exponential Moving Average decay.
    # Only used when memory_mode == "ema".
    beta: float = 0.99

    # Initial importance assigned to every vocabulary token.
    initial_importance: float = 1.0

    # -------------------------------------------------------
    # Importance Estimation
    # -------------------------------------------------------

    # Token importance metric computed from embedding gradients.
    #
    # Supported:
    #   "l2_squared"
    #   "l2"
    importance_metric: str = "l2_squared"

    # Whether to clip importance observations before updating memory.
    clip_importance: bool = True

    # Bounds for clipping individual importance observations
    # before they are incorporated into the importance memory.
    clip_max: float = 4.0
    clip_min: float = 0.25

    # Normalize the batch-level token importance observations
    # so that their mean equals one before the memory update.
    normalize_batch_importance: bool = True

    # Numerical stability constant.
    eps: float = 1e-8

    # -------------------------------------------------------
    # Weight Retrieval
    # -------------------------------------------------------

    # Compression applied before returning weights.
    #
    # Supported:
    #   "none"
    #   "log"
    compression: str = "none"

    # Final normalization applied to retrieved weights.
    #
    # Supported:
    #   "none"
    #   "mean"
    normalization: str = "mean"

    # -------------------------------------------------------
    # Configuration Validation
    # -------------------------------------------------------

    def __post_init__(self):

        if self.memory_mode not in {
            "ema",
            "previous_batch",
        }:
            raise ValueError(
                f"Unknown memory_mode: {self.memory_mode}. "
                "Expected 'ema' or 'previous_batch'."
            )

        if self.importance_metric not in {
            "l2",
            "l2_squared",
        }:
            raise ValueError(
                f"Unknown importance_metric: "
                f"{self.importance_metric}. "
                "Expected 'l2' or 'l2_squared'."
            )

        if self.compression not in {
            "none",
            "log",
        }:
            raise ValueError(
                f"Unknown compression: {self.compression}. "
                "Expected 'none' or 'log'."
            )

        if self.normalization not in {
            "none",
            "mean",
        }:
            raise ValueError(
                f"Unknown normalization: {self.normalization}. "
                "Expected 'none' or 'mean'."
            )

        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError(
                f"beta must be between 0 and 1, got {self.beta}."
            )

        if self.clip_min > self.clip_max:
            raise ValueError(
                "clip_min cannot be greater than clip_max."
            )

        if self.clip_min < 0.0:
            raise ValueError(
                f"clip_min must be non-negative, got {self.clip_min}."
            )

        if self.initial_importance <= 0.0:
            raise ValueError(
                "initial_importance must be positive."
            )

        if self.eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )