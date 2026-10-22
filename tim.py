"""
Token Importance Memory (TIM)

TIM-DPO

Maintains a vocabulary-level token importance estimate for
every vocabulary token.

The stored values are raw importance estimates.

Compression and normalization are only applied when
retrieving weights for training.
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from config import TIMConfig

class TokenImportanceMemory(nn.Module):
    """
    Token Importance Memory.

    Maintains a vocabulary-level importance estimate for each token.

    In EMA mode:

        I_v ← β I_v + (1-β) ĉ_v

    In previous-batch mode:

        I_v ← ĉ_v

    where ĉ_v is the batch-level mean importance observation
    for vocabulary token v, computed from embedding gradient signals.

    The importance metric is controlled by TIMConfig.
    """

    def __init__(
        self,
        vocab_size: int,
        tim_config: TIMConfig,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.tim_config = tim_config

        self.beta = tim_config.beta
        self.eps = tim_config.eps

        self.compression = tim_config.compression
        self.normalization = tim_config.normalization

        self.clip_importance = tim_config.clip_importance
        self.clip_min = tim_config.clip_min
        self.clip_max = tim_config.clip_max

        self.memory_mode = tim_config.memory_mode

        self.device = torch.device(device)
        self.normalize_batch_importance = tim_config.normalize_batch_importance

        # Every vocabulary token starts equally important.
        self.register_buffer(
            "ema_table",
            torch.full(
                (vocab_size,),
                tim_config.initial_importance,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.register_buffer(
            "update_counts",
            torch.zeros(
                vocab_size,
                dtype=torch.long,
                device=self.device,
            ),
        )    
       

    # -------------------------------------------------------
    # Weight Retrieval
    # -------------------------------------------------------

    @torch.no_grad()
    def get_weights(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns token weights for a tensor of token ids.

        Previously unseen tokens retain the neutral weight of 1.0.

        Pipeline:

            importance lookup
                ↓
            compression
                ↓
            normalization
        """

        weights = self.ema_table.index_select(
            0,
               token_ids.reshape(-1),
        ).view(token_ids.shape)

        updated = self.update_counts.index_select(
            0,
            token_ids.reshape(-1),
        ).view(token_ids.shape) > 0

        weights = self._compress(weights)

        if updated.any():
            weights_updated = self._normalize(weights[updated])
            weights = weights.clone()
            weights[updated] = weights_updated

        # Previously unseen tokens retain the neutral weight of 1.0.
        weights[~updated] = 1.0

        return weights

    # -------------------------------------------------------
    # Compression
    # -------------------------------------------------------

    def _compress(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.compression == "none":
            return x

        if self.compression == "log":
            return torch.log1p(x)

        raise ValueError(
            f"Unknown compression: {self.compression}"
        )

    # -------------------------------------------------------
    # Normalization
    # -------------------------------------------------------

    def _normalize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.normalization == "none":
            return x

        if self.normalization == "mean":

            low = 0.0
            high = 1.0

            def mean_weight(scale):
                return (scale * x).mean()

            # Find an upper bound whose mean is >= 1.
            while mean_weight(high) < 1.0:
                high *= 2.0

            # Find the scale giving mean ~= 1.
            for _ in range(40):

                mid = (low + high) / 2.0

                if mean_weight(mid) < 1.0:
                    low = mid
                else:
                    high = mid

            scale = (low + high) / 2.0

            return scale * x

        raise ValueError(
            f"Unknown normalization: {self.normalization}"
        )

    # -------------------------------------------------------
    # EMA Update
    # -------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        token_ids: torch.Tensor,
        importance: torch.Tensor,
    ):
        """
        Update token importance memory.

        Parameters
        ----------
        token_ids

            Flattened vocabulary ids.

        importance

            Gradient-derived importance observations for the
            corresponding token positions.

        Repeated occurrences of the same vocabulary token are
        averaged before the memory update.
        """

        token_ids = token_ids.reshape(-1)
        importance = importance.reshape(-1)

        if token_ids.numel() != importance.numel():
            raise ValueError(
                "token_ids and importance must have the same number of elements."
            )

        # Collapse repeated occurrences so that each vocabulary
        # token receives exactly one memory update per batch.
        unique_tokens, inverse = torch.unique(
            token_ids,
            sorted=False,
            return_inverse=True,
        )

        # Accumulate the total observed importance for each
        # unique vocabulary token.
        sums = torch.zeros(
            unique_tokens.numel(),
            device=importance.device,
            dtype=importance.dtype,
        )

        # Count how many times each unique token appears
        # in the current batch.
        counts = torch.bincount(inverse).to(sums.dtype)

        sums.index_add_(0, inverse, importance)

        # Average importance over all occurrences of each
        # token within the current batch.
        means = sums / counts

        if self.clip_importance:
            means = torch.clamp(
                means,
                min=self.clip_min,
                max=self.clip_max,
            )

        # Optionally normalize the token-level observations within
        # the current batch so their mean equals 1 before the
        # memory update.
        if self.normalize_batch_importance:

            batch_mean = means.mean().detach()

            if batch_mean > 0:
                means = means / batch_mean


        # Update the token importance memory.
        #
        # In EMA mode, β controls how much historical importance
        # is retained. Larger β results in slower but more stable updates.
        if self.memory_mode == "ema":

            self.ema_table[unique_tokens] = (
                self.beta
                * self.ema_table[unique_tokens]
                +
                (1.0 - self.beta)
                * means
            )

        elif self.memory_mode == "previous_batch":

            self.ema_table[unique_tokens] = means

        else:
            raise ValueError(
                f"Unknown memory_mode: {self.memory_mode}"
            )

        # Record how many token occurrences have contributed to
        # each token's importance estimate.
        self.update_counts[unique_tokens] += counts.long()

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    @torch.no_grad()
    def statistics(self) -> Dict[str, float]:

        table = self.ema_table

        return {

            "mean": table.mean().item(),

            "std": table.std().item(),

            "min": table.min().item(),

            "max": table.max().item(),

            "median": table.median().item(),

            "p95": torch.quantile(
                table,
                0.95,
            ).item(),

            "p99": torch.quantile(
                table,
                0.99,
            ).item(),
        }

    @torch.no_grad()
    def reset(self):

        self.ema_table.fill_(
            self.tim_config.initial_importance
        )

        self.update_counts.zero_()