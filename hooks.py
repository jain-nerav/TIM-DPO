"""
hooks.py

TIM-DPO

Captures gradients of the token embedding outputs

    g_t = dL / de_t

during backpropagation for token-level importance estimation.

The hook is attached once to the embedding layer and
automatically registers a tensor hook on every forward pass.

Gradients from multiple backward passes (e.g. gradient
accumulation) are buffered separately before being
consumed by the Token Importance Memory.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn


class EmbeddingGradientHook:
    """
    Captures embedding-output gradients.

    The stored gradients correspond to

        dL / de_t

    where

    L   = training loss
    e_t = embedding vector at token position t
        
    This class is responsible ONLY for gradient capture.
    It performs no importance estimation or EMA updates.
    """

    def __init__(
        self,
        importance_metric: str = "l2_squared",
    ):

        if importance_metric not in {"l2", "l2_squared"}:
            raise ValueError(
                f"Unknown importance metric: {importance_metric}. "
                "Expected 'l2' or 'l2_squared'."
            )

        self.importance_metric = importance_metric

        # Stores one gradient tensor for each backward pass.
        # Multiple captured tensors can therefore be consumed together
        # when gradient accumulation is used.
        self._gradient_buffer: List[torch.Tensor] = []

        self._forward_handle: Optional[
            torch.utils.hooks.RemovableHandle
        ] = None

    # -------------------------------------------------------
    # Registration
    # -------------------------------------------------------

    def register(
        self,
        embedding_module: nn.Module,
    ) -> None:
        """
        Register a forward hook on the model's embedding layer.

        This only needs to be called once.
        """

        self.remove()

        self._forward_handle = embedding_module.register_forward_hook(
            self._forward_hook
        )

    # -------------------------------------------------------
    # Forward Hook
    # -------------------------------------------------------

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor | tuple,
    ) -> None:
        """
        Automatically registers a tensor hook on the embedding
        output produced during this forward pass.
        """

        if isinstance(output, tuple):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Embedding output must be a Tensor."
            )

        # The embedding layer is frozen under LoRA, so its output may
        # not require gradients by default. TIM needs the gradient
        # flowing through this output in order to estimate token
        # importance.
        if not output.requires_grad:
            output.requires_grad_(True)

        # A new embedding output tensor is created on every forward pass,
        # so a tensor hook must be registered for every forward rather than
        # once during initialization.
        output.register_hook(self._backward_hook)

    # -------------------------------------------------------
    # Tensor Hook
    # -------------------------------------------------------

    def _backward_hook(
        self,
        grad: torch.Tensor,
    ) -> None:
        """
        Store the embedding-output gradients produced by a single
        backward pass.

        Each backward pass contributes one tensor to the gradient
        buffer, allowing gradient accumulation to be handled
        naturally before the Token Importance Memory is updated.
        """

        # Detach so stored gradients do not remain connected
        # to the computation graph.
        self._gradient_buffer.append(
            grad.detach()
        )

    # -------------------------------------------------------
    # Gradient Access
    # -------------------------------------------------------

    def has_gradients(self) -> bool:

        return len(self._gradient_buffer) > 0

    @torch.no_grad()
    def get_raw_gradients(
        self,
    ) -> List[torch.Tensor]:
        """
        Returns every captured embedding-output gradient.

        One tensor is returned for each backward pass.

        No aggregation or averaging is performed.
        """

        if not self.has_gradients():

            raise RuntimeError(
                "No embedding gradients have been captured."
            )

        # Return a shallow copy so callers cannot modify the
        # internal gradient buffer directly.
        return self._gradient_buffer.copy()

    @torch.no_grad()
    def gradient_importance(
        self,
    ) -> List[torch.Tensor]:
        """
        Computes one token-level importance value for every
        captured embedding-output gradient.

        importance_metric = "l2":
            ||g_t||_2

        importance_metric = "l2_squared":
            ||g_t||_2^2
        """

        importance = []

        for gradients in self.get_raw_gradients():

            if self.importance_metric == "l2":
                values = gradients.norm(dim=-1)

            # One scalar squared L2 norm per token.
            #
            # Shape:
            #   (batch_size, sequence_length, embedding_dim)
            #           ↓
            #   (batch_size, sequence_length)
            elif self.importance_metric == "l2_squared":
                values = gradients.pow(2).sum(dim=-1)

            else:
                raise RuntimeError(
                    f"Unknown importance metric: "
                    f"{self.importance_metric}"
                )

            importance.append(values)

        return importance

    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------

    @torch.no_grad()
    def clear(
        self,
    ) -> None:
        """
        Clear all captured gradients after they have been consumed
        by the Token Importance Memory.
        """

        self._gradient_buffer.clear()



    def remove(
        self,
    ) -> None:
        """
        Remove the registered forward hook.
        """

        if self._forward_handle is not None:

            self._forward_handle.remove()

            self._forward_handle = None