"""
weighted_dpo.py

TIM-DPO

Weighted Direct Preference Optimization.

Extends TRL's DPOTrainer with

• Token Importance Memory (TIM)
• Token-weighted sequence scoring
• Online importance updates
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TIMConfig

from trl.trainer.utils import (
    flush_left,
    flush_right,
    selective_log_softmax,
)

from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_torch_hpu_available,
    is_sagemaker_mp_enabled,
)

from accelerate.utils import DistributedType

from transformers.utils import logging

logger = logging.get_logger(__name__)

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

from trl import DPOTrainer

from tim import TokenImportanceMemory
from hooks import EmbeddingGradientHook


class WeightedDPOTrainer(DPOTrainer):
    """
    TIM-DPO trainer.
    """

    def __init__(
        self,
        *args,
        tim_config: TIMConfig | None = None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.tim_config = tim_config or TIMConfig()

        # Token Importance Memory (TIM).
        #
        # Maintains a vocabulary-level importance estimate for each token.
        # The stored estimates are used to weight token contributions
        # when computing sequence scores for the DPO objective.
        self.tim = TokenImportanceMemory(
            vocab_size=self.model.config.vocab_size,
            tim_config=self.tim_config,
            device=self.accelerator.device,
        )
        print(f"TIM device: {self.tim.ema_table.device}")

        self.gradient_hook = EmbeddingGradientHook(
            importance_metric=self.tim_config.importance_metric
        )

        self._label_buffer: list[torch.Tensor] = []

        embedding_layer = self.model.get_input_embeddings()

        # Register a backward hook on the input embedding layer.
        #
        # During backpropagation the hook captures embedding
        # gradients, allowing token-level importance to be
        # estimated after each training batch.
        self.gradient_hook.register(
            embedding_layer
        )

    def concatenated_forward(
            self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
        ) -> dict[str, torch.Tensor]:
            """
            Run a policy or reference model on the concatenated chosen
            and rejected inputs.

            Chosen and rejected sequences are concatenated so that they
            can be processed in a single forward pass.

            Token-level log probabilities are computed for response tokens
            and weighted using the current Token Importance Memory.

            Args:
                model:
                    Model used for the forward pass.

                batch:
                    Batch containing prompt, chosen, and rejected inputs.

                is_ref_model:
                    Whether the forward pass is for the reference model.
                    Reference-model forwards do not store labels for the
                    subsequent TIM importance update.
            """
            num_examples = batch["prompt_input_ids"].shape[0]
    
            concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
    
            model_kwargs = {"use_cache": False}
            if self.aux_loss_enabled:
                model_kwargs["output_router_logits"] = True
    
            # Add the pixel values and attention masks for vision models
            if "pixel_values" in concatenated_batch:
                model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
            if "image_sizes" in concatenated_batch:
                model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]
    
            prompt_input_ids = concatenated_batch["prompt_input_ids"]
            prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
            completion_input_ids = concatenated_batch["completion_input_ids"]
            completion_attention_mask = concatenated_batch["completion_attention_mask"]
            if self.is_encoder_decoder:
                labels = completion_input_ids
                labels[completion_attention_mask == 0] = self.label_pad_token_id
                outputs = model(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    labels=labels,  # we need the labels for the logits to be returned
                    **model_kwargs,
                )
                logits = outputs.logits
                loss_mask = completion_attention_mask.bool()
            else:
                # Concatenate the prompt and completion inputs
                input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
                attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
                # Mask the prompt but not the completion for the loss
                loss_mask = torch.cat(
                    (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                    dim=1,
                )
    
                # Flush and truncate
                if self.max_length is not None and self.max_length < attention_mask.size(1):
                    if self.truncation_mode == "keep_start":
                        # Flush left to reduce the memory usage
                        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                        attention_mask = attention_mask[:, : self.max_length]
                        input_ids = input_ids[:, : self.max_length]
                        loss_mask = loss_mask[:, : self.max_length]
                    elif self.truncation_mode == "keep_end":
                        # Flush right before truncating left, then flush left
                        # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                        #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                        attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                        input_ids = input_ids[:, -self.max_length :]
                        attention_mask = attention_mask[:, -self.max_length :]
                        loss_mask = loss_mask[:, -self.max_length :]
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    else:
                        raise ValueError(
                            f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                            "'keep_start']."
                        )
                else:
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
    
                if self.use_logits_to_keep:
                    # Compute logits_to_keep based on loss_mask pattern:
                    # [[0, 0, 0, x, x, x, x],
                    #  [0, 0, 0, x, x, x, 0]]
                    #         ^ start computing logits from here ([:, -(7-3+1):])
                    first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                    logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                    model_kwargs["logits_to_keep"] = logits_to_keep
    
                model_kwargs["output_hidden_states"] = True
    
                if self.padding_free:
                    # Flatten the input_ids, position_ids, and loss_mask
                    # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                    #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                    input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                    loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                    position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                    model_kwargs["position_ids"] = position_ids
                else:
                    model_kwargs["attention_mask"] = attention_mask
    
                outputs = model(input_ids, **model_kwargs)
                logits = outputs.logits
    
                # Offset the logits by one to align with the labels
                labels = torch.roll(input_ids, shifts=-1, dims=1)
                loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
    
                if self.use_logits_to_keep:
                    # Align labels with logits
                    # logits:    -,  -, [x2, x3, x4, x5, x6]
                    #                     ^ --------- ^       after logits[:, :-1, :]
                    # labels:   [y0, y1, y2, y3, y4, y5, y6]
                    #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                    # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                    labels = labels[:, -logits_to_keep:]
                    loss_mask = loss_mask[:, -logits_to_keep:]
    
            if logits.shape[:2] != labels.shape[:2]:
                # for llava, the returned logits include the image tokens (placed before the text tokens)
                seq_len = labels.shape[1]
                logits = logits[:, -seq_len:]
    
            # Compute the log probabilities of the labels
            labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
            per_token_logps = selective_log_softmax(logits, labels)
            per_token_logps[~loss_mask] = 0
            per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
    
            if self.padding_free:
                # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
                batch_size, seq_len = attention_mask.shape
                per_token_logps_ = torch.zeros(
                    batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
                )
                per_token_logps_[attention_mask.bool()] = per_token_logps
                per_token_logps = per_token_logps_

            # Retrieve TIM weights for the target tokens whose
            # log probabilities contribute to the DPO sequence score.
            token_ids = labels[:, 1:]

            weights = self.tim.get_weights(token_ids)


            #optional Shuffled TIM Control
            #this randomly assigns the computed token importance values to a different token
            #
            #
            # shuffled_weights = torch.empty_like(weights)

            # for i in range(weights.shape[0]):
            #     perm = torch.randperm(weights.shape[1], device=weights.device)
            #     shuffled_weights[i] = weights[i, perm]

            # weights = shuffled_weights



            # Restrict TIM weighting to the same tokens that participate in the
            # DPO objective. Masked tokens receive zero weight.
            weights = weights * loss_mask[:, 1:].to(weights.dtype)

            # Weight each token log-probability according to its
            # current importance estimate.
            weighted_logps = weights * per_token_logps[:, 1:]

            # Sequence score used by TIM-DPO.
            #
            # Unlike standard DPO, token contributions are weighted
            # according to their gradient-derived importance estimates
            # rather than contributing equally.
            all_logps = weighted_logps.sum(dim=-1)

            # Store labels only for the policy forward.
            # Reference-model forwards do not contribute embedding gradients
            # to the TIM update.
            if not is_ref_model:
                self._label_buffer.append(
                    (
                        labels.detach(),
                        loss_mask.detach(),
                    )
                )
    
            output = {}
            output["labels"] = labels
    
            if self.use_weighting:
                with torch.no_grad():
                    # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                    logprobs = F.log_softmax(logits, dim=-1)
                    weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                    per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                    all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                    chosen_weights = all_weights[:num_examples]
                    rejected_weights = all_weights[num_examples:]
                    output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)
    
            if self.args.rpo_alpha is not None or "sft" in self.loss_type:
                # Only use the chosen logits for the RPO loss or SFT loss
                chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
                chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]
    
                # Compute the log probabilities of the labels
                output["nll_loss"] = F.cross_entropy(
                    torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
                )
    
            if "ipo" in self.loss_type:
                all_logps = all_logps / loss_mask.sum(-1)
    
            if self.args.ld_alpha is not None and not is_ref_model:
                # Compute response lengths based on loss_mask
                completion_lengths = loss_mask.sum(dim=1)
    
                chosen_lengths = completion_lengths[:num_examples]
                rejected_lengths = completion_lengths[num_examples:]
                public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
                public_lengths = torch.cat([public_lengths, public_lengths], dim=0)
    
                seq_len = per_token_logps.size(1)
                position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)
    
                ld_mask = position_ids < public_lengths.unsqueeze(1)
                mask = position_ids < completion_lengths.unsqueeze(1)
    
                front_mask = (ld_mask & mask).float()
                rear_mask = (~ld_mask & mask).float()
                front_logps = (per_token_logps * front_mask).sum(dim=1)
                rear_logps = (per_token_logps * rear_mask).sum(dim=1)
    
                all_logps = front_logps + self.args.ld_alpha * rear_logps
    
            output["chosen_logps"] = all_logps[:num_examples]
            output["rejected_logps"] = all_logps[num_examples:]
    
            # Compute the mean logits
            if self.padding_free:
                # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
                # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
                # and the second half to the rejected tokens.
                # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
                split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
                mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
                mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
            else:
                mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
                mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()
    
            output["mean_chosen_logits"] = mean_chosen_logits
            output["mean_rejected_logits"] = mean_rejected_logits
    
            if self.aux_loss_enabled:
                output["aux_loss"] = outputs.aux_loss
    
            return output
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                try:
                    from apex import amp
                except ImportError as exc:
                    raise ImportError(
                        "Apex is required when use_apex=True."
                    ) from exc

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                    loss = loss / self.current_gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

                # ---------------------------------------
                # TIM update
                # ---------------------------------------

                # Retrieve token-level gradient-based importance estimates
                # captured during the backward pass.
                #
                # These serve as the importance observations used to
                # update the Token Importance Memory.
                importance_batches = self.gradient_hook.gradient_importance()

                assert len(importance_batches) == len(self._label_buffer)

                for (labels, loss_mask), importance in zip(
                    self._label_buffer,
                    importance_batches,
                ):

                    # DPO / causal-LM loss predicts labels[:, 1:] from
                    # input positions [:, :-1].
                    #
                    # Therefore:
                    #
                    #   input token importance [:, :-1]
                    #                ↓
                    #   target token IDs  [1:]
                    #
                    # IMPORTANT: do NOT shift importance in the same direction
                    # as the labels.

                    token_ids = labels[:, 1:]
                    # Update TIM only for valid response tokens.
                    # Prompt tokens are excluded by the response loss mask,
                    # and EOS is excluded from the importance memory update.
                    valid_mask = loss_mask[:, 1:].bool() & (
                            token_ids != self.processing_class.eos_token_id
                        )

                    aligned_importance = importance[:, :-1]

                    # Update the Token Importance Memory using response-token IDs
                    # and their corresponding gradient-derived importance observations.
                    self.tim.update(
                        token_ids=token_ids[valid_mask],
                        importance=aligned_importance[valid_mask],

                    )

            self.gradient_hook.clear()
            self._label_buffer.clear()

            return loss.detach()