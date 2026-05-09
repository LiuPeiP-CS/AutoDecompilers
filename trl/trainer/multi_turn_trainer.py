from typing import Callable, Optional, Union, Any, List, Dict, Tuple

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from .multi_turn_config import MTGRPOConfig
import copy
from trl.models.utils import create_reference_model, unwrap_model_for_generation
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.trainer.utils import pad, prepare_deepspeed
from ..extras.profiling import profiling_context, profiling_decorator
from ..models.utils import disable_gradient_checkpointing
from .utils import nanstd, nanmax, nanmin
if is_peft_available():
    from peft import PeftConfig

from trl.rewards import accuracy_reward

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[List, List], List[float]]]


class MTGRPOTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env,
            max_turns: int,
            # reward_funcs: Union[RewardFunc, List[RewardFunc]],
            reward_weights: Optional[List[float]] = None,
            only_final_turn_reward: bool = False,
            progress_alpha: float = 0.25, # progress_alpha ∈ [0.1, 0.3]
            same_adv: bool = True, # if true, all token are with the same advantages
            # no_turn_reward: Optional[bool] = None,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        if not args.use_vllm:
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")

        # self.reward_funcs = reward_funcs
        # self.num_reward_funcs = len(reward_funcs)
        #
        # if reward_weights is None:
        #     self.reward_weights = torch.ones(self.num_reward_funcs)
        # else:
        #     self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)

        self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)

        self.reward_weights.requires_grad_(False)

        # assert len(self.reward_weights) == self.num_reward_funcs, "There is wrong that the reward_weight does not match the reward_funcs"

        # self.no_turn_reward = no_turn_reward

        super().__init__(
            model=model,
            reward_funcs=accuracy_reward,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        ) # it will have an initial ref_model

        self.env = env # 此处的env，应该是trl.envs.code_env的类对象
        # self.sampling_params = self.env.sampling_args
        self.max_turns = max_turns

        self.only_final_turn_reward = only_final_turn_reward
        self.progress_alpha = progress_alpha
        self.same_adv = same_adv

    def _generate_completions(self, prompts):
        # this function is equal to the _generate_single_turn in GRPOTrainer
        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        # the "server" mode of vllm ↓
        if self.vllm_mode == "server":
            all_prompts = gather_object(prompts)
            if self.accelerator.is_main_process:
                with profiling_context(self, "vLLM.generate"):
                    env_result = self.env.generate( # the prompts are already duplicated
                        prompts=all_prompts, # this module is the same with line 1331 in original GRPOTrainer, i.e., output = self.rollout_func(rollout_prompts, self)
                        llm=self.vllm_client,
                        # sampling_params=self.sampling_params,
                        max_turns=self.max_turns,
                    )
                    payload = (env_result['ids'], env_result['messages'], env_result['mask'], env_result['prompt_ids'], env_result['trajectory_rewards'])
            else:
                payload = None

            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)

            completion_ids, completion_messages, completion_mask, gen_prompt_ids, trajectory_rewards = obj_list[0]

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )

            gen_prompt_ids = gen_prompt_ids[process_slice]
            completion_ids = completion_ids[process_slice]
            completion_messages = completion_messages[process_slice]
            completion_mask = completion_mask[process_slice]
            trajectory_rewards = trajectory_rewards[process_slice]
            # the "server" mode of vllm ↑

        # *******************************************************************************
        # the "colocate" mode of vllm ↓
        elif self.vllm_mode == "colocate":
            if self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                all_prompts = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts = prompts

            if self.args.vllm_enable_sleep_mode:
                self.llm.wake_up(tags=["kv_cache"])

            with profiling_context(self, "vLLM.generate"):
                env_result = self.env.generate( # the prompts are already duplicated
                    prompts=all_prompts, # this module is the same with line 1331 in original GRPOTrainer, i.e., output = self.rollout_func(rollout_prompts, self)
                    llm=self.llm,
                    # sampling_params=self.sampling_params,
                    max_turns=self.max_turns,
                )

            if self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = env_result['ids'][tp_slice]
                completion_messages = env_result['messages'][tp_slice]
                completion_mask = env_result['mask'][tp_slice]
                gen_prompt_ids = env_result['prompt_ids'][tp_slice]
                trajectory_rewards = env_result['trajectory_rewards'][tp_slice]
            else:
                completion_ids = env_result['ids']
                completion_messages = env_result['messages']
                completion_mask = env_result['mask']
                gen_prompt_ids = env_result['prompt_ids']
                trajectory_rewards = env_result['trajectory_rewards']

            if self.args.vllm_enable_sleep_mode:
                self.llm.sleep(level=2)

            # the "colocate" mode of vllm ↑


        return gen_prompt_ids, completion_ids, completion_mask, completion_messages, trajectory_rewards

    def _prepare_model_inputs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        return prompt_completion_ids, attention_mask, logits_to_keep

    def _compute_logps(self, prompt_completion_ids, attention_mask, logits_to_keep, batch_size):
        # with torch.no_grad():
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # if self.num_iterations > 1:
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                print("++++++++++++++++++++++++We are using the gradient_accumulation_steps check++++++++++++++++++++++++\n")
                # old_per_token_logps = self._get_per_token_logps(
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                # ref_per_token_logps = self._get_per_token_logps(
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    # ref_per_token_logps = self._get_per_token_logps(
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )

        return old_per_token_logps, ref_per_token_logps

    def pad_rewards_btk(self, rewards):
        """
        Convert ragged reward list into padded tensor.

        Returns:
        rewards_btk: (B, T_max, K)
        turn_mask: (B, T_max), 1 for valid turns
        T_max: the max turn number
        """
        device = self.accelerator.device

        B = len(rewards)
        T_max = max(len(r) for r in rewards)
        K = len(rewards[0][0])
        rewards_btk = torch.zeros(B, T_max, K, device=device, dtype=torch.float32)
        turn_mask = torch.zeros(B, T_max, device=device, dtype=torch.long)
        for b in range(B):
            for t, r_tk in enumerate(rewards[b]):
                rewards_btk[b, t] = torch.tensor(r_tk, dtype=torch.float32, device=device)
                turn_mask[b, t] = 1

        if self.reward_weights.device != device:
            w = self.reward_weights.to(device).view(1, 1, -1)
        else:
            w = self.reward_weights.view(1, 1, -1)

        agg_rewards = (rewards_btk * w).nansum(dim=-1)

        return agg_rewards, turn_mask, T_max  #

    # def _calculate_rewards(self, prompts, completions, reward_funcs, inputs=None):
    def _calculate_rewards(self, *args, **kwargs):

        aggregate_funcs_rewards =  args[0] if args else None
        pad_traj_mask = args[1] if args else None

        device = self.accelerator.device

        # trajectory_rewards = torch.tensor(trajectory_rewards, dtype=torch.float32, device=device)

        def get_last_valid(
                pad_trajectory_rewards: torch.Tensor, pad_trajectory_rewards_mask: torch.Tensor
        ) -> torch.Tensor:
            """
            using the torch.gather() for getting final turn reward
            """
            mask = pad_trajectory_rewards_mask.bool()
            lengths = mask.sum(dim=1)
            indices = torch.clamp(lengths - 1, 0).unsqueeze(-1).to(device)  # (batch, 1)

            # using gather for collecting
            result = torch.gather(pad_trajectory_rewards, 1, indices).squeeze(-1)
            return result

        def trajectory_reward_progress(
                pad_trajectory_rewards: torch.Tensor,  # (B, T)
                pad_trajectory_rewards_mask: torch.Tensor  # (B, T)
        ) -> torch.Tensor:
            """
            Compute per-turn reward progress:
                delta_r_t = r_t - r_{t-1}
            First valid turn always gets 0 reward.
            Padding positions get 0.
            """

            valid_mask = pad_trajectory_rewards_mask.clone().float()
            valid_mask[:, 0] = 0 # the first turn has no any reward
            valid_lengths = valid_mask.sum(dim=1).clamp(min=1)

            # rewards = pad_trajectory_rewards.float() # trace = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 10],   # 有效长度5])
            # mask = valid_mask.float() # mask = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 1]])

            # r_{t-1}
            prev_rewards = torch.zeros_like(pad_trajectory_rewards, device=device, dtype=torch.float32)
            prev_rewards[:, 1:] = pad_trajectory_rewards[:, :-1] # tensor([[0., 1., 2., 3., 0.], [0., 4., 5., 0., 0.], [0., 6., 7., 8., 9.]])
            reward_progress = pad_trajectory_rewards - prev_rewards # tensor([[ 1.,  1.,  1., -3.,  0.], [ 4.,  1., -5.,  0.,  0.], [ 6.,  1.,  1.,  1.,  1.]])

            # get the valid turn progress
            reward_progress = reward_progress * valid_mask # tensor([[0., 1., 1., -0., 0.], [0., 1., -0., 0., 0.], [0., 1., 1., 1., 1.]])

            total_reward_progress = self.progress_alpha * reward_progress.nansum(dim=1) / valid_lengths # normalize through different lengths (the number of turn)

            return total_reward_progress

        final_turn_rewards = get_last_valid(aggregate_funcs_rewards, pad_traj_mask) # (batch)

        if self.only_final_turn_reward:
            total_rewards = final_turn_rewards
        else:
            wise_turn_progress_rewards = trajectory_reward_progress(aggregate_funcs_rewards, pad_traj_mask)
            total_rewards = final_turn_rewards + wise_turn_progress_rewards

        return gather(total_rewards) # (B, )

    def _compute_normalized_advantages(self, rewards, slice_length=None):
        # mode = "train" if self.model.training else "eval"

        # Compute grouped-wise rewards
        print(f"rewards: {rewards.shape}")
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # (B, )
        print(f"mean_grouped_rewards: {mean_grouped_rewards.shape}")

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B*n,)
        print(f"mean_grouped_rewards repeat_interleave: {mean_grouped_rewards.shape}")
        advantages = rewards - mean_grouped_rewards # (B*n,)

        # Compute the group-level std
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        assert self.scale_rewards != "none", "There is the wrong scale_rewards setting!"
        advantages = advantages / (std_grouped_rewards + 1e-6) # (B*n,)
        advantages = advantages.t()

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * slice_length,
            (self.accelerator.process_index + 1) * slice_length,
        )

        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]
        print(f"process_index: {self.accelerator.process_index}, slice_length: {slice_length}, advantages: {advantages.shape}, all_process_advantages: {all_process_advantages.shape}")

        return mean_grouped_rewards, std_grouped_rewards, all_process_advantages, advantages # (B*n,)

    def _log_completion_samples(self, prompts, completions, rewards):
        prompts_to_log = gather_object(prompts)
        completions_to_log = gather_object(completions)
        rewards_to_log = rewards.tolist()

        if self.accelerator.is_main_process:
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(rewards),
                    "prompt": prompts_to_log,
                    "completion": completions_to_log,
                    "reward": rewards.tolist(),
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def _generate_and_score_completions(
            self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        mode = "train" if self.model.training else "eval"

        #  # the prompt list for all input data, only the conversation component
        # inputs = [{"prompt": xxx, "dependency": yyy, "solutions": zzz}, ...]
        prompts = inputs
        # prompt_ids, prompt_mask = self._prepare_prompt_inputs(prompts, mode)

        # First, update the vLLM weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        prompt_ids, completion_ids, completion_mask, completion_messages, trajectory_rewards = self._generate_completions(prompts) # the return is based on slice

        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        assert self.processing_class.pad_token_id == self.pad_token_id, "the two pad_token_id are different"

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        prompt_completion_ids, attention_mask, logits_to_keep = self._prepare_model_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        # the same with line 1854
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_completion_ids, attention_mask, logits_to_keep, batch_size
        )

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))

        assert len(prompt_ids) == len(trajectory_rewards), "there is wrong due to the difference of prompt number and trajectory number"

        traj_rewards, traj_mask, _ = self.pad_rewards_btk(trajectory_rewards) # (batch, max_turn)

        final_rewards = self._calculate_rewards(traj_rewards, traj_mask).unsqueeze(0) # float (1, B)

        global_mean_grouped_rewards, global_std_grouped_rewards, global_advantages, local_advantages = self._compute_normalized_advantages(
            final_rewards, len(prompts))

        # We also need to compute the global mean_reward and std_reward
        self._metrics[mode]["reward"].append(global_mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(global_std_grouped_rewards.mean().item())
        is_std_zero = torch.isclose(global_std_grouped_rewards, torch.zeros_like(global_std_grouped_rewards))
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        self._logs["advantages"].extend(global_advantages.tolist())

        # advantages = self._same_advantages(completion_mask, local_advantages) # (Bn, seq)
        advantages = local_advantages

        if not self.same_adv:
            comp_advantages = self._weighted_advantages(traj_rewards, traj_mask, completion_mask, local_advantages)
            # 是否需要将prompt的advantage计入
            prop_adv = prompt_mask.float() * local_advantages.unsqueeze(-1)
            advantages = torch.cat([prop_adv, comp_advantages], dim=1)

        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completion_samples(prompts, completion_messages, final_rewards)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def _same_advantages(self, completion_mask, local_advantages):
        """
        completion_mask: (B, T)  0/1
        local_advantages: (B,)
        """
        return completion_mask.float() * local_advantages.unsqueeze(-1) # (Bn, seq)

    def masked_linear_normalize_fixed(self, turns_rewards, turns_mask, dim=-1, eps=1e-8):
        """
        turns_rewards: (B, T)
        turns_mask:    (B, T), 0/1
        """
        device = turns_rewards.device
        turns_mask = turns_mask.to(dtype=turns_rewards.dtype)

        # zero out padded positions
        masked_rewards = turns_rewards * turns_mask

        # sum over valid positions
        sum_rewards = masked_rewards.sum(dim=dim, keepdim=True)

        # avoid divide-by-zero
        # 使用 torch.where 处理整个张量，避免维度不匹配
        normalized_scores = torch.where(
            sum_rewards > eps,
            masked_rewards / (sum_rewards + eps),  # 加 eps 确保安全
            torch.zeros_like(masked_rewards)
        )

        turns_num = turns_mask.sum(dim=-1)

        return normalized_scores.to(device), turns_num

    def batch_weighted_adv(self, mask, rewards, reward_mask):

        B, L = mask.shape # the shape of tokens_mask
        device = mask.device

        result = torch.zeros((B, L), dtype=rewards.dtype, device=device)

        for b in range(B):
            reward_idx = 0
            i = 0
            mask_b = mask[b]
            rewards_b = rewards[b]
            reward_mask_b = reward_mask[b]

            while i < L:
                # 如果是 1，说明是一个 segment 的开始
                if mask_b[i]:
                    # 找到 segment 结束位置
                    j = i
                    while j < L and mask_b[j]:
                        j += 1

                    # 检查是否有对应的有效 reward
                    if reward_idx >= len(rewards_b) or not reward_mask_b[reward_idx]:
                        # 没有更多有效奖励，后续的 segment 不填充
                        break

                    # 填充整个 segment
                    reward_val = rewards_b[reward_idx]
                    result[b, i:j] = reward_val

                    reward_idx += 1
                    i = j
                else:
                    i += 1

        return result

    def _weighted_advantages(self, turns_rewards, turns_mask, tokens_mask, overall_advantages):
        # float, long, long, float
        # we first get the sof_pro for the turns_rewards
        turns_weights, turns_num = self.masked_linear_normalize_fixed(turns_rewards, turns_mask) # float, long
        turn_advantages = (
                overall_advantages.unsqueeze(-1)
                * turns_num.float().unsqueeze(-1)
                * turns_weights
        ) # reconstruct the adv for each turn

        return self.batch_weighted_adv(tokens_mask, turn_advantages, turns_mask)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # the inputs is based on slice
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        # per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep) # per_token_logps是input_ids中每个token的logpro，shape是(Bn, seq)

        advantages = inputs["advantages"]
        print(f"advantages from inputs dict: {advantages.shape}")
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1) # (Bn, 1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        coef_1 = torch.exp(log_importance_weights)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high) # 一般设置epsilon_low与epsilon_high相等
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        if len(advantages.shape) == 2:
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
        else:
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)

        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        mode = "eval" if self.control.should_evaluate else "train"

        # is_clipped = (per_token_loss1 < per_token_loss2).float()
        # clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        # self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss