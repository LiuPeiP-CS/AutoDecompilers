import os  # ← 补上这个！
import asyncio
import random
import subprocess
from typing import List, Dict, Sequence, Any, Tuple, Union
from collections import defaultdict
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, Future

from datasets import Dataset
from vllm import LLM, SamplingParams
from abc import ABC

# from ..experimental.bco.bco_trainer import logger
import logging
logger = logging.getLogger(__name__)

from ..rewards.fuse_rewards import OverallRewards
import threading

_LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
TOKENIZER_NAME = "/workspace/qwen3-train/models/instruct-6.7b"

_REWARD_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(((os.cpu_count() or 8) // _LOCAL_WORLD_SIZE) * 0.75)),
    thread_name_prefix="reward_worker",
)

# 单次 reward 计算的超时秒数，可通过环境变量覆盖
_REWARD_TIMEOUT = float(os.environ.get("REWARD_TIMEOUT", "350"))


def dict_to_string(data: Union[Dict, List], indent: int = 0) -> str:
    """Convert dictionary/list to a formatted string representation."""
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{'  ' * indent}{key}:")
                lines.append(dict_to_string(value, indent + 1))
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return '\n'.join(lines)
    elif isinstance(data, list):
        lines = []
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"{'  ' * indent}[{i}]:")
                lines.append(dict_to_string(item, indent + 1))
            else:
                lines.append(f"{'  ' * indent}[{i}]: {item}")
        return '\n'.join(lines)
    else:
        return str(data)


def count_tokens_simple(tokenizer, data) -> int:
    """Count tokens in a text string."""
    text = dict_to_string(data)
    if not text or not isinstance(text, str):
        return 0

    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def load_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


class MultiTurnEnv(ABC):

    def __init__(
        self,
        sampling_args: Dict[str, Any] = {},
        mask_env_response: bool = True,
        # ⚡ max_workers → max_concurrent，默认扩到 16（协程比线程轻得多）
        max_concurrent: int = 32,
        max_steps: int = 5,
        # ⚡ sleep_time 默认 0.0：本地 vLLM 无外部 rate-limit，不需要限速 sleep
        sleep_time: float = 0.0,
        so_path=None,
        use_apted=False,
        judge_c=False,
        re_exe_reward=True,
        syntax_reward=False,
        semantic_reward=False,
        mode='train',
        dataset='exebench',
        train_id=0,
        mt_max_tokens=16384,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
            "truncate_prompt_tokens": None,
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_concurrent = max_concurrent
        self.sleep_time = sleep_time  # 保留字段，但默认 0.0
        self.max_steps = max_steps
        self.train_id = train_id
        self.mt_max_tokens = mt_max_tokens
        self.reward_model = OverallRewards(
            so_path=so_path,
            use_apted=use_apted,
            judge_c=judge_c,
            re_exe_reward=re_exe_reward,
            syntax_reward=syntax_reward,
            semantic_reward=semantic_reward,
            mode=mode,
            dataset=dataset,
        )
        self.ctokenizer = load_tokenizer()

    # ------------------------------------------------------------------
    # 工具方法（不变）
    # ------------------------------------------------------------------
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def is_completed(self, message: Dict[str, str], **kwargs: Any) -> bool:
        try:
            return message["content"] == "All right"
        except Exception:
            return False

    def run_code(self, code: str, **kwargs: Any) -> str:
        try:
            result = subprocess.run(
                ['python', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True,
            )
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip() if result.stdout else ""
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 10 seconds"

    # ------------------------------------------------------------------
    # ⚡ 修复版：_env_response_async
    #    新增：asyncio.wait_for 超时保护 + 异常兜底，防止单个 sample 卡死
    # ------------------------------------------------------------------
    async def _env_response_async(
            self,
            messages: List[Dict[str, str]],
            gt_code: str,
            dependency: Any,
    ) -> Tuple[Dict[str, str], List[float]]:
        loop = asyncio.get_running_loop()
        pre_decom_code = messages[-1]["content"]

        # 每次调用创建独立的 cancel_event
        cancel_event = threading.Event()

        async def _call():
            return await loop.run_in_executor(
                _REWARD_EXECUTOR,
                lambda: self.reward_model.get_code_rewards(
                    dependency=dependency,
                    response=pre_decom_code,
                    ground_truth=gt_code,
                    train_id=self.train_id,
                    cancel_event=cancel_event,  # ← 新增，透传进去
                ),
            )

        try:
            cur_reward, user_feedback = await asyncio.wait_for(
                _call(), timeout=_REWARD_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[_env_response_async] get_code_rewards timed out after "
                f"{_REWARD_TIMEOUT}s, killing subprocess..."
            )
            cancel_event.set()  # ← 新增，真正杀掉内部 subprocess
            cur_reward = [0, 0, 0, 0]
            user_feedback = "In the code execution and correction process, an exception occurred. Please regenerate the code."
        except Exception as e:
            logger.error(f"[_env_response_async] get_code_rewards raised: {e}")
            cancel_event.set()  # ← 异常时也 set，防止僵尸 subprocess
            cur_reward = [0, 0, 0, 0]
            user_feedback = "In the code execution and correction process, an exception occurred. Please regenerate the code."

        return {"role": "user", "content": user_feedback}, cur_reward

    # ------------------------------------------------------------------
    # ⚡ 修复版：_update_single_state_async
    #    新增：整个 state 更新本身也有异常兜底，避免单个 sample 异常
    #    导致 asyncio.gather 提前终止其他任务
    # ------------------------------------------------------------------
    async def _update_single_state_async(
        self,
        j: int,
        state: Dict[str, Any],
        llm_response: Any,
        sampling_params: SamplingParams,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[int, Dict[str, Any]]:
        async with semaphore:
            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time * random.random())

            state = state.copy()

            try:
                if len(state["anchor_prompt_ids"]) == 0:
                    state["anchor_prompt_ids"] = llm_response.prompt_token_ids

                state["messages"].append(
                    {"role": "assistant", "content": llm_response.outputs[0].text}
                )

                # ⚡ 已有超时保护的异步 reward 调用
                next_action, current_reward = await self._env_response_async(
                    state["messages"],
                    state["source_code"],
                    state["function_dependency"],
                )
                state["step_rewards"].append(current_reward)

                total_prev_len = (
                    len(state["anchor_prompt_ids"]) + len(state["completion_ids"])
                )
                env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
                new_completion_len = len(llm_response.outputs[0].token_ids)

                state["completion_mask"].extend([self.env_mask] * env_response_len)
                state["completion_mask"].extend([1] * new_completion_len)

                state["completion_ids"].extend(
                    llm_response.prompt_token_ids[total_prev_len:]
                )
                state["completion_ids"].extend(llm_response.outputs[0].token_ids)

                if self.is_completed(next_action) or len(
                    state["completion_ids"]
                ) > sampling_params.max_tokens:
                    state["completed"] = True
                    # state["completion_ids"] = state["completion_ids"][: sampling_params.max_tokens]
                    state["completion_ids"] = state["completion_ids"][: self.mt_max_tokens]
                    state["completion_mask"] = state["completion_mask"][: len(state["completion_ids"])]
                else:
                    state["messages"].append(next_action)

                if len(state["completion_mask"]) != len(state["completion_ids"]):
                    logger.error(
                        f"[state {j}] completion_mask length {len(state['completion_mask'])} "
                        f"!= completion_ids length {len(state['completion_ids'])}"
                    )
                    raise ValueError(
                        f"Completion mask and completion ids are not the same length for state {j}"
                    )

            except ValueError:
                # ValueError 是严重的数据一致性问题，继续上抛
                raise
            except Exception as e:
                # ⚡ 其他异常（如 reward 计算失败）：标记该 sample 为 completed
                # 避免一个坏 sample 卡死整个 batch
                logger.error(f"[state {j}] unexpected error in state update: {e}, marking as completed.")
                state["completed"] = True

            return j, state

    # ------------------------------------------------------------------
    # ⚡ 修复版：step_all_trace
    #    修复 asyncio.get_event_loop() 在 Python 3.10+ 的 DeprecationWarning
    #    统一用 asyncio.get_event_loop_policy 或直接 try/except 判断
    # ------------------------------------------------------------------
    def step_all_trace(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
    ) -> List[Dict[str, Any]]:
        sampling_params_obj = SamplingParams(**sampling_params)

        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        # # ======================= ⚡ 修复上下文超限问题 =======================
        messages_to_step = []
        valid_live_indices = []
        #

        MAX_SAFE_TOKENS = 8192
        #
        for i in live_indices:
            msgs = states[i]["messages"]
            anchor_len = states[i]['anchor_prompt_messages']
            # anchor_prompt_messages 记录了最开始原始 Prompt 的消息条数
            token_len = count_tokens_simple(self.ctokenizer, msgs)

            if token_len > MAX_SAFE_TOKENS and len(msgs) > anchor_len + 1:
                logger.warning(
                    f"[state {i}] Prompt too long ({token_len} tokens), "
                    f"truncating last message content."
                )
        #       # 只保留一轮对话的内容
                msgs = msgs[:anchor_len+1]
                states[i]["messages"] = msgs
                states[i]["completed"] = True
                continue

            messages_to_step.append(msgs)
            valid_live_indices.append(i)

        # 更新存活的 index
        live_indices = valid_live_indices
        #
        # 如果经过过滤后，所有的样本都被强制 completed 了，直接返回
        if not messages_to_step:
            return states

        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(
            messages_to_step, sampling_params=sampling_params_obj, use_tqdm=False
        )

        async def _run_all():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._update_single_state_async(
                    j=j,
                    state=states[j],
                    llm_response=llm_responses[i],
                    sampling_params=sampling_params_obj,
                    semaphore=semaphore,
                )
                for i, j in enumerate(live_indices)
            ]
            # ⚡ return_exceptions=True：单个 task 异常不会取消其他 task
            return await asyncio.gather(*tasks, return_exceptions=True)

        # ⚡ 修复：用 try/except 替代 get_event_loop()（Python 3.10+ 已弃用无参版本）
        try:
            loop = asyncio.get_running_loop()
            # 已在事件循环内，开独立线程运行新循环
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(asyncio.run, _run_all()).result()
        except RuntimeError:
            # 没有正在运行的事件循环，直接 run
            results = asyncio.run(_run_all())

        # 写回 states，跳过异常结果
        for item in results:
            if isinstance(item, Exception):
                logger.error(f"[step_all_trace] a task raised an unhandled exception: {item}")
                continue
            j, state = item
            states[j] = state

        return states

    # ------------------------------------------------------------------
    # generate（主循环，不变）
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: List[Dict[str, Any]],
        llm: LLM,
        max_turns: int = 5,
        mode: str = "train",
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:

        states = [
            {
                "messages": m["prompt"],
                "anchor_prompt_messages": len(m["prompt"]),
                "anchor_prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
                "function_dependency": m["dependency"],
                "step_rewards": [],
                "source_code": m["source"],
            }
            for i, m in enumerate(prompts)
        ]

        turn_iter = 0
        all_completed = False
        while not all_completed and turn_iter < max_turns:
            states = self.step_all_trace(states, llm, self.sampling_args)
            all_completed = all(s["completed"] for s in states)
            turn_iter += 1

        completion_messages = [s["messages"][s["anchor_prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        prompt_ids = [s["anchor_prompt_ids"] for s in states]
        trajectory_rewards = [s["step_rewards"] for s in states]

        return {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "prompt_ids": prompt_ids,
            "trajectory_rewards": trajectory_rewards,
        }

    # ------------------------------------------------------------------
    # step_api / eval_api（推理评估路径，逻辑不变）
    # ------------------------------------------------------------------
    def step_api(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, str]], bool]:
        messages_copy = messages.copy()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
            )
            assistant_msg = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            messages_copy.append(assistant_msg)

            if self.is_completed(messages_copy[-1]):
                return messages_copy, True
            else:
                env_msg, _ = self.env_response(messages_copy, "", {})
                messages_copy.append(env_msg)
                return messages_copy, False

        except Exception as e:
            messages_copy.append({"role": "assistant", "content": f"Error in API call: {str(e)}"})
            return messages_copy, True

    def eval_api(
        self,
        client: Any,
        model: str,
        max_concurrent: int = 16,
        timeout: int = 60,
        sampling_args: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        def run_evaluation():
            from asyncio import Semaphore
            from tqdm.asyncio import tqdm_asyncio

            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)
            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")

            eval_dataset = self.eval_dataset

            async def process_example(example, semaphore):
                async with semaphore:
                    prompt = example["prompt"]
                    messages = example["prompt"].copy()
                    answer = example["answer"]
                    initial_length = len(messages)

                    for _ in range(self.max_steps):
                        try:
                            loop = asyncio.get_event_loop()
                            messages, is_completed = await loop.run_in_executor(
                                None,
                                lambda: self.step_api(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    **sampling_args,
                                ),
                            )
                            if is_completed:
                                break
                        except Exception as e:
                            print(f"Error processing example {example.get('id', 'unknown')}: {e}")
                            break

                    return {
                        "prompt": prompt,
                        "completions": messages[initial_length:],
                        "answer": answer,
                    }

            async def run_all_examples():
                semaphore = Semaphore(max_concurrent)
                tasks = [process_example(ex, semaphore) for ex in eval_dataset]
                return await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples",
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()

            results_dict = {
                "prompt": [r["prompt"] for r in results],
                "answer": [r["answer"] for r in results],
                "completions": [r["completions"] for r in results],
            }
            reward_funcs = self.get_rubric()
            rewards = {}
            for reward_func in reward_funcs:
                func_rewards = reward_func(**results_dict)
                func_name = reward_func.__name__
                rewards[func_name] = sum(func_rewards) / len(func_rewards)
                print(f"{func_name}: {rewards[func_name]}")
            return rewards

        return run_evaluation()
