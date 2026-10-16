import os  
import asyncio
import random
import subprocess
import re
import copy
from typing import List, Dict, Sequence, Any, Tuple, Union, Optional
from collections import defaultdict
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, Future
import torch.distributed as dist

from datasets import Dataset
from vllm import LLM, SamplingParams
from abc import ABC

# from ..experimental.bco.bco_trainer import logger
import logging
logger = logging.getLogger(__name__)

from ..rewards.fuse_rewards import OverallRewards
import threading

_LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
TOKENIZER_NAME = "/path/to/sft/checkpoint/XXX"

_reward_max_workers_env = os.environ.get("REWARD_MAX_WORKERS", "").strip()
if _reward_max_workers_env:
    _reward_max_workers = max(1, int(_reward_max_workers_env))
elif os.environ.get("VLLM_MODE_HINT", "server") == "colocate":
    _reward_max_workers = max(1, int(((os.cpu_count() or 8) // _LOCAL_WORLD_SIZE) * 0.85))
else:
    _reward_max_workers = max(1, int((os.cpu_count() or 8) * 0.85))

_REWARD_EXECUTOR = ThreadPoolExecutor(
    max_workers=_reward_max_workers,
    thread_name_prefix="reward_worker",
)

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


_FENCED_BLOCK_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+\-]*)[ \t]*\r?\n(?P<code>.*?)```",
    flags=re.DOTALL,
)

_ATTRIBUTE_RE = (
    r"(?:__attribute__\s*\(\([^()]*(?:\([^()]*\)[^()]*)*\)\)\s*|"
    r"__declspec\s*\([^()]*\)\s*)*"
)

_WS_OR_COMMENT = r"(?:\s|//[^\n]*|/\*.*?\*/)*"

_C_FUNCTION_RE = re.compile(
    r"(?ms)^[ \t]*"
    + _ATTRIBUTE_RE +
    r"(?:[A-Za-z_]\w*(?:\s+|\s*\*\s*))+"   
                                           
    r"[A-Za-z_]\w*\s*"                     
    r"\([^;{}]*\)"                         
    + _WS_OR_COMMENT +                     
    r"\{"
)


def _find_matching_brace(text: str, open_index: int) -> Optional[int]:
    depth = 0
    i = open_index
    state = "normal"
    escaped = False

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if state == "line_comment":
            if ch == "\n":
                state = "normal"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "normal"
                i += 1
        elif state == "string":
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                state = "normal"
        elif state == "char":
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                state = "normal"
        else:
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block_comment"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    return None


def _extract_first_c_function(text: str) -> Optional[Tuple[str, int, int]]:
    for match in _C_FUNCTION_RE.finditer(text):
        open_index = match.end() - 1
        if text[open_index] != "{":
            continue
        close_index = _find_matching_brace(text, open_index)
        if close_index is not None:
            return text[match.start(): close_index + 1].strip(), match.start(), close_index + 1
    return None


def _looks_like_c_preamble(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    prose_markers = re.compile(
        r"\b(here is|the following|explanation|this code|decompiled code|"
        r"corrected code|analysis|note that|hope this helps)\b",
        flags=re.IGNORECASE,
    )
    if prose_markers.search(stripped):
        return False

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "//", "/*", "*", "*/")):
            continue
        if line.endswith((";", "{", "}", "\\", ",")):
            continue
        if re.match(
            r"^(typedef|struct|union|enum|static|extern|const|volatile|"
            r"unsigned|signed|long|short|void|char|int|float|double)\b",  # 修复: \\b -> \b
            line,
        ):
            continue
        return False

    return True


def _looks_like_c_suffix(text: str) -> bool:
    """Return True for empty/comments/additional C declarations or functions."""
    stripped = text.strip()
    if not stripped:
        return True
    if _extract_first_c_function(stripped) is not None:
        return _looks_like_c_preamble(stripped.split("{", 1)[0])
    return _looks_like_c_preamble(stripped)


def extract_c_code(response: str) -> Tuple[str, bool, str]:
    raw = (response or "").strip()
    if not raw:
        return "", False, "none"

    fenced_candidates = []
    for match in _FENCED_BLOCK_RE.finditer(raw):
        lang = match.group("lang").strip().lower()
        code = match.group("code").strip()
        if not code:
            continue

        function = _extract_first_c_function(code)
        has_function = function is not None
        language_bonus = 2 if lang in {"c", "h", "cpp", "c++", "cc"} else 0
        score = language_bonus + (10 if has_function else 0) + min(len(code), 20000) / 20000
        fenced_candidates.append((score, code))

    if fenced_candidates:
        _, candidate = max(fenced_candidates, key=lambda item: item[0])
        function = _extract_first_c_function(candidate)
        # Keep the complete fenced block when it contains code declarations/helpers.
        code = candidate if function is not None else ""
        return code.strip(), True, "fenced" if code else "none"

    function = _extract_first_c_function(raw)
    if function is None:
        return "", True, "none"

    code, start, end = function
    prefix = raw[:start]
    suffix = raw[end:]

    if _looks_like_c_preamble(prefix) and _looks_like_c_suffix(suffix):
        return raw, False, "raw"

    had_non_code_text = bool(prefix.strip() or suffix.strip())
    source = "function" if had_non_code_text else "raw"
    return code, had_non_code_text, source

class _Completion:
    __slots__ = ("text", "token_ids")
    def __init__(self, text, token_ids):
        self.text = text
        self.token_ids = token_ids

class _FakeRequestOutput:
    __slots__ = ("prompt_token_ids", "outputs")
    def __init__(self, prompt_token_ids, outputs):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs

class MultiTurnEnv(ABC):

    def __init__(
        self,
        sampling_args: Optional[Dict[str, Any]] = None,
        mask_env_response: bool = True,
        max_concurrent: int = 4,
        max_steps: int = 5,
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
        strict_code_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
            "truncate_prompt_tokens": None,
        }
        self.sampling_args.update(sampling_args or {})
        self.env_mask = 0 if mask_env_response else 1
        self.max_concurrent = max_concurrent
        self.sleep_time = sleep_time  
        self.max_steps = max_steps
        self.train_id = train_id
        self.mt_max_tokens = mt_max_tokens
        self.strict_code_only = strict_code_only
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

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def is_completed(self, message: Dict[str, str], **kwargs: Any) -> bool:
        """Return True only for an explicit validation-success message."""
        try:
            content = " ".join(str(message.get("content", "")).strip().lower().split())
        except Exception:
            return False

        content = content.rstrip(".!")
        return content in {
            "all right",
            "all tests passed",
            "the generated decompiled code is ok",
            "the generated decompiled is ok",
        }

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

    async def _env_response_async(
            self,
            generated_code: str,
            gt_code: str,
            dependency: Any,
            had_non_code_text: bool = False,
    ) -> Tuple[Dict[str, str], List[float], bool]:
        if not generated_code.strip():
            cur_reward = []
            if getattr(self.reward_model, "judge_c", False):
                cur_reward.append(-1.0)
            if getattr(self.reward_model, "re_exe_reward", False):
                cur_reward.append(-1.0)
            if getattr(self.reward_model, "syntax_reward", False):
                cur_reward.append(0.0)
            if getattr(self.reward_model, "semantic_reward", False):
                cur_reward.append(0.0)
            if not cur_reward:
                cur_reward = [0.0]
            feedback = (
                "No complete C function is generated, please re-decompile the pscode. "
            )
            return {"role": "user", "content": feedback}, cur_reward, False

        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        async def _call():
            return await loop.run_in_executor(
                _REWARD_EXECUTOR,
                lambda: self.reward_model.get_code_rewards(
                    dependency=dependency,
                    response=generated_code,
                    ground_truth=gt_code,
                    train_id=self.train_id,
                    cancel_event=cancel_event,
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
            cancel_event.set()
            cur_reward = [0, 0, 0, 0]
            user_feedback = (
                "An exception occurred during code validation. "
                "Please regenerate the complete C code."
            )
        except Exception as e:
            logger.error(f"[_env_response_async] get_code_rewards raised: {e}")
            cancel_event.set()
            cur_reward = [0, 0, 0, 0]
            user_feedback = (
                "An exception occurred during code validation. "
                "Please regenerate the complete C code."
            )

        success = self.is_completed({"content": user_feedback})

        # Compilation may succeed after extraction, but the original response
        # can still violate the required code-only output format.
        if had_non_code_text and self.strict_code_only:
            format_feedback = (
                "The extracted C code was validated, but the response contains "
                "natural-language text or Markdown wrappers. Return only the complete C code."
            )
            if success:
                success = False
                user_feedback = format_feedback
            else:
                user_feedback = f"{format_feedback} {user_feedback}"

            # Preserve a format-learning signal when the C-like reward is enabled.
            if getattr(self.reward_model, "judge_c", False) and cur_reward:
                cur_reward = list(cur_reward)
                cur_reward[0] = min(float(cur_reward[0]), 0.0)

        return {"role": "user", "content": user_feedback}, cur_reward, success

    async def _update_single_state_async(
        self,
        j: int,
        state: Dict[str, Any],
        llm_response: Any,
        sampling_params: SamplingParams,
        semaphore: asyncio.Semaphore,
        max_turns: int,
    ) -> Tuple[int, Dict[str, Any]]:
        async with semaphore:
            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time * random.random())

            state = state.copy()

            try:
                if len(state["anchor_prompt_ids"]) == 0:
                    state["anchor_prompt_ids"] = list(llm_response.prompt_token_ids)

                raw_response = llm_response.outputs[0].text
                state["messages"].append(
                    {"role": "assistant", "content": raw_response}
                )
                state["raw_responses"].append(raw_response)

                extracted_code, had_non_code_text, extraction_source = extract_c_code(
                    raw_response
                )
                # print("############################################\n")
                # print(f"the original response is: \n{raw_response}\n")
                # print(f"the extracted code is: \n{extracted_code}\n")
                # print("############################################\n")

                state["extracted_codes"].append(extracted_code)
                state["extraction_sources"].append(extraction_source)

                next_action, current_reward, validation_success = (
                    await self._env_response_async(
                        generated_code=extracted_code,
                        gt_code=state["source_code"],
                        dependency=state["function_dependency"],
                        had_non_code_text=had_non_code_text,
                    )
                )
                state["step_rewards"].append(current_reward)
                state["turn_count"] += 1

                total_prev_len = (
                    len(state["anchor_prompt_ids"]) + len(state["completion_ids"])
                )
                current_prompt_len = len(llm_response.prompt_token_ids)
                env_response_len = current_prompt_len - total_prev_len
                if env_response_len < 0:
                    raise ValueError(
                        f"Current prompt length ({current_prompt_len}) is shorter than "
                        f"the previously recorded trajectory length ({total_prev_len})."
                    )

                new_completion_len = len(llm_response.outputs[0].token_ids)

                state["completion_mask"].extend([self.env_mask] * env_response_len)
                state["completion_mask"].extend([1] * new_completion_len)

                state["completion_ids"].extend(
                    llm_response.prompt_token_ids[total_prev_len:]
                )
                state["completion_ids"].extend(llm_response.outputs[0].token_ids)

                total_context_tokens = (
                    len(state["anchor_prompt_ids"]) + len(state["completion_ids"])
                )
                reached_turn_limit = state["turn_count"] >= max_turns
                reached_token_limit = total_context_tokens >= self.mt_max_tokens

                if validation_success:
                    state["completed"] = True
                    state["stop_reason"] = "validation_success"
                elif reached_turn_limit:
                    state["completed"] = True
                    state["stop_reason"] = "max_turns"
                elif reached_token_limit:
                    state["completed"] = True
                    state["stop_reason"] = "context_token_limit"
                else:
                    # Append feedback only when another refinement turn will run.
                    state["messages"].append(next_action)

                if total_context_tokens > self.mt_max_tokens:
                    allowed_completion_len = max(
                        0, self.mt_max_tokens - len(state["anchor_prompt_ids"])
                    )
                    state["completion_ids"] = state["completion_ids"][
                        :allowed_completion_len
                    ]
                    state["completion_mask"] = state["completion_mask"][
                        :allowed_completion_len
                    ]

                if len(state["completion_mask"]) != len(state["completion_ids"]):
                    logger.error(
                        f"[state {j}] completion_mask length {len(state['completion_mask'])} "
                        f"!= completion_ids length {len(state['completion_ids'])}"
                    )
                    raise ValueError(
                        f"Completion mask and completion ids are not the same length for state {j}"
                    )

            except ValueError:
                raise
            except Exception as e:
                logger.error(
                    f"[state {j}] unexpected error in state update: {e}, "
                    "marking as completed."
                )
                state["completed"] = True
                state["stop_reason"] = "state_update_error"

            return j, state

    def _chat_via_vllm_client(
        self,
        llm,                        # VLLMClient
        messages_batch: List[List[Dict[str, Any]]],
        sampling_params: Union[SamplingParams, Dict[str, Any]],
        sampling_params_obj: SamplingParams,
    ) -> List[Any]:
        top_level_keys = (
            "n", "repetition_penalty", "temperature", "top_p",
            "top_k", "min_p", "max_tokens", "truncate_prompt_tokens",
        )
        gen_keys = ("stop", "skip_special_tokens", "spaces_between_special_tokens")

        if isinstance(sampling_params, dict):
            src = dict(sampling_params)
        else:
            src = {}
            for k in top_level_keys + gen_keys:
                if hasattr(sampling_params_obj, k):
                    v = getattr(sampling_params_obj, k)
                    if v is not None:
                        src[k] = v

        chat_kwargs = {k: v for k, v in src.items() if k in top_level_keys and v is not None}
        gen_kwargs = {k: v for k, v in src.items() if k in gen_keys and v is not None}
        if gen_kwargs:
            chat_kwargs["generation_kwargs"] = gen_kwargs

        raw_resp = llm.chat(messages_batch, **chat_kwargs)
        prompt_ids_batch = raw_resp["prompt_ids"]
        completion_ids_batch = raw_resp["completion_ids"]

        n = int(chat_kwargs.get("n", 1) or 1)
        if len(completion_ids_batch) != len(prompt_ids_batch) * n:
            raise RuntimeError(
                f"VLLMClient.chat Returned count mismatch: "
                f"completion={len(completion_ids_batch)}, "
                f"prompt={len(prompt_ids_batch)}, n={n}"
            )

        fake_responses: List[_FakeRequestOutput] = []
        for i, prompt_ids in enumerate(prompt_ids_batch):
            group = completion_ids_batch[i * n : (i + 1) * n]
            comps = []
            for cids in group:
                text = self.ctokenizer.decode(cids, skip_special_tokens=False)
                comps.append(_Completion(text=text, token_ids=list(cids)))
            fake_responses.append(
                _FakeRequestOutput(prompt_token_ids=list(prompt_ids), outputs=comps)
            )
        return fake_responses

    def step_all_trace(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: Union[SamplingParams, Dict[str, Any]],
        max_turns: int,
    ) -> List[Dict[str, Any]]:

        local_rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[rank{local_rank}] step_all_trace called, live_states={sum(1 for s in states if not s['completed'])}")

        sampling_params_obj = (
            sampling_params
            if isinstance(sampling_params, SamplingParams)
            else SamplingParams(**sampling_params)
        )

        live_indices = []
        for i, state in enumerate(states):
            if state["completed"]:
                continue
            if state["turn_count"] >= max_turns:
                state["completed"] = True
                state["stop_reason"] = "max_turns"
                continue
            live_indices.append(i)

        messages_to_step = []
        valid_live_indices = []

        # Reserve room for the next generated response. max_tokens is a
        # per-turn generation limit, not a multi-turn stopping threshold.
        per_turn_output_budget = int(getattr(sampling_params_obj, "max_tokens", 0) or 0)
        max_prompt_tokens = max(1, self.mt_max_tokens - per_turn_output_budget)

        for i in live_indices:
            msgs = states[i]["messages"]
            token_len = count_tokens_simple(self.ctokenizer, msgs)

            if token_len >= max_prompt_tokens:
                logger.warning(
                    f"[state {i}] Prompt is too long ({token_len} tokens); "
                    "stopping before the next generation."
                )
                states[i]["completed"] = True
                states[i]["stop_reason"] = "context_token_limit"
                continue

            messages_to_step.append(msgs)
            valid_live_indices.append(i)

        live_indices = valid_live_indices
        if not messages_to_step:
            return states

        messages_to_step = [states[i]["messages"] for i in live_indices]
        from ..extras.vllm_client import VLLMClient  
        if isinstance(llm, VLLMClient):
            llm_responses = self._chat_via_vllm_client(
                llm=llm,
                messages_batch=messages_to_step,
                sampling_params=sampling_params,           
                sampling_params_obj=sampling_params_obj,  
            )
        else:
            llm_responses = llm.chat(
                messages_to_step,
                sampling_params=sampling_params_obj,
                use_tqdm=False,
            )
        if len(llm_responses) != len(live_indices):
            raise RuntimeError(
                f"vLLM returned {len(llm_responses)} responses for "
                f"{len(live_indices)} live states."
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
                    max_turns=max_turns,
                )
                for i, j in enumerate(live_indices)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(asyncio.run, _run_all()).result()
        except RuntimeError:
            results = asyncio.run(_run_all())

        for j, item in zip(live_indices, results):
            if isinstance(item, Exception):
                logger.error(
                    f"[step_all_trace] state {j} raised an unhandled exception: {item}"
                )
                states[j]["completed"] = True
                states[j]["stop_reason"] = "state_update_error"
                continue
            returned_j, state = item
            states[returned_j] = state

        return states

    def generate(
        self,
        prompts: List[Dict[str, Any]],
        llm: LLM,
        max_turns: Optional[int] = None,
        mode: str = "train",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        effective_max_turns = self.max_steps if max_turns is None else int(max_turns)
        if max_turns is not None and int(max_turns) != int(self.max_steps):
            logger.warning(
                "generate(max_turns=%s) overrides env.max_steps=%s. "
                "Use one authoritative setting to avoid configuration confusion.",
                max_turns,
                self.max_steps,
            )
        if effective_max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")

        states = [
            {
                "messages": copy.deepcopy(m["prompt"]),
                "anchor_prompt_messages": len(m["prompt"]),
                "anchor_prompt_ids": [],
                "completed": False,
                "stop_reason": None,
                "turn_count": 0,
                "completion_ids": [],
                "completion_mask": [],
                "function_dependency": m["dependency"],
                "step_rewards": [],
                "source_code": m["source"],
                "raw_responses": [],
                "extracted_codes": [],
                "extraction_sources": [],
            }
            for m in prompts
        ]

        # One loop iteration corresponds to exactly one assistant generation
        # for each still-live state.
        for _ in range(effective_max_turns):
            if all(s["completed"] for s in states):
                break
            states = self.step_all_trace(
                states,
                llm,
                self.sampling_args,
                max_turns=effective_max_turns,
            )

        # Defensive finalization: every trajectory leaves with an explicit reason.
        for state in states:
            if not state["completed"]:
                state["completed"] = True
                state["stop_reason"] = "max_turns"

        completion_messages = [
            s["messages"][s["anchor_prompt_messages"]:] for s in states
        ]
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
            "raw_responses": [s["raw_responses"] for s in states],
            "extracted_codes": [s["extracted_codes"] for s in states],
            "turn_counts": [s["turn_count"] for s in states],
            "stop_reasons": [s["stop_reason"] for s in states],
        }

