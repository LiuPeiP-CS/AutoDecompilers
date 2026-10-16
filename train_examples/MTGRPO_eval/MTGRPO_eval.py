import os
import json
import argparse
import logging
from typing import List, Dict, Any
import copy
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from trl.envs.multiturn_env import MultiTurnEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  
            data = json.load(f)
        else:  
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def prepare_eval_samples(data_path: str) -> List[Dict[str, Any]]:
    
    raw = load_json_or_jsonl(data_path)
    samples = []
    for item in raw:
        prompt     = item.get("prompt")
        source     = item.get("source")
        dependency = item.get("dependency")
        orig_data  = item.get("orig_data")
        # if prompt and source and dependency is not None:
        if prompt is not None and source is not None and dependency is not None:
            samples.append({
                "prompt":     prompt,       
                "source":     source,       
                "dependency": dependency,   
                "orig_data":  orig_data,
            })
    logger.info(f"Loaded {len(samples)} eval samples from {data_path}")
    return samples


def load_model(model_dir: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85, batch_size: int=8, max_model_len: int = None) -> LLM:
    logger.info(f"Loading vLLM model from {model_dir} ...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size, # vllm_tensor_parallel_size
        gpu_memory_utilization=gpu_memory_utilization, # vllm_gpu_memory_utilization
        trust_remote_code=True,
        dtype="bfloat16", # torch.bfloat16
        max_num_seqs=batch_size,
        max_model_len=max_model_len,  
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    logger.info("Model loaded.")
    return llm

# Attention: During inference, the reference code is not visible, so we ignore the syntactic_reward and semantic_reward.
def _flatten_reward(reward_vec, weights=(0.5, 1.0)):
    if not reward_vec:
        return float("-inf")
    return sum(
        (reward_vec[i] if i < len(reward_vec) else 0.0) * w
        for i, w in enumerate(weights)
    )


def select_final_code(
    per_rollout_stop_reasons: List[str],
    per_rollout_extracted_codes: List[List[str]],
    per_rollout_trajectory_rewards: List[List[List[float]]],
    # per_rollout_raw_last_response: List[str],
) -> Dict[str, Any]:

    n = len(per_rollout_stop_reasons)

    
    for r_idx in range(n):
        if per_rollout_stop_reasons[r_idx] == "validation_success":
            codes = per_rollout_extracted_codes[r_idx]
            
            final_code = ""
            for code in reversed(codes):
                if code.strip():
                    final_code = code
                    break
            if final_code:
                return {
                    "final_code": final_code,
                    "selection_tier": "validation_success",
                    "rollout_idx": r_idx,
                    "turn_idx": len(codes) - 1,
                    "score": None,
                }
            
    best_code, best_r_idx, best_t_idx, best_score = "", -1, -1, float("-inf")
    for r_idx in range(n):
        codes = per_rollout_extracted_codes[r_idx]
        rewards = per_rollout_trajectory_rewards[r_idx]
        for t_idx, reward_vec in enumerate(rewards):
            if t_idx >= len(codes) or not codes[t_idx].strip():
                continue
            score = _flatten_reward(reward_vec)
            if score > best_score:
                best_score = score
                best_code = codes[t_idx]
                best_r_idx, best_t_idx = r_idx, t_idx

    if best_code:
        return {
            "final_code": best_code,
            "selection_tier": "best_reward",
            "rollout_idx": best_r_idx,
            "turn_idx": best_t_idx,
            "score": best_score,
        }

    
    fallback_code = next(
        (codes[0] for codes in per_rollout_extracted_codes if codes and codes[0].strip()),
        "",
    )
    return {
        "final_code": fallback_code,
        "selection_tier": "raw_fallback",
        "rollout_idx": -1,
        "turn_idx": -1,
        "score": None,
    }


def run_multiturn_eval(
    samples:      List[Dict[str, Any]],
    llm:          "LLM",
    env:          "MultiTurnEnv",
    sampling_params: "SamplingParams",
    max_turns:    int,
    n:            int = 4,          
    batch_size:   int = 8,          
) -> List[Dict[str, Any]]:
    
    results = []
    total = len(samples)

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        logger.info(
            f"[Best-of-{n}] batch {batch_start // batch_size + 1} / "
            f"{(total + batch_size - 1) // batch_size}  "
            f"(samples {batch_start}~{min(batch_start + batch_size, total) - 1})"
        )

        expanded_batch = []
        for orig_idx, sample in enumerate(batch):
            for rep_idx in range(n):
                replica = copy.deepcopy(sample)
                replica["_orig_idx"] = orig_idx
                replica["_rep_idx"] = rep_idx
                expanded_batch.append(replica)

        output = env.generate(
            prompts=expanded_batch,
            llm=llm,
            max_turns=max_turns,
            mode="eval",
        )

        assert len(output["stop_reasons"]) == len(expanded_batch), (
            f"generate() return length {len(output['stop_reasons'])} and input {len(expanded_batch)} are not matched"
        )

        
        grouped: Dict[int, List[int]] = defaultdict(list)
        for flat_idx, replica in enumerate(expanded_batch):
            grouped[replica["_orig_idx"]].append(flat_idx)

        for orig_idx, sample in enumerate(batch):
            flat_indices = grouped[orig_idx] 

            stop_reasons          = [output["stop_reasons"][i]         for i in flat_indices]
            extracted_codes_group = [output["extracted_codes"][i]      for i in flat_indices]
            reward_group          = [output["trajectory_rewards"][i]   for i in flat_indices]

            raw_last_responses = []
            for i in flat_indices:
                full_messages = expanded_batch[i]["prompt"] + output["messages"][i]
                first_response = ""
                for msg in full_messages:
                    if msg.get("role") == "assistant":
                        first_response = msg["content"].strip()
                        break
                raw_last_responses.append(first_response)

            selection = select_final_code(
                per_rollout_stop_reasons=stop_reasons,
                per_rollout_extracted_codes=extracted_codes_group,
                per_rollout_trajectory_rewards=reward_group,
                # per_rollout_raw_last_response=raw_last_responses,
            )

            
            chosen_flat_idx = (
                flat_indices[selection["rollout_idx"]]
                if selection["rollout_idx"] != -1
                else flat_indices[0]
            )
            chosen_full_messages = (
                expanded_batch[chosen_flat_idx]["prompt"] + output["messages"][chosen_flat_idx]
            )

            results.append({
                **sample["orig_data"],
                "source":            sample["source"],
                "messages":          chosen_full_messages,
                "c_func_decompile":  selection["final_code"],
                "dependencies":      sample["dependency"],
                "num_turns":         sum(
                    1 for m in chosen_full_messages if m.get("role") == "assistant"
                ),
                "best_of_n":         n,
                "selection_tier":    selection["selection_tier"],   
                "selection_rollout": selection["rollout_idx"],
                "selection_turn":    selection["turn_idx"],
                "selection_score":   selection["score"],
            })

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MTGRPO Eval Inference")

    
    parser.add_argument("--model_dir",  required=True,  help="the address saves checkpoint")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor_parallel_size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.93)

    
    parser.add_argument("--data_path",  default=None,   help="eval data file（JSON / JSONL）")
    parser.add_argument("--so_path",    default=None,   help="tree-sitter .so path")
    parser.add_argument("--output_dir", help="the eval results are in this dir")

    
    parser.add_argument("--max_turns",  type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p",      type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int,   default=4096) 

    
    parser.add_argument("--judge_c",        action="store_true", default=True)
    parser.add_argument("--re_exe_reward",  action="store_true", default=True)
    parser.add_argument("--use_apted",      action="store_true")
    parser.add_argument("--syntax_reward",  action="store_true", default=True)
    parser.add_argument("--semantic_reward",action="store_true", default=True)
    parser.add_argument("--max_model_len", type=int, default=16384)  
    parser.add_argument("--dataset", type=str, default="exebench")  

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    sampling_args = {
        "temperature": args.temperature,
        "top_p":       args.top_p,
        "max_tokens":  args.max_tokens,
        "stop":        [tokenizer.eos_token],
        "skip_special_tokens":          False,
        "spaces_between_special_tokens": False,
        "n": 1,
    }

    env = MultiTurnEnv(
        sampling_args=sampling_args,
        so_path=args.so_path,
        use_apted=args.use_apted,
        judge_c=args.judge_c,
        re_exe_reward=args.re_exe_reward,
        syntax_reward=args.syntax_reward,
        semantic_reward=args.semantic_reward,
        mode='eval',
        dataset=args.dataset,
        strict_code_only=False,  
    )

    llm = load_model(
        model_dir=args.model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
    )

    samples = prepare_eval_samples(args.data_path)
    if not samples:
        logger.error("the dataset is full, exit")
        return

    results = run_multiturn_eval(
        samples=samples,
        llm=llm,
        env=env,
        sampling_params=None,   
        max_turns=args.max_turns,
        batch_size=args.batch_size,
    )

    eval_model = os.path.basename(args.model_dir.rstrip('/'))
    filename = f"{eval_model}_max_turns_{args.max_turns}_eval_{args.dataset}_results.jsonl"

    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)

if __name__ == "__main__":
    main()
