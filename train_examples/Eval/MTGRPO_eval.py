import os
import json
import argparse
import logging
from typing import List, Dict, Any

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
    """自动加载 JSON 或 JSONL 文件，返回列表"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  # JSON 数组
            data = json.load(f)
        else:  # JSONL，每行一个 JSON 对象
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def prepare_eval_samples(data_path: str) -> List[Dict[str, Any]]:
    """返回包含 prompt / source / dependency 的样本列表。"""
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
                "prompt":     prompt,       # List[Dict]，与训练时相同格式
                "source":     source,       # ground-truth C 代码
                "dependency": dependency,   # 测试用例依赖信息
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
        max_model_len=max_model_len,  # 新增
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    logger.info("Model loaded.")
    return llm

def run_multiturn_eval(
    samples:      List[Dict[str, Any]],
    llm:          LLM,
    env:          MultiTurnEnv,
    sampling_params: SamplingParams,
    max_turns:    int,
    batch_size:   int = 8,
) -> List[Dict[str, Any]]:
    """
    对所有样本做 multi-turn rollout，返回每条样本的推理结果。
    """
    results = []
    total = len(samples)

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        logger.info(
            f"Processing batch {batch_start // batch_size + 1} / "
            f"{(total + batch_size - 1) // batch_size}  "
            f"(samples {batch_start}~{min(batch_start + batch_size, total) - 1})"
        )

        # env.generate 完全复用训练侧的 multi-turn rollout 逻辑
        # prompts 格式：[{"prompt": [...], "source": "...", "dependency": ...}, ...]
        output = env.generate(
            prompts=batch,
            llm=llm,
            max_turns=max_turns,
        )

        for i, sample in enumerate(batch):
            full_messages = sample["prompt"] + output["messages"][i]
            # 取最后一条 assistant 消息作为最终代码
            final_code = ""
            for msg in reversed(full_messages):
                if msg.get("role") == "assistant":
                    final_code = msg["content"].strip()
                    break

            # trajectory_rewards = output["trajectory_rewards"][i]  # List[List[float]]
            # turns = len(trajectory_rewards)

            results.append({
                **sample["orig_data"],
                # "prompt":             sample["prompt"],
                "source":             sample["source"],
                "messages":           full_messages, # complete reasoning trajectory
                "c_func_decompile":   final_code, # the decompilation code
                "dependencies":       sample['dependency'],
                "num_turns":          sum(1 for m in full_messages if m.get("role") == "assistant"),
            })

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            # messages 里含 dict，直接 dump
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MTGRPO Eval Inference")

    # 模型相关
    parser.add_argument("--model_dir",  required=True,  help="训练好的模型目录（或 checkpoint）")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM 张量并行数（默认 1）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # 数据相关
    parser.add_argument("--data_path",  default=None,   help="eval 数据文件（JSON / JSONL）")
    parser.add_argument("--so_path",    default=None,   help="tree-sitter .so 路径")
    parser.add_argument("--output_dir", help="the eval results are in this dir")

    # 推理相关
    parser.add_argument("--max_turns",  type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=8,   help="每批并发推理的样本数")
    parser.add_argument("--temperature", type=float, default=0.0, help="eval 时建议用 0（greedy）")
    parser.add_argument("--top_p",      type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int,   default=4096) # 1.3B和6.7B是4096，30B是65536

    # reward 开关（与训练侧保持一致）
    parser.add_argument("--judge_c",        action="store_true", default=True)
    parser.add_argument("--re_exe_reward",  action="store_true", default=True)
    parser.add_argument("--use_apted",      action="store_true")
    parser.add_argument("--syntax_reward",  action="store_true", default=True)
    parser.add_argument("--semantic_reward",action="store_true", default=True)
    parser.add_argument("--max_model_len", type=int, default=16384)  # prompt+response
    parser.add_argument("--dataset", type=str, default="exebench")  # 或者 humaneval

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # ── 采样参数：eval 用 greedy（temperature=0）确保可复现 ──
    sampling_args = {
        "temperature": args.temperature,
        "top_p":       args.top_p,
        "max_tokens":  args.max_tokens,
        "stop":        [tokenizer.eos_token],
        "skip_special_tokens":          False,
        "spaces_between_special_tokens": False,
        "n": 1,
    }

    # ── 构建 env（复用训练侧完全一致的逻辑）──
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
        mt_max_tokens=args.max_model_len,
    )

    # ── 加载模型 ──
    llm = load_model(
        model_dir=args.model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
    )

    samples = prepare_eval_samples(args.data_path)
    if not samples:
        logger.error("数据集为空，退出。")
        return

    results = run_multiturn_eval(
        samples=samples,
        llm=llm,
        env=env,
        sampling_params=None,   # env 内部使用 self.sampling_args，此参数仅占位
        max_turns=args.max_turns,
        batch_size=args.batch_size,
    )

    # eval_model = os.path.basename(os.path.normpath(args.model_dir))
    eval_model = os.path.basename(args.model_dir.rstrip('/'))
    # output_path = args.output_dir + eval_model + '_max_turns_' + str(args.max_turns) + "_eval_" + args.dataset +"_results.jsonl"
    filename = f"{eval_model}_max_turns_{args.max_turns}_eval_{args.dataset}_results.jsonl"

    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)

if __name__ == "__main__":
    main()
