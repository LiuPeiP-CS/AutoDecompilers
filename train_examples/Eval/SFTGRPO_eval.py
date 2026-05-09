import os
import json
import argparse
import logging
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        if prompt is not None and source is not None and dependency is not None:
            samples.append({
                "prompt":     prompt,
                "source":     source,
                "dependency": dependency,
                "orig_data":  orig_data,
            })
    logger.info(f"Loaded {len(samples)} eval samples from {data_path}")
    return samples


def load_model(model_dir: str, tensor_parallel_size: int = 1,
               gpu_memory_utilization: float = 0.85,
               batch_size: int=8,
               max_model_len: int = None) -> LLM:
    logger.info(f"Loading model from {model_dir} ...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_num_seqs=batch_size,
        max_model_len=max_model_len,  # 新增
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    logger.info("Model loaded.")
    return llm


def run_singleturn_eval(
    samples:    List[Dict[str, Any]],
    llm:        LLM,
    sampling_params: SamplingParams,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:

    results = []
    total   = len(samples)

    for batch_start in range(0, total, batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        logger.info(
            f"Processing batch {batch_start // batch_size + 1} / "
            f"{(total + batch_size - 1) // batch_size}  "
            f"(samples {batch_start}~{min(batch_start + batch_size, total) - 1})"
        )

        # 每条样本的 prompt 是 List[Dict]（role/content 格式），直接批量传给 llm.chat()
        messages_batch = [s["prompt"] for s in batch]
        outputs = llm.chat(messages_batch, sampling_params=sampling_params, use_tqdm=False)

        for i, (sample, output) in enumerate(zip(batch, outputs)):
            final_code = output.outputs[0].text.strip()  # 单次生成，取第 0 个

            results.append({
                **(sample["orig_data"] or {}),
                "source":           sample["source"],
                "c_func_decompile": final_code,
                "dependencies":     sample["dependency"],
            })

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single Turn Eval Inference")

    parser.add_argument("--model_dir",              required=True)
    parser.add_argument("--data_path",              required=True)
    parser.add_argument("--output_dir",             help="the eval results are in this dir")
    parser.add_argument("--tensor_parallel_size",   type=int,   default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--batch_size",             type=int,   default=8)
    parser.add_argument("--temperature",            type=float, default=0.0)
    parser.add_argument("--top_p",                  type=float, default=1.0)
    parser.add_argument("--max_tokens",             type=int,   default=8192) # response
    parser.add_argument("--dataset",                type=str, default="exebench")  # 或者 humaneval
    parser.add_argument("--max_model_len",          type=int, default=16384) # prompt+response
    return parser.parse_args()

def main():

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[tokenizer.eos_token],
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    llm     = load_model(args.model_dir, args.tensor_parallel_size,
                         args.gpu_memory_utilization, args.batch_size,
                         args.max_model_len)
    samples = prepare_eval_samples(args.data_path)

    if not samples:
        logger.error("数据集为空，退出。")
        return

    results = run_singleturn_eval(
        samples=samples,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
    )


    # eval_model = os.path.basename(os.path.normpath(args.model_dir))
    # output_path = args.output_dir + eval_model + "_eval_" + args.dataset +"_results.jsonl"
    # save_results(results, output_path)
    eval_model = os.path.basename(os.path.normpath(args.model_dir))
    # output_path = args.output_dir + eval_model + '_max_turns_' + str(args.max_turns) + "_eval_" + args.dataset +"_results.jsonl"
    filename = f"{eval_model}_eval_{args.dataset}_results.jsonl"

    output_path = os.path.join(args.output_dir, filename)
    save_results(results, output_path)


if __name__ == "__main__":
    main()
