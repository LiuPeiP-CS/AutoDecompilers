import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from trl.rewards.fuse_rewards import OverallRewards
from typing import List, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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


def prepare_dataset_for_grpo(data_path):
    # 加载数据文件（可以是 JSON 或 JSONL 格式）
    data = load_json_or_jsonl(data_path)

    formatted_data = []

    for item in data:
        prompt = item.get('prompt', None)
        source = item.get('source', None)
        dependency = item.get('dependency', None)

        if prompt and source and dependency:
            formatted_data.append({
                'prompt': prompt,
                'source': source,
                'dependency': dependency
            })

    if len(formatted_data) > 0:
        return Dataset.from_list(formatted_data)
    else:
        return None


def create_grpo_reward_function(overall_rewards_computer, reward_weights, train_id):

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    cores_per_rank = (os.cpu_count() or 8) // local_world_size # cpu num for each process
    # MAX_WORKERS = max(1, cores_per_rank // 2)
    MAX_WORKERS = max(1, int(cores_per_rank * 0.75))
    SAMPLE_TIMEOUT = 300  # 单样本超时，要大于 _DEFAULT_CMD_TIMEOUT

    def grpo_reward_function(completions, **kwargs):
        source_codes = kwargs.get('source', None)
        dependencies = kwargs.get('dependency', None)

        if not source_codes or not dependencies or not completions:
            print("=========== There is wrong when computing rewards in GRPO ==========")
            return [0.0] * len(completions)

        predicted_codes = [completion[-1]["content"] for completion in completions]
        n = len(predicted_codes)
        cancel_events = [threading.Event() for _ in range(n)]

        def compute_one(idx, dep, pred, gt):
            try:
                reward, _ = overall_rewards_computer.get_code_rewards(
                    dep, pred, gt, train_id,
                    cancel_event=cancel_events[idx]
                )
                print(f"---------------------reward {reward}-------------------")
                assert len(reward) == len(reward_weights), \
                    "The length of rewards is not equal to the length of reward_weights"
                return idx, float(sum(r * w for r, w in zip(reward, reward_weights)))
            except AssertionError as e:
                print(f"[ERROR] reward assertion failed at sample {idx}: {e}")
                return idx, 0.0
            except Exception as e:
                print(f"[WARN] reward computation failed at sample {idx}: {e}")
                return idx, 0.0

        rewards = [0.0] * n
        n_workers = min(n, MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(compute_one, idx, dep, pred, gt): idx
                for idx, (dep, pred, gt) in enumerate(
                    zip(dependencies, predicted_codes, source_codes)
                )
            }
            try:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        _, val = future.result(timeout=SAMPLE_TIMEOUT)
                        rewards[idx] = val
                    except TimeoutError:
                        print(f"[WARN] sample {idx} exceeded {SAMPLE_TIMEOUT}s, killing...")
                        cancel_events[idx].set()  # 通知内层杀掉 subprocess
                        rewards[idx] = 0.0
                    except Exception as e:
                        print(f"[WARN] sample {idx} failed: {e}")
                        rewards[idx] = 0.0
            except TimeoutError:
                print(f"[WARN] Global reward timeout (300s), filling remaining samples with 0.0")
                for future, idx in futures.items():
                    if not future.done():
                        future.cancel()

        return rewards

    return grpo_reward_function

# ------------------------------
# Reward 函数（适配 GRPO）
# ------------------------------
def _create_grpo_reward_function(overall_rewards_computer, reward_weights):

    def grpo_reward_function(completions, **kwargs):
        rewards = []
        source_codes = kwargs.get('source', None)
        dependencies = kwargs.get('dependency', None)

        if not source_codes or not dependencies or not completions:
            print("=========== There is wrong when computing rewards in GRPO ==========")
            return

        predicted_codes = [completion[-1]["content"] for completion in completions]

        for dep, pred, gt in zip(dependencies, predicted_codes, source_codes):
            reward, _ = overall_rewards_computer.get_code_rewards(dep, pred, gt)
            print(f"this is reward {reward}")
            print(f"this is reward_weights {reward_weights}")
            assert len(reward) == len(
                reward_weights), "++++++++++ The length of rewards is not equal to the length of reward_weights +++++++++++"

            weighted_reward = sum(r * w for r, w in zip(reward, reward_weights))
            rewards.append(float(weighted_reward))

        return rewards

    return grpo_reward_function


# ------------------------------
# 训练函数
# ------------------------------
def train(train_fpath, save_dir, model_dir, overall_rewards_computer, reward_weights, batch_size, train_id):
    # 转换为 GRPO 格式（需要包含 'prompt' 字段）
    grpo_dataset = prepare_dataset_for_grpo(train_fpath)

    if grpo_dataset is None or len(grpo_dataset) == 0:
        print("❌ 错误: 数据集为空或格式不正确，请检查数据集是否包含 'prompt' 字段")
        return

    print(f"✅ Dataset prepared: {len(grpo_dataset)} samples")

    # GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=save_dir,

        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,

        learning_rate=1e-5,
        lr_scheduler_type='cosine',
        warmup_ratio=0.2,
        # warmup_steps=100,
        bf16=True,

        num_train_epochs=1,
        # max_steps=1,
        # generation_batch_size = 8,

        num_generations=4,  # GRPO: 每个 prompt 生成多个响应

        gradient_checkpointing=True,  # Enable gradient checkpointing
        use_vllm=True,
        vllm_tensor_parallel_size=1,
        vllm_mode="colocate",
        # vllm_mode="server",  # 使用 VLLM 服务器模式进行推理
        # vllm_server_host="127.0.0.1",  # 将 vllm_server_host 改为本地地址或服务器地址
        # vllm_device='auto',
        vllm_gpu_memory_utilization=0.3,  # Control GPU memory usage
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 50,
        },

        logging_steps=10,
        save_strategy='steps',
        save_steps=0.2,
        # save_total_limit=3,

        seed=1234,

        remove_unused_columns=False,
        # report_to="tensorboard",  # 启用 TensorBoard
        logging_dir=save_dir + "/training_logs",
        max_prompt_length=4096,
        max_completion_length = 4096,
        vllm_max_model_length=8192
    )

    reward_func = create_grpo_reward_function(overall_rewards_computer, reward_weights, train_id)

    trainer = GRPOTrainer(
        model=model_dir,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=reward_func,
    )
    # 开始训练
    print(">>> Starting TRL GRPO training...")
    trainer.train()
    print(f"✅ Model saved to {save_dir}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--abs_path', type=str, help="the absolute path from system to trl-main") # 系统到trl-main，即xxx/trl-main

    parser.add_argument('--llm_size', type=str, help="llm size for saving", choices=['1.3', '6.7', '30'])
    parser.add_argument('--llm_step', type=str, help="the step for finding the best for training rl")
    parser.add_argument('--sft_model_dir', type=str, help="directory of pretrained sft model") # 对应于llm_size和llm_step

    parser.add_argument('--judge_c',action="store_true")
    # parser.add_argument('--judge_c', type=bool, help="whether we use the judge_c as a reward", choices=[True, False], default=False)
    parser.add_argument('--judge_c_weight', type=float, default=0)

    parser.add_argument('--re_exe_reward', action="store_true")
    # parser.add_argument('--re_exe_reward', type=bool, help="whether we use the re_exe_reward as a reward", choices=[True, False], default=False)
    parser.add_argument('--re_exe_reward_weight', type=float, default=0)

    parser.add_argument("--use_apted", action="store_true")
    # parser.add_argument('--use_apted', type=bool, help="whether we use the the apted to compute the syntax_reward and semantic_reward", choices=[True, False], default=False)

    parser.add_argument('--syntax_reward', action="store_true")
    # type=bool, help="whether we use the syntax_reward as a reward", choices=[True, False], default=False)
    parser.add_argument('--syntax_reward_weight', type=float, default=0)

    parser.add_argument('--semantic_reward',action="store_true")
    # parser.add_argument('--semantic_reward', type=bool, help="whether we use the semantic_reward as a reward", choices=[True, False], default=False)
    parser.add_argument('--semantic_reward_weight', type=float, default=0)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_id', type=int, default=0)
    parser.add_argument('--mode', type=str, default="e2e")

    args = parser.parse_args()

    save_dir = os.path.join(args.abs_path, "trained_checkpoints", "GRPO", args.llm_size + '-' + args.llm_step + '-' + args.mode)

    reward_weights = []

    if args.use_apted:
        save_dir = save_dir + '-use_apted'

    if args.judge_c:
        # save_dir = save_dir + '-jc' + str(args.judge_c_weight) + '-'
        reward_weights.append(args.judge_c_weight)

    if args.re_exe_reward:
        # save_dir = save_dir + '-re_exe' + str(args.re_exe_reward_weight) + '-'
        reward_weights.append(args.re_exe_reward_weight)

    if args.syntax_reward:
        # save_dir = save_dir + '-syntax' + str(args.syntax_reward_weight) + '-'
        reward_weights.append(args.syntax_reward_weight)

    if args.semantic_reward:
        # save_dir = save_dir + '-semantic' + str(args.semantic_reward_weight) + '-'
        reward_weights.append(args.semantic_reward_weight)

    os.makedirs(save_dir, exist_ok=True)

    so_path = os.path.join(args.abs_path, 'rewards_tools/parser/my-languages.so')
    train_dataset = os.path.join(args.abs_path, 'datasets/train/rl4096train4end2end.jsonl')

    overall_rewards_computer = OverallRewards(
        so_path=so_path,
        use_apted=args.use_apted,
        judge_c=args.judge_c,
        re_exe_reward=args.re_exe_reward,
        syntax_reward=args.syntax_reward,
        semantic_reward=args.semantic_reward,
    )

    # the following code is used for debugging
    # print(args.use_apted)

    train(train_dataset, save_dir, args.sft_model_dir, overall_rewards_computer, reward_weights, args.batch_size, args.train_id)

if __name__ == '__main__':
    main()