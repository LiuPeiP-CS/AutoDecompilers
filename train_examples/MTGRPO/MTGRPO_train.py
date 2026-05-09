import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
from trl.envs.multiturn_env import MultiTurnEnv
from trl import MTGRPOTrainer, GRPOConfig
from utils import get_model_and_tokenizer
from typing import List, Dict, Any, Optional
from datasets import Dataset
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

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


# ------------------------------
def train(train_fpath, save_dir, model_dir, mtgrpo_env, max_turns, only_final_turn_reward, progress_alpha, same_adv,
              reward_weights, training_args):
    # Convert to GRPO format (must include the 'prompt' field)
    grpo_dataset = prepare_dataset_for_grpo(train_fpath)

    if grpo_dataset is None or len(grpo_dataset) == 0:
        print("❌ 错误: 数据集为空或格式不正确，请检查数据集是否包含 'prompt' 字段")
        return

    print(f"✅ Dataset prepared: {len(grpo_dataset)} samples")

    model, tokenizer = get_model_and_tokenizer(model_dir)
    trainer = MTGRPOTrainer(
        model=model,
        env=mtgrpo_env,
        max_turns=max_turns,
        reward_weights=reward_weights,
        only_final_turn_reward=only_final_turn_reward,
        progress_alpha=progress_alpha,
        same_adv=same_adv,
        args=training_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )
    # 开始训练
    print(">>> Starting TRL Multi-Turns GRPO training...")
    trainer.train()
    print(f"✅ Model saved to {save_dir}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--abs_path', type=str, help="the absolute path from system to trl-main") # 系统到trl-main，即xxx/trl-main

    parser.add_argument('--llm_size', type=str, help="llm size for saving", choices=['0.5', '1.3', '6.7', '30'])
    parser.add_argument('--llm_step', type=str, help="the step for finding the best for training rl")
    parser.add_argument('--sft_model_dir', type=str, help="directory of pretrained sft model") # 对应于llm_size和llm_step

    parser.add_argument('--judge_c',action="store_true")
    # parser.add_argument('--judge_c', type=bool, help="whether we use the judge_c as a reward", choices=[True, False], default=True)
    parser.add_argument('--judge_c_weight', type=float, default=0.5)

    parser.add_argument('--re_exe_reward', action="store_true")
    # parser.add_argument('--re_exe_reward', type=bool, help="whether we use the re_exe_reward as a reward", choices=[True, False], default=True)
    parser.add_argument('--re_exe_reward_weight', type=float, default=1.0)

    parser.add_argument("--use_apted", action="store_true")
    # parser.add_argument('--use_apted', type=bool, help="whether we use the the apted to compute the syntax_reward and semantic_reward", choices=[True, False], default=False)

    parser.add_argument('--syntax_reward', action="store_true")
    # parser.add_argument('--syntax_reward', type=bool, help="whether we use the syntax_reward as a reward", choices=[True, False], default=True)
    parser.add_argument('--syntax_reward_weight', type=float, default=0.3)

    parser.add_argument('--semantic_reward',action="store_true")
    # parser.add_argument('--semantic_reward', type=bool, help="whether we use the semantic_reward as a reward", choices=[True, False], default=True)
    parser.add_argument('--semantic_reward_weight', type=float, default=0.2)

    #===================================================================================================#

    parser.add_argument('--max_turns', type=int, help="the max turns for conversation between llm and envs", choices=[2, 3, 4, 5])
    parser.add_argument('--only_final_turn_reward', action="store_true")
    # parser.add_argument('only_final_turn_reward', type=bool, help="whether we only use the final turn reward for computing the advantage", choices=[True, False], default=True)
    parser.add_argument('--progress_alpha', type=float, help="the coefficient for process rewards is used to assess its importance in contributing to the final result.", default=0.2) # progress_alpha ∈ [0.1, 0.3]
    parser.add_argument('--same_adv', action="store_true")
    # parser.add_argument('same_adv', type=bool, help="whether we assign the same value to all tokens when calculating the advantage", choices=[True, False], default=False) # same_adv

    # --------------------------------------------------------------------------------------------------#

    # parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
    parser.add_argument('--num_generations', type=int, default=4, help='Rollouts per prompt (default: 2), we can set 2 or 4')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='Per device train batch size (default: 12)')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--num_iterations', type=int, default=2, help='Number of iterations (default: 2)')
    parser.add_argument('--save_model_ratio', type=float, default=0.2, help='Frequency of saving the model (relative to total steps)')
    parser.add_argument('--beta', type=float, default=0.02, help='Beta parameter for KL divergence (default: 0.01)')
    # ===========================================================================================================================================================================
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1)
    parser.add_argument('--dec_mode', type=str, default='e2e')
    parser.add_argument('--dataset_path', type=str)
    # parser.add_argument('--max_tokens', type=int)
    parser.add_argument('--train_id', type=int, default=0)

    args = parser.parse_args()

    # ===========================================================================================================================================================================
    save_dir = os.path.join(args.abs_path, "trained_checkpoints", "MTGRPO", args.llm_size + '-' + args.llm_step + f'-{args.dec_mode}-')

    reward_weights = []

    if args.use_apted:
        save_dir = save_dir + '-use_apted'

    if args.judge_c:
        # ===========================================================================================================================================================================
        # save_dir = save_dir + '-judge_c'
        reward_weights.append(args.judge_c_weight)

    if args.re_exe_reward:
        # ===========================================================================================================================================================================
        # save_dir = save_dir + '-re_exe_reward'
        reward_weights.append(args.re_exe_reward_weight)

    if args.syntax_reward:
        # ===========================================================================================================================================================================
        # save_dir = save_dir + '-syntax_reward'
        reward_weights.append(args.syntax_reward_weight)

    if args.semantic_reward:
        # ===========================================================================================================================================================================
        # save_dir = save_dir + '-semantic_reward'
        reward_weights.append(args.semantic_reward_weight)

    save_dir = save_dir + f"max_turns_{args.max_turns}"

    if args.only_final_turn_reward:
        save_dir = save_dir + '-only_final_turn_reward'
    else:
        save_dir = save_dir + f'-progress_alpha_{args.progress_alpha}'

    if args.same_adv:
        save_dir = save_dir + '-same_adv'

    os.makedirs(save_dir, exist_ok=True)

    so_path = os.path.join(args.abs_path, 'rewards_tools/parser/my-languages.so')
    # ===========================================================================================================================================================================
    train_dataset = os.path.join(args.abs_path, 'datasets', 'train', args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir)
    sampling_kwargs={
        "stop": [tokenizer.eos_token],
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "n": 1,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        # "min_p": 0.05,
        "max_tokens": 4096,
        "repetition_penalty": 1.05,
    }

    # sampling_params = SamplingParams(**sampling_kwargs)
    sampling_params = sampling_kwargs
    mt_env = MultiTurnEnv(
        sampling_args=sampling_params, # 大模型的输入设置，即llm.chat()的参数
        so_path=so_path,
        use_apted=args.use_apted,
        judge_c=args.judge_c,
        re_exe_reward=args.re_exe_reward,
        syntax_reward=args.syntax_reward,
        semantic_reward=args.semantic_reward,
        max_concurrent=32,
        train_id=args.train_id,
    )

    training_args = GRPOConfig(
        # use_cpu=True,
        output_dir=save_dir, # 模型和检查点保存目录
        # run_name=run_name,
        learning_rate=args.learning_rate, # 初始学习率
        lr_scheduler_type="constant_with_warmup", # 带热身的恒定学习率
        warmup_steps=20, # 前20步线性增加学习率到设定值
        num_train_epochs=1,  # 训练轮数（GRPO通常单轮即可）
        bf16=True,
        adam_beta1=0.9, # Adam优化器的一阶动量衰减率
        adam_beta2=0.999, # Adam优化器的二阶动量衰减率
        max_grad_norm=0.1, # 梯度裁剪的最大范数，防止梯度爆炸
        num_iterations=args.num_iterations, # 每个数据点的优化次数
        beta=args.beta, # 0.04
        max_prompt_length=8192,
        max_completion_length=8192,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations, # 每个prompt产生几个回应
        gradient_accumulation_steps=args.grad_accum_steps, # args.grad_accum_steps
        gradient_checkpointing=True,
        save_strategy="steps", # 按步骤保存而非按epoch，或者"epoch"
        save_steps=args.save_model_ratio, # 如果是100就指每100步保存一次，也可以是[0,1]之间的小数，认为是total step的ratio
        save_only_model=True,
        use_vllm=True,
        # vllm_device=f"cuda:{args.num_gpus - 1}",
        # vllm_gpu_memory_utilization=0.7 if args.num_gpus > 1 else 0.3,
        logging_steps=10,
        log_on_each_node=False,
        log_completions=True,
        # report_to="wandb", # 使用Weights & Biases记录实验
        # max_steps=args.max_steps, # 最大训练步数
        # reward_weights=reward_weights,
        logging_dir=save_dir + "/training_logs",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,  # Control GPU memory usage
    )

    train(train_dataset, save_dir, args.sft_model_dir, mt_env, args.max_turns, args.only_final_turn_reward, args.progress_alpha, args.same_adv, reward_weights, training_args)

if __name__ == '__main__':
    main()

