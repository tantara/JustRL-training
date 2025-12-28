"""
JustRL Training Script using veRL Framework

This implements the simple RL recipe from JustRL using ByteDance's veRL framework:
- Single-stage training with fixed hyperparameters
- GRPO (Group Relative Policy Optimization) via veRL
- Binary outcome rewards from lightweight DAPO verifier
- Max context length: 16K tokens (1k prompt + 15k response)
- Simple prompt suffix

Hardware Configuration (from paper):
- 32 A800-80GB GPUs (4 nodes × 8 GPUs per node)
- Training duration: ~15 days

Based on: JustRL: Scaling a 1.5B LLM with a Simple RL Recipe
Paper: https://arxiv.org/abs/2512.16649
veRL: https://github.com/volcengine/verl
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="JustRL Training Script using veRL")
    
    # Model config
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Base model name")
    parser.add_argument("--max_length", type=int, default=16384,
                       help="Maximum context length")
    
    # Training config (matching JustRL paper)
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                       help="Learning rate (paper: 1e-6 constant)")
    parser.add_argument("--train_batch_size", type=int, default=256,
                       help="Global train batch size across all GPUs (paper: 256, with 32 GPUs = 8 per GPU). "
                            "WARNING: Reducing nodes while keeping this constant will increase per-GPU batch size and may cause OOM!")
    parser.add_argument("--max_steps", type=int, default=4000,
                       help="Maximum training steps")
    
    # GRPO config (matching JustRL paper)
    parser.add_argument("--num_rollouts", type=int, default=8,
                       help="Number of rollouts per problem (paper: 8)")
    parser.add_argument("--ppo_mini_batch_size", type=int, default=64,
                       help="PPO mini batch size (paper: 64)")
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=1,
                       help="PPO micro batch size per GPU (paper: 1)")
    parser.add_argument("--clip_ratio_low", type=float, default=0.2,
                       help="Lower clip ratio (paper: 0.2, range [0.8, 1.28])")
    parser.add_argument("--clip_ratio_high", type=float, default=0.28,
                       help="Upper clip ratio (paper: 0.28, range [0.8, 1.28])")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (paper: 1.0)")
    parser.add_argument("--rollout_engine", type=str, default="sglang",
                       choices=["vllm", "sglang"],
                       help="Rollout engine: vllm or sglang (default: sglang)")
    
    # Length config (matching JustRL paper)
    parser.add_argument("--max_prompt_length", type=int, default=1000,
                       help="Max prompt length (paper: 1k)")
    parser.add_argument("--max_response_length", type=int, default=15000,
                       help="Max response length (paper: 15k)")
    
    # Regularization (matching JustRL paper - both disabled)
    parser.add_argument("--use_kl_loss", action="store_true", default=False,
                       help="Use KL loss (paper: No)")
    parser.add_argument("--use_entropy_regularization", action="store_true", default=False,
                       help="Use entropy regularization (paper: No)")
    
    # Data config
    parser.add_argument("--dataset_name", type=str, default="BytedTsinghua-SIA/DAPO-Math-17k",
                       help="Dataset name (HuggingFace)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Local data path (parquet file)")
    parser.add_argument("--prompt_suffix", type=str,
                       default="Please reason step by step, and put your final answer within \\boxed{}.",
                       help="Prompt suffix")
    
    # Other config
    parser.add_argument("--output_dir", type=str, default="./justrl_outputs",
                       help="Output directory")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--wandb_project", type=str, default="justrl_training",
                       help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="WandB entity/team name (optional)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # veRL specific (matching JustRL paper)
    parser.add_argument("--n_gpus_per_node", type=int, default=8,
                       help="Number of GPUs per node (paper: 8 GPUs/node, 32 total)")
    parser.add_argument("--nnodes", type=int, default=4,
                       help="Number of nodes (paper: 4 nodes × 8 GPUs = 32 A800-80GB GPUs)")
    parser.add_argument("--gpu_type", type=str, default="A800-80GB",
                       help="GPU type (paper: A800-80GB)")
    parser.add_argument("--training_days", type=float, default=15.0,
                       help="Expected training duration in days (paper: ~15 days)")
    
    args = parser.parse_args()
    
    # Set WandB entity environment variable if provided
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    
    # Check if veRL is available
    try:
        import verl
        logger.info(f"veRL version: {verl.__version__ if hasattr(verl, '__version__') else 'unknown'}")
    except ImportError:
        logger.error("veRL is not installed. Please install it with: pip install verl")
        logger.info("Or install from source: pip install git+https://github.com/volcengine/verl.git")
        return
    
    # Prepare data path
    if args.data_path is None:
        # Use HuggingFace dataset - need to convert to parquet first
        logger.info(f"Using HuggingFace dataset: {args.dataset_name}")
        logger.info("Note: You may need to preprocess the dataset to parquet format first")
        logger.info("See examples/data_preprocess/math_dataset.py for reference")
        data_path = args.dataset_name  # veRL can handle HF datasets
    else:
        data_path = args.data_path
    
    # Build veRL command matching JustRL paper hyperparameters
    # veRL uses Hydra config system, so we pass configs as command-line arguments
    cmd_parts = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator=grpo",  # Paper: GRPO
        f"data.train_files=['{data_path}']",
        f"data.train_batch_size={args.train_batch_size}",  # Paper: 256 (global batch size, 8 per GPU with 32 GPUs)
        f"data.max_prompt_length={args.max_prompt_length}",  # Paper: 1k
        f"data.max_response_length={args.max_response_length}",  # Paper: 15k
        f"data.filter_overlong_prompts=True",
        f"actor_rollout_ref.model.path={args.model_name}",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",  # Paper: 1e-6 constant
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",  # Paper: 64
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",  # Paper: 1
        f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
    ]
    
    # KL Loss: Paper says "No"
    if args.use_kl_loss:
        cmd_parts.append(f"actor_rollout_ref.actor.use_kl_loss=True")
        cmd_parts.append(f"actor_rollout_ref.actor.kl_loss_coef=0.001")
    else:
        cmd_parts.append(f"actor_rollout_ref.actor.use_kl_loss=False")
    
    # Entropy Regularization: Paper says "No"
    if args.use_entropy_regularization:
        cmd_parts.append(f"actor_rollout_ref.actor.entropy_coeff=0.01")
    else:
        cmd_parts.append(f"actor_rollout_ref.actor.entropy_coeff=0.0")
    
    # Clip Ratio Range: Paper [0.8, 1.28] = [1-0.2, 1+0.28]
    # This means clip_ratio_low=0.2, clip_ratio_high=0.28
    cmd_parts.extend([
        f"actor_rollout_ref.actor.clip_ratio={args.clip_ratio_high}",  # Use high as base
        f"actor_rollout_ref.actor.clip_ratio_low={args.clip_ratio_low}",  # Paper: 0.2
        f"actor_rollout_ref.actor.clip_ratio_high={args.clip_ratio_high}",  # Paper: 0.28
    ])
    
    cmd_parts.extend([
        f"actor_rollout_ref.rollout.n={args.num_rollouts}",  # Paper: 8
        f"actor_rollout_ref.rollout.name={args.rollout_engine}",  # Use vLLM or SGLang for generation
        f"actor_rollout_ref.rollout.temperature={args.temperature}",  # Paper: 1.0
    ])
    
    # SGLang-specific configuration
    if args.rollout_engine == "sglang":
        cmd_parts.extend([
            f"actor_rollout_ref.rollout.mode=async",
            f"actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton",
            f"actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        ])
    
    cmd_parts.extend([
        f"algorithm.use_kl_in_reward=False",  # Binary rewards, no KL in reward
        f"trainer.critic_warmup=0",  # No critic warmup for GRPO
        f"trainer.logger='[\"console\",\"wandb\"]'",  # Use both console and wandb
        f"trainer.project_name={args.wandb_project}",
        f"trainer.experiment_name={args.model_name.split('/')[-1]}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.save_freq={args.save_steps}",
        f"trainer.total_epochs=1",  # Single-stage training
        f"trainer.max_steps={args.max_steps}",
    ])
    
    # Add reward manager config for DAPO verifier
    # DAPO reward manager uses binary outcome rewards (1.0 or 0.0)
    # with lightweight rule-based verifier (no SymPy) for fast training
    cmd_parts.extend([
        f"trainer.reward_manager.name=dapo",  # Use DAPO reward manager (binary rewards, no SymPy)
        f"trainer.reward_manager.num_examine=5",  # Print first 5 examples for debugging
    ])
    
    cmd = " ".join(cmd_parts)
    
    logger.info("=" * 80)
    logger.info("JustRL Training Configuration (Matching Paper)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Max steps: {args.max_steps}")
    total_gpus = args.n_gpus_per_node * args.nnodes
    batch_size_per_gpu = args.train_batch_size // total_gpus
    logger.info(f"Train batch size: {args.train_batch_size} global (paper: 256)")
    logger.info(f"  - Batch size per GPU: {batch_size_per_gpu} (with {total_gpus} GPUs)")
    if total_gpus < 32 and args.train_batch_size == 256:
        logger.warning(f"  ⚠️  WARNING: With {total_gpus} GPUs, each GPU handles {batch_size_per_gpu} samples.")
        logger.warning(f"     This is {batch_size_per_gpu / 8:.1f}x more than the paper's 8 samples per GPU.")
        logger.warning(f"     Consider reducing --train_batch_size to {batch_size_per_gpu * 32} to maintain 8 samples per GPU.")
    logger.info(f"PPO mini batch size: {args.ppo_mini_batch_size} (paper: 64)")
    logger.info(f"PPO micro batch size/GPU: {args.ppo_micro_batch_size_per_gpu} (paper: 1)")
    gradient_accumulation = args.ppo_mini_batch_size // args.ppo_micro_batch_size_per_gpu
    logger.info(f"  - Gradient accumulation: {gradient_accumulation} steps (auto-calculated by veRL)")
    logger.info(f"Max prompt length: {args.max_prompt_length} (paper: 1k)")
    logger.info(f"Max response length: {args.max_response_length} (paper: 15k)")
    logger.info(f"Learning rate: {args.learning_rate} (paper: 1e-6 constant)")
    logger.info(f"Number of rollouts: {args.num_rollouts} (paper: 8)")
    logger.info(f"Temperature: {args.temperature} (paper: 1.0)")
    logger.info(f"Rollout engine: {args.rollout_engine} (default: sglang)")
    if args.rollout_engine == "sglang":
        logger.info(f"  - SGLang mode: async")
        logger.info(f"  - SGLang attention backend: triton")
        logger.info(f"  - GPU memory utilization: 0.7")
    logger.info(f"Clip ratio range: [{1-args.clip_ratio_low:.2f}, {1+args.clip_ratio_high:.2f}] (paper: [0.8, 1.28])")
    logger.info(f"  - clip_ratio_low: {args.clip_ratio_low} (paper: 0.2)")
    logger.info(f"  - clip_ratio_high: {args.clip_ratio_high} (paper: 0.28)")
    logger.info(f"Use KL Loss: {args.use_kl_loss} (paper: No)")
    logger.info(f"Use Entropy Regularization: {args.use_entropy_regularization} (paper: No)")
    logger.info(f"GPU Configuration:")
    logger.info(f"  - Total GPUs: {total_gpus} (paper: 32 A800-80GB)")
    logger.info(f"  - GPUs per node: {args.n_gpus_per_node} (paper: 8)")
    logger.info(f"  - Number of nodes: {args.nnodes} (paper: 4)")
    logger.info(f"  - GPU type: {args.gpu_type} (paper: A800-80GB)")
    logger.info(f"  - Expected training time: ~{args.training_days} days (paper: ~15 days)")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"WandB project: {args.wandb_project}")
    if args.wandb_entity:
        logger.info(f"WandB entity: {args.wandb_entity}")
    logger.info("=" * 80)
    logger.info("\nTo run training, execute:")
    logger.info(cmd)
    logger.info("\nOr run this script with --execute flag to run directly")
    
    # Optionally execute
    if "--execute" in sys.argv:
        logger.info("Executing training...")
        os.system(cmd)
    else:
        logger.info("\nNote: Add --execute flag to run training directly")
        logger.info("Or copy the command above and run it manually")


if __name__ == "__main__":
    main()
