# JustRL Training

This directory contains the training code for JustRL, implementing the simple RL recipe described in the paper.

**Note**: This implementation uses ByteDance's [veRL framework](https://github.com/volcengine/verl) for RL training, as mentioned in the JustRL paper. veRL provides the default GRPO implementation used by JustRL.

## Overview

JustRL uses a minimal approach:
- **Single-stage training** with fixed hyperparameters
- **GRPO** (Group Relative Policy Optimization) algorithm
- **Binary outcome rewards** from lightweight DAPO verifier (no SymPy)
- **Max context length**: 16K tokens
- **Simple prompt suffix**: "Please reason step by step, and put your final answer within \boxed{}."
- **Clip higher** technique for stability

## Files

- `train.py`: Main training script implementing GRPO
- `verifier.py`: Lightweight rule-based verifier for binary rewards
- `requirements.txt`: Python dependencies

## Installation

First, install veRL framework:

```bash
pip install verl
# or from source:
# pip install git+https://github.com/volcengine/verl.git
```

Then install other dependencies:

```bash
pip install -r requirements.txt
```

## Hardware Requirements

According to the JustRL paper:
- **GPUs**: 32 A800-80GB GPUs (4 nodes × 8 GPUs per node)
- **Training Duration**: ~15 days per model
- **Models**: DeepSeek-R1-Distill-Qwen-1.5B or OpenMath-Nemotron-1.5B

The script defaults to this configuration but can be adjusted for different hardware setups.

## Usage

### Step 1: Prepare Data

Convert the HuggingFace dataset to parquet format:

```bash
python prepare_data.py \
    --dataset_name BytedTsinghua-SIA/DAPO-Math-17k \
    --output_path ./data/dapo_math_17k.parquet
```

### Step 2: Train with veRL

Use the veRL-based training script:

```bash
python train_verl.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path ./data/dapo_math_17k.parquet \
    --max_steps 4000 \
    --output_dir ./justrl_outputs \
    --execute
```

**Using vLLM instead of SGLang (default):**

```bash
python train_verl.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path ./data/dapo_math_17k.parquet \
    --rollout_engine vllm \
    --max_steps 4000 \
    --output_dir ./justrl_outputs \
    --execute
```

### Training JustRL-Nemotron-1.5B

```bash
python train_verl.py \
    --model_name nvidia/OpenMath-Nemotron-1.5B \
    --data_path ./data/dapo_math_17k.parquet \
    --max_steps 4000 \
    --batch_size 32 \
    --num_rollouts 32 \
    --output_dir ./justrl_nemotron_outputs \
    --execute
```

### Alternative: Direct veRL Command

You can also run veRL directly:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['./data/dapo_math_17k.parquet']" \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=32 \
    trainer.reward_manager.name=dapo \
    trainer.max_steps=4000 \
    trainer.save_freq=500
```

## Hyperparameters

The default hyperparameters match the paper's configuration:

- Learning rate: 1e-6
- Batch size: 32
- Number of rollouts: 32 per problem
- Max steps: 4000+
- Max context length: 16K tokens
- Clip epsilon: 0.2
- Clip higher: True (only clip upper bound)
- Entropy coefficient: 0.01

## Key Features

### GRPO Algorithm

Group Relative Policy Optimization is similar to PPO but uses group-based relative rewards. For JustRL, we use binary outcome rewards (0 or 1) based on whether the answer is correct.

### Binary Rewards

The verifier uses a lightweight rule-based approach:
- Extracts answers from `\boxed{}` commands
- Normalizes answers for string matching
- Returns 1 if correct, 0 otherwise
- No SymPy dependency (faster, simpler)

### Clip Higher

JustRL uses "clip higher", a well-established practice for stability in long-horizon RL training. Instead of standard PPO clipping (both upper and lower bounds), "clip higher" only clips the upper bound:

- **Standard PPO**: `ratio` is clamped to `[1 - ε, 1 + ε]` (e.g., `[0.8, 1.2]` with ε=0.2)
- **Clip Higher**: `ratio` is clamped to `[0, 1 + ε]` (e.g., `[0, 1.2]` with ε=0.2)

This prevents reward collapse while maintaining exploration. In veRL, this is configured via:
- `clip_ratio_low=0.0` (no lower clipping)
- `clip_ratio_high=0.2` (only upper clipping)

The training script enables this by default with the `--clip_higher` flag.

## Training Process

1. **Data Loading**: Loads DAPO-Math-17k dataset
2. **Rollout Generation**: For each problem, generates multiple responses (default: 32)
3. **Reward Computation**: Uses DAPO verifier to compute binary rewards
4. **Policy Update**: Updates policy using GRPO with clip higher
5. **Checkpointing**: Saves checkpoints every 500 steps

## Notes

- The implementation is simplified for clarity. A production version would include:
  - Proper value function estimation
  - More efficient rollout generation
  - Distributed training support
  - Better memory management

- For the exact implementation used in the paper, refer to the veRL framework which implements GRPO.

## References

- Paper: [JustRL: Scaling a 1.5B LLM with a Simple RL Recipe](https://arxiv.org/abs/2512.16649)
- Dataset: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- veRL Framework: [HybridFlow](https://github.com/BytedanceAI/HybridFlow)

