"""
Prepare DAPO-Math-17k dataset for veRL training

Converts HuggingFace dataset to parquet format expected by veRL.

Note: The original DAPO-Math-17k dataset has a known duplication issue (~100x),
resulting in ~1.79M rows instead of ~17k unique problems. This script automatically
deduplicates based on the problem text to get the true ~17k unique problems.
"""

import argparse
import logging
from pathlib import Path
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare DAPO-Math-17k dataset for veRL")
    parser.add_argument("--dataset_name", type=str, default="BytedTsinghua-SIA/DAPO-Math-17k",
                       help="HuggingFace dataset name")
    parser.add_argument("--output_path", type=str, default="./data/dapo_math_17k.parquet",
                       help="Output parquet file path")
    parser.add_argument("--prompt_suffix", type=str,
                       default="Please reason step by step, and put your final answer within \\boxed{}.",
                       help="Prompt suffix to append")
    parser.add_argument("--no-deduplicate", action="store_false", dest="deduplicate", default=True,
                       help="Disable deduplication (deduplication is enabled by default to remove ~100x duplication)")
    parser.add_argument("--deduplicate_key", type=str, default="problem",
                       choices=["problem", "prompt"],
                       help="Key to use for deduplication: 'problem' (before adding suffix) or 'prompt' (after adding suffix)")
    
    args = parser.parse_args()
    
    logger.info(f"Loading dataset: {args.dataset_name}")
    
    try:
        dataset = load_dataset(args.dataset_name)
        
        # Handle different dataset structures
        if "train" in dataset:
            data = dataset["train"]
        elif "default" in dataset:
            data = dataset["default"]
        else:
            data = list(dataset.values())[0]
        
        logger.info(f"Loaded {len(data)} examples")
        
        # Convert to format expected by veRL
        formatted_data = []
        for item in data:
            problem = item.get("problem", item.get("question", ""))
            answer = item.get("answer", item.get("solution", ""))
            
            # Format prompt
            prompt = f"{problem}\n\n{args.prompt_suffix}"
            
            formatted_data.append({
                "prompt": prompt,
                "ground_truth": answer,
                "problem": problem,  # Keep original problem for deduplication
                "data_source": "math_dapo",  # veRL uses this to select reward function
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(formatted_data)
        
        # Deduplicate if requested (default: True, handles ~100x duplication bug)
        if args.deduplicate:
            original_count = len(df)
            if args.deduplicate_key == "problem":
                # Deduplicate based on problem text (before adding suffix)
                # This is preferred as it catches duplicates even if answers differ slightly
                df = df.drop_duplicates(subset=["problem"], keep="first")
            else:
                # Deduplicate based on prompt (after adding suffix)
                df = df.drop_duplicates(subset=["prompt"], keep="first")
            
            removed_count = original_count - len(df)
            logger.info(f"Deduplication: Removed {removed_count} duplicate rows ({removed_count/original_count*100:.1f}%)")
            logger.info(f"Unique examples: {len(df)} (down from {original_count})")
        
        # Remove 'problem' column from final output (veRL only needs 'prompt' and 'ground_truth')
        if "problem" in df.columns:
            df = df.drop(columns=["problem"])
        
        # Create output directory if needed
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} examples to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    main()

