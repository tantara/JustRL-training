"""
Lightweight rule-based verifier from DAPO for training.

This implements "binary outcome rewards" as described in the JustRL paper:
- Reward is either 1.0 (correct) or 0.0 (incorrect) - no partial credits
- Uses lightweight string matching - no SymPy for computational efficiency
- Extracts answers from \\boxed{} commands
- Normalizes answers for comparison using simple string operations

Key difference from evaluation verifier:
- Training: Fast, string-only matching (this file)
- Evaluation: Robust, uses SymPy fallback (evals/utils.py)

Reference: DAPO [Yu et al., 2025] - lightweight verifier without SymPy
"""
import re
from typing import Optional


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        idx = solution.rfind("\\fbox")
        if idx < 0:
            return None
    
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(solution):
        if solution[i] == "{":
            num_left_braces_open += 1
        if solution[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        return None
    
    boxed_str = solution[idx:right_brace_idx + 1]
    # Remove \boxed{ and }
    if boxed_str.startswith("\\boxed{"):
        return boxed_str[7:-1]
    elif boxed_str.startswith("\\fbox{"):
        return boxed_str[6:-1]
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for string matching (lightweight, no SymPy)"""
    if not answer:
        return ""
    
    # Remove whitespace
    answer = answer.strip()
    
    # Remove LaTeX commands that don't affect value
    answer = re.sub(r"\\text\{([^}]+)\}", r"\1", answer)
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\\!", "", answer)
    answer = re.sub(r"\\%", "%", answer)
    answer = re.sub(r"\\$", "$", answer)
    
    # Remove dollar signs and percentage
    answer = answer.replace("$", "").replace("%", "")
    
    # Normalize fractions
    answer = re.sub(r"\\tfrac|\\dfrac", "\\frac", answer)
    
    # Remove spaces
    answer = answer.replace(" ", "")
    
    # Lowercase for text answers
    answer = answer.lower()
    
    return answer


def grade_answer_binary(given_answer: str, ground_truth: str) -> bool:
    """
    Binary outcome reward: returns True if answer matches, False otherwise.
    
    This is the lightweight DAPO verifier without SymPy, as described in JustRL paper.
    The reward signal is binary (1.0 or 0.0) - no partial credits or continuous values.
    
    Process:
    1. Extract answers from \\boxed{} commands
    2. Normalize both answers using string operations (no SymPy)
    3. Compare normalized strings
    4. Return True (reward=1.0) if match, False (reward=0.0) otherwise
    
    Args:
        given_answer: Model's response (should contain \\boxed{answer})
        ground_truth: Correct answer (may or may not be boxed)
        
    Returns:
        bool: True if answer is correct (reward=1.0), False otherwise (reward=0.0)
    """
    if not ground_truth:
        return False
    
    # Extract boxed answers
    if "\\boxed" in ground_truth:
        ground_truth = extract_boxed_answer(ground_truth)
    
    given_answer = extract_boxed_answer(given_answer)
    
    if given_answer is None:
        return False
    
    # Normalize both answers
    gt_norm = normalize_answer(ground_truth)
    given_norm = normalize_answer(given_answer)
    
    # Simple string matching
    return gt_norm == given_norm

