"""
Evaluate a trained model on ARC-AGI tasks.

Usage:
    python eval_arc.py --out_dir=out-arc-arc-residual --split=val
    python eval_arc.py --out_dir=out-arc-arc-residual --split=eval --num_samples=3
    python eval_arc.py --out_dir=out-arc-arc-residual --constrained=True

Metrics:
    - Exact match: entire predicted grid matches ground truth
    - Cell accuracy: fraction of correctly predicted cells
    - pass@k: probability at least 1 of k samples is correct
"""
import os
import json
import argparse
import pickle
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GPTConfig, GPT


# =============================================================================
# Vocabulary (must match prepare.py)
# =============================================================================

PAD, BOS, EOS = 0, 1, 2
TRAIN, TEST, EX = 3, 4, 5
IN, OUT, ROW, SEP = 6, 7, 8, 9

def h_token(height: int) -> int:
    return 10 + height

def w_token(width: int) -> int:
    return 41 + width

def c_token(color: int) -> int:
    return 72 + color

def token_to_height(t: int) -> int:
    return t - 10

def token_to_width(t: int) -> int:
    return t - 41

def token_to_color(t: int) -> int:
    return t - 72

def is_height_token(t: int) -> bool:
    return 10 <= t <= 40

def is_width_token(t: int) -> bool:
    return 41 <= t <= 71

def is_color_token(t: int) -> bool:
    return 72 <= t <= 81

VOCAB_SIZE = 82

# Build itos for decoding
def build_itos():
    itos = {
        PAD: '<PAD>', BOS: '<BOS>', EOS: '<EOS>',
        TRAIN: '<TRAIN>', TEST: '<TEST>', EX: '<EX>',
        IN: 'IN', OUT: 'OUT', ROW: '<ROW>', SEP: '<SEP>'
    }
    for h in range(31):
        itos[h_token(h)] = f'H{h:02d}'
    for w in range(31):
        itos[w_token(w)] = f'W{w:02d}'
    for c in range(10):
        itos[c_token(c)] = f'C{c}'
    return itos

ITOS = build_itos()


# =============================================================================
# Grid Serialization (for building prompts)
# =============================================================================

def serialize_grid(grid: list[list[int]]) -> list[int]:
    """Serialize a 2D grid into tokens."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    tokens = [h_token(height), w_token(width)]
    for row in grid:
        tokens.append(ROW)
        for cell in row:
            tokens.append(c_token(cell))
    return tokens


def serialize_pair(pair: dict, include_output: bool = True) -> list[int]:
    """Serialize an input/output pair."""
    tokens = [IN]
    tokens.extend(serialize_grid(pair['input']))
    if include_output:
        tokens.append(OUT)
        tokens.extend(serialize_grid(pair['output']))
    return tokens


def build_prompt(task: dict, test_idx: int = 0) -> list[int]:
    """
    Build prompt for a task: all train demos as context, then test input.
    Returns tokens up to and including the final OUT token.
    """
    tokens = [BOS, TRAIN]
    
    # Add all training demonstrations as context
    for demo in task['train']:
        tokens.append(EX)
        tokens.extend(serialize_pair(demo, include_output=True))
    
    # Add test input
    tokens.extend([TEST, EX])
    tokens.extend(serialize_pair({'input': task['test'][test_idx]['input'], 'output': []}, include_output=False))
    tokens.append(OUT)
    
    return tokens


# =============================================================================
# Grid Parsing (from generated tokens)
# =============================================================================

def parse_grid_from_tokens(tokens: list[int]) -> tuple[list[list[int]] | None, str]:
    """
    Parse a grid from generated tokens.
    
    Expected format: H## W## <ROW> C# C# ... <ROW> C# C# ... [EOS]
    
    Returns:
        (grid, status) where grid is 2D list or None, status is 'ok' or error message
    """
    if len(tokens) < 3:
        return None, "too_short"
    
    # First two tokens should be height and width
    if not is_height_token(tokens[0]):
        return None, f"expected_height_got_{tokens[0]}"
    if not is_width_token(tokens[1]):
        return None, f"expected_width_got_{tokens[1]}"
    
    height = token_to_height(tokens[0])
    width = token_to_width(tokens[1])
    
    if height == 0 or width == 0:
        return None, "zero_dimension"
    
    # Parse the grid
    grid = []
    current_row = []
    idx = 2
    
    while idx < len(tokens):
        t = tokens[idx]
        
        if t == EOS:
            break
        elif t == ROW:
            if current_row:
                grid.append(current_row)
                current_row = []
        elif is_color_token(t):
            current_row.append(token_to_color(t))
        else:
            # Unexpected token, stop parsing
            break
        idx += 1
    
    # Don't forget the last row
    if current_row:
        grid.append(current_row)
    
    # Validate dimensions
    if len(grid) != height:
        return grid, f"height_mismatch_{len(grid)}_vs_{height}"
    
    for i, row in enumerate(grid):
        if len(row) != width:
            return grid, f"width_mismatch_row{i}_{len(row)}_vs_{width}"
    
    return grid, "ok"


# =============================================================================
# Constrained Generation
# =============================================================================

def generate_constrained(model, idx, max_new_tokens, temperature=1.0, device='cuda'):
    """
    Generate with constraints for ARC grid output.
    
    Constraints:
    1. First token must be H## (height)
    2. Second token must be W## (width)
    3. After that: only <ROW> and C0-C9 allowed
    4. Stop at EOS or when grid is complete
    """
    model.eval()
    
    # Token masks for constrained decoding
    height_mask = torch.zeros(VOCAB_SIZE, device=device)
    for h in range(1, 31):  # H01-H30 (skip H00)
        height_mask[h_token(h)] = 1
    
    width_mask = torch.zeros(VOCAB_SIZE, device=device)
    for w in range(1, 31):  # W01-W30
        width_mask[w_token(w)] = 1
    
    grid_mask = torch.zeros(VOCAB_SIZE, device=device)
    grid_mask[ROW] = 1
    for c in range(10):
        grid_mask[c_token(c)] = 1
    grid_mask[EOS] = 1
    
    generated = []
    expected_height = None
    expected_width = None
    cells_generated = 0
    rows_generated = 0
    
    for i in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        # Forward pass
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Apply constraints
        if i == 0:
            # First token: must be height
            logits = logits + (height_mask - 1) * 1e9
        elif i == 1:
            # Second token: must be width
            logits = logits + (width_mask - 1) * 1e9
        else:
            # Grid tokens only
            logits = logits + (grid_mask - 1) * 1e9
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        token = idx_next.item()
        
        # Track grid structure
        if i == 0:
            expected_height = token_to_height(token)
        elif i == 1:
            expected_width = token_to_width(token)
        elif token == ROW:
            rows_generated += 1
        elif is_color_token(token):
            cells_generated += 1
        
        generated.append(token)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop conditions
        if token == EOS:
            break
        
        # Stop if grid is complete (all rows and cells generated)
        if expected_height and expected_width:
            expected_cells = expected_height * expected_width
            if rows_generated >= expected_height and cells_generated >= expected_cells:
                generated.append(EOS)
                break
    
    return generated


def generate_unconstrained(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Standard generation without constraints."""
    model.eval()
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        if idx_next.item() == EOS:
            break
    
    return idx[0, -max_new_tokens:].tolist()


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(predicted: list[list[int]] | None, ground_truth: list[list[int]]) -> dict:
    """Compute evaluation metrics for a single prediction."""
    metrics = {
        'exact_match': False,
        'cell_accuracy': 0.0,
        'valid_grid': predicted is not None,
    }
    
    if predicted is None:
        return metrics
    
    gt_height = len(ground_truth)
    gt_width = len(ground_truth[0]) if gt_height > 0 else 0
    pred_height = len(predicted)
    
    # Exact match
    metrics['exact_match'] = (predicted == ground_truth)
    
    # Cell accuracy (compare overlapping region, handling ragged rows)
    if pred_height > 0:
        correct = 0
        total = gt_height * gt_width
        
        for i in range(min(pred_height, gt_height)):
            pred_row = predicted[i]
            gt_row = ground_truth[i]
            for j in range(min(len(pred_row), len(gt_row))):
                if pred_row[j] == gt_row[j]:
                    correct += 1
        
        metrics['cell_accuracy'] = correct / total if total > 0 else 0.0
    
    return metrics


def compute_pass_at_k(predictions: list[list[list[int]] | None], ground_truth: list[list[int]], k: int) -> float:
    """
    Compute pass@k: probability that at least one of k samples is correct.
    
    For a single task with n total samples where c are correct:
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    n = len(predictions)
    c = sum(1 for p in predictions if p == ground_truth)
    
    if n < k:
        return float(c > 0)
    
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0
    
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # = 1 - (n-c)! * (n-k)! / ((n-c-k)! * n!)
    # Use log to avoid overflow
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


# =============================================================================
# Main Evaluation
# =============================================================================

def load_tasks(directory: Path) -> list[dict]:
    """Load all JSON task files from a directory."""
    tasks = []
    json_files = sorted(directory.glob('*.json'))
    
    for fpath in json_files:
        with open(fpath, 'r') as f:
            task = json.load(f)
            task['_filename'] = fpath.stem
            tasks.append(task)
    
    return tasks


def evaluate_task(model, task: dict, num_samples: int, constrained: bool, 
                  temperature: float, device: str, max_tokens: int = 1000) -> dict:
    """Evaluate a single task."""
    results = {
        'task_id': task['_filename'],
        'test_results': []
    }
    
    # Evaluate each test case in the task
    for test_idx, test_case in enumerate(task['test']):
        ground_truth = test_case['output']
        prompt_tokens = build_prompt(task, test_idx)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, ...]
        
        predictions = []
        all_metrics = []
        
        for sample_idx in range(num_samples):
            # Generate
            with torch.no_grad():
                if constrained:
                    generated = generate_constrained(
                        model, prompt_tensor.clone(), max_tokens, 
                        temperature=temperature, device=device
                    )
                else:
                    generated = generate_unconstrained(
                        model, prompt_tensor.clone(), max_tokens,
                        temperature=temperature, top_k=50
                    )
            
            # Parse grid
            pred_grid, parse_status = parse_grid_from_tokens(generated)
            predictions.append(pred_grid)
            
            # Compute metrics
            metrics = compute_metrics(pred_grid, ground_truth)
            metrics['parse_status'] = parse_status
            metrics['sample_idx'] = sample_idx
            all_metrics.append(metrics)
        
        # Aggregate metrics for this test case
        test_result = {
            'test_idx': test_idx,
            'samples': all_metrics,
            'best_exact_match': any(m['exact_match'] for m in all_metrics),
            'best_cell_accuracy': max(m['cell_accuracy'] for m in all_metrics),
            'pass_at_1': compute_pass_at_k(predictions, ground_truth, 1) if num_samples >= 1 else 0,
            'pass_at_3': compute_pass_at_k(predictions, ground_truth, 3) if num_samples >= 3 else 0,
        }
        results['test_results'].append(test_result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI model")
    parser.add_argument("--out_dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "eval"],
                        help="Which split to evaluate (train/val from training, eval from evaluation)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per task (for pass@k)")
    parser.add_argument("--constrained", type=bool, default=True, help="Use constrained decoding")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--max_tasks", type=int, default=None, help="Max tasks to evaluate (for debugging)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = args.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Temperature handling (0 means greedy)
    temperature = args.temperature if args.temperature > 0 else 1e-8
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load tasks
    data_dir = Path("data/arc-agi")
    if args.split == "eval":
        task_dir = data_dir / "evaluation"
    else:
        task_dir = data_dir / "training"
    
    print(f"Loading tasks from {task_dir}...")
    tasks = load_tasks(task_dir)
    
    # For train/val split, we need to match the prepare.py split
    if args.split in ["train", "val"]:
        # Load the same split used during training
        np.random.seed(42)  # Same seed as prepare.py
        indices = np.random.permutation(len(tasks))
        val_size = int(len(tasks) * 0.1)
        
        if args.split == "val":
            tasks = [tasks[i] for i in indices[:val_size]]
        else:
            tasks = [tasks[i] for i in indices[val_size:]]
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    print(f"Evaluating {len(tasks)} tasks...")
    
    # Evaluate
    all_results = []
    exact_matches = 0
    total_tests = 0
    cell_accuracies = []
    
    for task in tqdm(tasks, desc="Evaluating"):
        result = evaluate_task(
            model, task, args.num_samples, args.constrained,
            temperature, device
        )
        all_results.append(result)
        
        for test_result in result['test_results']:
            total_tests += 1
            if test_result['best_exact_match']:
                exact_matches += 1
            cell_accuracies.append(test_result['best_cell_accuracy'])
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Split: {args.split}")
    print(f"Tasks: {len(tasks)}")
    print(f"Test cases: {total_tests}")
    print(f"Samples per test: {args.num_samples}")
    print(f"Constrained decoding: {args.constrained}")
    print(f"Temperature: {args.temperature}")
    print("-"*60)
    print(f"Exact Match: {exact_matches}/{total_tests} ({100*exact_matches/total_tests:.1f}%)")
    print(f"Cell Accuracy: {100*np.mean(cell_accuracies):.1f}%")
    
    if args.num_samples >= 3:
        pass_at_1 = np.mean([r['test_results'][0]['pass_at_1'] for r in all_results])
        pass_at_3 = np.mean([r['test_results'][0]['pass_at_3'] for r in all_results])
        print(f"pass@1: {100*pass_at_1:.1f}%")
        print(f"pass@3: {100*pass_at_3:.1f}%")
    
    print("="*60)
    
    # Save detailed results
    results_path = os.path.join(args.out_dir, f"eval_results_{args.split}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'summary': {
                'exact_match': exact_matches / total_tests,
                'cell_accuracy': float(np.mean(cell_accuracies)),
                'total_tasks': len(tasks),
                'total_tests': total_tests,
            },
            'results': all_results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
