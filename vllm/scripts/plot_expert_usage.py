#!/usr/bin/env python3
"""
Plot MoE expert usage histogram from routing log.

Usage:
    python plot_expert_usage.py moe_routes.jsonl
"""

import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="Path to JSONL log file")
    parser.add_argument("--output", "-o", default="expert_hist.png")
    args = parser.parse_args()

    # Load data
    expert_counts = Counter()
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("type") == "route":
                for eid in record.get("topk_ids", []):
                    expert_counts[eid] += 1

    # Prepare data
    num_experts = max(expert_counts.keys()) + 1
    experts = list(range(num_experts))
    counts = [expert_counts.get(e, 0) for e in experts]

    # Plot
    plt.figure(figsize=(14, 5))
    plt.bar(experts, counts, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Expert ID')
    plt.ylabel('Selection Count')
    plt.title('MoE Expert Usage Distribution')
    
    # Show x-axis ticks - skip some if too many experts
    if num_experts <= 20:
        plt.xticks(experts)
    else:
        step = max(1, num_experts // 20)
        plt.xticks(range(0, num_experts, step))
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
