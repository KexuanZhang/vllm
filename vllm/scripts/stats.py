#!/usr/bin/env python3
"""
Analyze MoE expert usage statistics from routing log.

Usage:
    python analyze_expert_usage.py moe_routes.jsonl
"""

import argparse
import json
import math
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="Path to JSONL log file")
    args = parser.parse_args()

    # Load data
    expert_counts = Counter()
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("type") == "route":
                for eid in record.get("topk_ids", []):
                    expert_counts[eid] += 1

    num_experts = max(expert_counts.keys()) + 1
    total = sum(expert_counts.values())

    # a.Top-3 experts
    print("Top-3 Experts")
    for rank, (eid, cnt) in enumerate(expert_counts.most_common(3), 1):
        print(f"  #{rank}: Expert {eid} -> {cnt:,} selections ({100*cnt/total:.1f}%) \n")

    # b.Normalized distribution
    print("Normalized Distribution")
    for eid in range(num_experts):
        cnt = expert_counts.get(eid, 0)
        pct = 100 * cnt / total
        bar = "#" * int(pct * 2)
        print(f"  Expert {eid:2d}: {pct:5.2f}% {bar}\n")

    # c.Entropy
    print("Load Balance Metric")
    entropy = 0.0
    for eid in range(num_experts):
        p = expert_counts.get(eid, 0) / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    max_entropy = math.log2(num_experts)
    normalized_entropy = entropy / max_entropy

    print(f"  Entropy: {entropy:.3f} bits (max: {max_entropy:.3f})")
    print(f"  Normalized entropy: {normalized_entropy:.3f}")


if __name__ == "__main__":
    main()
