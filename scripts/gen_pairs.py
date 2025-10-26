# file: scripts/gen_pairs.py

"""
生成训练/验证/测试对
支持:
1. 正负样本对 (user_nid, item_nid, label)
2. Contrastive learning 负样本 (K个)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='data/mooc/processed')
    parser.add_argument('--out-dir', type=str, default='data/mooc/pairs')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--num-negatives', type=int, default=4, help='每个正样本的负样本数 (contrastive)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_data(proc_dir):
    """加载预处理数据"""
    with open(proc_dir / 'id_maps.json', 'r') as f:
        id_maps = json.load(f)

    train_logs = pd.read_parquet(proc_dir / 'train_logs_early.parquet')
    test_logs = pd.read_parquet(proc_dir / 'test_logs_early.parquet')

    return id_maps, train_logs, test_logs


def build_interaction_dict(logs):
    """构建用户-课程交互字典"""
    user_items = defaultdict(set)

    for _, row in logs.iterrows():
        user_nid = row['user_nid']
        course_nid = row['course_nid']
        user_items[user_nid].add(course_nid)

    return user_items


def sample_negatives(user_nid, positive_items, all_items, num_neg):
    """为用户采样负样本"""
    negatives = []
    candidates = list(all_items - positive_items)

    if len(candidates) < num_neg:
        negatives = candidates
    else:
        negatives = random.sample(candidates, num_neg)

    return negatives


def generate_pairs(logs, user_items_dict, all_items, num_negatives, split_name):
    """生成正负样本对"""
    pairs = []
    contrastive_pairs = []

    # 按 enroll_id 聚合
    enroll_groups = logs.groupby('enroll_id').agg({
        'user_nid': 'first',
        'course_nid': 'first',
        'truth': 'first'  # dropout label
    }).reset_index()

    for _, row in tqdm(enroll_groups.iterrows(), total=len(enroll_groups), desc=f"Gen {split_name} pairs"):
        user_nid = row['user_nid']
        course_nid = row['course_nid']
        label = 1 - int(row['truth'])  # 1-dropout -> 1=完成, 0=dropout

        # 基础正样本
        pairs.append({
            'user_nid': int(user_nid),
            'item_nid': int(course_nid),
            'label': label
        })

        # Contrastive 负样本
        if label == 1:  # 只对正样本生成 contrastive
            positive_items = user_items_dict.get(user_nid, set())
            negatives = sample_negatives(user_nid, positive_items, all_items, num_negatives)

            contrastive_pairs.append({
                'user_nid': int(user_nid),
                'item_nid': int(course_nid),
                **{f'neg_{i + 1}': int(neg) for i, neg in enumerate(negatives)}
            })

    return pairs, contrastive_pairs


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    proc_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("[1/4] Loading data...")
    id_maps, train_logs, test_logs = load_data(proc_dir)

    all_items = set(range(id_maps['num_courses']))

    # 2. 构建交互字典
    print("[2/4] Building interaction dict...")
    train_user_items = build_interaction_dict(train_logs)
    test_user_items = build_interaction_dict(test_logs)

    # 3. 划分训练/验证集 (基于 train_logs)
    print("[3/4] Splitting train/val...")
    train_enrolls = train_logs['enroll_id'].unique()
    train_enrolls, val_enrolls = train_test_split(
        train_enrolls,
        test_size=args.val_ratio,
        random_state=args.seed
    )

    train_subset = train_logs[train_logs['enroll_id'].isin(train_enrolls)]
    val_subset = train_logs[train_logs['enroll_id'].isin(val_enrolls)]

    # 4. 生成 pairs
    print("[4/4] Generating pairs...")

    train_pairs, train_contrastive = generate_pairs(
        train_subset, train_user_items, all_items, args.num_negatives, 'train'
    )

    val_pairs, _ = generate_pairs(
        val_subset, train_user_items, all_items, 0, 'val'
    )

    test_pairs, _ = generate_pairs(
        test_logs, test_user_items, all_items, 0, 'test'
    )

    # 5. 保存
    print("[5/5] Saving pairs...")
    pd.DataFrame(train_pairs).to_csv(out_dir / 'train_pairs.csv', index=False)
    pd.DataFrame(val_pairs).to_csv(out_dir / 'val_pairs.csv', index=False)
    pd.DataFrame(test_pairs).to_csv(out_dir / 'test_pairs.csv', index=False)

    if train_contrastive:
        pd.DataFrame(train_contrastive).to_csv(out_dir / 'train_contrastive.csv', index=False)

    print(f"Done!")
    print(f"  - Train pairs: {len(train_pairs)}")
    print(f"  - Val pairs: {len(val_pairs)}")
    print(f"  - Test pairs: {len(test_pairs)}")
    print(f"  - Train contrastive: {len(train_contrastive)}")
    print(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()