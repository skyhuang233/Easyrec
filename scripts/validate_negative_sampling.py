"""
负采样验证脚本
功能：
1. 统计关系/随机负样本比例（期望≈1:3）
2. 检测假负样本（负样本不应在用户正集合中）
3. 输出分布报告与抽样示例

运行示例：
python scripts/validate_negative_sampling.py --data_path data/mooc
"""

import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


def parse_args():
    parser = argparse.ArgumentParser(description='Validate Negative Sampling Distribution')
    parser.add_argument('--data_path', type=str, default='data/mooc',
                        help='数据集根目录')
    parser.add_argument('--rel_window', type=int, default=5,
                        help='关系负采样窗口（与训练脚本保持一致）')
    parser.add_argument('--topk_next', type=int, default=20,
                        help='全局next-map的top-K（与训练脚本保持一致）')
    parser.add_argument('--sample_users', type=int, default=10,
                        help='抽样展示的用户数')
    return parser.parse_args()


def build_global_next_map_from_train(train_df, topk_per_item=20):
    """
    从训练原始数据构建全局next-map
    """
    g = defaultdict(Counter)
    for _, hist in train_df.sort_values(["student_id", "date"]).groupby("student_id"):
        seq = hist["courses"].tolist()
        for t in range(len(seq) - 1):
            g[seq[t]][seq[t + 1]] += 1

    next_map = {}
    for it, cnt in g.items():
        next_map[it] = set([x for x, _ in cnt.most_common(topk_per_item)])
    return next_map


def parse_seq(s):
    """
    解析逗号分隔的序列字符串，过滤padding（0）
    """
    try:
        arr = [int(x) for x in str(s).split(",")]
        return [x for x in arr if x != 0]
    except:
        return []


def validate_train_set(args, raw_dir, proc_dir):
    """
    验证训练集负采样分布
    """
    print("\n" + "=" * 60)
    print("验证训练集负采样分布")
    print("=" * 60)

    # 加载原始训练数据（用于构建next-map和用户历史）
    train_raw = pd.read_csv(os.path.join(proc_dir, 'train_data.csv'))
    train_tf = pd.read_csv(os.path.join(proc_dir, 'train_data_transformed.csv'))

    print(f"训练集样本数: {len(train_tf):,}")
    print(f"正样本数: {(train_tf['label'] == 1).sum():,}")
    print(f"负样本数: {(train_tf['label'] == 0).sum():,}")

    # 构建全局next-map
    print("\n构建全局next-map...")
    gnext = build_global_next_map_from_train(train_raw, topk_per_item=args.topk_next)

    # 构建用户历史集合
    print("构建用户历史集合...")
    user_hist = train_raw.groupby("student_id")["courses"].apply(
        lambda s: set(s.astype(int))
    ).to_dict()

    # 分析负样本
    print("\n分析负样本类型...")
    rel_cnt = 0
    rnd_cnt = 0
    bad_eq_cnt = 0
    per_user_neg = Counter()

    neg_df = train_tf[train_tf["label"] == 0]

    for _, r in neg_df.iterrows():
        uid = r["student_id"]
        cand = int(r["candidate"])
        his = parse_seq(r["courses"])

        # 检查假负样本（负样本在用户正集合中）
        if cand in user_hist.get(uid, set()):
            bad_eq_cnt += 1

        # 判断是否为关系负样本
        # 取历史最近rel_window步的全局next候选
        rel_pool = set()
        for h in his[-args.rel_window:]:
            rel_pool.update(gnext.get(h, set()))
        rel_pool -= user_hist.get(uid, set())

        if cand in rel_pool:
            rel_cnt += 1
        else:
            rnd_cnt += 1

        per_user_neg[uid] += 1

    total = rel_cnt + rnd_cnt

    # 输出统计结果
    print("\n" + "=" * 60)
    print("训练集负采样统计")
    print("=" * 60)
    print(f"负样本总数: {total:,}")
    print(f"关系感知负样本: {rel_cnt:,} ({rel_cnt / max(1, total) * 100:.2f}%)")
    print(f"随机负样本: {rnd_cnt:,} ({rnd_cnt / max(1, total) * 100:.2f}%)")
    print(f"期望比例: ~25% 关系 : ~75% 随机")
    print(f"\n假负样本数（负样本在用户正集合中）: {bad_eq_cnt}")
    if bad_eq_cnt > 0:
        print("⚠️  警告：存在假负样本！")
    else:
        print("✓ 无假负样本")

    # 每用户负样本数分布
    neg_counts = list(per_user_neg.values())
    print(f"\n每用户负样本数统计:")
    print(f"  平均: {np.mean(neg_counts):.2f}")
    print(f"  中位数: {np.median(neg_counts):.2f}")
    print(f"  最小: {np.min(neg_counts)}, 最大: {np.max(neg_counts)}")

    # 抽样展示
    print(f"\n抽样展示（前{args.sample_users}个用户）:")
    sample_users = list(per_user_neg.most_common(args.sample_users))
    for uid, cnt in sample_users:
        print(f"  用户 {uid}: {cnt} 个负样本")

    return {
        'total_neg': total,
        'relation': rel_cnt,
        'random': rnd_cnt,
        'bad': bad_eq_cnt,
        'relation_ratio': rel_cnt / max(1, total)
    }


def validate_test_set(args, raw_dir, proc_dir, set_name='test'):
    """
    验证测试/验证集负采样分布
    """
    print("\n" + "=" * 60)
    print(f"验证{set_name}集负采样分布")
    print("=" * 60)

    # 加载数据
    pos_file = f'{set_name}_pos.csv'
    ans_file = f'{set_name}_ans.csv'
    tf_file = f'{set_name}_data_transformed.csv'

    test_raw = pd.concat([
        pd.read_csv(os.path.join(proc_dir, pos_file)),
        pd.read_csv(os.path.join(proc_dir, ans_file))
    ])
    test_tf = pd.read_csv(os.path.join(proc_dir, tf_file))

    print(f"{set_name}集样本数: {len(test_tf):,}")
    print(f"正样本数: {(test_tf['label'] == 1).sum():,}")
    print(f"负样本数: {(test_tf['label'] == 0).sum():,}")

    # 构建全局next-map（基于测试集历史数据）
    print("\n构建全局next-map...")
    gnext = build_global_next_map_from_train(test_raw, topk_per_item=args.topk_next)

    # 构建用户历史集合
    print("构建用户历史集合...")
    user_hist = test_raw.groupby("student_id")["courses"].apply(
        lambda s: set(s.astype(int))
    ).to_dict()

    # 分析负样本
    print("\n分析负样本类型...")
    rel_cnt = 0
    rnd_cnt = 0
    bad_eq_cnt = 0
    per_user_neg = Counter()

    neg_df = test_tf[test_tf["label"] == 0]

    for _, r in neg_df.iterrows():
        uid = r["student_id"]
        cand = int(r["candidate"])
        his = parse_seq(r["courses"])

        # 检查假负样本
        if cand in user_hist.get(uid, set()):
            bad_eq_cnt += 1

        # 判断关系负样本
        rel_pool = set()
        for h in his[-args.rel_window:]:
            rel_pool.update(gnext.get(h, set()))
        rel_pool -= user_hist.get(uid, set())

        if cand in rel_pool:
            rel_cnt += 1
        else:
            rnd_cnt += 1

        per_user_neg[uid] += 1

    total = rel_cnt + rnd_cnt

    # 输出统计结果
    print("\n" + "=" * 60)
    print(f"{set_name}集负采样统计")
    print("=" * 60)
    print(f"负样本总数: {total:,}")
    print(f"关系感知负样本: {rel_cnt:,} ({rel_cnt / max(1, total) * 100:.2f}%)")
    print(f"随机负样本: {rnd_cnt:,} ({rnd_cnt / max(1, total) * 100:.2f}%)")
    print(f"期望比例: ~25% 关系 : ~75% 随机")
    print(f"\n假负样本数: {bad_eq_cnt}")
    if bad_eq_cnt > 0:
        print("⚠️  警告：存在假负样本！")
    else:
        print("✓ 无假负样本")

    # 每用户负样本数统计
    neg_counts = list(per_user_neg.values())
    print(f"\n每用户负样本数统计:")
    print(f"  平均: {np.mean(neg_counts):.2f}")
    print(f"  中位数: {np.median(neg_counts):.2f}")
    print(f"  最小: {np.min(neg_counts)}, 最大: {np.max(neg_counts)}")

    return {
        'total_neg': total,
        'relation': rel_cnt,
        'random': rnd_cnt,
        'bad': bad_eq_cnt,
        'relation_ratio': rel_cnt / max(1, total)
    }


def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print("负采样验证脚本")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"关系窗口: {args.rel_window}")
    print(f"Top-K next: {args.topk_next}")

    raw_dir = os.path.join(args.data_path, 'raw')
    proc_dir = os.path.join(args.data_path, 'processed')

    # 验证训练集
    train_stats = validate_train_set(args, raw_dir, proc_dir)

    # 验证验证集
    val_stats = validate_test_set(args, raw_dir, proc_dir, set_name='val')

    # 验证测试集
    test_stats = validate_test_set(args, raw_dir, proc_dir, set_name='test')

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    all_stats = [
        ('训练集', train_stats),
        ('验证集', val_stats),
        ('测试集', test_stats)
    ]

    print(f"\n{'数据集':<10} {'关系比例':<12} {'假负样本':<12} {'状态'}")
    print("-" * 60)
    for name, stats in all_stats:
        ratio = stats['relation_ratio'] * 100
        bad = stats['bad']
        status = "✓ 通过" if (20 <= ratio <= 30 and bad == 0) else "⚠️  警告"
        print(f"{name:<10} {ratio:>6.2f}%      {bad:>6}        {status}")

    print("\n期望标准:")
    print("  - 关系负样本比例: 20% ~ 30% (目标25%)")
    print("  - 假负样本数: 0")

    # 判断是否通过验证
    all_pass = all(
        20 <= stats['relation_ratio'] * 100 <= 30 and stats['bad'] == 0
        for _, stats in all_stats
    )

    if all_pass:
        print("\n" + "=" * 60)
        print("✓ 所有数据集验证通过！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  部分数据集验证未通过，请检查")
        print("=" * 60)


if __name__ == '__main__':
    main()