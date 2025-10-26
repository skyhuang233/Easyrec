# file: scripts/mooc_prepare.py

"""
MOOC 数据预处理脚本
功能:
1. 解压 tracking logs
2. 读取 dropout prediction dataset
3. 构建 ID 映射 (user2nid, course2nid, enroll2id)
4. 过滤早期窗口行为 (前 30% 时长)
5. 保存为 parquet 格式
"""

import os
import json
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=str, default='data/mooc/raw',
                        help='原始数据目录')
    parser.add_argument('--out-dir', type=str, default='data/mooc/processed',
                        help='输出目录')
    parser.add_argument('--early-window-percent', type=float, default=30.0,
                        help='早期窗口百分比 (0-100)')
    parser.add_argument('--chunk-size', type=int, default=500000,
                        help='分块读取大小')
    return parser.parse_args()


def extract_tracking_logs(raw_dir, out_dir):
    """解压 tracking log tar.gz 文件"""
    tar_path = Path(raw_dir) / '20150801-20160801-activity.tar.gz'
    extract_to = Path(out_dir) / 'tracking_logs'

    if extract_to.exists():
        print(f"[INFO] Tracking logs already extracted to {extract_to}")
        return extract_to

    print(f"[INFO] Extracting {tar_path}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

    print(f"[INFO] Extraction complete: {extract_to}")
    return extract_to


def load_course_info(raw_dir):
    """加载课程信息"""
    course_path = Path(raw_dir) / 'course_info.csv'
    df = pd.read_csv(course_path)

    # 解析时间
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df['duration_days'] = (df['end'] - df['start']).dt.days

    return df


def load_dropout_data(raw_dir):
    """加载 dropout prediction 数据集"""
    train_log = pd.read_csv(Path(raw_dir) / 'train_log.csv')
    test_log = pd.read_csv(Path(raw_dir) / 'test_log.csv')
    train_truth = pd.read_csv(Path(raw_dir) / 'train_truth.csv')
    test_truth = pd.read_csv(Path(raw_dir) / 'test_truth.csv')

    # 合并 log 和 truth
    train_log = train_log.merge(train_truth, on='enroll_id', how='left')
    test_log = test_log.merge(test_truth, on='enroll_id', how='left')

    # 解析时间
    train_log['time'] = pd.to_datetime(train_log['time'], errors='coerce')
    test_log['time'] = pd.to_datetime(test_log['time'], errors='coerce')

    return train_log, test_log


def build_id_mappings(train_log, test_log, course_info):
    """构建 ID 映射"""
    # User mapping
    all_users = pd.concat([train_log['username'], test_log['username']]).unique()
    user2nid = {u: i for i, u in enumerate(sorted(all_users))}

    # Course mapping
    all_courses = pd.concat([
        train_log['course_id'],
        test_log['course_id'],
        course_info['course_id']
    ]).unique()
    course2nid = {c: i for i, c in enumerate(sorted(all_courses))}

    # Enroll mapping
    all_enrolls = pd.concat([train_log['enroll_id'], test_log['enroll_id']]).unique()
    enroll2id = {e: i for i, e in enumerate(sorted(all_enrolls))}

    id_maps = {
        'user2nid': user2nid,
        'course2nid': course2nid,
        'enroll2id': enroll2id,
        'num_users': len(user2nid),
        'num_courses': len(course2nid),
        'num_enrolls': len(enroll2id)
    }

    return id_maps


def filter_early_window(log_df, course_info, window_percent):
    """过滤早期窗口行为"""
    log_df = log_df.merge(
        course_info[['course_id', 'start', 'end', 'duration_days']],
        on='course_id',
        how='left'
    )

    # 计算每个行为相对课程开始的时间
    log_df['days_since_start'] = (log_df['time'] - log_df['start']).dt.days

    # 计算窗口阈值
    log_df['window_threshold'] = log_df['duration_days'] * (window_percent / 100.0)

    # 过滤
    early_df = log_df[log_df['days_since_start'] <= log_df['window_threshold']].copy()

    print(f"[INFO] Original logs: {len(log_df)}, Early window logs: {len(early_df)} "
          f"({len(early_df) / len(log_df) * 100:.2f}%)")

    return early_df


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解压 tracking logs (可选,如果需要使用)
    # extract_tracking_logs(raw_dir, out_dir)

    # 2. 加载数据
    print("[1/5] Loading data...")
    course_info = load_course_info(raw_dir)
    train_log, test_log = load_dropout_data(raw_dir)

    # 3. 构建 ID 映射
    print("[2/5] Building ID mappings...")
    id_maps = build_id_mappings(train_log, test_log, course_info)

    # 添加 nid 列
    train_log['user_nid'] = train_log['username'].map(id_maps['user2nid'])
    train_log['course_nid'] = train_log['course_id'].map(id_maps['course2nid'])
    test_log['user_nid'] = test_log['username'].map(id_maps['user2nid'])
    test_log['course_nid'] = test_log['course_id'].map(id_maps['course2nid'])

    # 4. 过滤早期窗口
    print(f"[3/5] Filtering early window ({args.early_window_percent}%)...")
    train_early = filter_early_window(train_log, course_info, args.early_window_percent)
    test_early = filter_early_window(test_log, course_info, args.early_window_percent)

    # 5. 保存
    print("[4/5] Saving processed data...")
    train_early.to_parquet(out_dir / 'train_logs_early.parquet', index=False)
    test_early.to_parquet(out_dir / 'test_logs_early.parquet', index=False)
    course_info.to_parquet(out_dir / 'course_info.parquet', index=False)

    # 保存 ID 映射
    with open(out_dir / 'id_maps.json', 'w', encoding='utf-8') as f:
        json.dump(id_maps, f, indent=2, ensure_ascii=False)

    # 保存元数据
    metadata = {
        'early_window_percent': args.early_window_percent,
        'num_users': id_maps['num_users'],
        'num_courses': id_maps['num_courses'],
        'num_enrolls': id_maps['num_enrolls'],
        'train_logs': len(train_early),
        'test_logs': len(test_early),
        'processed_at': datetime.now().isoformat()
    }

    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("[5/5] Done!")
    print(f"  - Train logs (early): {len(train_early)}")
    print(f"  - Test logs (early): {len(test_early)}")
    print(f"  - Unique users: {id_maps['num_users']}")
    print(f"  - Unique courses: {id_maps['num_courses']}")
    print(f"  - Output directory: {out_dir}")


if __name__ == '__main__':
    main()