# file: scripts/gen_profiles.py

"""
生成用户和课程的文本画像
输出格式: JSONL (每行一个 JSON 对象)
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='data/mooc/processed')
    parser.add_argument('--raw-dir', type=str, default='data/mooc/raw')
    parser.add_argument('--out-dir', type=str, default='data/mooc/profiles')
    return parser.parse_args()


def load_user_profile(raw_dir):
    """加载用户画像原始文件"""
    path = Path(raw_dir) / 'user_profile.csv'
    if not path.exists():
        print(f"[WARN] {path} not found, skipping user demographics")
        return None
    return pd.read_csv(path)


def generate_item_profiles(course_info, train_logs, id_maps):
    """生成课程画像"""
    profiles = []
    course2nid = id_maps['course2nid']

    # 统计每门课程的互动数
    course_stats = train_logs.groupby('course_id').agg({
        'enroll_id': 'nunique',  # 注册人数
        'action': 'count'  # 行为数
    }).rename(columns={'enroll_id': 'num_enrolls', 'action': 'num_actions'})

    course_info = course_info.merge(course_stats, on='course_id', how='left')
    course_info['num_enrolls'] = course_info['num_enrolls'].fillna(0).astype(int)
    course_info['num_actions'] = course_info['num_actions'].fillna(0).astype(int)

    for _, row in tqdm(course_info.iterrows(), total=len(course_info), desc="Gen item profiles"):
        course_id = row['course_id']
        nid = course2nid.get(course_id)

        if nid is None:
            continue

        # 构建画像文本
        category = row.get('category', 'Unknown')
        course_type = 'self-paced' if row.get('course_type') == 1 else 'instructor-paced'
        num_enrolls = row['num_enrolls']

        text = (f"This is a {course_type} course in {category}. "
                f"It has {num_enrolls} enrolled students.")

        profile = {
            'nid': int(nid),
            'course_id': course_id,
            'text': text,
            'category': category,
            'course_type': course_type,
            'num_enrolls': int(num_enrolls),
            'num_actions': int(row['num_actions'])
        }

        profiles.append(profile)

    return profiles


def generate_user_profiles(train_logs, course_info, user_demo, id_maps):
    """生成用户画像"""
    profiles = []
    user2nid = id_maps['user2nid']

    # 统计每个用户的行为
    user_course_actions = train_logs.groupby(['username', 'course_id']).size().reset_index(name='action_count')

    # 关联课程类别
    user_course_actions = user_course_actions.merge(
        course_info[['course_id', 'category']],
        on='course_id',
        how='left'
    )

    # 按用户聚合
    user_stats = defaultdict(lambda: {'categories': Counter(), 'total_actions': 0, 'num_courses': 0})

    for _, row in user_course_actions.iterrows():
        username = row['username']
        category = row.get('category', 'Unknown')
        actions = row['action_count']

        user_stats[username]['categories'][category] += actions
        user_stats[username]['total_actions'] += actions
        user_stats[username]['num_courses'] += 1

    for username, stats in tqdm(user_stats.items(), desc="Gen user profiles"):
        nid = user2nid.get(username)
        if nid is None:
            continue

        # 取 top 类别
        top_categories = stats['categories'].most_common(3)
        category_str = ', '.join([cat for cat, _ in top_categories])

        # 人口统计特征 (如果有)
        gender = 'unknown'
        age_bucket = 'unknown'

        if user_demo is not None:
            user_row = user_demo[user_demo['user_id'] == username]
            if not user_row.empty:
                gender = user_row.iloc[0].get('gender', 'unknown')
                birth = user_row.iloc[0].get('birth', None)
                if pd.notna(birth):
                    age = 2024 - int(birth)
                    if age < 20:
                        age_bucket = 'under_20'
                    elif age < 30:
                        age_bucket = '20-30'
                    elif age < 40:
                        age_bucket = '30-40'
                    else:
                        age_bucket = 'over_40'

        text = (f"This user has enrolled in {stats['num_courses']} courses, "
                f"primarily in {category_str}. "
                f"Total learning actions: {stats['total_actions']}.")

        profile = {
            'nid': int(nid),
            'username': username,
            'text': text,
            'top_categories': [cat for cat, _ in top_categories],
            'num_courses': stats['num_courses'],
            'total_actions': stats['total_actions'],
            'gender': gender,
            'age_bucket': age_bucket
        }

        profiles.append(profile)

    return profiles


def main():
    args = parse_args()

    proc_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("[1/4] Loading data...")
    with open(proc_dir / 'id_maps.json', 'r', encoding='utf-8') as f:
        id_maps = json.load(f)

    train_logs = pd.read_parquet(proc_dir / 'train_logs_early.parquet')
    course_info = pd.read_parquet(proc_dir / 'course_info.parquet')
    user_demo = load_user_profile(raw_dir)

    # 2. 生成课程画像
    print("[2/4] Generating item profiles...")
    item_profiles = generate_item_profiles(course_info, train_logs, id_maps)

    # 3. 生成用户画像
    print("[3/4] Generating user profiles...")
    user_profiles = generate_user_profiles(train_logs, course_info, user_demo, id_maps)

    # 4. 保存
    print("[4/4] Saving profiles...")
    with open(out_dir / 'item_profile.jsonl', 'w', encoding='utf-8') as f:
        for profile in item_profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')

    with open(out_dir / 'user_profile.jsonl', 'w', encoding='utf-8') as f:
        for profile in user_profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')

    print(f"Done! Generated {len(item_profiles)} item profiles and {len(user_profiles)} user profiles.")
    print(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()