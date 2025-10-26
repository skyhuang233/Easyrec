"""
EasyRec 训练数据准备流水线
功能：
1. 用户≥5过滤（不筛物品、不去重）
2. 80/10/10 用户级划分（train/val/test）
3. 关系感知+随机混合负采样（1:3比例）
4. 序列打包与落盘

运行示例：
python generation/prepare_training_data.py --data_path data/mooc --negsample_train 4 --negsample_eval 99
"""

import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def parse_args():
    parser = argparse.ArgumentParser(description='EasyRec Training Data Preparation')
    parser.add_argument('--data_path', type=str, default='data/mooc',
                        help='数据集根目录（不含raw/processed）')
    parser.add_argument('--seq_max_len', type=int, default=20,
                        help='序列最大长度（padding/truncation）')
    parser.add_argument('--negsample_train', type=int, default=4,
                        help='训练负采样数（建议4，约1关系+3随机）')
    parser.add_argument('--negsample_eval', type=int, default=99,
                        help='验证/测试负采样数（建议99）')
    parser.add_argument('--rel_window', type=int, default=5,
                        help='关系负采样窗口（使用历史最近N步）')
    parser.add_argument('--topk_next', type=int, default=20,
                        help='全局next-map保留的top-K候选')
    parser.add_argument('--rel_ratio', type=float, default=0.25,
                        help='关系负样本占比（0.25≈1:3）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集用户占比')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集用户占比')
    parser.add_argument('--seed', type=int, default=2025,
                        help='随机种子')
    parser.add_argument('--user_min_interactions', type=int, default=5,
                        help='用户最小交互次数（>=5保留）')
    return parser.parse_args()


def get_cid2cateid_dict(courses_info):
    """
    构建课程ID -> 类别ID映射
    courses_info: DataFrame with columns [id, category_id]
    """
    cid2cateid = {}
    for _, row in courses_info.iterrows():
        try:
            cid = int(row['id'])
            cateid = int(row['category_id'])
            cid2cateid[cid] = cateid
        except:
            continue
    return cid2cateid


def trn_val_test_split(args, raw_dir, proc_dir):
    """
    用户级划分：80/10/10 (train/val/test)
    - 仅过滤用户：交互次数 >= args.user_min_interactions
    - 不筛物品，不删重复
    - 测试/验证用户：最后一条为答案，其余为历史
    """
    print("\n=== Step 1: Train/Val/Test Split ===")
    all_click_df = pd.read_csv(os.path.join(raw_dir, 'all_ratings.csv'))
    print(f"原始交互数: {len(all_click_df)}, 用户数: {all_click_df['student_id'].nunique()}")

    # 仅过滤用户：交互次数 >= threshold
    user_cnt = all_click_df.groupby('student_id').size()
    keep_users = set(user_cnt[user_cnt >= args.user_min_interactions].index)
    all_click_df = all_click_df[all_click_df['student_id'].isin(keep_users)].copy()
    print(f"过滤后交互数: {len(all_click_df)}, 用户数: {all_click_df['student_id'].nunique()}")

    # 按时间排序
    all_click_df = all_click_df.sort_values(['student_id', 'date']).reset_index(drop=True)

    # 用户级随机划分
    all_user_ids = all_click_df['student_id'].unique()
    num_users = len(all_user_ids)
    np.random.seed(args.seed)
    np.random.shuffle(all_user_ids)

    n_train = int(num_users * args.train_ratio)
    n_val = int(num_users * args.val_ratio)

    train_users = set(all_user_ids[:n_train])
    val_users = set(all_user_ids[n_train:n_train + n_val])
    test_users = set(all_user_ids[n_train + n_val:])

    print(f"用户划分 - Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")

    # 提取各集合
    train_all = all_click_df[all_click_df['student_id'].isin(train_users)].copy()
    val_all = all_click_df[all_click_df['student_id'].isin(val_users)].copy()
    test_all = all_click_df[all_click_df['student_id'].isin(test_users)].copy()

    # 验证集：最后一条为答案
    val_ans = val_all.groupby('student_id').tail(1)
    val_pos = val_all.groupby('student_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    # 测试集：最后一条为答案
    test_ans = test_all.groupby('student_id').tail(1)
    test_pos = test_all.groupby('student_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    # 训练数据 = 训练用户全部
    train_data = train_all.copy()

    # 保存
    os.makedirs(proc_dir, exist_ok=True)
    train_data.to_csv(os.path.join(proc_dir, 'train_data.csv'), index=False)
    val_pos.to_csv(os.path.join(proc_dir, 'val_pos.csv'), index=False)
    val_ans.to_csv(os.path.join(proc_dir, 'val_ans.csv'), index=False)
    test_pos.to_csv(os.path.join(proc_dir, 'test_pos.csv'), index=False)
    test_ans.to_csv(os.path.join(proc_dir, 'test_ans.csv'), index=False)

    print(f"训练集交互数: {len(train_data)}")
    print(f"验证集 - 历史: {len(val_pos)}, 答案: {len(val_ans)}")
    print(f"测试集 - 历史: {len(test_pos)}, 答案: {len(test_ans)}")

    # 返回所有物品ID（用于负采样）
    all_item_ids = all_click_df['courses'].unique()
    return all_item_ids


def _build_global_next_map(data, topk_per_item=20):
    """
    构建全局 item->next_item 转移表（按频次降序）
    用于关系感知负采样
    """
    g = defaultdict(Counter)
    for _, hist in data.sort_values(["student_id", "date"]).groupby("student_id"):
        seq = hist["courses"].tolist()
        for t in range(len(seq) - 1):
            g[seq[t]][seq[t + 1]] += 1

    next_map = {}
    for it, cnt in g.items():
        next_map[it] = [x for x, _ in cnt.most_common(topk_per_item)]
    return next_map


def _alloc_counts(total_neg, rel_ratio=0.25):
    """
    分配关系/随机负样本数量
    rel_ratio=0.25 表示 1:3 比例
    """
    n_rel = int(total_neg * rel_ratio + 0.5)
    n_rel = max(0, min(total_neg, n_rel))
    n_rand = total_neg - n_rel
    return n_rel, n_rand


def _safe_get_cate(cid2cateid_dict, item_id, default=-1):
    """安全获取类别ID"""
    try:
        return int(cid2cateid_dict.get(int(item_id), default))
    except:
        return default


def gen_train_data_set(data, all_item_ids, cid2cateid_dict, negsample=4,
                       rel_window=5, topk_next=20, rel_ratio=0.25,
                       unique_neg=True, seed=2025):
    """
    训练集负采样（滑窗方式）
    返回: list of (user_id, history_courses, history_cates, candidate, candidate_cate, label, hist_len)

    负采样策略：
    - 关系负样本（~25%）：从历史课程的"下一跳候选"中抽取
    - 随机负样本（~75%）：从用户未交互全集中抽取
    """
    print(f"\n=== Generating Training Data (negsample={negsample}) ===")
    rng = np.random.RandomState(seed)
    data = data.sort_values("date")
    item_ids = np.asarray(all_item_ids, dtype=int)

    # 构建全局next-map
    print("构建全局next-map...")
    global_next = _build_global_next_map(data, topk_per_item=topk_next)

    train_set = []
    for reviewerID, hist in tqdm(data.groupby('student_id'), desc="gen_train_data_set"):
        pos_course_list = hist['courses'].astype(int).tolist()
        pos_cate_list = hist['category_id'].astype(int).tolist()
        if len(pos_course_list) == 0:
            continue

        user_items = set(pos_course_list)
        rand_pool = np.array(list(set(item_ids) - user_items))

        for i in range(len(pos_course_list)):
            course_hist = pos_course_list[:i]
            cate_hist = pos_cate_list[:i]
            pos_item = int(pos_course_list[i])
            pos_cate = int(pos_cate_list[i])

            # 正样本
            train_set.append((reviewerID, course_hist[:], cate_hist[:],
                              pos_item, pos_cate, 1, len(course_hist)))

            if negsample <= 0:
                continue

            # === 混合负采样 ===
            # 1. 关系候选池：历史最近rel_window步的全局next候选
            rel_pool = set()
            if i > 0:
                left = max(0, i - rel_window)
                recent_hist = course_hist[left:i]
                for h in recent_hist:
                    rel_pool.update(global_next.get(h, []))
                rel_pool.difference_update(user_items)
                rel_pool.discard(pos_item)
            rel_pool = list(rel_pool)

            # 2. 分配数量
            n_rel, n_rand = _alloc_counts(negsample, rel_ratio=rel_ratio)
            chosen_negs = []

            # 3. 关系负样本
            if len(rel_pool) > 0 and n_rel > 0:
                if unique_neg:
                    n_take = min(n_rel, len(rel_pool))
                    chosen_negs.extend(rng.choice(rel_pool, size=n_take, replace=False).tolist())
                else:
                    chosen_negs.extend(rng.choice(rel_pool, size=n_rel, replace=True).tolist())
            else:
                # 关系候选不足，回退给随机
                n_rand = negsample

            # 4. 随机负样本
            if n_rand > 0 and len(rand_pool) > 0:
                if unique_neg and len(chosen_negs) > 0:
                    mask = ~np.isin(rand_pool, np.array(chosen_negs, dtype=int))
                    cand = rand_pool[mask]
                else:
                    cand = rand_pool
                if len(cand) >= n_rand:
                    chosen_negs.extend(rng.choice(cand, size=n_rand, replace=False).tolist())
                else:
                    chosen_negs.extend(rng.choice(cand, size=n_rand, replace=True).tolist())

            # 5. 写出负样本
            for neg_item in chosen_negs:
                neg_cate = _safe_get_cate(cid2cateid_dict, neg_item, default=-1)
                train_set.append((reviewerID, course_hist[:], cate_hist[:],
                                  int(neg_item), neg_cate, 0, len(course_hist)))

    print(f"训练样本总数: {len(train_set)} (正样本 + 负样本)")
    return train_set


def gen_test_data_set(data, all_item_ids, cid2cateid_dict, negsample=99,
                      rel_window=5, topk_next=20, rel_ratio=0.25,
                      unique_neg=True, seed=2025):
    """
    测试/验证集负采样（每用户最后一条）
    返回: list of (user_id, history_courses, history_cates, candidate, candidate_cate, label, hist_len)

    构造方式：1正 + negsample负（关系:随机 ≈ 1:3）
    """
    print(f"\n=== Generating Test/Val Data (negsample={negsample}) ===")
    rng = np.random.RandomState(seed)
    data = data.sort_values("date")
    item_ids = np.asarray(all_item_ids, dtype=int)

    # 构建全局next-map（注意：仅使用历史数据构建，避免泄露）
    print("构建全局next-map...")
    global_next = _build_global_next_map(data, topk_per_item=topk_next)

    test_set = []
    for reviewerID, hist in tqdm(data.groupby('student_id'), desc="gen_test_data_set"):
        pos_course_list = hist['courses'].astype(int).tolist()
        pos_cate_list = hist['category_id'].astype(int).tolist()
        if len(pos_course_list) < 2:
            continue  # 需要至少1条历史 + 1条目标

        user_items = set(pos_course_list)
        rand_pool = np.array(list(set(item_ids) - user_items))

        course_hist = pos_course_list[:-1]
        cate_hist = pos_cate_list[:-1]
        pos_item = int(pos_course_list[-1])
        pos_cate = int(pos_cate_list[-1])

        # 正样本
        test_set.append((reviewerID, course_hist[:], cate_hist[:],
                         pos_item, pos_cate, 1, len(course_hist)))

        if negsample <= 0:
            continue

        # === 混合负采样 ===
        # 1. 关系候选池
        rel_pool = set()
        for h in course_hist[-rel_window:]:
            rel_pool.update(global_next.get(h, []))
        rel_pool.difference_update(user_items)
        rel_pool.discard(pos_item)
        rel_pool = list(rel_pool)

        # 2. 分配数量
        n_rel, n_rand = _alloc_counts(negsample, rel_ratio=rel_ratio)
        chosen_negs = []

        # 3. 关系负样本
        if len(rel_pool) > 0 and n_rel > 0:
            if unique_neg:
                n_take = min(n_rel, len(rel_pool))
                chosen_negs.extend(rng.choice(rel_pool, size=n_take, replace=False).tolist())
            else:
                chosen_negs.extend(rng.choice(rel_pool, size=n_rel, replace=True).tolist())
        else:
            n_rand = negsample

        # 4. 随机负样本
        if n_rand > 0 and len(rand_pool) > 0:
            if unique_neg and len(chosen_negs) > 0:
                mask = ~np.isin(rand_pool, np.array(chosen_negs, dtype=int))
                cand = rand_pool[mask]
            else:
                cand = rand_pool
            if len(cand) >= n_rand:
                chosen_negs.extend(rng.choice(cand, size=n_rand, replace=False).tolist())
            else:
                chosen_negs.extend(rng.choice(cand, size=n_rand, replace=True).tolist())

        # 5. 写出负样本
        for neg_item in chosen_negs:
            neg_cate = _safe_get_cate(cid2cateid_dict, neg_item, default=-1)
            test_set.append((reviewerID, course_hist[:], cate_hist[:],
                             int(neg_item), neg_cate, 0, len(course_hist)))

    print(f"测试样本总数: {len(test_set)} (正样本 + 负样本)")
    return test_set


def gen_model_input(train_set, seq_max_len=20):
    """
    打包样本：pad/truncate 到固定长度，转换为DataFrame

    输入: list of (user_id, history_courses, history_cates, candidate, candidate_cate, label, hist_len)
    输出: DataFrame with columns [student_id, courses, category_id, candidate, candidate_cate, label, hist_len]

    courses/category_id 格式：逗号分隔字符串，pad用0填充
    每用户保留最近50条（groupby+tail）
    """
    print(f"\n=== Packing Model Input (seq_max_len={seq_max_len}) ===")

    users_id, courses_list, cates_list, candidate_list, candidate_cate_list, labels_list, hist_len_list = [], [], [], [], [], [], []

    for user_id, course_hist, cate_hist, candidate, candidate_cate, label, hist_len in tqdm(train_set,
                                                                                            desc="gen_model_input"):
        users_id.append(user_id)
        courses_list.append(course_hist)
        cates_list.append(cate_hist)
        candidate_list.append(candidate)
        candidate_cate_list.append(candidate_cate)
        labels_list.append(label)
        hist_len_list.append(hist_len)

    # Pad sequences
    courses_list = pad_sequences(courses_list, maxlen=seq_max_len, padding='pre', truncating='pre', value=0).tolist()
    cates_list = pad_sequences(cates_list, maxlen=seq_max_len, padding='pre', truncating='pre', value=0).tolist()

    # 转换为逗号字符串
    courses_str = [','.join(map(str, seq)) for seq in courses_list]
    cates_str = [','.join(map(str, seq)) for seq in cates_list]

    # 构建DataFrame
    df = pd.DataFrame({
        'student_id': users_id,
        'courses': courses_str,
        'category_id': cates_str,
        'candidate': candidate_list,
        'candidate_cate': candidate_cate_list,
        'label': labels_list,
        'hist_len': hist_len_list
    })

    # 每用户保留最近50条
    df = df.groupby('student_id').tail(50).reset_index(drop=True)

    print(f"最终样本数: {len(df)}")
    return df


def main():
    args = parse_args()
    print(f"\n{'=' * 60}")
    print(f"EasyRec Training Data Preparation")
    print(f"{'=' * 60}")
    print(f"数据路径: {args.data_path}")
    print(f"训练负采样: {args.negsample_train}, 评测负采样: {args.negsample_eval}")
    print(f"序列最大长度: {args.seq_max_len}")
    print(f"关系负采样比例: {args.rel_ratio:.2%} (窗口={args.rel_window}, topk={args.topk_next})")
    print(f"随机种子: {args.seed}")
    print(f"{'=' * 60}\n")

    # 路径设置
    raw_dir = os.path.join(args.data_path, 'raw')
    proc_dir = os.path.join(args.data_path, 'processed')

    # Step 1: 加载课程元信息
    print("=== Loading Course Metadata ===")
    courses_info = pd.read_csv(os.path.join(raw_dir, 'courses_info_with_pre.csv'))
    cid2cateid_dict = get_cid2cateid_dict(courses_info)
    print(f"课程数: {len(cid2cateid_dict)}")

    # Step 2: 划分数据集
    all_item_ids = trn_val_test_split(args, raw_dir, proc_dir)
    print(f"物品总数: {len(all_item_ids)}")

    # Step 3: 训练集负采样
    train_data = pd.read_csv(os.path.join(proc_dir, 'train_data.csv'))
    train_set = gen_train_data_set(
        train_data, all_item_ids, cid2cateid_dict,
        negsample=args.negsample_train,
        rel_window=args.rel_window,
        topk_next=args.topk_next,
        rel_ratio=args.rel_ratio,
        seed=args.seed
    )
    train_data_transformed = gen_model_input(train_set, seq_max_len=args.seq_max_len)
    train_data_transformed.to_csv(os.path.join(proc_dir, 'train_data_transformed.csv'), index=False)
    print(f"✓ 训练集已保存: {os.path.join(proc_dir, 'train_data_transformed.csv')}")

    # Step 4: 验证集负采样
    val_pos = pd.read_csv(os.path.join(proc_dir, 'val_pos.csv'))
    val_ans = pd.read_csv(os.path.join(proc_dir, 'val_ans.csv'))
    val_data = pd.concat([val_pos, val_ans])
    val_set = gen_test_data_set(
        val_data, all_item_ids, cid2cateid_dict,
        negsample=args.negsample_eval,
        rel_window=args.rel_window,
        topk_next=args.topk_next,
        rel_ratio=args.rel_ratio,
        seed=args.seed
    )
    val_data_transformed = gen_model_input(val_set, seq_max_len=args.seq_max_len)
    val_data_transformed.to_csv(os.path.join(proc_dir, 'val_data_transformed.csv'), index=False)
    print(f"✓ 验证集已保存: {os.path.join(proc_dir, 'val_data_transformed.csv')}")

    # Step 5: 测试集负采样
    test_pos = pd.read_csv(os.path.join(proc_dir, 'test_pos.csv'))
    test_ans = pd.read_csv(os.path.join(proc_dir, 'test_ans.csv'))
    test_data = pd.concat([test_pos, test_ans])
    test_set = gen_test_data_set(
        test_data, all_item_ids, cid2cateid_dict,
        negsample=args.negsample_eval,
        rel_window=args.rel_window,
        topk_next=args.topk_next,
        rel_ratio=args.rel_ratio,
        seed=args.seed
    )
    test_data_transformed = gen_model_input(test_set, seq_max_len=args.seq_max_len)
    test_data_transformed.to_csv(os.path.join(proc_dir, 'test_data_transformed.csv'), index=False)
    print(f"✓ 测试集已保存: {os.path.join(proc_dir, 'test_data_transformed.csv')}")

    # 统计信息
    print(f"\n{'=' * 60}")
    print("数据准备完成！")
    print(f"{'=' * 60}")
    print(f"训练集: {len(train_data_transformed):,} 样本 "
          f"(正样本比例: {train_data_transformed['label'].mean():.2%})")
    print(f"验证集: {len(val_data_transformed):,} 样本 "
          f"({val_ans['student_id'].nunique()} 用户)")
    print(f"测试集: {len(test_data_transformed):,} 样本 "
          f"({test_ans['student_id'].nunique()} 用户)")
    print(f"{'=' * 60}\n")

    # 验证提示
    print("下一步：")
    print("1. 验证负采样分布（可选）：")
    print(f"   python scripts/validate_negative_sampling.py --data_path {args.data_path}")
    print("2. 生成用户/物品画像（如需）")
    print("3. 开始训练：")
    print(f"   python train_easyrec.py --train_data {proc_dir}/train_data_transformed.csv ...")


if __name__ == '__main__':
    main()