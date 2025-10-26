"""
构建 MOOC 元数据(items_meta.json, users_meta.json)
优先使用 prediction_data/train_log.csv,回退到 activity.tar.gz
修复版本:改进列名映射和数据加载逻辑
"""
import argparse
import gzip
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--early_ratio", type=float, default=0.3)
    return p.parse_args()


def load_course_info(raw_dir: Path):
    """加载课程信息"""
    path = raw_dir / "course_info.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # 标准化列名
    df.columns = df.columns.str.strip()

    courses = {}
    for _, row in df.iterrows():
        cid = row.get("id")
        course_id = str(row["course_id"]).strip()

        # 处理 cid: 尝试转换为 int,失败则保留字符串或 None
        cid_value = None
        if pd.notna(cid):
            try:
                cid_value = int(cid)
            except (ValueError, TypeError):
                cid_value = str(cid).strip()

        courses[course_id] = {
            "cid": cid_value,
            "course_id": course_id,
            "title": str(row.get("title", course_id)).strip() if pd.notna(row.get("title")) else course_id,
            "category": str(row.get("category", "unknown")).strip() if pd.notna(row.get("category")) else "unknown",
            "course_type": int(row["course_type"]) if pd.notna(row.get("course_type")) else -1,
            "start": str(row["start"]) if pd.notna(row.get("start")) else "",
            "end": str(row["end"]) if pd.notna(row.get("end")) else "",
        }
    return courses


def load_user_info(raw_dir: Path):
    """加载用户信息"""
    path = raw_dir / "user_info.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    users = {}
    current_year = 2025

    for _, row in df.iterrows():
        uid = str(row["user_id"]).strip()

        # 人口统计
        gender = str(row.get("gender", "")).strip() if pd.notna(row.get("gender")) else ""
        edu = str(row.get("education", "")).strip() if pd.notna(row.get("education")) else ""
        birth = row.get("birth")

        # 年龄段
        age_group = ""
        if pd.notna(birth):
            try:
                age = current_year - int(birth)
                if age < 18:
                    age_group = "18岁以下"
                elif age <= 25:
                    age_group = "18-25岁"
                elif age <= 35:
                    age_group = "26-35岁"
                else:
                    age_group = "36岁以上"
            except:
                pass

        # 组装 demo 字符串
        demo_parts = []
        if gender:
            demo_parts.append(f"性别 {gender}")
        if edu:
            demo_parts.append(f"教育 {edu}")
        if age_group:
            demo_parts.append(f"年龄段 {age_group}")

        users[uid] = {
            "uid": uid,
            "demo": ";".join(demo_parts) if demo_parts else "信息有限",
        }
    return users


def try_load_train_log(raw_dir: Path):
    """尝试从 prediction_data.tar.gz 读取 train_log.csv"""
    tar_path = raw_dir / "prediction_data.tar.gz"
    if not tar_path.exists():
        print(f"      未找到 {tar_path}")
        return None

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if "train_log.csv" in member.name:
                    print(f"      找到 {member.name}")
                    f = tar.extractfile(member)
                    df = pd.read_csv(f, encoding="utf-8")
                    df.columns = df.columns.str.strip()

                    print(f"      原始列名: {list(df.columns)}")

                    # 改进的列名映射逻辑
                    column_map = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        # 明确映射 username -> user_id
                        if col_lower == 'username':
                            column_map[col] = 'user_id'
                        elif 'user' in col_lower and 'id' in col_lower:
                            column_map[col] = 'user_id'
                        elif 'course' in col_lower and 'id' in col_lower:
                            column_map[col] = 'course_id'
                        elif 'time' in col_lower or 'date' in col_lower:
                            column_map[col] = 'time'
                        elif col_lower == 'action':
                            column_map[col] = 'action'
                        elif 'event' in col_lower:
                            column_map[col] = 'action'

                    if column_map:
                        print(f"      应用映射: {column_map}")
                        df.rename(columns=column_map, inplace=True)

                    # 验证必需列
                    required = ['user_id', 'course_id', 'time']
                    missing = [col for col in required if col not in df.columns]

                    if missing:
                        print(f"      [WARN] 缺少必需列: {missing}")
                        print(f"      映射后列名: {list(df.columns)}")
                        return None

                    print(f"      成功加载 train_log: {len(df)} 条记录")
                    print(f"      最终列名: {list(df.columns)}")
                    return df

    except Exception as e:
        print(f"      [WARN] 无法从 prediction_data.tar.gz 读取 train_log: {e}")
    return None


def load_activity_tracking(raw_dir: Path):
    """回退:加载 activity.tar.gz 的 tracking JSON"""
    records = []

    tar_files = [
        "20150801-20160801-activity.tar.gz",
        "20160801-20170801-activity.tar.gz"
    ]

    for fname in tar_files:
        tar_path = raw_dir / fname
        if not tar_path.exists():
            print(f"      未找到 {fname}")
            continue

        print(f"      读取 {fname}")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                member_count = 0
                for member in tar.getmembers():
                    if not member.isfile():
                        continue

                    member_count += 1
                    if member_count % 100 == 0:
                        print(f"        处理第 {member_count} 个文件,已提取 {len(records)} 条记录")

                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        for line_num, line in enumerate(f, 1):
                            try:
                                obj = json.loads(line)
                                username = obj.get("username", "")
                                context = obj.get("context", {})
                                course_id = context.get("course_id", "") if isinstance(context, dict) else ""

                                # 确保有有效的用户名和课程ID
                                if username and course_id:
                                    records.append({
                                        "user_id": str(username),
                                        "course_id": str(course_id),
                                        "time": obj.get("time", ""),
                                        "event": str(obj.get("event_type", "")),
                                    })
                            except json.JSONDecodeError:
                                # 跳过无效的 JSON 行
                                continue
                            except Exception as e:
                                # 跳过其他解析错误
                                if line_num == 1:  # 只在第一行报告错误
                                    print(f"          解析行出错: {str(e)[:50]}")
                                continue

                    except Exception as e:
                        print(f"        处理文件 {member.name} 出错: {e}")
                        continue

                print(f"      从 {fname} 提取 {len(records)} 条记录")

        except Exception as e:
            print(f"      [ERROR] 无法打开 {fname}: {e}")
            continue

    if not records:
        print("      [ERROR] 未能从 activity 文件中提取任何记录")
        return None

    print(f"      总共提取 {len(records)} 条 activity 记录")
    df = pd.DataFrame(records)

    # 过滤空值
    df = df[df["user_id"].str.len() > 0]
    df = df[df["course_id"].str.len() > 0]

    print(f"      过滤后剩余 {len(df)} 条有效记录")

    if len(df) == 0:
        return None

    return df


def compute_early_window_cutoff(courses: dict, early_ratio: float):
    """计算每个课程的早期窗口截止时间"""
    cutoffs = {}
    for cid, info in courses.items():
        start_str = info["start"]
        end_str = info["end"]
        if not start_str or not end_str:
            cutoffs[cid] = None
            continue
        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
            delta = (end_dt - start_dt).total_seconds() * early_ratio
            cutoff_dt = start_dt + pd.Timedelta(seconds=delta)
            cutoffs[cid] = cutoff_dt
        except:
            cutoffs[cid] = None
    return cutoffs


def filter_early_window(df: pd.DataFrame, cutoffs: dict):
    """过滤出早期窗口内的交互(向量化优化)"""
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time_dt"])

    # 向量化:映射每个 course_id 到其 cutoff
    df["cutoff_dt"] = df["course_id"].map(cutoffs)

    # 过滤:只保留 time_dt <= cutoff_dt 且 cutoff 非空的行
    mask = df["cutoff_dt"].notna() & (df["time_dt"] <= df["cutoff_dt"])
    df_early = df[mask].copy()

    # 清理临时列
    df_early.drop(columns=["cutoff_dt"], inplace=True)

    return df_early


def build_items_meta(courses: dict, df: pd.DataFrame, from_tracking=False):
    """构建课程元数据"""
    # 统计课程层 top_actions
    if from_tracking:
        action_col = "event"
    else:
        action_col = "action"

    action_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        cid = row["course_id"]
        act = str(row.get(action_col, "")).strip()
        # 清洗非字母数字
        act = "".join(c for c in act if c.isalnum() or c in "_-")
        if act:
            action_counts[cid][act] += 1

    items_meta = []
    for cid, info in courses.items():
        top_acts = [a for a, _ in action_counts[cid].most_common(5)]

        items_meta.append({
            "cid": info["cid"],
            "course_id": info["course_id"],
            "title": info["title"],
            "category": info["category"],
            "course_type": info["course_type"],
            "start": info["start"],
            "end": info["end"],
            "top_actions": top_acts,
            "keywords": []
        })

    return items_meta


def build_users_meta(users: dict, df_early: pd.DataFrame, courses: dict, from_tracking=False):
    """构建用户元数据(仅训练+早期窗口)"""
    if df_early.empty:
        # 若无早期数据,仅返回基础 demo
        return [{"uid": u["uid"], "demo": u["demo"], "prefcats": [], "early_behaviors": [], "sampled_cids": []}
                for u in users.values()]

    # 确保必需列存在
    if "user_id" not in df_early.columns:
        print(f"[ERROR] df_early 缺少 user_id 列,实际列: {list(df_early.columns)}")
        return [{"uid": u["uid"], "demo": u["demo"], "prefcats": [], "early_behaviors": [], "sampled_cids": []}
                for u in users.values()]

    action_col = "event" if from_tracking else "action"

    # 确保 action_col 存在
    if action_col not in df_early.columns:
        print(f"[WARN] df_early 缺少 {action_col} 列,使用默认值")
        action_col = None

    # 按用户聚合
    try:
        user_groups = df_early.groupby("user_id")
    except Exception as e:
        print(f"[ERROR] groupby 失败: {e}")
        print(f"df_early 列: {list(df_early.columns)}")
        print(f"df_early shape: {df_early.shape}")
        return [{"uid": u["uid"], "demo": u["demo"], "prefcats": [], "early_behaviors": [], "sampled_cids": []}
                for u in users.values()]

    users_meta = []
    for uid, uinfo in users.items():
        if uid not in user_groups.groups:
            users_meta.append({
                "uid": uid,
                "demo": uinfo["demo"],
                "prefcats": [],
                "early_behaviors": [],
                "sampled_cids": []
            })
            continue

        grp = user_groups.get_group(uid)

        # prefcats: Top-3 类别
        cats = [courses[c]["category"] for c in grp["course_id"] if c in courses]
        cat_counter = Counter(cats)
        prefcats = [c for c, _ in cat_counter.most_common(3) if c != "unknown"]

        # early_behaviors
        behaviors = []

        # 时间分布
        if "time_dt" in grp.columns:
            hours = grp["time_dt"].dt.hour
            evening_ratio = (hours >= 20).sum() / len(hours) if len(hours) > 0 else 0
            morning_ratio = (hours < 12).sum() / len(hours) if len(hours) > 0 else 0

            if evening_ratio > 0.5:
                behaviors.append("晚间活跃")
            elif morning_ratio > 0.5:
                behaviors.append("上午活跃")

        # Top-3 行为
        if action_col and action_col in grp.columns:
            act_counter = Counter(grp[action_col].dropna())
            for act, _ in act_counter.most_common(3):
                behaviors.append(f"常做:{act}")

        behaviors = behaviors[:6]

        # sampled_cids
        sampled = list(grp["course_id"].drop_duplicates().head(5))

        users_meta.append({
            "uid": uid,
            "demo": uinfo["demo"],
            "prefcats": prefcats,
            "early_behaviors": behaviors,
            "sampled_cids": sampled
        })

    return users_meta


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 加载 course_info.csv")
    courses = load_course_info(raw_dir)
    print(f"      加载 {len(courses)} 门课程")

    print("[2/5] 加载 user_info.csv")
    users = load_user_info(raw_dir)
    print(f"      加载 {len(users)} 个用户")

    print("[3/5] 尝试读取 train_log.csv")
    df = try_load_train_log(raw_dir)
    from_tracking = False

    if df is None:
        print("      未找到 train_log,回退到 activity.tar.gz")
        df = load_activity_tracking(raw_dir)
        from_tracking = True
        if df is None:
            print("      [ERROR] 无法加载任何日志数据")
            return
    else:
        print(f"      成功加载 train_log: {len(df)} 条记录")

    print("[4/5] 构建 items_meta.json")
    items_meta = build_items_meta(courses, df, from_tracking)
    items_path = out_dir / "items_meta.json"
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(items_meta, f, ensure_ascii=False, indent=2)
    print(f"      已写入 {items_path}")

    print("[5/5] 构建 users_meta.json")
    cutoffs = compute_early_window_cutoff(courses, args.early_ratio)
    df_early = filter_early_window(df, cutoffs)
    print(f"      早期窗口过滤后: {len(df_early)} 条记录")

    users_meta = build_users_meta(users, df_early, courses, from_tracking)
    users_path = out_dir / "users_meta.json"
    with open(users_path, "w", encoding="utf-8") as f:
        json.dump(users_meta, f, ensure_ascii=False, indent=2)
    print(f"      已写入 {users_path}")

    print("\n[完成] items_meta.json, users_meta.json 已生成")


if __name__ == "__main__":
    main()