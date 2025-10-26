# -*- coding: utf-8 -*-
"""
用户画像生成脚本（DeepSeek LLM，完整替换版）

功能：
1) 从 processed/train_data.csv 与 processed/test_pos.csv 读取用户历史
2) 按时间排序，取最近 L=20 的课程标题序列 + Top-2 类别
3) 调用 DeepSeek（通过 llm_utils.batch_chat）为每位用户生成 3 条不同表述的 30~60 字摘要
4) 训练/评测严格分流，不读取答案标签；输出到 repr/ 下的新文件

输出：
- data/mooc/repr/user_profiles_train.jsonl
- data/mooc/repr/user_profiles_eval.jsonl
- data/mooc/repr/user_train_summary.cache.json（缓存）
- data/mooc/repr/user_eval_summary.cache.json（缓存）

安全：
- 不改动任何原始/评测答案文件
- 仅用历史与课程元信息

运行示例（WSL）：
export DATA_ROOT="data/mooc"
cd /mnt/d/EasyRec-main/EasyRec-main
python scripts/build_user_profiles_llm.py

运行示例（Windows PowerShell）：
$env:DATA_ROOT="data/mooc"
cd D:\EasyRec-main\EasyRec-main
python scripts/build_user_profiles_llm.py
"""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd

# 导入 LLM 工具模块（确保 scripts/llm_utils.py 存在且提供 batch_chat）
import sys
sys.path.insert(0, os.path.dirname(__file__))
from llm_utils import batch_chat


# ==================== 环境与常量 ====================
DATA_ROOT = os.environ.get("DATA_ROOT", "data/mooc")
RAW_DIR   = os.path.join(DATA_ROOT, "raw")
PROC_DIR  = os.path.join(DATA_ROOT, "processed")
REPR_DIR  = os.path.join(DATA_ROOT, "repr")

COURSES_INFO_PATH = os.path.join(RAW_DIR,  "courses_info_with_pre.csv")
TRAIN_DATA_PATH   = os.path.join(PROC_DIR, "train_data.csv")
TEST_POS_PATH     = os.path.join(PROC_DIR, "test_pos.csv")

OUT_TRAIN_JSONL   = os.path.join(REPR_DIR, "user_profiles_train.jsonl")
OUT_EVAL_JSONL    = os.path.join(REPR_DIR, "user_profiles_eval.jsonl")
CACHE_TRAIN_JSON  = os.path.join(REPR_DIR, "user_train_summary.cache.json")
CACHE_EVAL_JSON   = os.path.join(REPR_DIR, "user_eval_summary.cache.json")

# 控制参数
L = 20                   # 最近历史窗口
MIN_INTERACTIONS = 5     # 过滤历史不足用户（与您的数据约束一致）
MAX_CONCURRENCY  = int(os.environ.get("MAX_CONCURRENCY", "30"))
REQUEST_TIMEOUT  = int(os.environ.get("REQUEST_TIMEOUT", "120"))
TEMPERATURE      = float(os.environ.get("TEMPERATURE", "0.2"))

# 系统提示（更严格）
SYSTEM_PROMPT = (
    "你是教育推荐系统的特征工程助手。回答需客观、精炼、专业，不得口语化或营销化，禁止虚构。"
    "不得逐字复述课程名；不要写学分/证书/时长/考试等未给出的信息；保持 30~60 字。"
)

# 违规/虚构词规则（可按需增删）
FORBIDDEN_PATTERNS = [
    r'\d+学时', r'\d+小时', r'\d+课时', r'\d+门课',
    r'\d+学分', r'GPA', r'绩点', r'评分[:：]\d',
    r'颁发证书', r'官方认证', r'权威认证',
    r'保过', r'包会', r'通关', r'最全', r'最佳', r'全网最好', r'全网最'
]


# ==================== 工具函数 ====================
def contains_forbidden(text: str) -> bool:
    return any(re.search(p, text) for p in FORBIDDEN_PATTERNS)

def clean_text(s: str, min_len=30, max_len=70) -> str:
    """长度/字符清洗 + 违规词过滤"""
    s = re.sub(r'\s+', ' ', (s or "").strip())
    if not s:
        return ""
    if contains_forbidden(s):
        return ""
    if len(s) < min_len:
        return ""
    if len(s) > max_len:
        s = s[:max_len-1] + "。"
    return s

def parse_json_array(resp: str) -> List[str]:
    """鲁棒解析 JSON 数组：直接解析→正则提取→兜底单条"""
    if resp is None:
        return []
    txt = str(resp)
    try:
        arr = json.loads(txt)
        if isinstance(arr, list) and arr:
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # 修正常见中文/花引号
    txt2 = (txt.replace('，', ',')
                .replace('“', '"').replace('”', '"')
                .replace('‘', '"').replace('’', '"'))
    m = re.search(r'\[.*\]', txt2, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr:
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [txt.strip()] if txt.strip() else []

def jaccard(a: str, b: str) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A or B else 1.0

def dedup_texts(texts: List[str], thr=0.70) -> List[str]:
    """字级 Jaccard 去重"""
    out = []
    for t in texts:
        if all(jaccard(t, x) <= thr for x in out):
            out.append(t)
    return out

def load_course_info() -> Dict[int, Dict]:
    """{course_id: {'title':..., 'category':...}}"""
    if not Path(COURSES_INFO_PATH).exists():
        raise FileNotFoundError(f"未找到课程文件: {COURSES_INFO_PATH}")
    df = pd.read_csv(COURSES_INFO_PATH)
    info = {}
    for _, r in df.iterrows():
        cid = int(r['id'])
        info[cid] = {
            'title': str(r.get('merged_title', 'Unknown')),
            'category': str(r.get('category', 'Unknown'))
        }
    return info

def build_user_meta(history_cids: List[int], course_info: Dict) -> Tuple[str, str, int]:
    """最近 L=20 的标题串 & Top-2 类别；返回 (titles_str, cats_str, n_hist)"""
    seq = history_cids[-L:] if len(history_cids) > L else history_cids
    titles, cats = [], []
    for cid in seq:
        if cid in course_info:
            titles.append(course_info[cid]['title'])
            cats.append(course_info[cid]['category'])
    titles_str = "、".join(titles[:10]) + (f"等{len(titles)}门" if len(titles) > 10 else "")
    top2 = [c for c, _ in Counter([c for c in cats if c]).most_common(2)]
    cats_str = "、".join(top2) if top2 else "综合"
    return titles_str, cats_str, len(seq)

def build_user_prompt(titles_str: str, cats_str: str, n_hist: int) -> str:
    """统一且严格的提示词（要求输出 JSON 数组）"""
    return (
        "你是教育推荐系统的特征工程助手。根据“最近课程标题列表”和“Top-2 类别”，"
        "生成 3 条 30~60 字、不同表述的一句话摘要，概括学习者的主题兴趣与能力倾向。"
        "要求：客观专业、无口语；不得逐字复述课程名；禁止虚构考试/学分/证书/时长等信息。\n\n"
        f"- 最近课程（按时间，{n_hist} 门）：{titles_str}\n"
        f"- 关注领域（Top-2）：{cats_str}\n\n"
        "仅输出 JSON 数组（如 [\"...\",\"...\",\"...\"]），不要输出其他文字。"
    )


# ==================== 数据集处理 ====================
async def process_dataset(
    data_path: str,
    out_jsonl: str,
    cache_json: str,
    course_info: Dict,
    dataset_name: str
):
    print(f"\n{'='*60}\n处理{dataset_name}\n{'='*60}")
    print(f"输入：{data_path}\n输出：{out_jsonl}\n缓存：{cache_json}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"未找到输入：{data_path}")

    # 读取与清洗
    df = pd.read_csv(data_path)
    if 'date' not in df.columns:
        raise ValueError(f"{data_path} 缺少 date 列")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # courses → int，过滤 NaN
    df['courses'] = pd.to_numeric(df['courses'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['courses'])
    df['courses'] = df['courses'].astype(int)

    # 时间排序
    df = df.sort_values(['student_id', 'date'])

    # 聚合历史，并过滤最少交互
    hist = df.groupby('student_id')['courses'].apply(list)
    hist = hist[hist.map(len) >= MIN_INTERACTIONS]

    # 构造提示（保序：按 user_id 排序）
    user_ids = sorted(hist.index.tolist())
    prompts: List[str] = []
    prompt_to_uid: Dict[str, int] = {}

    for uid in user_ids:
        titles_str, cats_str, n_hist = build_user_meta(hist[uid], course_info)
        p = build_user_prompt(titles_str, cats_str, n_hist)
        prompts.append(p)
        prompt_to_uid[p] = int(uid)

    print(f"待请求用户数：{len(prompts)}")

    # 调用 LLM（向后兼容：batch_chat 可能不支持某些参数）
    batch_kwargs = dict(
        prompts=prompts,
        cache_json=cache_json,
        system=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_concurrency=MAX_CONCURRENCY,
    )
    # 尝试可选参数（如果 llm_utils 不支持，会自动降级）
    for k, v in [('request_timeout', REQUEST_TIMEOUT), ('max_retries', 3), ('keepalive', True)]:
        try:
            batch_kwargs[k] = v
        except Exception:
            pass

    try:
        results = await batch_chat(**batch_kwargs)
    except TypeError:
        # 降级（只保留最基本参数）
        results = await batch_chat(
            prompts=prompts,
            cache_json=cache_json,
            system=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_concurrency=MAX_CONCURRENCY
        )

    # 写结果（严格按 prompts 顺序防错位），清洗/去重/补齐
    Path(REPR_DIR).mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for p in prompts:
            uid = prompt_to_uid[p]
            raw = results.get(p, "")
            arr = parse_json_array(raw)

            cleaned = [clean_text(x) for x in arr if x]
            cleaned = [x for x in cleaned if x]
            cleaned = dedup_texts(cleaned, thr=0.70)

            while len(cleaned) < 3:
                cleaned.append("该学习者在所关注领域呈现持续兴趣与进阶意愿，具备稳健的学习与迁移能力。")

            for s in cleaned[:3]:
                f.write(json.dumps({"user_id": uid, "text": f"用户画像：{s}\n用户ID：{uid}"}, ensure_ascii=False) + "\n")

    print(f"✓ 已保存：{out_jsonl}")


# ==================== 主程序 ====================
async def main():
    print("\n" + "="*60)
    print("用户画像生成（DeepSeek LLM，完整替换版）")
    print("="*60)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"并发: {MAX_CONCURRENCY} | 超时: {REQUEST_TIMEOUT}s | 温度: {TEMPERATURE}")
    print("="*60 + "\n")

    # 课程元数据（标题/类别映射）
    print("📖 加载课程元数据 ...")
    course_info = load_course_info()
    print(f"✓ 课程数：{len(course_info)}")

    # 训练集（仅 train_data.csv）
    await process_dataset(
        data_path=TRAIN_DATA_PATH,
        out_jsonl=OUT_TRAIN_JSONL,
        cache_json=CACHE_TRAIN_JSON,
        course_info=course_info,
        dataset_name="训练集"
    )

    # 评测集（仅 test_pos.csv，不读取答案）
    await process_dataset(
        data_path=TEST_POS_PATH,
        out_jsonl=OUT_EVAL_JSONL,
        cache_json=CACHE_EVAL_JSON,
        course_info=course_info,
        dataset_name="评测集"
    )

    print("\n完成：")
    print(f"  训练画像：{OUT_TRAIN_JSONL}")
    print(f"  评测画像：{OUT_EVAL_JSONL}")
    print(f"  缓存：{CACHE_TRAIN_JSON}, {CACHE_EVAL_JSON}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

