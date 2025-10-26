# -*- coding: utf-8 -*-
"""
物品画像生成脚本（DeepSeek LLM）
功能：
1) 从 courses_info_with_pre.csv 读取课程信息
2) 调用 DeepSeek API 生成每个课程的简介（20~40字）
3) 输出 JSONL 格式的物品画像（使用“先修标题串”，非数字ID）
4) 生成质量检查 + 本地回退（长度/复述/敏感词）

运行示例（WSL）：
  DATA_ROOT=data/mooc python scripts/build_item_profiles_llm.py
"""

import os
import re
import json
import asyncio
import pandas as pd
from pathlib import Path

# 导入 LLM 工具模块
import sys
sys.path.insert(0, os.path.dirname(__file__))
from llm_utils import batch_chat

# ===================== 环境与路径 =====================
DATA_ROOT = os.environ.get("DATA_ROOT", "data/mooc")
RAW_DIR   = os.path.join(DATA_ROOT, "raw")
REPR_DIR  = os.path.join(DATA_ROOT, "repr")

COURSES_INFO_PATH = os.path.join(RAW_DIR,  "courses_info_with_pre.csv")
OUTPUT_JSONL      = os.path.join(REPR_DIR, "item_profiles.jsonl")
CACHE_JSON        = os.path.join(REPR_DIR, "item_summary.cache.json")

# ===================== 提示词 =====================
SYSTEM_PROMPT = (
    "你是教育推荐系统的内容助手。回答必须客观、精炼、专业，不得复述课程名，"
    "不得使用口语/感叹号/营销语，不得虚构章节、时长、评分、证书或合作方。"
    "输出长度应在20~40个中文字符左右，仅输出一句话。"
)

def build_title_and_req_maps(courses_df: pd.DataFrame):
    """构建 id->标题 与 id->先修标题串 的映射"""
    title_map = dict(zip(courses_df["id"], courses_df["merged_title"]))
    req_title_map = {}
    for _, r in courses_df.iterrows():
        rid = int(r["id"])
        req_ids = []
        for k in ("required_course1", "required_course2", "required_course3"):
            v = r.get(k, 0)
            try:
                v = int(v)
            except Exception:
                v = 0
            if v != 0:
                req_ids.append(v)
        req_titles = [title_map[i] for i in req_ids if i in title_map]
        req_title_map[rid] = "；".join(req_titles) if req_titles else "无"
    return title_map, req_title_map

def build_item_prompt(item_id: int, title_map: dict, req_title_map: dict, category: str) -> str:
    """构造 LLM 用户提示词（使用先修标题，非ID）"""
    title  = title_map.get(item_id, f"课程ID {item_id}")
    req_str= req_title_map.get(item_id, "无")
    return (
        "你是一个专业的在线教育课程分析师。"
        "请基于给定信息，生成一句20~40字的课程简介，客观概述核心内容与学习目标。"
        "要求：不得复述课程名称，不使用口语或感叹号，不得虚构章节/时长/评分/证书。\n\n"
        f"课程标题：{title}\n"
        f"类别：{category}\n"
        f"先修要求：{req_str}\n\n"
        "仅输出简介一句话。"
    )

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def violates_rules(summary: str, title: str) -> bool:
    """本地质量检查：长度、复述标题、敏感词"""
    s = clean_text(summary)
    if len(s) < 16 or len(s) > 60:
        return True
    title_norm = clean_text(title).lower()
    s_norm     = s.lower()
    if len(title_norm) >= 4 and title_norm in s_norm:
        return True
    ban = ["证书", "官方认证", "学分", "保过", "全网最好"]
    if any(b in s for b in ban):
        return True
    return False

def fallback_summary(req_str: str) -> str:
    """生成回退简介（不含虚构）"""
    core = "系统讲解核心概念与实践要点，帮助学习者建立结构化理解并掌握应用方法。"
    return (f"面向具备相关先修基础的学习者，{core}" if req_str != "无"
            else f"面向初学者，{core}")

def format_item_profile(item_id: int, title_map: dict, req_title_map: dict, category: str, summary: str) -> dict:
    title  = title_map.get(item_id, f"课程ID {item_id}")
    req_str= req_title_map.get(item_id, "无")
    text = f"课程：{title}\n类别：{category}\n先修要求：{req_str}\n简介：{summary}\n课程ID：{item_id}"
    return {"item_id": item_id, "text": text}

# ===================== 主流程 =====================
async def main():
    print("\n" + "="*60)
    print("物品画像生成（DeepSeek LLM）")
    print("="*60)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"输入文件: {COURSES_INFO_PATH}")
    print(f"输出文件: {OUTPUT_JSONL}")
    print(f"缓存文件: {CACHE_JSON}")
    print("="*60 + "\n")

    # 1) 加载课程表
    if not Path(COURSES_INFO_PATH).exists():
        raise FileNotFoundError(f"❌ 文件不存在: {COURSES_INFO_PATH}")
    courses_df = pd.read_csv(COURSES_INFO_PATH)
    print(f"✓ 共 {len(courses_df)} 门课程")

    # 2) 构建映射
    title_map, req_title_map = build_title_and_req_maps(courses_df)

    # 3) 组装 prompts 与元数据
    prompts = []
    meta    = []  # [(item_id, category)]
    for _, row in courses_df.iterrows():
        item_id  = int(row["id"])
        category = str(row.get("category", "未标注"))
        prompts.append(build_item_prompt(item_id, title_map, req_title_map, category))
        meta.append((item_id, category))
    print(f"✓ 共生成 {len(prompts)} 条提示词")

    # 4) 批量请求（带缓存/重试/并发）
    results = await batch_chat(
        prompts=prompts,
        cache_json=CACHE_JSON,
        system=SYSTEM_PROMPT,
        temperature=0.3,
        max_concurrency=5,     # 视你的key和网络情况调整
    )

    # 5) 生成 JSONL（带本地质检 + 回退）
    Path(REPR_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for (prompt, raw_summary), (item_id, category) in zip(results.items(), meta):
            title  = title_map.get(item_id, f"课程ID {item_id}")
            req_str= req_title_map.get(item_id, "无")
            summ   = clean_text(raw_summary)
            if not summ or violates_rules(summ, title):
                summ = fallback_summary(req_str)
            profile = format_item_profile(item_id, title_map, req_title_map, category, summ)
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")
    print(f"✓ 已保存: {OUTPUT_JSONL}")

    # 6) 示例输出
    print("\n示例输出（前 3 条）：")
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            print(f"[{i+1}] item_id={obj['item_id']}")
            print(obj["text"].split("\n简介：")[0] + " ...")

    print("\n✓ 物品画像生成完成")

if __name__ == "__main__":
    asyncio.run(main())
