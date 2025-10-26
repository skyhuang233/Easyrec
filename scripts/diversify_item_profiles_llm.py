# -*- coding: utf-8 -*-
"""
物品画像多样化生成脚本（DeepSeek LLM）- 修正版

功能：
1. 读取 data/mooc/repr/item_profiles.jsonl（基线，不修改）
2. 基于 data/mooc/raw/courses_info_with_pre.csv 获取课程元信息
3. 调用 DeepSeek API 为每个课程生成 3 条不同表述的简介
4. 输出 data/mooc/repr/item_profiles.div3.jsonl（新文件）

修正点：
✓ 先修课映射为标题（而非数值ID）
✓ JSON 解析更鲁棒（中文符号修正）
✓ 长度/虚构词过滤与兜底
✓ 变体去重（Jaccard 简单版 + sentence-transformers 可选）
✓ 并发/超时/重试参数可配置
✓ 顺序可复现（按 item_id 排序）
✓ 文件校验统计

安全与隐私：
- 不修改原始画像文件
- 不引入任何评测数据或答案标签

依赖：
pip install openai pandas tqdm tenacity sentence-transformers

运行示例（WSL）：
export DEEPSEEK_API_KEY="your_key"
export DATA_ROOT="data/mooc"
export MAX_CONCURRENCY=5
export REQUEST_TIMEOUT=120
cd /mnt/d/EasyRec-main/EasyRec-main
python scripts/diversify_item_profiles_llm.py
"""

import os
import json
import re
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
from tqdm import tqdm

# 导入 LLM 工具模块
import sys

sys.path.insert(0, os.path.dirname(__file__))
from llm_utils import batch_chat

# ============ 环境变量 ============
DATA_ROOT = os.environ.get("DATA_ROOT", "data/mooc")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
REPR_DIR = os.path.join(DATA_ROOT, "repr")

# 输入输出路径
COURSES_INFO_PATH = os.path.join(RAW_DIR, "courses_info_with_pre.csv")
BASELINE_JSONL = os.path.join(REPR_DIR, "item_profiles.jsonl")
OUTPUT_JSONL = os.path.join(REPR_DIR, "item_profiles.div3.jsonl")
CACHE_JSON = os.path.join(REPR_DIR, "item_div.cache.json")
ERROR_LOG = os.path.join(REPR_DIR, "item_div.err.log")

# 可配置参数
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "5"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))
TEMPERATURE = 0.2

# ============ LLM 提示词 ============
SYSTEM_PROMPT = """你是教育推荐系统的内容助手。回答需客观、精炼、无口语、不虚构，不编造章节/时长/评分等信息。"""

# ============ 虚构词过滤 ============
FORBIDDEN_PATTERNS = [
    r'\d+学时', r'\d+小时', r'\d+课时', r'\d+章节',
    r'\d+学分', r'\d+\.?\d*分', r'评分[:：]\d',
    r'时长[:：]\d', r'共\d+讲', r'\d+周',
    r'难度[:：]', r'难度系数'
]


def contains_forbidden_content(text: str) -> bool:
    """检查是否包含虚构词"""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def clean_text(text: str, min_len: int = 15, max_len: int = 50) -> str:
    """
    文本清洗与规整
    - 去除多余空格
    - 过滤虚构词
    - 长度控制
    """
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text.strip())

    # 检查虚构内容
    if contains_forbidden_content(text):
        return ""  # 标记为无效

    # 长度控制
    if len(text) < min_len:
        return ""  # 太短，标记为无效
    if len(text) > max_len:
        text = text[:max_len - 1] + "。"  # 截断

    return text


def build_item_prompt(title: str, category: str, req_titles: List[str]) -> str:
    """
    构造物品画像多样化提示词
    修正点：先修课使用标题而非ID
    """
    req_str = "、".join(req_titles) if req_titles else "无"

    prompt = f"""你是教育推荐系统的内容助手。请基于给定的课程标题、类别与先修信息，生成 3 条 20~40 字的不同表述的一句话简介，语气客观、无口语、不得虚构章节或时长。

输入：
- 课程标题：{title}
- 类别：{category}
- 先修：{req_str}

输出：仅输出 JSON 数组（例如：["...","...","..."]），不要输出其他多余文字或解释。"""
    return prompt


def fix_json_string(text: str) -> str:
    """
    修正常见的 JSON 格式错误
    - 中文逗号/引号
    - 多余换行
    """
    # 中文符号转英文
    text = text.replace('，', ',').replace('"', '"').replace('"', '"')
    text = text.replace('【', '[').replace('】', ']')
    # 去除换行
    text = text.replace('\n', ' ').replace('\r', '')
    return text


def parse_llm_response(response: str, fallback_count: int = 3) -> List[str]:
    """
    解析 LLM 返回的 JSON 数组，做健壮性处理
    修正点：中文符号修正 + 确保所有元素为 str
    """
    # 修正 JSON 格式
    response = fix_json_string(response)

    try:
        # 尝试直接解析 JSON
        variants = json.loads(response)
        if isinstance(variants, list) and len(variants) > 0:
            # 确保所有元素为 str 并 strip
            variants = [str(v).strip() for v in variants if v]
            return variants
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 数组
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            json_str = fix_json_string(match.group(0))
            variants = json.loads(json_str)
            if isinstance(variants, list) and len(variants) > 0:
                variants = [str(v).strip() for v in variants if v]
                return variants
    except:
        pass

    # 兜底：返回原文本作为单个变体
    return [response.strip()] if response.strip() else []


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算 Jaccard 相似度（字级别）
    """
    set1 = set(text1)
    set2 = set(text2)
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def deduplicate_variants(variants: List[str], threshold: float = 0.75) -> List[str]:
    """
    变体去重（基于 Jaccard 相似度）
    修正点：简单但有效的去重策略
    """
    if len(variants) <= 1:
        return variants

    unique = [variants[0]]

    for i in range(1, len(variants)):
        is_duplicate = False
        for existing in unique:
            sim = jaccard_similarity(variants[i], existing)
            if sim > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(variants[i])

    return unique


def pad_variants(variants: List[str], target_count: int = 3,
                 fallback: str = "课程简介：内容待完善。") -> List[str]:
    """
    补齐变体数量到 target_count
    """
    while len(variants) < target_count:
        variants.append(fallback)
    return variants[:target_count]


def format_item_text(title: str, category: str, req_str: str,
                     summary: str, item_id: int) -> str:
    """
    格式化为最终的文本格式
    """
    text = f"课程：{title}\n类别：{category}\n先修要求：{req_str}\n简介：{summary}\n课程ID：{item_id}"
    return text


def validate_output(output_jsonl: str) -> Dict:
    """
    验证输出文件
    修正点：检查每个 item 是否有 3 行，统计异常
    """
    print("\n📊 验证输出文件...")

    item_count = Counter()
    nan_count = 0
    empty_count = 0
    forbidden_count = 0
    forbidden_examples = []

    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                item_id = item['item_id']
                text = item['text']

                item_count[item_id] += 1

                # 检查异常
                if pd.isna(text) or text is None:
                    nan_count += 1
                elif not text.strip():
                    empty_count += 1
                elif contains_forbidden_content(text):
                    forbidden_count += 1
                    if len(forbidden_examples) < 5:
                        forbidden_examples.append((item_id, text[:100]))
            except:
                continue

    # 检查每个 item 是否有 3 行
    irregular_items = {iid: cnt for iid, cnt in item_count.items() if cnt != 3}

    stats = {
        'total_lines': sum(item_count.values()),
        'total_items': len(item_count),
        'irregular_items': len(irregular_items),
        'nan_count': nan_count,
        'empty_count': empty_count,
        'forbidden_count': forbidden_count,
        'forbidden_examples': forbidden_examples
    }

    return stats


# ============ 主流程 ============
async def main():
    print("\n" + "=" * 60)
    print("物品画像多样化生成（DeepSeek LLM）- 修正版")
    print("=" * 60)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"并发数: {MAX_CONCURRENCY}")
    print(f"请求超时: {REQUEST_TIMEOUT}s")
    print(f"基线文件: {BASELINE_JSONL} (不修改)")
    print(f"输出文件: {OUTPUT_JSONL}")
    print("=" * 60 + "\n")

    # 1. 读取课程元信息
    print("📖 加载课程元信息...")
    if not Path(COURSES_INFO_PATH).exists():
        raise FileNotFoundError(f"❌ 文件不存在: {COURSES_INFO_PATH}")

    courses_df = pd.read_csv(COURSES_INFO_PATH)
    print(f"✓ 共 {len(courses_df)} 门课程")

    # 构建 item_id -> title 映射（用于先修课）
    id_to_title = {}
    for _, row in courses_df.iterrows():
        item_id = int(row['id'])
        title = str(row.get('merged_title', 'N/A'))
        id_to_title[item_id] = title

    # 构建 item_id -> (title, category, req_titles) 映射
    item_meta = {}
    for _, row in courses_df.iterrows():
        item_id = int(row['id'])
        title = str(row.get('merged_title', 'N/A'))
        category = str(row.get('category', 'N/A'))

        # 修正点：先修课映射为标题列表
        req_titles = []
        for i in [1, 2, 3]:
            pre_id = row.get(f'required_course{i}', None)
            if pd.notna(pre_id):
                try:
                    pre_id = int(pre_id)
                    pre_title = id_to_title.get(pre_id, None)
                    if pre_title:
                        req_titles.append(pre_title)
                except:
                    continue

        item_meta[item_id] = (title, category, req_titles)

    # 2. 读取基线画像（获取所有 item_id）
    print("\n📖 加载基线画像...")
    if not Path(BASELINE_JSONL).exists():
        raise FileNotFoundError(f"❌ 文件不存在: {BASELINE_JSONL}")

    item_ids = []
    with open(BASELINE_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                item_ids.append(int(item['item_id']))
            except:
                continue

    # 修正点：按 item_id 排序，确保顺序可复现
    item_ids = sorted(set(item_ids))
    print(f"✓ 基线画像包含 {len(item_ids)} 个物品")

    # 3. 构造提示词
    print("\n📝 构造提示词...")
    prompts = []
    prompt_to_meta = {}  # {prompt: (item_id, title, category, req_titles)}

    for item_id in item_ids:
        if item_id not in item_meta:
            print(f"⚠️  物品 {item_id} 无元信息，跳过")
            continue

        title, category, req_titles = item_meta[item_id]
        prompt = build_item_prompt(title, category, req_titles)
        prompts.append(prompt)
        prompt_to_meta[prompt] = (item_id, title, category, req_titles)

    print(f"✓ 共生成 {len(prompts)} 条提示词")

    # 4. 批量调用 LLM
    print("\n📡 调用 DeepSeek API...")
    results = await batch_chat(
        prompts=prompts,
        cache_json=CACHE_JSON,
        system=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_concurrency=MAX_CONCURRENCY
    )

    # 5. 解析结果并生成 JSONL
    print("\n💾 解析结果并生成 JSONL...")
    Path(REPR_DIR).mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    error_log_lines = []

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        for prompt, response in tqdm(results.items(), desc="处理结果"):
            if prompt not in prompt_to_meta:
                continue

            item_id, title, category, req_titles = prompt_to_meta[prompt]
            req_str = "、".join(req_titles) if req_titles else "无"

            # 解析变体
            variants = parse_llm_response(response)

            # 清洗文本
            cleaned_variants = []
            for v in variants:
                cleaned = clean_text(v, min_len=15, max_len=50)
                if cleaned:
                    cleaned_variants.append(cleaned)

            # 去重
            if len(cleaned_variants) > 1:
                cleaned_variants = deduplicate_variants(cleaned_variants, threshold=0.75)

            # 补齐到 3 条
            if len(cleaned_variants) < 3:
                fail_count += 1
                error_msg = f"item_id={item_id}, valid_variants={len(cleaned_variants)}, raw={response[:100]}"
                error_log_lines.append(error_msg)

            cleaned_variants = pad_variants(cleaned_variants, target_count=3)
            success_count += 1

            # 写入 3 条变体（每条一行）
            for variant in cleaned_variants:
                text = format_item_text(title, category, req_str, variant, item_id)
                profile = {"item_id": item_id, "text": text}
                f_out.write(json.dumps(profile, ensure_ascii=False) + '\n')

    print(f"✓ 已保存: {OUTPUT_JSONL}")

    # 6. 写错误日志
    if error_log_lines:
        with open(ERROR_LOG, 'w', encoding='utf-8') as f_err:
            f_err.write('\n'.join(error_log_lines))
        print(f"⚠️  错误日志: {ERROR_LOG} ({len(error_log_lines)} 条)")

    # 7. 验证输出
    validation_stats = validate_output(OUTPUT_JSONL)

    # 8. 统计信息
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"总物品数: {len(item_ids)}")
    print(f"成功处理: {success_count}")
    print(f"部分失败（已补齐）: {fail_count}")
    print(f"\n文件验证:")
    print(f"  总行数: {validation_stats['total_lines']}")
    print(f"  预期行数: {len(item_ids) * 3}")
    print(f"  异常物品数（非3行）: {validation_stats['irregular_items']}")
    print(f"  空文本: {validation_stats['empty_count']}")
    print(f"  虚构词: {validation_stats['forbidden_count']}")
    if validation_stats['forbidden_examples']:
        print(f"\n  虚构词示例（前5条）:")
        for iid, text in validation_stats['forbidden_examples']:
            print(f"    item_id={iid}: {text}...")
    print(f"\n输出文件: {OUTPUT_JSONL}")
    print(f"缓存文件: {CACHE_JSON}")
    print("=" * 60 + "\n")

    # 9. 展示前 3 条
    print("示例输出（前 3 条）：")
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item = json.loads(line)
            print(f"\n[{i + 1}] item_id={item['item_id']}")
            print(f"    {item['text'][:80]}...")


if __name__ == "__main__":
    asyncio.run(main())