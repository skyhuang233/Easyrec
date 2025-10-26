# -*- coding: utf-8 -*-
"""
用户画像多样化生成脚本（DeepSeek LLM）- 修正版

功能：
1. 读取训练/评测数据（train_data.csv, test_pos.csv）
2. 为每个用户基于最近 20 门课程生成 3 条不同表述的画像
3. 输出：
   - data/mooc/repr/user_profiles_train.div3.jsonl（训练集）
   - data/mooc/repr/user_profiles_eval.div3.jsonl（评测集）

修正点：
✓ 日期字段正确排序（pd.to_datetime）
✓ JSON 解析更鲁棒
✓ 长度/虚构词过滤
✓ 变体去重（Jaccard）
✓ 并发/超时可配置
✓ 顺序可复现（按 user_id 排序）
✓ 文件校验统计

安全与隐私：
- ✓ 训练多样化只读取 train_data.csv
- ✓ 评测多样化只读取 test_pos.csv（历史）
- ✓ 不修改原始画像文件

依赖：
pip install openai pandas tqdm tenacity

运行示例（WSL）：
export DEEPSEEK_API_KEY="your_key"
export DATA_ROOT="data/mooc"
export MAX_CONCURRENCY=5
cd /mnt/d/EasyRec-main/EasyRec-main
python scripts/diversify_user_profiles_llm.py
"""

import os
import json
import re
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm

# 导入 LLM 工具模块
import sys

sys.path.insert(0, os.path.dirname(__file__))
from llm_utils import batch_chat

# ============ 环境变量 ============
DATA_ROOT = os.environ.get("DATA_ROOT", "data/mooc")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROC_DIR = os.path.join(DATA_ROOT, "processed")
REPR_DIR = os.path.join(DATA_ROOT, "repr")

# 输入路径
COURSES_INFO_PATH = os.path.join(RAW_DIR, "courses_info_with_pre.csv")
TRAIN_DATA_PATH = os.path.join(PROC_DIR, "train_data.csv")
TEST_POS_PATH = os.path.join(PROC_DIR, "test_pos.csv")

# 输出路径
OUTPUT_TRAIN_JSONL = os.path.join(REPR_DIR, "user_profiles_train.div3.jsonl")
OUTPUT_EVAL_JSONL = os.path.join(REPR_DIR, "user_profiles_eval.div3.jsonl")
CACHE_TRAIN_JSON = os.path.join(REPR_DIR, "user_train_div.cache.json")
CACHE_EVAL_JSON = os.path.join(REPR_DIR, "user_eval_div.cache.json")
ERROR_LOG_TRAIN = os.path.join(REPR_DIR, "user_train_div.err.log")
ERROR_LOG_EVAL = os.path.join(REPR_DIR, "user_eval_div.err.log")

# 可配置参数
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "5"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))
TEMPERATURE = 0.2

# ============ LLM 提示词 ============
SYSTEM_PROMPT = """你是教育推荐系统的特征工程助手。回答需客观、精炼、无口语、禁止虚构。"""

# ============ 虚构词过滤 ============
FORBIDDEN_PATTERNS = [
    r'\d+学时', r'\d+小时', r'\d+课时', r'\d+门课',
    r'\d+学分', r'GPA', r'绩点',
    r'考试', r'作业', r'实验报告'
]


def contains_forbidden_content(text: str) -> bool:
    """检查是否包含虚构词"""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def clean_text(text: str, min_len: int = 25, max_len: int = 70) -> str:
    """
    文本清洗与规整
    """
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text.strip())

    # 检查虚构内容
    if contains_forbidden_content(text):
        return ""

    # 长度控制
    if len(text) < min_len:
        return ""
    if len(text) > max_len:
        text = text[:max_len - 1] + "。"

    return text


def build_user_prompt(titles_str: str, cats_str: str) -> str:
    """
    构造用户画像多样化提示词
    """
    prompt = f"""你是教育推荐系统的特征工程助手。根据"最近课程标题列表"和"Top-2 类别"，生成 3 条 30~60 字、不同表述的一句话学习者兴趣/能力摘要。避免口语，不复述课程名，禁止虚构。

输入：
- 最近课程（按时间）：{titles_str}
- 关注领域：{cats_str}

输出：仅输出 JSON 数组（例如：["...","...","..."]），不要输出其他多余文字或解释。"""
    return prompt


def fix_json_string(text: str) -> str:
    """修正常见的 JSON 格式错误"""
    text = text.replace('，', ',').replace('"', '"').replace('"', '"')
    text = text.replace('【', '[').replace('】', ']')
    text = text.replace('\n', ' ').replace('\r', '')
    return text


def parse_llm_response(response: str, fallback_count: int = 3) -> List[str]:
    """解析 LLM 返回的 JSON 数组"""
    response = fix_json_string(response)

    try:
        variants = json.loads(response)
        if isinstance(variants, list) and len(variants) > 0:
            variants = [str(v).strip() for v in variants if v]
            return variants
    except json.JSONDecodeError:
        pass

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

    return [response.strip()] if response.strip() else []


def jaccard_similarity(text1: str, text2: str) -> float:
    """计算 Jaccard 相似度（字级别）"""
    set1 = set(text1)
    set2 = set(text2)
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def deduplicate_variants(variants: List[str], threshold: float = 0.75) -> List[str]:
    """变体去重"""
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
                 fallback: str = "该学习者摘要：内容待完善。") -> List[str]:
    """补齐变体数量"""
    while len(variants) < target_count:
        variants.append(fallback)
    return variants[:target_count]


def format_user_text(summary: str, user_id: int) -> str:
    """格式化为最终的文本格式"""
    text = f"用户画像：{summary}\n用户ID：{user_id}"
    return text


def load_course_info() -> Dict[int, Tuple[str, str]]:
    """加载课程信息"""
    courses_df = pd.read_csv(COURSES_INFO_PATH)
    course_info = {}
    for _, row in courses_df.iterrows():
        course_id = int(row['id'])
        title = str(row.get('merged_title', 'Unknown'))
        category = str(row.get('category', 'Unknown'))
        course_info[course_id] = (title, category)
    return course_info


def build_user_meta(user_history: List[int], course_info: Dict,
                    max_courses: int = 20) -> Tuple[str, str]:
    """构建用户元信息"""
    recent_courses = user_history[-max_courses:] if len(user_history) > max_courses else user_history

    titles = []
    categories = []

    for cid in recent_courses:
        if cid in course_info:
            title, category = course_info[cid]
            titles.append(title)
            categories.append(category)

    titles_str = "、".join(titles[:10])
    if len(titles) > 10:
        titles_str += f"等{len(titles)}门"

    category_counter = Counter(categories)
    top_cats = [cat for cat, _ in category_counter.most_common(2)]
    cats_str = "、".join(top_cats) if top_cats else "综合"

    return titles_str, cats_str


def validate_output(output_jsonl: str) -> Dict:
    """验证输出文件"""
    print("\n📊 验证输出文件...")

    user_count = Counter()
    nan_count = 0
    empty_count = 0
    forbidden_count = 0
    forbidden_examples = []

    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                user = json.loads(line)
                user_id = user['user_id']
                text = user['text']

                user_count[user_id] += 1

                if pd.isna(text) or text is None:
                    nan_count += 1
                elif not text.strip():
                    empty_count += 1
                elif contains_forbidden_content(text):
                    forbidden_count += 1
                    if len(forbidden_examples) < 5:
                        forbidden_examples.append((user_id, text[:100]))
            except:
                continue

    irregular_users = {uid: cnt for uid, cnt in user_count.items() if cnt != 3}

    stats = {
        'total_lines': sum(user_count.values()),
        'total_users': len(user_count),
        'irregular_users': len(irregular_users),
        'nan_count': nan_count,
        'empty_count': empty_count,
        'forbidden_count': forbidden_count,
        'forbidden_examples': forbidden_examples
    }

    return stats


# ============ 处理单个数据集 ============
async def process_dataset(
        data_path: str,
        output_jsonl: str,
        cache_json: str,
        error_log: str,
        course_info: Dict,
        dataset_name: str
):
    """处理单个数据集"""
    print(f"\n{'=' * 60}")
    print(f"处理{dataset_name}数据集")
    print(f"{'=' * 60}")
    print(f"输入文件: {data_path}")
    print(f"输出文件: {output_jsonl}")
    print(f"缓存文件: {cache_json}")

    # 1. 读取交互数据
    print("\n📖 加载交互数据...")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"❌ 文件不存在: {data_path}")

    data_df = pd.read_csv(data_path)
    print(f"✓ 共 {len(data_df)} 条交互记录")

    # 修正点：日期字段正确排序
    print("📅 转换日期格式...")
    data_df['date'] = pd.to_datetime(data_df['date'], errors='coerce')

    # 删除日期缺失的行（如果有）
    before_len = len(data_df)
    data_df = data_df.dropna(subset=['date'])
    if len(data_df) < before_len:
        print(f"⚠️  删除 {before_len - len(data_df)} 条日期缺失记录")

    # 2. 按用户聚合历史（按日期排序）
    print("\n📊 聚合用户历史...")
    data_df = data_df.sort_values(['student_id', 'date'])
    user_history = data_df.groupby('student_id')['courses'].apply(list).to_dict()

    # 修正点：按 user_id 排序，确保顺序可复现
    user_ids = sorted(user_history.keys())
    print(f"✓ 共 {len(user_ids)} 个用户")

    # 3. 构造提示词
    print("\n📝 构造提示词...")
    prompts = []
    prompt_to_uid = {}

    for uid in user_ids:
        history = user_history[uid]
        titles_str, cats_str = build_user_meta(history, course_info, max_courses=20)
        prompt = build_user_prompt(titles_str, cats_str)
        prompts.append(prompt)
        prompt_to_uid[prompt] = uid

    print(f"✓ 共生成 {len(prompts)} 条提示词")

    # 4. 批量调用 LLM
    print("\n📡 调用 DeepSeek API...")
    results = await batch_chat(
        prompts=prompts,
        cache_json=cache_json,
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

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for prompt, response in tqdm(results.items(), desc="处理结果"):
            if prompt not in prompt_to_uid:
                continue

            uid = prompt_to_uid[prompt]

            # 解析变体
            variants = parse_llm_response(response)

            # 清洗文本
            cleaned_variants = []
            for v in variants:
                cleaned = clean_text(v, min_len=25, max_len=70)
                if cleaned:
                    cleaned_variants.append(cleaned)

            # 去重
            if len(cleaned_variants) > 1:
                cleaned_variants = deduplicate_variants(cleaned_variants, threshold=0.75)

            # 补齐到 3 条
            if len(cleaned_variants) < 3:
                fail_count += 1
                error_msg = f"user_id={uid}, valid_variants={len(cleaned_variants)}, raw={response[:100]}"
                error_log_lines.append(error_msg)

            cleaned_variants = pad_variants(cleaned_variants, target_count=3)
            success_count += 1

            # 写入 3 条变体
            for variant in cleaned_variants:
                text = format_user_text(variant, uid)
                profile = {"user_id": uid, "text": text}
                f_out.write(json.dumps(profile, ensure_ascii=False) + '\n')

    print(f"✓ 已保存: {output_jsonl}")

    # 6. 写错误日志
    if error_log_lines:
        with open(error_log, 'w', encoding='utf-8') as f_err:
            f_err.write('\n'.join(error_log_lines))
        print(f"⚠️  错误日志: {error_log} ({len(error_log_lines)} 条)")

    # 7. 验证输出
    validation_stats = validate_output(output_jsonl)

    # 8. 统计信息
    print("\n" + "=" * 60)
    print(f"{dataset_name}生成完成！")
    print("=" * 60)
    print(f"总用户数: {len(user_ids)}")
    print(f"成功处理: {success_count}")
    print(f"部分失败（已补齐）: {fail_count}")
    print(f"\n文件验证:")
    print(f"  总行数: {validation_stats['total_lines']}")
    print(f"  预期行数: {len(user_ids) * 3}")
    print(f"  异常用户数（非3行）: {validation_stats['irregular_users']}")
    print(f"  空文本: {validation_stats['empty_count']}")
    print(f"  虚构词: {validation_stats['forbidden_count']}")
    if validation_stats['forbidden_examples']:
        print(f"\n  虚构词示例（前5条）:")
        for uid, text in validation_stats['forbidden_examples']:
            print(f"    user_id={uid}: {text}...")
    print(f"\n输出文件: {output_jsonl}")
    print("=" * 60 + "\n")

    # 9. 展示前 3 条
    print(f"示例输出（前 3 条）：")
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            user = json.loads(line)
            print(f"\n[{i + 1}] user_id={user['user_id']}")
            print(f"    {user['text'][:80]}...")


# ============ 主流程 ============
async def main():
    print("\n" + "=" * 60)
    print("用户画像多样化生成（DeepSeek LLM）- 修正版")
    print("=" * 60)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"并发数: {MAX_CONCURRENCY}")
    print(f"请求超时: {REQUEST_TIMEOUT}s")
    print("安全约束：训练多样化不读取任何 eval/答案数据")
    print("=" * 60 + "\n")

    # 1. 加载课程信息
    print("📖 加载课程元数据...")
    course_info = load_course_info()
    print(f"✓ 共 {len(course_info)} 门课程")

    # 2. 处理训练集
    await process_dataset(
        data_path=TRAIN_DATA_PATH,
        output_jsonl=OUTPUT_TRAIN_JSONL,
        cache_json=CACHE_TRAIN_JSON,
        error_log=ERROR_LOG_TRAIN,
        course_info=course_info,
        dataset_name="训练集"
    )

    # 3. 处理评测集
    await process_dataset(
        data_path=TEST_POS_PATH,
        output_jsonl=OUTPUT_EVAL_JSONL,
        cache_json=CACHE_EVAL_JSON,
        error_log=ERROR_LOG_EVAL,
        course_info=course_info,
        dataset_name="评测集"
    )

    # 4. 总结
    print("\n" + "=" * 60)
    print("所有用户画像多样化生成完成！")
    print("=" * 60)
    print(f"训练集: {OUTPUT_TRAIN_JSONL}")
    print(f"评测集: {OUTPUT_EVAL_JSONL}")
    print(f"缓存文件: {CACHE_TRAIN_JSON}, {CACHE_EVAL_JSON}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

"""
快速检查脚本（验证生成结果）：
python - <<'PY'
import json
from collections import Counter

for p in ["data/mooc/repr/user_profiles_train.div3.jsonl",
          "data/mooc/repr/user_profiles_eval.div3.jsonl"]:
    print(f"\n{'='*60}\n{p}\n{'='*60}")

    # 统计每个用户的行数
    user_count = Counter()
    with open(p, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user_count[obj['user_id']] += 1

    print(f"总用户数: {len(user_count)}")
    print(f"总行数: {sum(user_count.values())}")
    irregular = {uid: cnt for uid, cnt in user_count.items() if cnt != 3}
    print(f"异常用户（非3行）: {len(irregular)}")
    if irregular:
        print(f"  示例: {list(irregular.items())[:5]}")

    # 展示前 3 条
    print("\n前 3 条:")
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(f"[{i+1}] {json.loads(line)}")
PY

合并基线与多样化（可选）：
# WSL:
cat data/mooc/repr/user_profiles_train.jsonl data/mooc/repr/user_profiles_train.div3.jsonl > data/mooc/repr/user_profiles_train.all.jsonl
cat data/mooc/repr/user_profiles_eval.jsonl data/mooc/repr/user_profiles_eval.div3.jsonl > data/mooc/repr/user_profiles_eval.all.jsonl

# Windows PowerShell:
Get-Content data/mooc/repr/user_profiles_train.jsonl, data/mooc/repr/user_profiles_train.div3.jsonl | Set-Content data/mooc/repr/user_profiles_train.all.jsonl
Get-Content data/mooc/repr/user_profiles_eval.jsonl, data/mooc/repr/user_profiles_eval.div3.jsonl | Set-Content data/mooc/repr/user_profiles_eval.all.jsonl
"""