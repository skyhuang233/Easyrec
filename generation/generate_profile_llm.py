"""
生成课程/用户画像(LLM 驱动)
支持并发、幂等、重试、质检
"""
import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from prompts_zh import (
    SYSTEM_ITEM_ZH, USER_ITEM_TEXT_ZH,
    SYSTEM_USER_ZH, USER_USER_TEXT_ZH
)
from qc import quality_check, retry_hint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--items_meta", type=str, help="课程元数据 JSON")
    p.add_argument("--users_meta", type=str, help="用户元数据 JSON")
    p.add_argument("--item_profile", type=str, help="已生成的课程画像 JSONL(用于用户画像)")
    p.add_argument("--out", type=str, required=True, help="输出 JSONL 路径")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--soft_timeout", type=int, default=600)
    p.add_argument("--force", type=str, default="False", help="是否强制重新生成")
    return p.parse_args()


def setup_logging():
    log_dir = Path("data/mooc/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "stage2_generation.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_done_ids(out_path: Path) -> set[str]:
    """加载已完成的 ID 集合"""
    if not out_path.exists():
        return set()

    done = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            done.add(str(obj["id"]))
    return done


def append_jsonl(path: Path, record: dict):
    """追加写入 JSONL"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_retry_queue(kind: str, item_id: str, reason: str, attempts: int):
    """记录失败队列"""
    retry_path = Path("data/mooc/logs/retry_queue.jsonl")
    retry_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "kind": kind,
        "id": item_id,
        "reason": reason,
        "attempts": attempts
    }
    append_jsonl(retry_path, record)


def summarize_item_brief(text: str, max_len_cn: int = 30) -> str:
    """截断课程画像为简短摘要"""
    import re
    cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(cn_chars) <= max_len_cn:
        return text
    return "".join(cn_chars[:max_len_cn]) + "..."


def build_item_prompt(meta: dict) -> tuple[str, str]:
    """构建课程画像 Prompt"""
    course_type_str = {
        0: "0=教师主导",
        1: "1=自主进度",
        -1: "未知"
    }.get(meta.get("course_type", -1), "未知")

    top_actions = ", ".join(meta.get("top_actions", [])) if meta.get("top_actions") else "无"
    keywords = ", ".join(meta.get("keywords", [])) if meta.get("keywords") else "无"

    user_msg = USER_ITEM_TEXT_ZH.format(
        title=meta.get("title", "未知"),
        category=meta.get("category", "未知"),
        course_type=course_type_str,
        start=meta.get("start", "未知"),
        end=meta.get("end", "未知"),
        top_actions=top_actions,
        keywords=keywords
    )

    return SYSTEM_ITEM_ZH, user_msg


def load_item_brief_map(item_profile_path: Path) -> dict[str, str]:
    """加载课程画像简短摘要映射"""
    if not item_profile_path or not item_profile_path.exists():
        return {}

    brief_map = {}
    with open(item_profile_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = str(obj["id"])
            text = obj["text"]
            brief_map[cid] = summarize_item_brief(text, 30)

    return brief_map


def build_user_prompt(meta: dict, item_brief_map: dict[str, str], courses_meta: dict) -> tuple[str, str]:
    """构建用户画像 Prompt"""
    demo = meta.get("demo", "信息有限")
    prefcats = ", ".join(meta.get("prefcats", [])) if meta.get("prefcats") else "无"
    early_behaviors = ", ".join(meta.get("early_behaviors", [])) if meta.get("early_behaviors") else "无"

    # 构建课程采样摘要
    sampled_cids = meta.get("sampled_cids", [])
    sampled_briefs = []
    for cid in sampled_cids[:5]:
        brief = item_brief_map.get(str(cid))
        if brief:
            sampled_briefs.append(f"{cid}: {brief}")
        else:
            # 回退:使用原始元数据
            if str(cid) in courses_meta:
                c = courses_meta[str(cid)]
                fallback = f"{c.get('title', cid)} ({c.get('category', '未知')})"
                sampled_briefs.append(fallback)
            else:
                sampled_briefs.append(str(cid))

    sampled_str = "; ".join(sampled_briefs) if sampled_briefs else "无"

    user_msg = USER_USER_TEXT_ZH.format(
        demo=demo,
        pref_categories=prefcats,
        early_behaviors=early_behaviors,
        sampled_items_with_brief=sampled_str
    )

    return SYSTEM_USER_ZH, user_msg


def generate_one_profile(
    client: LLMClient,
    kind: str,
    item_id: str,
    system_msg: str,
    user_msg: str,
    max_retries: int = 2
) -> tuple[str, dict | None, str]:
    """
    生成单条画像

    Returns:
        (text, usage, error_msg)
    """
    temperature_base = 0.35

    for attempt in range(max_retries + 1):
        try:
            # 增加温度随机性
            temp = temperature_base + random.uniform(-0.05, 0.05)

            text, usage = client.chat(
                system=system_msg,
                user=user_msg,
                temperature=temp,
                max_tokens=360
            )

            # 质检
            passed, issues = quality_check(text, kind)

            if passed:
                return text, usage, ""

            # 质检失败,生成重试提示
            if attempt < max_retries:
                hint = retry_hint(kind, issues)
                user_msg += hint
                logging.warning(f"[{kind}:{item_id}] 质检失败(尝试 {attempt+1}/{max_retries}): {issues}")
                time.sleep(1)
                continue
            else:
                return "", None, f"质检失败: {', '.join(issues)}"

        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"[{kind}:{item_id}] API 错误(尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
                continue
            else:
                return "", None, f"API 错误: {str(e)}"

    return "", None, "未知错误"


def process_item_profile(
    client: LLMClient,
    meta: dict,
    out_path: Path,
    logger: logging.Logger
) -> tuple[bool, dict | None]:
    """处理单个课程画像"""
    item_id = meta.get("course_id", meta.get("cid", "unknown"))

    try:
        system_msg, user_msg = build_item_prompt(meta)
        text, usage, error = generate_one_profile(client, "item", item_id, system_msg, user_msg)

        if error:
            logger.error(f"课程 {item_id} 生成失败: {error}")
            log_retry_queue("item", item_id, error, 1)
            return False, None

        record = {"id": item_id, "text": text}
        append_jsonl(out_path, record)
        logger.info(f"课程 {item_id} 完成")

        return True, usage

    except Exception as e:
        logger.error(f"课程 {item_id} 处理异常: {e}")
        log_retry_queue("item", item_id, str(e), 1)
        return False, None


def process_user_profile(
    client: LLMClient,
    meta: dict,
    item_brief_map: dict,
    courses_meta: dict,
    out_path: Path,
    logger: logging.Logger
) -> tuple[bool, dict | None]:
    """处理单个用户画像"""
    uid = meta.get("uid", "unknown")

    try:
        system_msg, user_msg = build_user_prompt(meta, item_brief_map, courses_meta)
        text, usage, error = generate_one_profile(client, "user", uid, system_msg, user_msg)

        if error:
            logger.error(f"用户 {uid} 生成失败: {error}")
            log_retry_queue("user", uid, error, 1)
            return False, None

        record = {"id": uid, "text": text}
        append_jsonl(out_path, record)
        logger.info(f"用户 {uid} 完成")

        return True, usage

    except Exception as e:
        logger.error(f"用户 {uid} 处理异常: {e}")
        log_retry_queue("user", uid, str(e), 1)
        return False, None


def main():
    args = parse_args()
    logger = setup_logging()

    force = args.force.lower() == "true"
    out_path = Path(args.out)

    # 初始化客户端
    client = LLMClient()

    # 确定模式
    if args.items_meta:
        kind = "item"
        logger.info(f"[模式] 生成课程画像")

        with open(args.items_meta, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        # 加载已完成
        if not force:
            done_ids = load_done_ids(out_path)
            logger.info(f"已完成 {len(done_ids)} 个课程,跳过")
            meta_list = [m for m in meta_list if str(m.get("course_id", m.get("cid"))) not in done_ids]

        logger.info(f"待处理: {len(meta_list)} 个课程")

        # 并发处理
        success_count = 0
        fail_count = 0
        total_usage = defaultdict(int)

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_item_profile, client, meta, out_path, logger): meta
                for meta in meta_list
            }

            for future in as_completed(futures):
                success, usage = future.result()
                if success:
                    success_count += 1
                    if usage:
                        for k, v in usage.items():
                            total_usage[k] += v
                else:
                    fail_count += 1

        logger.info(f"\n=== 完成 ===")
        logger.info(f"成功: {success_count}, 失败: {fail_count}")
        if total_usage:
            logger.info(f"Token 用量: {dict(total_usage)}")

    elif args.users_meta:
        kind = "user"
        logger.info(f"[模式] 生成用户画像")

        with open(args.users_meta, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        # 加载课程画像摘要
        item_profile_path = Path(args.item_profile) if args.item_profile else None
        item_brief_map = load_item_brief_map(item_profile_path)
        logger.info(f"加载 {len(item_brief_map)} 个课程画像摘要")

        # 加载课程元数据(用于回退)
        courses_meta = {}
        if item_profile_path:
            items_meta_path = item_profile_path.parent.parent / "processed" / "items_meta.json"
            if items_meta_path.exists():
                with open(items_meta_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    courses_meta = {str(i.get("course_id", i.get("cid"))): i for i in items}

        # 加载已完成
        if not force:
            done_ids = load_done_ids(out_path)
            logger.info(f"已完成 {len(done_ids)} 个用户,跳过")
            meta_list = [m for m in meta_list if str(m.get("uid")) not in done_ids]

        logger.info(f"待处理: {len(meta_list)} 个用户")

        # 并发处理
        success_count = 0
        fail_count = 0
        total_usage = defaultdict(int)

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_user_profile, client, meta, item_brief_map, courses_meta, out_path, logger
                ): meta
                for meta in meta_list
            }

            for future in as_completed(futures):
                success, usage = future.result()
                if success:
                    success_count += 1
                    if usage:
                        for k, v in usage.items():
                            total_usage[k] += v
                else:
                    fail_count += 1

        logger.info(f"\n=== 完成 ===")
        logger.info(f"成功: {success_count}, 失败: {fail_count}")
        if total_usage:
            logger.info(f"Token 用量: {dict(total_usage)}")

    else:
        logger.error("必须指定 --items_meta 或 --users_meta")
        return


if __name__ == "__main__":
    main()