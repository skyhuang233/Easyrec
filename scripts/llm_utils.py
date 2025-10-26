# -*- coding: utf-8 -*-
"""
LLM 工具模块 - DeepSeek/OpenAI 兼容接口
功能：
1. 异步并发批量请求（AsyncOpenAI）
2. 指数退避重试（tenacity）
3. JSON 缓存机制
4. 自动长度控制与回退
"""

import os
import json
import asyncio
from typing import List, Dict, Optional
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from tqdm.asyncio import tqdm


# ============ 环境变量读取 ============
def get_env(key: str, default: str = "") -> str:
    """安全读取环境变量"""
    return os.environ.get(key, default)


DEEPSEEK_API_KEY = get_env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL_CHAT = get_env("DEEPSEEK_MODEL_CHAT", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise ValueError("❌ 环境变量 DEEPSEEK_API_KEY 未设置，请先配置")

# ============ 异步客户端 ============
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=1790.0  # 接近 30 分钟
)


# ============ 重试装饰器 ============
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
async def _call_chat_once(
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int = 512
) -> str:
    """
    单次调用 DeepSeek Chat API（带重试）
    """
    try:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL_CHAT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=1790.0
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"⚠️  请求失败: {e}")
        raise


# ============ 缓存读写 ============
def load_cache(cache_json: str) -> Dict[str, str]:
    """加载缓存（不存在则返回空字典）"""
    if Path(cache_json).exists():
        with open(cache_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache_json: str, cache: Dict[str, str]):
    """保存缓存到 JSON"""
    Path(cache_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_json, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ============ 并发批量请求 ============
async def batch_chat(
        prompts: List[str],
        cache_json: str,
        system: str = "你是一个专业的数据分析助手。",
        temperature: float = 0.3,
        max_concurrency: int = 15,
        fallback_text: str = "内容待完善。"
) -> Dict[str, str]:
    """
    批量异步调用 DeepSeek Chat API（带缓存与并发控制）

    参数：
    - prompts: 用户提示列表
    - cache_json: 缓存文件路径（*.cache.json）
    - system: 系统提示词
    - temperature: 生成温度
    - max_concurrency: 最大并发数（默认 5）
    - fallback_text: 失败时的回退文本

    返回：
    - Dict[prompt, response] 字典
    """
    # 1. 加载缓存
    cache = load_cache(cache_json)

    # 2. 过滤已缓存的 prompts
    todo_prompts = [p for p in prompts if p not in cache]

    if not todo_prompts:
        print(f"✓ 所有 {len(prompts)} 条记录已缓存，无需重新请求")
        return cache

    print(f"📡 开始请求 {len(todo_prompts)} 条记录（并发={max_concurrency}）")

    # 3. 异步信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _fetch_one(prompt: str) -> tuple:
        """单条请求（带并发控制）"""
        async with semaphore:
            try:
                response = await _call_chat_once(prompt, system, temperature)
                if not response:
                    response = fallback_text
                return prompt, response
            except Exception as e:
                print(f"❌ 请求失败（使用回退）: {e}")
                return prompt, fallback_text

    # 4. 并发执行所有任务
    tasks = [_fetch_one(p) for p in todo_prompts]
    results = []

    # 使用 tqdm 显示进度
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="请求进度"):
        result = await coro
        results.append(result)

    # 5. 更新缓存
    for prompt, response in results:
        cache[prompt] = response

    save_cache(cache_json, cache)
    print(f"✓ 缓存已更新: {cache_json}")

    return cache


# ============ 长度控制工具 ============
def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本到指定长度（中文友好）"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def validate_length(text: str, min_len: int, max_len: int) -> bool:
    """验证文本长度是否在范围内"""
    return min_len <= len(text) <= max_len


# ============ 测试入口 ============
if __name__ == "__main__":
    async def _test():
        test_prompts = [
            "请用一句话介绍 Python 编程语言。",
            "请用一句话介绍机器学习。"
        ]

        results = await batch_chat(
            prompts=test_prompts,
            cache_json="test_cache.json",
            system="你是一个专业的技术顾问。",
            temperature=0.3,
            max_concurrency=2
        )

        for prompt, response in results.items():
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")


    asyncio.run(_test())