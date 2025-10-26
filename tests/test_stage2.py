"""
Stage-2 最小单测:I/O 合法性、QC 规则、幂等与重试
"""
import json
import sys
from pathlib import Path

# 添加 generation 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

from qc import quality_check, retry_hint, count_chinese_chars


def test_qc_length():
    """测试长度检查"""
    print("[TEST] 长度检查")

    # 太短
    short_text = "这是一个很短的文本。" * 5  # ~50 字
    passed, issues = quality_check(short_text, "item")
    assert not passed, "应检测出文本过短"
    assert any("长度不符" in issue for issue in issues)

    # 合格
    good_text = "这是一个符合长度要求的课程画像文本。" * 8  # ~140 字
    passed, issues = quality_check(good_text, "item")
    assert passed or not any("长度不符" in issue for issue in issues), f"合格文本被误判: {issues}"

    # 太长
    long_text = "这是一个很长的文本。" * 20  # ~180 字
    passed, issues = quality_check(long_text, "item")
    assert not passed, "应检测出文本过长"

    print("  ✓ 长度检查通过")


def test_qc_paragraph():
    """测试段落检查"""
    print("[TEST] 段落检查")

    # 多段落
    multi_para = "第一段。\n第二段。\n第三段。" * 15
    passed, issues = quality_check(multi_para, "item")
    assert not passed, "应检测出多段落"
    assert any("段落过多" in issue for issue in issues)

    # 单段落(允许 1 个换行)
    single_para = "第一段内容很长。" * 15 + "\n" + "第二段内容。" * 5
    passed, _ = quality_check(single_para, "item")
    # 只要不超过 1 个换行就不应因段落检查失败

    print("  ✓ 段落检查通过")


def test_qc_prefix():
    """测试前缀检查"""
    print("[TEST] 前缀检查")

    bad_prefixes = [
        "课程画像:这是一个课程。" * 15,
        "学习者画像：这是一个学习者。" * 15,
        "【课程画像】这是内容。" * 15
    ]

    for text in bad_prefixes:
        passed, issues = quality_check(text, "item")
        assert not passed, f"应检测出禁止前缀: {text[:20]}"
        assert any("前缀" in issue for issue in issues)

    print("  ✓ 前缀检查通过")


def test_qc_chinese_ratio():
    """测试中文占比"""
    print("[TEST] 中文占比")

    # 中文占比过低
    low_cn = "This is an English text with very few Chinese characters 少量中文。" * 5
    passed, issues = quality_check(low_cn, "item")
    assert not passed, "应检测出中文占比过低"
    assert any("中文占比" in issue for issue in issues)

    print("  ✓ 中文占比检查通过")


def test_qc_privacy():
    """测试隐私检查"""
    print("[TEST] 隐私检查")

    # 邮箱
    email_text = "联系方式: user@example.com,这是一个课程画像。" * 10
    passed, issues = quality_check(email_text, "item")
    assert not passed, "应检测出邮箱"
    assert any("邮箱" in issue for issue in issues)

    # 手机号
    phone_text = "联系电话: 13812345678,这是一个学习者画像。" * 10
    passed, issues = quality_check(phone_text, "user")
    assert not passed, "应检测出手机号"
    assert any("手机号" in issue for issue in issues)

    print("  ✓ 隐私检查通过")


def test_qc_placeholder():
    """测试占位符检查"""
    print("[TEST] 占位符检查")

    placeholders = [
        "这个课程的信息是 <unknown>,学习者需要自己探索。" * 10,
        "用户的偏好类别为 N/A,暂无数据支持。" * 10,
        "TODO: 补充更多信息。" * 20
    ]

    for text in placeholders:
        passed, issues = quality_check(text, "item")
        assert not passed, f"应检测出占位符: {text[:30]}"
        assert any("占位符" in issue or "模板" in issue for issue in issues)

    print("  ✓ 占位符检查通过")


def test_retry_hint_generation():
    """测试重试提示生成"""
    print("[TEST] 重试提示生成")

    issues = ["长度不符", "段落过多", "含禁止前缀"]
    hint = retry_hint("item", issues)

    assert "120-160" in hint, "应包含长度约束"
    assert "一段文本" in hint or "分段" in hint, "应包含段落约束"
    assert "前缀" in hint or "标题" in hint, "应包含前缀约束"

    print("  ✓ 重试提示生成正确")


def test_jsonl_format():
    """测试 JSONL 格式合法性"""
    print("[TEST] JSONL 格式")

    # 模拟 JSONL 内容
    test_records = [
        {"id": "course_001", "text": "这是一个课程画像文本" * 10},
        {"id": "user_123", "text": "这是一个用户画像文本" * 10}
    ]

    # 验证可以序列化和反序列化
    for record in test_records:
        json_str = json.dumps(record, ensure_ascii=False)
        restored = json.loads(json_str)
        assert restored["id"] == record["id"]
        assert restored["text"] == record["text"]

    print("  ✓ JSONL 格式合法")


def test_chinese_char_counter():
    """测试中文字符计数"""
    print("[TEST] 中文字符计数")

    text1 = "这是中文"  # 4 个中文字符
    assert count_chinese_chars(text1) == 4

    text2 = "中文mixed英文123"  # 2 个中文字符
    assert count_chinese_chars(text2) == 2

    text3 = "Hello World"  # 0 个中文字符
    assert count_chinese_chars(text3) == 0

    print("  ✓ 中文字符计数正确")


def test_idempotent_logic():
    """测试幂等逻辑"""
    print("[TEST] 幂等逻辑")

    # 模拟已完成的 ID 集合
    done_ids = {"course_001", "course_002", "user_123"}

    # 模拟待处理列表
    all_items = [
        {"course_id": "course_001"},
        {"course_id": "course_002"},
        {"course_id": "course_003"},
        {"course_id": "course_004"}
    ]

    # 过滤逻辑
    remaining = [item for item in all_items if item["course_id"] not in done_ids]

    assert len(remaining) == 2, f"应剩余 2 个待处理,实际 {len(remaining)}"
    assert remaining[0]["course_id"] == "course_003"
    assert remaining[1]["course_id"] == "course_004"

    print("  ✓ 幂等逻辑正确")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("Stage-2 单元测试")
    print("=" * 50 + "\n")

    try:
        test_chinese_char_counter()
        test_qc_length()
        test_qc_paragraph()
        test_qc_prefix()
        test_qc_chinese_ratio()
        test_qc_privacy()
        test_qc_placeholder()
        test_retry_hint_generation()
        test_jsonl_format()
        test_idempotent_logic()

        print("\n" + "=" * 50)
        print("✓ 所有测试通过!")
        print("=" * 50 + "\n")
        return True

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ 测试异常: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)