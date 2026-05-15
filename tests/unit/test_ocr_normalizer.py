"""OCRNormalizer 单元测试"""

import pytest
from app.parsers.ocr_parser import OCRNormalizer


class TestOCRNormalizer:
    """OCRNormalizer 测试类

    修订后的规则：只有当 l/O 前后都是数字时才转换
    - 12l3 → 1213 (l 前后都是数字)
    - l23 → l23 (l 前不是数字)
    - O123 → O123 (O 前不是数字)
    - 12O3 → 1203 (O 前后都是数字)
    """

    def test_l_between_digits(self):
        """测试 l 在两个数字之间时转换为 1"""
        assert OCRNormalizer.normalize("12l3") == "1213"
        assert OCRNormalizer.normalize("2l2") == "212"

    def test_l_at_start_of_number(self):
        """测试 l 在数字串开头时不转换"""
        assert OCRNormalizer.normalize("l23") == "l23"
        assert OCRNormalizer.normalize("l234") == "l234"
        assert OCRNormalizer.normalize("l2l") == "l2l"  # 第一个 l 前不是数字

    def test_l_at_end_of_number(self):
        """测试 l 在数字串结尾时不转换"""
        assert OCRNormalizer.normalize("123l") == "123l"

    def test_l_in_letter_context_unchanged(self):
        """测试 l 在字母上下文中不转换"""
        assert OCRNormalizer.normalize("label") == "label"
        assert OCRNormalizer.normalize("Hello") == "Hello"
        assert OCRNormalizer.normalize("linux") == "linux"
        assert OCRNormalizer.normalize("item1") == "item1"

    def test_O_between_digits(self):
        """测试 O 在两个数字之间时转换为 0"""
        assert OCRNormalizer.normalize("12O3") == "1203"
        assert OCRNormalizer.normalize("0O0") == "000"  # 0O0: O前后都是0(数字)

    def test_O_at_start_of_number(self):
        """测试 O 在数字串开头时不转换"""
        assert OCRNormalizer.normalize("O123") == "O123"
        assert OCRNormalizer.normalize("O234") == "O234"

    def test_O_in_letter_context_unchanged(self):
        """测试 O 在字母上下文中不转换"""
        assert OCRNormalizer.normalize("GB") == "GB"
        assert OCRNormalizer.normalize("OK") == "OK"
        assert OCRNormalizer.normalize("NO") == "NO"
        assert OCRNormalizer.normalize("BOOK") == "BOOK"
        assert OCRNormalizer.normalize("OO") == "OO"  # O前后都不是数字

    def test_O0O_combinations(self):
        """测试 O0O 类组合"""
        assert OCRNormalizer.normalize("O0O") == "O0O"  # O前后不是数字

    def test_multiple_spaces_normalized(self):
        """测试多余空格合并"""
        assert OCRNormalizer.normalize("键  宽   尺寸") == "键 宽 尺寸"
        assert OCRNormalizer.normalize("a  b  c") == "a b c"

    def test_leading_trailing_spaces_removed(self):
        """测试首尾空格去除（换行符保留）"""
        assert OCRNormalizer.normalize("  text  ") == "text"
        assert OCRNormalizer.normalize("text  ") == "text"
        assert OCRNormalizer.normalize("  text") == "text"

    def test_newline_with_indent_normalized(self):
        """测试换行符后带缩进的处理"""
        assert OCRNormalizer.normalize("line1\n  line2") == "line1\nline2"

    def test_mixed_ocr_errors(self):
        """测试混合 OCR 错误"""
        assert OCRNormalizer.normalize("GB/T 1568-2008  键") == "GB/T 1568-2008 键"

    def test_l2O0l2O0_original_issue(self):
        """原始问题: l2O0l2O0 转换验证

        正确理解：l2O0l2O0 经过 l->1 和 O->0 的连续转换后变为 l2001200
        - O 在 "12O3" 中前后都是数字 → O→0
        - l 在 "l2" 中前不是数字 → 不变
        """
        # l2O0l2O0 中：
        # - 第一个 l: 前不是数字，不转换
        # - O 在 2O0 中前后都是数字 (2和0)，O→0
        # - 第二个 l: 前不是数字 (O)，不转换
        # - 最后一个 O: 前是 2 (数字)，后是数字 → O→0
        result = OCRNormalizer.normalize("l2O0l2O0")
        # l2O0l2O0 → l2001200
        assert result == "l2001200", f"Expected l2001200, got {result}"

    def test_empty_string(self):
        """测试空字符串"""
        assert OCRNormalizer.normalize("") == ""
        assert OCRNormalizer.normalize("   ") == ""

    def test_digit_boundary_cases(self):
        """测试数字边界的边界情况"""
        assert OCRNormalizer.normalize("a1b") == "a1b"  # 1 前后都是字母，不转换
        assert OCRNormalizer.normalize("1a") == "1a"  # 1 前是数字，后是字母
        assert OCRNormalizer.normalize("a1") == "a1"  # 1 前是字母，后是数字


if __name__ == "__main__":
    pytest.main([__file__, "-v"])