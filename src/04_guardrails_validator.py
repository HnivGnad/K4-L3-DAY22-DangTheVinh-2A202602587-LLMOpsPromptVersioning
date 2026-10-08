"""Bước 4: tự viết PII detector và JSON repair bằng Guardrails FIX."""
import ast
import json
import re
from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from guardrails.types import OnFailAction
from utils.evidence import evidence_log

@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Redact bằng regex; ưu tiên số thẻ để tránh match một phần thành phone."""
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "CREDIT_CARD": r"(?<![\d-])(?:\d{16}|\d{4}(?P<card_sep>[- ])\d{4}(?P=card_sep)\d{4}(?P=card_sep)\d{4})(?![\d-])",
        "SSN": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
        "PHONE": r"(?<!\w)(?:(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}|(?:\+84|0)[35789]\d{8})(?!\w)",
    }

    def validate(self, value: str, metadata: dict):
        """Trả FailResult.fix_value khi phát hiện PII, PassResult khi sạch."""
        redacted = value
        found = []
        for kind, pattern in self.PII_PATTERNS.items():
            redacted, count = re.subn(pattern, f"[{kind}_REDACTED]", redacted)
            if count:
                found.append(f"{kind}={count}")
        if found:
            return FailResult(error_message="Phát hiện PII: " + ", ".join(found), fix_value=redacted)
        return PassResult()

@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Gỡ fences, sửa string nháy đơn và dấu phẩy thừa ngoài string."""
    @staticmethod
    def _repair(text: str) -> str:
        """Quét string để không phá apostrophe hay chuỗi chứa ',}'."""
        text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        output, i = [], 0
        while i < len(text):
            char = text[i]
            if char in ("'", '"'):
                start, quote = i, char
                i += 1
                while i < len(text):
                    if text[i] == "\\":
                        i += 2
                        continue
                    if text[i] == quote:
                        i += 1
                        break
                    i += 1
                token = text[start:i]
                if quote == "'":
                    try:
                        token = json.dumps(ast.literal_eval(token), ensure_ascii=False)
                    except (ValueError, SyntaxError):
                        pass
                output.append(token)
            elif char == ",":
                next_nonspace = i + 1
                while next_nonspace < len(text) and text[next_nonspace].isspace():
                    next_nonspace += 1
                if next_nonspace >= len(text) or text[next_nonspace] not in "}]":
                    output.append(char)
                i += 1
            else:
                output.append(char)
                i += 1
        return "".join(output).strip()

    @staticmethod
    def _parse(text):
        """Từ chối các hằng NaN/Infinity không thuộc JSON chuẩn."""
        def reject_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")
        return json.loads(text, parse_constant=reject_constant)

    def validate(self, value: str, metadata: dict):
        """JSON hợp lệ giữ nguyên; lỗi được FIX hoặc thay bằng JSON dự phòng."""
        try:
            self._parse(value)
            return PassResult()
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            parsed = self._parse(self._repair(value))
            return FailResult(error_message="JSON lỗi, đã tự sửa",
                              fix_value=json.dumps(parsed, ensure_ascii=False, indent=2, allow_nan=False))
        except (ValueError, json.JSONDecodeError):
            fallback = json.dumps({"error": "Không thể phân tích JSON", "raw": value[:200]}, ensure_ascii=False)
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)

def demo_pii_guard():
    """Demo 7 trường hợp với dữ liệu giả; kiểm tra output thực sự bị che."""
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    cases = [
        ("Email", "Contact john.doe@example.com.", "Contact [EMAIL_REDACTED]."),
        ("Phone", "Call (555) 867-5309.", "Call [PHONE_REDACTED]."),
        ("SSN", "SSN: 123-45-6789", "SSN: [SSN_REDACTED]"),
        ("Credit Card", "Card: 4532 1234 5678 9010", "Card: [CREDIT_CARD_REDACTED]"),
        ("Multi-PII", "alice@example.com / 555-123-4567", "[EMAIL_REDACTED] / [PHONE_REDACTED]"),
        ("Vietnam phone", "Phone: 0901234567", "Phone: [PHONE_REDACTED]"),
        ("Clean", "No sensitive information here.", "No sensitive information here."),
    ]
    print("Demo: PII Detection & Redaction (synthetic data)")
    for label, text, expected in cases:
        result = guard.validate(text)
        if result.validated_output != expected:
            raise AssertionError(f"PII demo failed: {label}")
        print(f"\n[{label}] PASS\n  Input: {text}\n  Output: {result.validated_output}")

def demo_json_guard():
    """Demo đủ các lỗi yêu cầu, cùng edge case để tránh sửa sai nội dung string."""
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```', {"name": "Bob"}),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}", {"name": "Charlie", "score": 95}),
        ("Trailing comma", '{"key": "value",}', {"key": "value"}),
        ("Nested arrays", "{'items': [1, 2,],}", {"items": [1, 2]}),
        ("Preserve strings", """{"text": "don't change ,}",}""", {"text": "don't change ,}"}),
        ("Truly invalid", "This is not JSON at all: ??? {]", None),
    ]
    print("Demo: JSON Formatting & Repair")
    for label, text, expected in cases:
        result = guard.validate(text)
        parsed = json.loads(result.validated_output)
        if expected is not None and parsed != expected:
            raise AssertionError(f"JSON demo failed: {label}")
        if expected is None and "error" not in parsed:
            raise AssertionError("Missing fallback error")
        print(f"\n[{label}] PASS\n  Input: {text}\n  Output: {result.validated_output}")

def main():
    """Lưu hai demo riêng vào evidence từ cùng lần chạy thật."""
    with evidence_log("04_pii_demo_log.txt"):
        demo_pii_guard()
    with evidence_log("04_json_demo_log.txt"):
        demo_json_guard()
    print("✅ Bước 4 hoàn thành: PII redacted, JSON hợp lệ hoặc fallback.")

if __name__ == "__main__":
    main()