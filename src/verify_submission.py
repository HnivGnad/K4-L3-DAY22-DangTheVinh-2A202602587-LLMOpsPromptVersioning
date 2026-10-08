"""Audit bài nộp: chỉ báo các thiếu sót, không tạo evidence giả."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    """Cho phép bỏ screenshot theo yêu cầu, còn log/report vẫn bắt buộc."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-screenshots", action="store_true")
    args = parser.parse_args()
    problems = []
    tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.splitlines()
    if ".env" in tracked:
        problems.append(".env đang bị Git track")
    for filename in tracked:
        if filename == ".env.example" or not filename.endswith((".py", ".md", ".txt", ".json")):
            continue
        path = ROOT / filename
        if path.exists() and re.search(r"sk-[A-Za-z0-9_-]{20,}|lsv2_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{25,}",
                                      path.read_text(encoding="utf-8")):
            problems.append(f"Có chuỗi giống API key trong {filename} (không in giá trị)")
    required = ["02_ab_routing_log.txt", "03_ragas_report.json",
                "04_pii_demo_log.txt", "04_json_demo_log.txt"]
    if not args.skip_screenshots:
        required += ["01_langsmith_traces.png", "02_prompt_hub.png", "03_ragas_scores.png"]
    for filename in required:
        path = ROOT / "evidence" / filename
        if not path.exists() or not path.stat().st_size:
            problems.append(f"Thiếu evidence/{filename}")
    log_path = ROOT / "evidence" / "02_ab_routing_log.txt"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8")
        requests = re.findall(r"^\[\d+\] \[prompt-v[12]\] request_id=(req-\d+)", text, re.M)
        if len(requests) != 50 or len(set(requests)) != 50:
            problems.append("A/B log cần đúng 50 request ID riêng biệt")
        if not all(f"[prompt-{v}]" in text for v in ("v1", "v2")) or text.count("↓ Đã pull") < 2:
            problems.append("A/B log thiếu hai version hoặc bằng chứng pull Hub")
    report_path = ROOT / "evidence" / "03_ragas_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["sample_count_per_version"] != 50:
                problems.append("RAGAS không đủ 50 QA/version")
            for version in ("prompt_v1_scores", "prompt_v2_scores"):
                for metric in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
                    score = report[version][metric]
                    if not isinstance(score, (float, int)) or not math.isfinite(score):
                        problems.append(f"Điểm không hợp lệ: {version}/{metric}")
            if not report["target_met"]:
                problems.append("Faithfulness chưa đạt 0.8")
        except (ValueError, KeyError, TypeError):
            problems.append("Report sai cấu trúc/JSON")
    for filename, minimum in [("04_pii_demo_log.txt", 5), ("04_json_demo_log.txt", 4)]:
        path = ROOT / "evidence" / filename
        if path.exists() and len(re.findall(r"^\[.+\] PASS$", path.read_text(encoding="utf-8"), re.M)) < minimum:
            problems.append(f"{filename} thiếu demo PASS")
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        print("PASS: kiểm tra local bài nộp.")
    if args.skip_screenshots:
        print("Screenshot được bỏ qua; bổ sung ba ảnh trước khi nộp.")
    print("Cần kiểm tra thêm URL/project LangSmith và quyền truy cập khi nộp LMS.")
    return bool(problems)

if __name__ == "__main__":
    raise SystemExit(main())