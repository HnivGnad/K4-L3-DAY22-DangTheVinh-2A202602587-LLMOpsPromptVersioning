"""Xác minh evidence trên LangSmith server, không dựa vào log console."""
import config
from datetime import datetime, timezone
import json
from langsmith import Client
from utils.evidence import ROOT

def main():
    """Đếm root queries thành công và kiểm tra context/answer đã upload."""
    if not config.validate():
        raise SystemExit(1)
    client = Client(api_key=config.LANGSMITH_API_KEY)
    project = client.read_project(project_name=config.LANGSMITH_PROJECT)
    counts = {"rag-query": 0, "ab-rag-query": 0}
    with_context = dict.fromkeys(counts, 0)
    examples = {}
    for run in client.list_runs(project_name=config.LANGSMITH_PROJECT, is_root=True,
                                filter='or(eq(name, "rag-query"), eq(name, "ab-rag-query"))'):
        if run.name not in counts or run.error:
            continue
        counts[run.name] += 1
        outputs = run.outputs or {}
        context = outputs.get("context") or outputs.get("contexts")
        if context and outputs.get("answer"):
            with_context[run.name] += 1
            examples.setdefault(run.name, str(run.id))
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "project_name": config.LANGSMITH_PROJECT,
        "project_url": project.url,
        "successful_root_queries": counts,
        "queries_with_context_and_answer": with_context,
        "example_run_ids": examples,
        "target_met": all(with_context[name] >= 50 for name in counts),
    }
    (ROOT / "evidence" / "01_langsmith_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["target_met"]:
        raise SystemExit("Server chưa có đủ 50 query + context/answer ở mỗi bước.")

if __name__ == "__main__":
    main()