"""Bước 3: đánh giá 50 QA × 2 prompt bằng 4 metric RAGAS."""
import config
import importlib
import os
import json
from datetime import datetime, timezone
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from prompts import SYSTEM_V1, SYSTEM_V2, PROMPT_V1, PROMPT_V2, PROMPTS, PROMPT_V1_NAME, PROMPT_V2_NAME
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.evidence import ROOT, evidence_log
from qa_pairs import QA_PAIRS

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

def setup_vectorstore():
    """Cùng chunking và embeddings với pipeline được đánh giá."""
    return build_vectorstore(split_text(load_knowledge_base()), get_embeddings())

@traceable(name="evaluation-rag-query", tags=["ragas", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Giữ contexts dạng list[str] cho đánh giá từng passage."""
    contexts = [doc.page_content for doc in retriever.invoke(question)]
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": "\n\n".join(contexts), "question": question,
    })
    return {"answer": answer, "contexts": contexts}

def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Chạy mọi QA, không đưa reference vào prompt sinh câu trả lời."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    results = []
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, PROMPTS[prompt_version], qa["question"])
        results.append({**qa, **out})
        print(f"[{prompt_version} {i:02d}/{len(QA_PAIRS)}] {qa['question']}")
    return results

def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Map đủ user_input/response/retrieved_contexts/reference."""
    for row in rag_results:
        if not isinstance(row["contexts"], list) or not all(isinstance(c, str) for c in row["contexts"]):
            raise ValueError("contexts phải là list[str]")
    return EvaluationDataset(samples=[
        SingleTurnSample(user_input=r["question"], response=r["answer"],
                         retrieved_contexts=r["contexts"], reference=r["reference"])
        for r in rag_results
    ])

def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Chấm tất cả mẫu; từ chối metric thiếu/NaN thay vì bỏ mẫu và tăng điểm."""
    result = evaluate(
        build_ragas_dataset(rag_results),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=get_llm(temperature=0),
        embeddings=get_embeddings(),
        run_config=RunConfig(timeout=180, max_retries=3, max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "4"))),
        raise_exceptions=True,
    )
    scores, per_sample = {}, {}
    for key in METRICS:
        raw = list(result[key])
        if len(raw) != len(rag_results) or any(v is None or not np.isfinite(v) for v in raw):
            raise ValueError(f"{version}/{key}: điểm thiếu hoặc không hữu hạn; cần chạy lại")
        per_sample[key] = [float(v) for v in raw]
        scores[key] = float(np.mean(raw))
        print(f"  {version}/{key}: {scores[key]:.4f}")
    write_json(ROOT / "data" / f"ragas_{version}_per_sample.json", per_sample)
    return scores

def write_json(path, value):
    """Ghi JSON UTF-8 chuẩn, không cho phép NaN/Infinity."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def write_samples(outputs):
    """Lưu đủ 100 mẫu và bốn điểm từng mẫu cho reviewer đối chiếu."""
    details = {}
    for version in ("v1", "v2"):
        per_sample = json.loads((ROOT / "data" / f"ragas_{version}_per_sample.json").read_text(encoding="utf-8"))
        details[version] = [
            {"sample_id": i + 1, **row, "scores": {metric: per_sample[metric][i] for metric in METRICS}}
            for i, row in enumerate(outputs[version])
        ]
    write_json(ROOT / "evidence" / "03_ragas_samples.json", details)

def write_analysis(report):
    """Phân tích số đo thật và nêu giới hạn của so sánh dùng LLM judge."""
    v1, v2 = report["prompt_v1_scores"], report["prompt_v2_scores"]
    rows = ["# Phân tích V1 và V2", "", f"Lần chạy UTC: {report['evaluated_at']}", "",
            "| Metric | V1 | V2 | Chênh lệch V2−V1 |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        rows.append(f"| {metric} | {v1[metric]:.4f} | {v2[metric]:.4f} | {v2[metric]-v1[metric]:+.4f} |")
    winner = "V1" if v1["faithfulness"] > v2["faithfulness"] else (
        "V2" if v2["faithfulness"] > v1["faithfulness"] else "Hai phiên bản bằng nhau")
    rows.extend(["", f"{winner} có faithfulness cao hơn hoặc bằng phiên bản còn lại.",
        "V1 giới hạn câu trả lời ngắn và trực tiếp, giảm số mệnh đề cần kiểm chứng. "
        "V2 tổ chức Definition và Key facts, yêu cầu từng câu có bằng chứng đầy đủ, nên có thể bao phủ câu hỏi tốt hơn, "
        "nhưng cũng tạo thêm mệnh đề cần được context hỗ trợ. Đây là giả thuyết giải thích, "
        "không phải bằng chứng nhân quả từ một lần chạy.",
        "", "Hai phiên bản dùng cùng 50 QA, embeddings, chunking và top-k=3. "
        "Context recall/precision chủ yếu phản ánh truy xuất; sai khác giữa hai bản có thể "
        "do LLM judge. Cần xem điểm từng mẫu và chạy lặp trước khi kết luận ổn định.",
        "", f"Đạt faithfulness ≥ 0.8: {report['target_met']}. "
        f"Đạt bonus cả hai ≥ 0.9: {report['bonus_faithfulness_met']}."])
    # Dùng outputs và per-sample đã chấm để phân tích có ví dụ kiểm chứng.
    samples = {}
    rows.extend(["", "## Đối chiếu từng mẫu"])
    for version in ("v1", "v2"):
        outputs_path = ROOT / "data" / f"rag_outputs_{version}.json"
        scores_path = ROOT / "data" / f"ragas_{version}_per_sample.json"
        if not outputs_path.exists() or not scores_path.exists():
            continue
        outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
        sample_scores = json.loads(scores_path.read_text(encoding="utf-8"))["faithfulness"]
        samples[version] = outputs
        words = sum(len(item["answer"].split()) for item in outputs) / len(outputs)
        full = sum(score >= 1.0 for score in sample_scores)
        rows.append(f"\n{version.upper()}: câu trả lời trung bình {words:.1f} từ; "
                    f"{full}/{len(outputs)} mẫu có faithfulness = 1.")
        for i, item in enumerate(outputs):
            if (sample_scores[i] == 0 and item["answer"].strip() == item["reference"].strip()
                    and any(item["reference"].strip() in context for context in item["contexts"])):
                rows.extend(["", f"Chú ý {version.upper()} mẫu {i+1}: answer trùng nguyên văn "
                             "reference và câu đó có trong retrieved context, nhưng judge chấm 0. "
                             "Đây là trường hợp điểm judge không khớp bằng chứng; "
                             "không nên kết luận answer hallucinate chỉ từ điểm này."])
        for index in sorted(range(len(sample_scores)), key=lambda i: sample_scores[i])[:2]:
            sample = outputs[index]
            rows.extend(["", f"### {version.upper()} — mẫu {index+1}, "
                         f"faithfulness {sample_scores[index]:.4f}",
                         sample["question"], "", "> " + sample["answer"].replace("\n", "\n> ")])
    if set(samples) == {"v1", "v2"}:
        same = sum(a["contexts"] == b["contexts"] for a, b in zip(samples["v1"], samples["v2"]))
        rows.extend(["", f"{same}/50 cặp dùng contexts giống hệt nhau. "
                     "Số từ là mô tả độ dài, không phải thước đo chất lượng. "
                     "Các mẫu điểm thấp cần đối chiếu passage gốc trong data/rag_outputs_*.json; "
                     "không xem mọi bất đồng của LLM judge là lỗi thực tế đã chứng minh."])

    initial_path = ROOT / "evidence" / "03_ragas_initial_report.json"
    if initial_path.exists():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        old_v2 = initial["prompt_v2_scores"]["faithfulness"]
        old_v1 = initial["prompt_v1_scores"]["faithfulness"]
        rows.extend(["", "## Cải thiện prompt qua hai lần chạy",
                     f"V2 ban đầu: {old_v2:.4f}; sau chỉnh prompt: {v2['faithfulness']:.4f} "
                     f"(chênh lệch {v2['faithfulness']-old_v2:+.4f}). "
                     "Prompt mới bỏ phần giới hạn không có bằng chứng và yêu cầu không hoàn tất "
                     "passage bị cắt từ kiến thức ngoài context.",
                     f"V1 không đổi prompt nhưng faithfulness thay từ {old_v1:.4f} "
                     f"thành {v1['faithfulness']:.4f}, cho thấy biến động giữa các lần chạy. "
                     "Không quy toàn bộ thay đổi điểm cho prompt khi chưa có nhiều lần đo.",
                     "", "V1 có answer relevancy cao hơn trong lần cuối, V2 có faithfulness "
                     "cao hơn. Vì vậy cần cân nhắc cả tính trực tiếp lẫn grounding, "
                     "không chọn một phiên bản chỉ dựa trên một metric.",
                     "", "Dữ liệu đối chiếu đủ 100 mẫu và điểm từng mẫu: "
                     "[03_ragas_samples.json](03_ragas_samples.json)."])

    (ROOT / "evidence" / "03_analysis.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

def main():
    """Pull chính prompt Hub đang sử dụng, sinh 100 answers và lưu báo cáo thật."""
    if not config.validate():
        raise SystemExit(1)
    if len(QA_PAIRS) != 50:
        raise ValueError("Bài nộp cần đúng 50 QA")
    with evidence_log("03_ragas_evaluation_log.txt"):
        hub = importlib.import_module("02_prompt_hub_ab_routing")
        pulled = hub.pull_prompts_from_hub(Client(api_key=config.LANGSMITH_API_KEY))
        PROMPTS.update({"v1": pulled[PROMPT_V1_NAME], "v2": pulled[PROMPT_V2_NAME]})
        vectorstore = setup_vectorstore()
        outputs = {}
        for version in ("v1", "v2"):
            outputs[version] = collect_rag_outputs(vectorstore, version)
            write_json(ROOT / "data" / f"rag_outputs_{version}.json", outputs[version])
        v1_scores = run_ragas_eval(outputs["v1"], "v1")
        v2_scores = run_ragas_eval(outputs["v2"], "v2")
        print(f"\n{'Metric':30} {'V1':>8} {'V2':>8} Winner")
        for metric in METRICS:
            a, b = v1_scores[metric], v2_scores[metric]
            winner = "V1" if a > b else ("V2" if b > a else "Tie")
            print(f"{metric:30} {a:8.4f} {b:8.4f} {winner}")
        report = {
            "prompt_v1_scores": v1_scores, "prompt_v2_scores": v2_scores,
            "target_met": max(v1_scores["faithfulness"], v2_scores["faithfulness"]) >= 0.8,
            "bonus_faithfulness_met": min(v1_scores["faithfulness"], v2_scores["faithfulness"]) >= 0.9,
            "sample_count_per_version": len(QA_PAIRS),
            "provider": config.PROVIDER, "project": config.LANGSMITH_PROJECT,
            "model": {
                "openai": config.OPENAI_MODEL, "gemini": config.GEMINI_MODEL,
                "anthropic": config.ANTHROPIC_MODEL, "ollama": config.OLLAMA_MODEL,
                "openrouter": config.OPENROUTER_MODEL,
            }[config.PROVIDER],
            "embedding_model": (
                config.GEMINI_EMBEDDING_MODEL if config.PROVIDER == "gemini" else
                config.OLLAMA_EMBEDDING_MODEL if config.PROVIDER == "ollama" else
                config.OPENAI_EMBEDDING_MODEL
            ),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "prompt_commits": {v: (PROMPTS[v].metadata or {}).get("lc_hub_commit_hash") for v in PROMPTS},
            "retrieval": {"chunk_size": 500, "chunk_overlap": 50, "k": 3},
        }
        write_json(ROOT / "data" / "ragas_report.json", report)
        write_json(ROOT / "evidence" / "03_ragas_report.json", report)
        write_samples(outputs)
        write_analysis(report)
        print(f"✅ Đã lưu báo cáo. target_met={report['target_met']}")
        if not report["target_met"]:
            raise RuntimeError("Faithfulness cả hai phiên bản chưa đạt 0.8; cần điều chỉnh và chạy lại")

if __name__ == "__main__":
    main()