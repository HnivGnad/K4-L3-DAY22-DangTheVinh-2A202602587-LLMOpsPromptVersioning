"""Kiểm tra local: không gọi provider và không tạo evidence giả."""
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS
from guardrails import Guard
from guardrails.types import OnFailAction
from utils.data_loader import load_knowledge_base, split_text
from qa_pairs import QA_PAIRS, SAMPLE_QUESTIONS
from prompts import PROMPT_V1, PROMPT_V2, PROMPT_V1_NAME, PROMPT_V2_NAME
step1 = importlib.import_module("01_langsmith_rag_pipeline")
step2 = importlib.import_module("02_prompt_hub_ab_routing")
step3 = importlib.import_module("03_ragas_evaluation")
step4 = importlib.import_module("04_guardrails_validator")

class PipelineTests(unittest.TestCase):
    def test_dataset_and_chunking(self):
        self.assertEqual(len(QA_PAIRS), 50)
        self.assertEqual(SAMPLE_QUESTIONS, [qa["question"] for qa in QA_PAIRS])
        chunks = split_text(load_knowledge_base())
        self.assertGreater(len(chunks), 50)
        self.assertTrue(all(0 < len(c) <= 500 for c in chunks))

    def test_real_faiss_and_lcel_preserve_context(self):
        store = FAISS.from_texts(["retrieval evidence one", "retrieval evidence two"],
                                DeterministicFakeEmbedding(size=16))
        with patch.object(step1, "get_llm", return_value=FakeListChatModel(responses=["grounded answer"])):
            chain, _ = step1.build_rag_chain(store)
            out = step1.ask(chain, "a question")
        self.assertEqual(out["question"], "a question")
        self.assertEqual(out["answer"], "grounded answer")
        self.assertIn("retrieval evidence", out["context"])

    def test_routing_stability_and_distribution(self):
        requests = [f"req-{i:04d}" for i in range(50)]
        first = [step2.get_prompt_version(r) for r in requests]
        self.assertEqual(first, [step2.get_prompt_version(r) for r in requests])
        self.assertEqual(set(first), {PROMPT_V1_NAME, PROMPT_V2_NAME})

    def test_prompt_contract(self):
        self.assertEqual(set(PROMPT_V1.input_variables), {"context", "question"})
        self.assertEqual(set(PROMPT_V2.input_variables), {"context", "question"})
        self.assertNotEqual(PROMPT_V1.messages[0].prompt.template, PROMPT_V2.messages[0].prompt.template)

    def test_hub_failure_is_not_silent(self):
        class UnavailableHub:
            def pull_prompt(self, name):
                raise ConnectionError("offline")
        with self.assertRaises(RuntimeError):
            step2.pull_prompts_from_hub(UnavailableHub())
        fallback = step2.pull_prompts_from_hub(UnavailableHub(), allow_local_fallback=True)
        self.assertEqual(set(fallback), {PROMPT_V1_NAME, PROMPT_V2_NAME})

    def test_ragas_schema(self):
        rows = [{**qa, "answer": "local fixture", "contexts": ["local context"]} for qa in QA_PAIRS]
        dataset = step3.build_ragas_dataset(rows)
        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.samples[0].retrieved_contexts, ["local context"])
        self.assertEqual(dataset.samples[0].reference, QA_PAIRS[0]["reference"])
        with self.assertRaises(ValueError):
            step3.build_ragas_dataset([{**rows[0], "contexts": "wrong type"}])

    def test_nan_scores_rejected(self):
        rows = [{"question": "q", "reference": "r", "answer": "a", "contexts": ["c"]}]
        with patch.object(step3, "get_llm"), patch.object(step3, "get_embeddings"),              patch.object(step3, "evaluate", return_value={"faithfulness": [float("nan")]}):
            with self.assertRaises(ValueError):
                step3.run_ragas_eval(rows, "v1")

class ValidatorTests(unittest.TestCase):
    def test_guard_fixes_overlapping_pii_and_phone_parenthesis(self):
        guard = Guard().use(step4.PIIDetector(on_fail=OnFailAction.FIX))
        text = "a@example.com a@example.com +1 (555) 123-4567 123-45-6789 4532-1234-5678-9010"
        output = guard.validate(text).validated_output
        self.assertEqual(output, "[EMAIL_REDACTED] [EMAIL_REDACTED] [PHONE_REDACTED] [SSN_REDACTED] [CREDIT_CARD_REDACTED]")
        clean = "Version 1.2, no sensitive data."
        self.assertEqual(guard.validate(clean).validated_output, clean)

    def test_json_preserves_apostrophe_comma_and_boolean(self):
        import json
        guard = Guard().use(step4.JSONFormatter(on_fail=OnFailAction.FIX))
        text = """{'message': "don't change ,}", 'enabled': true, 'items': [1, 2,],}"""
        output = guard.validate(text).validated_output
        self.assertEqual(json.loads(output), {"message": "don't change ,}", "enabled": True, "items": [1, 2]})

    def test_json_valid_unchanged_and_invalid_fallback(self):
        import json
        guard = Guard().use(step4.JSONFormatter(on_fail=OnFailAction.FIX))
        valid = '{"keep": [null, true, 1.5]}'
        self.assertEqual(guard.validate(valid).validated_output, valid)
        for malformed in ["not json", '{"x": NaN}', "{broken"]:
            output = json.loads(guard.validate(malformed).validated_output)
            self.assertIn("error", output)
            self.assertEqual(output["raw"], malformed)

if __name__ == "__main__":
    unittest.main()