"""Bước 2: Prompt Hub và A/B routing MD5 tất định."""
import config
import hashlib
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError
from prompts import PROMPT_V1_NAME, PROMPT_V2_NAME, SYSTEM_V1, SYSTEM_V2, PROMPT_V1, PROMPT_V2
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from utils.evidence import evidence_log
from qa_pairs import SAMPLE_QUESTIONS

def push_prompts_to_hub(client: Client):
    """Push hai prompt riêng, chấp nhận Nothing to commit khi chạy lại."""
    for name, prompt, description in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1: concise, direct, grounded answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2: structured definitions and mechanisms"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"✅ Đã push '{name}' → {url}")
        except LangSmithConflictError as exc:
            if "nothing to commit" not in str(exc).lower():
                raise
            print(f"ℹ️ '{name}' không đổi (Nothing to commit).")

def pull_prompts_from_hub(client: Client, allow_local_fallback=False) -> dict:
    """Mặc định phải pull thành công; fallback tường minh chỉ dùng debug."""
    prompts = {}
    for name, local in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompt = client.pull_prompt(name)
            if set(prompt.input_variables) != {"context", "question"}:
                raise ValueError(f"Prompt {name} phải có context và question")
            prompts[name] = prompt
            commit = (prompt.metadata or {}).get("lc_hub_commit_hash", "unknown")
            print(f"↓ Đã pull '{name}' từ Hub | commit={commit}")
        except Exception as exc:
            if not allow_local_fallback:
                raise RuntimeError(f"Không pull được '{name}' ({type(exc).__name__})") from exc
            print(f"⚠️ Local fallback '{name}' — chỉ debug, không phải evidence Hub")
            prompts[name] = local
    return prompts

def get_prompt_version(request_id: str) -> str:
    """Hash chẵn → V1, lẻ → V2; độc lập với Python hash seed."""
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str, request_id="") -> dict:
    """Trace gồm request_id, version, contexts và answer."""
    contexts = [doc.page_content for doc in retriever.invoke(question)]
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": "\n\n".join(contexts), "question": question,
    })
    return {"question": question, "answer": answer, "contexts": contexts,
            "version": version, "request_id": request_id}

def setup_vectorstore():
    """Cùng cấu hình truy xuất như Bước 1 và 3."""
    return build_vectorstore(split_text(load_knowledge_base()), get_embeddings())

def main():
    """Lưu log từ một lần chạy A/B thật."""
    if not config.validate():
        raise SystemExit(1)
    with evidence_log("02_ab_routing_log.txt"):
        client = Client(api_key=config.LANGSMITH_API_KEY)
        push_prompts_to_hub(client)
        prompts = pull_prompts_from_hub(client)
        retriever = setup_vectorstore().as_retriever(search_kwargs={"k": 3})
        llm = get_llm()
        counts = {"v1": 0, "v2": 0}
        for i, question in enumerate(SAMPLE_QUESTIONS):
            request_id = f"req-{i:04d}"
            key = get_prompt_version(request_id)
            version = "v1" if key == PROMPT_V1_NAME else "v2"
            result = ask_ab(retriever, llm, prompts[key], question, version, request_id)
            counts[version] += 1
            print(f"[{i+1:02d}] [prompt-{version}] request_id={request_id} Q: {question}")
            print(f"  A: {result['answer']}")
        if not all(counts.values()):
            raise RuntimeError("Routing chưa phân phối đến cả hai prompt")
        print(f"📊 Routing: V1={counts['v1']} | V2={counts['v2']} | Tổng={sum(counts.values())}")
        print("✅ Hoàn thành. Kiểm tra Prompt Hub và traces trên LangSmith.")

if __name__ == "__main__":
    main()