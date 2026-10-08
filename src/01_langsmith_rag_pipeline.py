"""Bước 1: FAISS, LCEL RAG và 50 traces có context/answer."""
import config  # Trước mọi import LangChain.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS

def setup_vectorstore():
    """Chia 500 ký tự, overlap 50 rồi tạo FAISS."""
    chunks = split_text(load_knowledge_base(), chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")
    return build_vectorstore(chunks, get_embeddings())

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer in the question's language using only the following context. "
     "If evidence is missing, say you do not know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def build_rag_chain(vectorstore):
    """Giữ context trong output để trace gốc cũng có bằng chứng."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    inputs = {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    chain = inputs | RunnablePassthrough.assign(answer=RAG_PROMPT | get_llm() | StrOutputParser())
    return chain, retriever

@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> dict:
    """Trả question/context/answer; các run con ghi retriever/prompt/LLM."""
    return chain.invoke(question)

def main():
    """Chạy đủ 50 câu; lỗi API dừng bước."""
    if not config.validate():
        raise SystemExit(1)
    chain, _ = build_rag_chain(setup_vectorstore())
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        result = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {result['answer']}\n")
    print(f"✅ Đã chạy {len(SAMPLE_QUESTIONS)} truy vấn. Kiểm tra traces trong "
          f"project '{config.LANGSMITH_PROJECT}' tại https://smith.langchain.com.")

if __name__ == "__main__":
    main()