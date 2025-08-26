from rag_engine.llm import load_llm
from rag_engine.embedder import get_embedder
from rag_engine.vector_store import load_vector_db
from rag_engine.chain import build_search_chain

if __name__ == "__main__":
    embedder = get_embedder()
    vectordb = load_vector_db(embedder)
    retriever = vectordb.as_retriever()
    llm = load_llm()
    chain = build_search_chain(llm, retriever)

    print("\n RAG 시스템 실행 준비 완료")
    while True:
        query = input("\n 질문을 입력하세요 (종료: q): ")
        if query.strip().lower() == "q":
            break
        response = chain.invoke({"question": query})
        print("\n 답변:", response)