from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
from rag_engine.prompt import get_search_prompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

def build_search_chain(llm, retriever):
    prompt = get_search_prompt()

    def format_with_source(inputs):
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)

        # 전체 context 합치기
        full_context = "\n".join([doc.page_content for doc in docs])

        # context = "\n".join([doc.page_content for doc in docs])
        # sources = ", ".join(sorted(set([doc.metadata.get("source", "?") + f" (p{doc.metadata.get('page', '?')})" for doc in docs])))
        
        
        # 토큰 길이 제한 (예: 800토큰만 사용)
        tokens = tokenizer.tokenize(full_context)
        if len(tokens) > 800:
            tokens = tokens[:800]
        context = tokenizer.convert_tokens_to_string(tokens)

        sources = ", ".join(sorted(set(
            [doc.metadata.get("source", "?") + f" (p{doc.metadata.get('page', '?')})" for doc in docs]
        )))
        
        return {
            "question": question,
            "context": context,
            "sources": sources
        }

    chain = (
        RunnableLambda(format_with_source)
        | prompt
        | llm
        | RunnableLambda(lambda x: x + "\n\n 출처: " + x["sources"] if isinstance(x, dict) else x)
        | StrOutputParser()
    )
    return chain


def build_chat_chain(llm, retriever):
    prompt = get_search_prompt()

    def merge_chat_history(input):
        history = input.get("chat_history", [])
        chat_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])
        full_question = f"{chat_str}\nUser: {input['question']}"
        docs = retriever.get_relevant_documents(full_question)
        context = "\n".join([doc.page_content for doc in docs])
        sources = ", ".join(sorted(set([doc.metadata.get("source", "?") + f" (p{doc.metadata.get('page', '?')})" for doc in docs])))
        return {
            "question": full_question,
            "context": context,
            "sources": sources
        }

    chain = (
        RunnableLambda(merge_chat_history)
        | prompt
        | llm
        | RunnableLambda(lambda x: x + "\n\n 출처: " + x["sources"] if isinstance(x, dict) else x)
        | StrOutputParser()
    )
    return chain
