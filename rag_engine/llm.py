from llama_cpp import Llama

def load_llm():
    try:
        print("🚀 llama.cpp 모델 로딩 시작")

        llm = Llama(
            model_path="models/llama-3.2-Korean-Bllossom-3B-Q6_K_L.gguf",
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

        print("✅ llama.cpp 모델 로딩 완료")
        return llm  # 이제 하나만 반환함

    except Exception as e:
        import traceback
        print("❌ LLM 로딩 중 예외 발생!")
        traceback.print_exc()
        raise RuntimeError(f"LLM 로딩 실패: {e}")

