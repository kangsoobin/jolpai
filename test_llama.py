from llama_cpp import Llama

llm = Llama(
    model_path="models/llama-3.2-Korean-Bllossom-3B-Q6_K_L.gguf",
    n_ctx=2048,           # context window
    n_threads=4,          # Mac에서 4~8 추천
    verbose=True
)

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 29 Oct 2024

너는 이제부터 매우 친절한 한국어 도우미야. 사용자의 질문에 자세하고 따뜻하게 대답해줘.<|eot_id|><|start_header_id|>user<|end_header_id|>
서울에서 가을에 가기 좋은 여행지 알려줘<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

output = llm(prompt, max_tokens=200, stop=["<|eot_id|>"], echo=False)

print("\n💬 모델 응답:\n")
print(output["choices"][0]["text"])
