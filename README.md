# RAG pipeline using LangChain

A simple retrieval-augmented generation framework using LangChain.

For embeddings, I use the **all-mpnet-base-v2** model from HuggingFace.
For the knowledge base I use Chromadb, which is a vector management library. It is light weight and an easy alternative for vector databases (for small prototyping or dev projects).
For the LLM, I use **declare-lab/flan-alpaca-large*** from HuggingFace.

## LLM 설정: LLaMA-Ko
```python
model_id = "beomi/llama-2-ko-7b"
```
> HuggingFace Hub에서 자동 로딩되며 `transformers` + `pipeline()`을 활용합니다.


## 프로젝트 구조

```
project_root/
├── main.py                  # 질문-응답 RAG 체인 실행
├── cli_runner.py            # 여러 개 PDF를 벡터DB에 저장하는 CLI
├── report_generator.py      # 주제 기반 보고서 생성기
├── rag_engine/              # 핵심 모듈 구성
│   ├── __init__.py
│   ├── loader.py            # PDF → 텍스트 분할
│   ├── embedder.py          # HuggingFace 임베딩
│   ├── vector_store.py      # ChromaDB 저장/로드
│   ├── llm.py               # LLaMA-Ko 모델 불러오기
│   ├── prompt.py            # 프롬프트 템플릿
│   ├── chain.py             # 검색/챗 체인 구성
│   └── process.py           # PDF 임베딩 및 저장 함수
└── vectordb/                # 생성된 ChromaDB 저장소 (자동 생성됨)


To run:

1. Install requirements using <br>
`pip install -r requirements.txt`

2. Create your knoweldge base by adding pdfs to the data folder. <br>
'python cli_runner.py'

3. Generate reports or run Q&A <br>
-To generate a report:
```bash
'python report_generator.py'

-To test question-answering:
'python main.py'

conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
