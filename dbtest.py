# dump_chroma_db.py
# (이 파일을 'report_generator.py'와 동일한 'RAG_pipeline-main_copy' 폴더에 저장하세요)

import os
from pprint import pprint

# 1. 'vector_store.py'에서 DB 접속 함수와 설정값 임포트
try:
    from rag_engine.vector_store import (
        get_chroma_collection, 
        _PERSIST_DIR_DEFAULT, 
        _COLLECTION_NAME_DEFAULT
    )
except ImportError as e:
    print(f"🔥🔥🔥 필수 모듈 임포트 실패: {e}")
    print("이 스크립트를 'report_generator.py'와 동일한 위치에서 실행하고 있는지,")
    print("'rag_engine' 폴더가 있는지 확인하세요.")
    exit()

def dump_all_documents():
    """
    ChromaDB에 저장된 모든 문서를 출력합니다.
    """
    print("="*60)
    print("ChromaDB 전체 데이터 덤프를 시작합니다.")
    print(f"DB 경로: '{_PERSIST_DIR_DEFAULT}'")
    print(f"컬렉션 이름: '{_COLLECTION_NAME_DEFAULT}'")
    print("="*60)

    # 1. ChromaDB 컬렉션 가져오기
    try:
        print(f"🔧 1. ChromaDB 컬렉션에 접속합니다...")
        collection = get_chroma_collection(
            persist_dir=_PERSIST_DIR_DEFAULT,
            collection_name=_COLLECTION_NAME_DEFAULT
        )
        print("✅ 1. 컬렉션 접속 성공.")
    except Exception as e:
        print(f"🔥🔥🔥 1. 컬렉션 접속 실패: {e}")
        print(f"'{_PERSIST_DIR_DEFAULT}' 경로에 vectordb가 실제로 있는지 확인하세요.")
        return

    # 2. 전체 문서 개수 확인
    try:
        count = collection.count()
        print(f"\n📊 2. 총 {count}개의 문서(청크)가 저장되어 있습니다.")
        if count == 0:
            print("DB가 비어있습니다.")
            return
    except Exception as e:
        print(f"🔥🔥🔥 2. 문서 개수 확인 실패: {e}")
        return

    # 3. 모든 데이터 가져오기 (문서 내용 + 메타데이터)
    try:
        print("\n🔍 3. 모든 문서와 메타데이터를 가져옵니다...")
        all_data = collection.get(
            include=["documents", "metadatas"]
        )
        print("✅ 3. 데이터 로드 완료.")
    except Exception as e:
        print(f"🔥🔥🔥 3. 데이터 가져오기 실패: {e}")
        return

    # 4. 결과 출력
    print("\n" + "="*60)
    print("📄 저장된 전체 문서 목록")
    print("="*60)

    documents = all_data.get('documents', [])
    metadatas = all_data.get('metadatas', [])
    ids = all_data.get('ids', [])

    if not documents:
        print("내용(documents)이 없습니다.")
        return

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        print(f"\n--- [문서 {i+1}] ---")
        
        # 메타데이터 (출처) 확인
        source = "N/A"
        if meta and "source" in meta:
            source = meta["source"]
        print(f"Source (출처 파일): {source}")
        
        # 기타 메타데이터 (있다면)
        # print(f"Metadata (전체): {meta}")
        
        # 내용 확인
        print("Content (내용):")
        print(doc.strip()) # .strip()으로 앞뒤 공백 제거
        
    print("\n" + "="*60)
    print(f"총 {len(documents)}개 문서 출력 완료.")
    print("="*60)


if __name__ == "__main__":
    dump_all_documents()