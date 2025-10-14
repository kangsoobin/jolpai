# retrievers/pg_vector.py

from __future__ import annotations
from typing import List, Callable
from .base import Retriever, Query, Document
import psycopg
import json # ⬅️ [추가] metadata 처리를 위해 import

class PostgresVectorRetriever(Retriever):
    """
    pgvector 테이블에서 벡터 검색을 수행하고,
    검색된 테이블 row를 LLM이 이해하기 쉬운 자연어 문장으로 변환합니다.
    """
    def __init__(self, dsn: str, table: str, embed_fn: Callable[[str], List[float]], n_results: int = 8):
        self.dsn = dsn
        self.table = table
        self.embed_fn = embed_fn
        self.n_results = n_results
        print(f"✅ PostgresVectorRetriever 초기화 완료: {table} 테이블 대상")

    def retrieve(self, query: Query) -> List[Document]:
        qvec = self.embed_fn(query.text)
        n = query.top_k or self.n_results

        # ❗️ [수정] content 컬럼 대신 모든 컬럼(*)을 가져오도록 SQL 변경
        #     유사도(score) 계산은 그대로 유지합니다.
        sql = f"""
          SELECT *, 1 - (embedding <=> %s::vector) AS score
          FROM {self.table}
          WHERE embedding IS NOT NULL
          ORDER BY embedding <=> %s::vector
          LIMIT %s
        """
        
        out: List[Document] = []
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                # ❗️ [수정] SQL 파라미터 전달 방식 변경
                cur.execute(sql, (qvec, qvec, n))
                rows = cur.fetchall()
                # DB 커서로부터 컬럼 이름들을 가져옵니다.
                column_names = [desc[0] for desc in cur.description]

        # ❗️ [핵심] 가져온 테이블 row를 자연어 문장으로 변환하는 부분
        for row in rows:
            # row 데이터를 (컬럼 이름: 값) 형태의 딕셔너리로 변환
            row_dict = dict(zip(column_names, row))
            
            doc_id = str(row_dict.get("id", "N/A"))
            score = float(row_dict.get("score", 0.0))
            
            # 'embedding', 'score' 등 내부용 컬럼은 최종 문장에서 제외
            content_items = {
                key: value
                for key, value in row_dict.items()
                if key not in ["embedding", "score"] and value is not None
            }
            
            # 딕셔너리를 "컬럼1: 값1, 컬럼2: 값2, ..." 형태의 문자열로 변환
            formatted_content = ", ".join([f"{key}: {value}" for key, value in content_items.items()])
            
            # 최종적으로 Document 객체 생성
            out.append(Document(
                id=doc_id,
                content=f"데이터 정보 ({self.table}): {formatted_content}", # ⬅️ 자연어 문장으로!
                metadata={"source": f"db_{self.table}", "db_id": doc_id},
                score=score
            ))
            
        return out