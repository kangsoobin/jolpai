# rag_engine/tagger.py
from __future__ import annotations
import re, json
from typing import Callable, List, Optional

BASEBALL_TAG_TAXONOMY = [
    "승리","패배","역전","역전승","끝내기","연승","연패",
    "선발승","구원승","세이브","완봉","무실점","호투","난조",
    "멀티홈런","멀티히트","결승타","쐐기타","동점타",
    "솔로홈런","2점포","3점포","만루홈런",
    "수비실책","도루","폭투","볼넷","삼진"
]

def _build_tag_prompt(article: str, topic: str, taxonomy: List[str]) -> str:
    return f"""
당신은 야구 기사 태깅 엔진입니다.
아래 기사 본문과 주제를 바탕으로 검색 및 분류에 유용한 키워드 태그를 최대 5개 추출하세요.

규칙:
- 한국어 태그만
- 택소노미에 해당하면 우선 포함
- 선수/팀 고유명사는 그대로 포함
- 일반 단어(경기/기사/오늘/진행/내용/요약/승리/패배)는 제외
- 반드시 JSON만 출력: {{"tags": ["태그1","태그2"]}}

택소노미: {taxonomy}

[주제]
{topic}

[본문]
{article.strip()}
""".strip()

def _parse_json_safely(text: str) -> List[str]:
    try:
        return json.loads(text).get("tags", [])
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)).get("tags", [])
            except Exception:
                return []
        return []

def _heuristic_tags(article: str) -> List[str]:
    a = article
    tags = set()
    if re.search(r"연승", a): tags.add("연승")
    if re.search(r"연패", a): tags.add("연패")
    if re.search(r"끝내기", a): tags.add("끝내기")
    if re.search(r"완봉", a): tags.add("완봉")
    if re.search(r"무실점", a): tags.add("무실점")
    if re.search(r"(솔로\s*홈런|솔로홈런)", a): tags.add("솔로홈런")
    if re.search(r"(2점\s*포|2점포)", a): tags.add("2점포")
    if re.search(r"(3점\s*포|3점포|스리런)", a): tags.add("3점포")
    if re.search(r"(만루\s*홈런|만루홈런|그랜드\s*슬램)", a): tags.add("만루홈런")
    if a.count("홈런") >= 2: tags.add("멀티홈런")
    if re.search(r"(멀티히트|2안타.*이상|3안타|4안타)", a): tags.add("멀티히트")
    for k in ["결승타","쐐기타","동점타","세이브","호투","난조","수비실책","도루","폭투","볼넷","삼진"]:
        if k in a: tags.add(k)
    return list(tags)

def _extract_proper_nouns(article: str, limit: int = 6) -> List[str]:
    cands = set()
    for m in re.finditer(r"([가-힣]{2,4})(?:\s*(선수|투수|타자|포수))", article):
        cands.add(m.group(1))
    for m in re.finditer(r"([A-Za-z가-힣]+)\s*팀", article):
        name = m.group(1).strip()
        if 1 < len(name) <= 10: cands.add(name)
    return list(cands)[:limit]

def generate_keyword_tags(
    article: str,
    topic: str,
    llm_fn: Optional[Callable[[str, int], str]] = None,
    max_tags: int = 12
) -> List[str]:
    # 1) LLM 시도(콜백 주입)
    llm_tags: List[str] = []
    if llm_fn is not None:
        try:
            prompt = _build_tag_prompt(article, topic, BASEBALL_TAG_TAXONOMY)
            llm_text = llm_fn(prompt, 512)
            llm_tags = _parse_json_safely(llm_text)
        except Exception:
            llm_tags = []

    # 2) 휴리스틱 + 3) 고유명사
    heur = _heuristic_tags(article)
    proper = _extract_proper_nouns(article)

    raw = llm_tags + heur + proper
    seen, normed = set(), []

    def _norm(t: str) -> str:
        t = t.strip().strip('"“”‘’')
        stop = {"경기","기사","오늘","진행","내용","요약","상세"}
        if t in stop or len(t) < 2 or len(t) > 20: return ""
        return t

    for t in raw:
        nt = _norm(t)
        if nt and nt not in seen:
            seen.add(nt); normed.append(nt)

    return normed[:max_tags]
