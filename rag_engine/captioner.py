# rag_engine/captioner.py
from __future__ import annotations
import re
from typing import Callable, Dict, List

PLATFORM_LIMIT = {
    "x": 100,        # 트윗(짧게) — 한글 기준 50자 내외 감안
    "kakao": 120,    # 카카오톡 미리보기 한 줄 권장
    "default": 120,
}

def _trim_to_limit(text: str, limit: int) -> str:
    """문자 길이 제한 내로 자르고 말줄임표 처리"""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"

def _build_prompt(article: str, topic: str, style: str, limit: int) -> str:
    """
    style: "x" | "kakao"
    """
    return f"""
당신은 스포츠 소셜 카피라이터입니다.
아래 기사 내용을 바탕으로 **세 줄 요약 문장**을 만들어주세요.

요구사항:
- 한국어 문장 3 줄
- 각 줄은 최대 {limit//2}자 내외로 간결하게
- 핵심(승패, 팀/선수, 결정적 장면)을 포함
- 과장/감탄사/이모지/해시태그 금지
- 결과는 **문장만 3 줄** 출력 (불필요한 설명 금지)

플랫폼: {style.upper()}
주제: {topic}

[기사]
{article.strip()}
""".strip()

def _fallback_captions(article: str, topic: str, limit: int) -> List[str]:
    """LLM 실패 시 단순 3줄 축약"""
    text = re.sub(r"\s+", " ", article.strip())
    if len(text) < 10:
        text = topic.strip()

    # 단순히 앞부분 잘라서 3등분
    total_len = len(text)
    chunk_size = max(1, total_len // 3)
    chunks = [text[i:i+chunk_size] for i in range(0, total_len, chunk_size)]
    chunks = [_trim_to_limit(c.strip(), limit) for c in chunks[:3]]

    # 부족하면 복제
    while len(chunks) < 3:
        chunks.append(chunks[-1])

    return chunks

def generate_captions(
    article: str,
    topic: str,
    llm_fn: Callable[[str, int], str] | None = None,
) -> Dict[str, str]:
    """
    article: 생성된 기사 본문
    topic: 사용자 주제
    llm_fn: prompt와 max_tokens를 받아 문자열을 반환하는 콜백(예: call_claude)
    """
    outputs: Dict[str, str] = {}
    for style in ("x", "kakao"):
        limit = PLATFORM_LIMIT.get(style, PLATFORM_LIMIT["default"])
        prompt = _build_prompt(article, topic, style, limit)

        lines: List[str] = []
        if llm_fn is not None:
            try:
                raw = llm_fn(prompt, 200).strip()
                # LLM이 여러 줄을 주면 3줄만 추출
                lines = [ln.strip('"“”‘’ ') for ln in raw.splitlines() if ln.strip()]
                lines = [_trim_to_limit(ln, limit) for ln in lines[:3]]
            except Exception:
                lines = []

        if not lines or len(lines) < 3:
            lines = _fallback_captions(article, topic, limit)

        outputs[style] = "\n".join(lines)  # 개행 포함 문자열로 반환

    return outputs