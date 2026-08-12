#!/usr/bin/env python3
"""전사문 정규화 — CER을 공정하게 재기 위한 전처리.

두 시스템(로컬 Whisper / 클라우드 LLM)의 출력을 같은 잣대로 비교하려면
표기 차이에서 오는 잡음을 먼저 걷어내야 한다. 이 모듈이 하는 일은
"틀린 것을 봐주는" 게 아니라 "같은 말인데 다르게 적힌 것"만 통일하는 것이다.

정규화 항목:
  1. 문장부호·특수문자 제거      — ASR 시스템마다 구두점 정책이 다르다
  2. 공백 제거                   — 한국어 CER의 통상 관행 (띄어쓰기는 별도 문제)
  3. 아라비아 숫자 → 한글 읽기   — Zeroth 정답문은 "이천 십 이 년", ASR 출력은 "2012년"
  4. 영문 소문자 통일

★ 3번이 없으면 두 시스템 모두 부당하게 나쁜 점수를 받는다.
  (상대 비교는 유지되지만 절대 수치가 왜곡된다)
"""

from __future__ import annotations

import re
import unicodedata

# ── 숫자 → 한글 ──────────────────────────────────────────────────────────
_DIGITS = ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_SMALL_UNITS = ["", "십", "백", "천"]
_BIG_UNITS = ["", "만", "억", "조", "경"]


def _read_four(n: int) -> str:
    """0~9999를 한글 읽기로. 십의 자리 이상에서 '일'은 생략한다 (일십 → 십)."""
    out = []
    for pos in range(3, -1, -1):
        d = (n // (10 ** pos)) % 10
        if d == 0:
            continue
        if d == 1 and pos > 0:
            out.append(_SMALL_UNITS[pos])
        else:
            out.append(_DIGITS[d] + _SMALL_UNITS[pos])
    return "".join(out)


def number_to_korean(n: int) -> str:
    """정수를 한글 읽기로 바꾼다. 예: 2012 → 이천십이"""
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스" + number_to_korean(-n)

    chunks = []
    idx = 0
    while n > 0:
        n, rem = divmod(n, 10000)
        if rem:
            chunks.append(_read_four(rem) + _BIG_UNITS[idx])
        idx += 1
        if idx >= len(_BIG_UNITS):
            break
    return "".join(reversed(chunks))


def _sub_numbers(text: str) -> str:
    """문자열 속 아라비아 숫자를 한글 읽기로 치환한다.

    자릿수 구분 쉼표(1,234)는 먼저 붙여 읽는다.
    소수점은 '점'으로 읽는다 (3.5 → 삼점오).
    """
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    def repl_decimal(m: re.Match[str]) -> str:
        whole, frac = m.group(1), m.group(2)
        frac_read = "".join(_DIGITS[int(c)] for c in frac)
        return number_to_korean(int(whole)) + "점" + frac_read

    text = re.sub(r"\b(\d+)\.(\d+)\b", repl_decimal, text)
    text = re.sub(r"\d+", lambda m: number_to_korean(int(m.group())), text)
    return text


# ── 본 정규화 ────────────────────────────────────────────────────────────
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str, *, drop_spaces: bool = True, korean_numbers: bool = True) -> str:
    """CER 계산용으로 전사문을 정규화한다.

    Args:
        text: 원본 전사문
        drop_spaces: 공백을 완전히 제거할지 (한국어 CER 관행)
        korean_numbers: 아라비아 숫자를 한글 읽기로 바꿀지
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").strip()

    if korean_numbers:
        text = _sub_numbers(text)

    text = _PUNCT.sub(" ", text)
    text = text.lower()
    text = _WS.sub(" ", text).strip()

    if drop_spaces:
        text = text.replace(" ", "")
    return text


def cer(reference: str, hypothesis: str, **kw) -> float:
    """문자오류율. 0에 가까울수록 좋다. 참조문이 비면 정의되지 않으므로 -1을 돌려준다."""
    import jiwer

    ref = normalize(reference, **kw)
    hyp = normalize(hypothesis, **kw)
    if not ref:
        return -1.0
    return jiwer.cer(ref, hyp)


if __name__ == "__main__":
    samples = [
        ("이천 십 이 년 맥북", "2012년 맥북"),
        ("십 일 이십 사 일 홍콩행", "10월 24일 홍콩행"),
        ("여든 일곱 채", "87채"),
        ("오늘은 무릎이 좀 시큰거려서 산책은 못 나갔다.", "오늘은 무릎이 좀 시큰거려서 산책은 못 나갔다"),
    ]
    print(f"{'참조':38} {'가설':30} {'정규화 후 CER':>12}")
    print("─" * 84)
    for ref, hyp in samples:
        print(f"{ref[:36]:38} {hyp[:28]:30} {cer(ref, hyp):>12.4f}")
    print()
    print("숫자 변환 예시:")
    for n in (0, 7, 12, 87, 2012, 1234, 30000, 123456789):
        print(f"  {n:>10,} → {number_to_korean(n)}")
