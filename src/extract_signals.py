#!/usr/bin/env python3
"""특이사항 추출 — 전사문에서 보호자가 알아야 할 신호를 뽑아낸다.

기존 실서비스에는 이 단계가 아예 없다. 전사된 텍스트가 대화 생성 로직으로
바로 들어가고 끝이라, 대화 속에 섞여 나온 중요한 신호가 그대로 묻힌다.

    "오늘은 무릎이 좀 시큰거려서 산책은 못 나갔다"
    "약은 어, 아침에 먹었나 모르겠네"

떨어져 사는 가족에게는 이런 문장이 대화 전체보다 중요하다.
그런데 지금은 이걸 알려면 대화 로그를 처음부터 끝까지 읽어야 한다.

★ 왜 규칙 기반인가 (LLM을 쓰지 않은 이유)

   이 PoC의 핵심 주장은 "음성이 기기 밖으로 나가지 않고, 비용이 0원"이다.
   추출 단계에 클라우드 LLM을 붙이면 **전사에서 막은 외부 전송이 여기서 도로 뚫린다.**
   전사문은 음성보다 오히려 더 노골적인 개인정보다.

   그래서 이 단계는 의도적으로 **완전 로컬·결정적·설명 가능한 규칙 기반**으로 구현했다.
   왜 그렇게 판정했는지 근거 문구(evidence)를 함께 돌려주므로,
   보호자가 결과를 신뢰할지 스스로 판단할 수 있다.

   로컬 소형 LLM으로의 확장은 다음 스텝으로 남겼다 (docs/limitations.md).

★ 이것은 진단이 아니다.
   의학적 판단을 하지 않는다. "한 번 더 들여다볼 거리"를 짚어주는 데서 멈춘다.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# ── 신호 정의 ────────────────────────────────────────────────────────────
# concern=True 는 "가족이 확인해볼 만한 신호", False 는 "안심 신호"다.
# 둘 다 뽑는 이유: 보호자에게는 "오늘 잘 드셨다"도 알고 싶은 정보다.

@dataclass(frozen=True)
class Rule:
    signal_type: str
    label: str
    patterns: tuple[str, ...]
    concern: bool = True
    priority: int = 2          # 1=높음(먼저 보여줌), 2=보통, 3=낮음
    negated_label: str = ""    # 부정 표현으로 뒤집혔을 때 쓸 라벨
                               # (예: "식사 양호"가 뒤집히면 "식사량 부족")


RULES: tuple[Rule, ...] = (
    # ── 건강 ────────────────────────────────────────────────────────────
    Rule("health", "통증 호소", (
        r"(무릎|허리|어깨|다리|팔|머리|목|가슴|속|배|손목|발목)[가이는을]?\s*(좀\s*)?(아프|아파|쑤시|시큰|결리|저리)",
        r"(아프|아파|쑤셔|쑤시|시큰거|결려|욱신)",
    ), priority=1),
    Rule("health", "어지럼증", (
        r"(어지러|핑\s*돌|현기증|어질어질)",
    ), priority=1),
    Rule("health", "소화기 이상", (
        r"(속이|배가|위가)\s*(좀\s*)?(안\s*좋|쓰리|쓰려|더부룩|메스꺼|아프)",
        r"(체했|토했|설사|변비)",
    ), priority=1),
    Rule("health", "감기·호흡기 증상", (
        r"(기침|가래|콧물|열이\s*나|목이\s*아프|감기\s*기운|몸살)",
    ), priority=2),
    Rule("health", "건망 호소", (
        r"(깜빡깜빡|자꾸\s*잊|기억이\s*안\s*나|생각이\s*안\s*나|헷갈리)",
    ), priority=1),
    Rule("health", "병원 일정", (
        r"(병원|진료|검진|외래)[에을를]?[^,.!?]{0,8}(가기로|간다|갔|예약|잡)",
        r"(정기\s*검진|진료\s*받)",
    ), concern=False, priority=2),
    Rule("health", "활동 양호", (
        r"(산책|운동|마당|경로당|나갔다\s*왔|걸었)",
    ), concern=False, priority=3, negated_label="활동 제한"),

    # ── 복약 ────────────────────────────────────────────────────────────
    Rule("medication", "복약 누락 의심", (
        # 약 언급과 부정 표현 사이에 이유가 길게 끼어드는 경우가 흔하다.
        #   "약을 먹으면 속이 쓰려서 어제는 그냥 안 먹었어"
        # 문장부호를 넘지 않으므로 한 문장 안으로 범위가 제한된다.
        r"약[을은는]?[^,.!?]{0,20}(안\s*먹|못\s*먹|거르|빼먹|건너뛰)",
    ), priority=1),
    Rule("medication", "복약 여부 불확실", (
        r"(먹었나|먹었는지)\s*(잘\s*)?(모르|기억이\s*안)",
    ), priority=1),
    Rule("medication", "복약 유지", (
        r"약[을은는]?[^,.!?]{0,12}(꼬박꼬박|잘\s*먹|챙겨\s*먹|안\s*빼먹)",
        r"(혈압약|당뇨약|약)[을은는]?[^,.!?]{0,10}먹고\s*있",
    ), concern=False, priority=3, negated_label="복약 누락 의심"),

    # ── 식사 ────────────────────────────────────────────────────────────
    Rule("meal", "식욕 저하", (
        r"(입맛이|밥맛이)\s*(통\s*)?(없|떨어)",
        r"(반\s*공기|조금밖에|몇\s*술)[^,.!?]{0,6}(못\s*먹|밖에)",
    ), priority=1),
    Rule("meal", "결식", (
        r"(아침|점심|저녁|밥)[을는]?[^,.!?]{0,8}(걸렀|굶|안\s*먹|못\s*먹)",
    ), priority=1),
    Rule("meal", "식사 양호", (
        r"(잘\s*먹었|맛있게|다\s*먹었|한\s*공기\s*다)",
        r"(끓여|해\s*먹|차려)\s*(서\s*)?먹",
    ), concern=False, priority=3, negated_label="식사량 부족"),

    # ── 수면 ────────────────────────────────────────────────────────────
    Rule("sleep", "수면 곤란", (
        r"(잠을?\s*(설쳤|못\s*잤|안\s*와)|뒤척|불면)",
        r"(새벽|밤중)에?\s*\S{0,4}(깨|일어나)",
    ), priority=2),
    Rule("sleep", "수면 양호", (
        r"(잘\s*잤|푹\s*잤)",
    ), concern=False, priority=3, negated_label="수면 곤란"),
    Rule("sleep", "낮잠", (
        r"낮잠",
    ), concern=False, priority=3),

    # ── 기분·정서 ───────────────────────────────────────────────────────
    Rule("mood", "고립감·우울", (
        r"(적적|외로|쓸쓸|심심|재미가\s*없|사람이\s*안\s*오)",
        r"(그냥\s*그래|별로야|기운이\s*없)",
    ), priority=1),
    Rule("mood", "활동 저하", (
        r"(누워\s*있었|종일\s*누워|나가지도\s*않)",
    ), priority=2),
    Rule("mood", "정서 긍정", (
        r"(좋을\s*수가\s*없|기분\s*좋|웃었|반가|재밌|즐거)",
        r"(전화|손주|아들|딸)[^,.!?]{0,12}(왔|했더라|보냈|붙여)",
    ), concern=False, priority=3, negated_label="정서 저하"),
    Rule("mood", "회피성 응답", (
        # "괜찮아, 걱정하지 마" — 실제 상태와 다를 수 있어 주의 신호로 본다
        r"(괜찮아|걱정\s*(하지\s*)?마|아무렇지도\s*않)",
    ), priority=2),
)

# 부정·불확실 표현. 안심 신호 앞에 붙으면 신호를 뒤집는다.
_NEGATION = re.compile(r"(안|못|없|아니|말고|글쎄|모르)")

SIGNAL_TYPE_KO = {
    "health": "건강",
    "medication": "복약",
    "meal": "식사",
    "sleep": "수면",
    "mood": "기분",
}


@dataclass
class Signal:
    signal_type: str
    signal_type_ko: str
    label: str
    concern: bool
    priority: int
    evidence: str          # 판정 근거가 된 실제 문구
    context: str           # 그 문구가 들어있던 문장

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sentences(text: str) -> list[str]:
    """문장 단위로 쪼갠다. 구어체라 종결 부호가 없을 수 있어 관대하게 처리한다."""
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


_SENT_PUNCT = re.compile(r"[.!?。]+")


def _segments(text: str) -> list[str]:
    """패턴을 적용할 구간 목록.

    ★ ASR이 찍은 구두점을 신뢰하지 않는다.

    실측에서 Whisper가 "누워 있었어"를 "누워. 있었어"로 적어
    문장이 엉뚱하게 쪼개졌고, 그 탓에 신호를 통째로 놓쳤다.
    구두점은 화자가 말한 것이 아니라 **모델이 지어낸 것**이므로,
    문장 단위 검사에 더해 **구두점을 걷어낸 전체 텍스트**로도 한 번 더 훑는다.

    거리 제한(`{0,N}`)이 그대로 걸려 있어 무한정 멀리 있는 표현끼리
    엮이지는 않는다.
    """
    segs = _sentences(text)
    flattened = _SENT_PUNCT.sub(" ", text)
    flattened = re.sub(r"\s+", " ", flattened).strip()
    if flattened and flattened not in segs:
        segs.append(flattened)
    return segs


def extract(text: str) -> list[Signal]:
    """전사문에서 특이사항 신호를 뽑는다.

    같은 (type, label) 조합은 한 번만 담는다. 여러 문장에서 걸리면 첫 근거를 쓴다.
    """
    if not text or not text.strip():
        return []

    found: dict[tuple[str, str], Signal] = {}

    for sent in _segments(text):
        for rule in RULES:
            for pat in rule.patterns:
                m = re.search(pat, sent)
                if not m:
                    continue

                concern = rule.concern
                label = rule.label

                # 안심 신호 주변에 부정 표현이 붙으면 주의 신호로 뒤집는다.
                #   "산책은 못 나갔다"  → 활동 양호(X) → 활동 제한(O)
                #
                # ★ 매치된 문구 자체는 검사 범위에서 제외한다.
                #   "좋을 수가 없어"처럼 부정어를 품은 긍정 관용구를
                #   잘못 뒤집는 것을 막기 위해서다.
                if not rule.concern:
                    before = sent[max(0, m.start() - 10): m.start()]
                    after = sent[m.end(): m.end() + 10]
                    if _NEGATION.search(before) or _NEGATION.search(after):
                        concern = True
                        label = rule.negated_label or f"{rule.label} 아님"

                key = (rule.signal_type, label)
                if key in found:
                    continue
                found[key] = Signal(
                    signal_type=rule.signal_type,
                    signal_type_ko=SIGNAL_TYPE_KO.get(rule.signal_type, rule.signal_type),
                    label=label,
                    concern=concern,
                    priority=rule.priority,
                    evidence=m.group(0).strip(),
                    context=sent,
                )
                break

    return sorted(
        found.values(),
        key=lambda s: (not s.concern, s.priority, s.signal_type),
    )


def format_report(text: str, signals: list[Signal]) -> str:
    """보호자가 실제로 받아볼 형태로 정리한다."""
    lines = ["─" * 58, "오늘의 특이사항", "─" * 58]

    concerns = [s for s in signals if s.concern]
    goods = [s for s in signals if not s.concern]

    if concerns:
        lines.append("")
        lines.append("[ 확인이 필요한 신호 ]")
        for s in concerns:
            lines.append(f"  · ({s.signal_type_ko}) {s.label}")
            lines.append(f"      근거: \"{s.context}\"")
    if goods:
        lines.append("")
        lines.append("[ 괜찮아 보이는 신호 ]")
        for s in goods:
            lines.append(f"  · ({s.signal_type_ko}) {s.label}")
    if not signals:
        lines.append("")
        lines.append("  특별히 잡힌 신호가 없습니다.")

    lines.append("")
    lines.append("─" * 58)
    lines.append("※ 이것은 의학적 진단이 아닙니다. 참고용으로만 보세요.")
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    if len(sys.argv) > 1:
        texts = [(f"arg{i}", t) for i, t in enumerate(sys.argv[1:], 1)]
    else:
        corpus = json.loads((root / "data" / "reference" / "corpus.json").read_text(encoding="utf-8"))
        texts = [(u["id"], u["text"]) for u in corpus["utterances"]]

    for uid, text in texts:
        sigs = extract(text)
        marks = " ".join(
            f"[{'!' if s.concern else '+'}{s.signal_type_ko}:{s.label}]" for s in sigs
        ) or "(없음)"
        print(f"{uid}  {text}")
        print(f"      → {marks}")
        print()
