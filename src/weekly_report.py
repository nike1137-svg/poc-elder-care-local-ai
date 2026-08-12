#!/usr/bin/env python3
"""주간 집계 리포트 — 하루치가 아니라 흐름을 본다.

특이사항 추출은 발화 한 건씩 처리한다. 그런데 가족이 실제로 알고 싶은 건
"오늘 무릎이 아프시다"가 아니라 **"요새 계속 무릎 얘기를 하신다"**이다.

한 번은 지나가는 말이고, 세 번은 신호다.
그 차이를 보려면 며칠치를 모아야 한다.

이 스크립트는 여러 날의 음성을 처리해 다음을 낸다.
  - 신호 유형별 발생 빈도
  - 반복 신호 (같은 신호가 여러 날 나온 것) ← 가장 중요
  - 확인 필요 신호의 날짜별 추이
  - 무응답·전사 실패 건수

사용법:
    python src/weekly_report.py --demo                     # 데모(합성 코퍼스를 7일로 배치)
    python src/weekly_report.py data/audio/*.m4a           # 파일 직접 지정
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_signals import extract  # noqa: E402

RESULTS = ROOT / "results"

# 며칠 이상 반복되면 '반복 신호'로 올릴지
REPEAT_THRESHOLD = 2


def demo_schedule() -> list[tuple[str, list[str]]]:
    """합성 코퍼스 20건을 7일치 대화로 배치한다.

    실제 운영에서는 날짜별 실제 발화가 들어온다.
    여기서는 '흐름을 본다'는 기능을 보여주기 위한 배치다.
    무릎 통증(u02, u10)과 복약 문제(u03, u15)가 여러 날 걸치도록 의도적으로 배열했다.
    """
    plan = [
        ["u01", "u02", "u03"],        # 1일차: 무릎 통증, 복약 불확실
        ["u04", "u05"],               # 2일차: 식욕 저하, 수면 곤란
        ["u06", "u07"],               # 3일차: 좋은 날
        ["u08", "u10"],               # 4일차: 적적함, 허리 통증
        ["u09", "u11", "u12"],        # 5일차: 무난
        ["u13", "u15", "u16"],        # 6일차: 결식, 복약 누락, 어지럼
        ["u14", "u17", "u18", "u19", "u20"],  # 7일차: 혼재
    ]
    start = date.today() - timedelta(days=len(plan) - 1)
    return [((start + timedelta(days=i)).isoformat(), ids) for i, ids in enumerate(plan)]


def analyze(days: list[dict[str, Any]]) -> dict[str, Any]:
    type_counter: Counter[str] = Counter()
    label_days: dict[str, set[str]] = defaultdict(set)
    label_counter: Counter[str] = Counter()
    concern_by_day: dict[str, int] = {}
    failures = 0
    total_utts = 0

    for d in days:
        concern_by_day[d["date"]] = 0
        for u in d["utterances"]:
            total_utts += 1
            if not u["ok"]:
                failures += 1
                continue
            for s in u["signals"]:
                type_counter[s["signal_type_ko"]] += 1
                key = f"{s['signal_type_ko']}:{s['label']}"
                label_counter[key] += 1
                label_days[key].add(d["date"])
                if s["concern"]:
                    concern_by_day[d["date"]] += 1

    repeated = sorted(
        [
            {
                "signal": k,
                "days": len(v),
                "occurrences": label_counter[k],
                "dates": sorted(v),
            }
            for k, v in label_days.items()
            if len(v) >= REPEAT_THRESHOLD
        ],
        key=lambda x: (-x["days"], -x["occurrences"]),
    )

    return {
        "period": {"from": days[0]["date"], "to": days[-1]["date"], "days": len(days)},
        "utterances_total": total_utts,
        "transcription_failures": failures,
        "signal_type_counts": dict(type_counter.most_common()),
        "repeated_signals": repeated,
        "concern_by_day": concern_by_day,
        "top_signals": [
            {"signal": k, "count": c} for k, c in label_counter.most_common(8)
        ],
    }


def render(report: dict[str, Any]) -> str:
    p = report["period"]
    lines = [
        "═" * 62,
        f"주간 리포트  {p['from']} ~ {p['to']}  ({p['days']}일)",
        "═" * 62,
        "",
        f"대화 {report['utterances_total']}건 처리 · 전사 실패 {report['transcription_failures']}건",
        "",
    ]

    rep = report["repeated_signals"]
    lines.append("[ 반복해서 나온 신호 ]  ← 한 번은 지나가는 말, 여러 번은 신호")
    if rep:
        for r in rep:
            lines.append(f"  · {r['signal']}")
            lines.append(f"      {r['days']}일에 걸쳐 {r['occurrences']}회 ({', '.join(r['dates'])})")
    else:
        lines.append("  반복된 신호는 없습니다.")
    lines.append("")

    lines.append("[ 유형별 발생 횟수 ]")
    for k, v in report["signal_type_counts"].items():
        bar = "█" * min(v, 30)
        lines.append(f"  {k:6} {v:>3}회  {bar}")
    lines.append("")

    lines.append("[ 날짜별 '확인 필요' 신호 수 ]")
    for d, n in report["concern_by_day"].items():
        bar = "▇" * n if n else "·"
        lines.append(f"  {d}  {n:>2}  {bar}")
    lines.append("")

    lines.append("═" * 62)
    lines.append("※ 이것은 의학적 진단이 아닙니다. 참고용으로만 보세요.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="주간 집계 리포트")
    ap.add_argument("audio", nargs="*", help="음성 파일들 (미지정 시 --demo 필요)")
    ap.add_argument("--demo", action="store_true", help="합성 코퍼스를 7일치로 배치해 시연")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.audio and not args.demo:
        ap.error("음성 파일을 지정하거나 --demo 를 쓰세요.")

    import os

    from stt_whisper import WhisperBackend

    backend = WhisperBackend(model_size=args.model or os.environ.get("WHISPER_MODEL", "small"))
    if not args.json:
        print("모델 준비 중...")
    backend.warmup()
    if not args.json:
        print("준비 완료\n")

    audio_dir = ROOT / "data" / "audio"

    if args.demo:
        schedule = demo_schedule()
    else:
        # 파일을 하루로 몰아서 처리
        schedule = [(date.today().isoformat(), [Path(p).stem for p in args.audio])]
        audio_dir = Path(args.audio[0]).parent

    days: list[dict[str, Any]] = []
    for day, ids in schedule:
        utts = []
        for uid in ids:
            p = audio_dir / f"{uid}.m4a"
            r = backend.transcribe(p, audio_id=uid)
            sigs = extract(r.text) if r.ok else []
            utts.append({
                "id": uid, "ok": r.ok, "failure": r.failure.value,
                "text": r.text,
                "signals": [s.to_dict() for s in sigs],
            })
            if not args.json:
                mark = "OK" if r.ok else f"실패({r.failure.value})"
                print(f"  {day} {uid} [{mark}] {r.text[:46]}")
        days.append({"date": day, "utterances": utts})

    report = analyze(days)
    report["days_detail"] = days

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "weekly_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print()
        print(render(report))
        print(f"\n결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
