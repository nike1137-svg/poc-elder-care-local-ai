#!/usr/bin/env python3
"""엔드투엔드 파이프라인 — 음성 파일 하나를 넣으면 보호자용 리포트가 나온다.

    음성(m4a)  →  로컬 Whisper 전사  →  특이사항 추출  →  구조화 리포트

이것이 이 PoC의 본체다. 전 과정이 로컬에서 돌고, 외부로 나가는 요청이 0건이다.

사용법:
    python src/pipeline.py data/audio/u02.m4a
    python src/pipeline.py data/audio/*.m4a --json
    python src/pipeline.py data/audio/u13.m4a --model medium
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_signals import extract, format_report  # noqa: E402
from stt_whisper import WhisperBackend  # noqa: E402


def load_env() -> None:
    """`.env` 로더는 src/envfile.py 로 일원화했다 (빈 값이 실제 키를 가리던 버그)."""
    from envfile import load_env as _load

    _load()


def process(backend: WhisperBackend, audio: Path) -> dict[str, Any]:
    """음성 한 건을 처리해 전사문과 특이사항을 돌려준다."""
    started = time.perf_counter()
    stt = backend.transcribe(audio)

    # ★ 실패를 삼키지 않는다. 실패는 실패로 보고한다.
    if not stt.ok:
        return {
            "audio": str(audio),
            "ok": False,
            "failure": stt.failure.value,
            "error_detail": stt.error_detail,
            "transcript": "",
            "signals": [],
            "total_sec": round(time.perf_counter() - started, 3),
        }

    signals = extract(stt.text)
    return {
        "audio": str(audio),
        "ok": True,
        "failure": "none",
        "transcript": stt.text,
        "audio_duration_sec": round(stt.audio_duration_sec, 2),
        "stt_latency_sec": round(stt.latency_sec, 3),
        "realtime_factor": round(stt.realtime_factor, 3),
        "stt_meta": stt.meta,
        "signals": [s.to_dict() for s in signals],
        "concern_count": sum(1 for s in signals if s.concern),
        "external_requests": 0,   # 로컬 파이프라인 — 정의상 0
        "total_sec": round(time.perf_counter() - started, 3),
    }


def print_human(res: dict[str, Any]) -> None:
    name = Path(res["audio"]).name
    print("=" * 58)
    print(f"입력: {name}")
    print("=" * 58)

    if not res["ok"]:
        print()
        print(f"  ✗ 전사 실패 [{res['failure']}]")
        print(f"    {res['error_detail']}")
        print()
        print("  ※ 기존 방식은 이 상황을 빈 문자열로 넘겨 조용히 무시했습니다.")
        print("     여기서는 실패를 실패로 보고합니다.")
        return

    print()
    print(f"음성 길이 {res['audio_duration_sec']}s  →  전사 {res['stt_latency_sec']}s "
          f"(실시간 대비 {res['realtime_factor']}배)")
    print(f"외부 전송 {res['external_requests']}건   비용 0원")
    print()
    print("[ 전사 결과 ]")
    print(f"  {res['transcript']}")
    print()

    from extract_signals import Signal

    sigs = [Signal(**s) for s in res["signals"]]
    print(format_report(res["transcript"], sigs))


def main() -> int:
    ap = argparse.ArgumentParser(description="음성 → 전사 → 특이사항 리포트")
    ap.add_argument("audio", nargs="+", help="음성 파일 경로")
    ap.add_argument("--model", default=None, help="Whisper 모델 크기")
    ap.add_argument("--json", action="store_true", help="JSON으로 출력")
    args = ap.parse_args()

    load_env()
    backend = WhisperBackend(model_size=args.model or os.environ.get("WHISPER_MODEL", "small"))

    if not args.json:
        print(f"모델 준비 중... ({backend.name}, CPU)")
        t0 = time.perf_counter()
        backend.warmup()
        print(f"준비 완료 {time.perf_counter() - t0:.1f}s\n")
    else:
        backend.warmup()

    results = [process(backend, Path(p)) for p in args.audio]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print_human(r)
            print()

    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
