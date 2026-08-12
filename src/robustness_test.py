#!/usr/bin/env python3
"""강건성 테스트 — 일부러 무너뜨려서 어디서 깨지는지 본다.

잘 되는 입력만 넣어보는 건 검증이 아니다. 실서비스에서 실제로 들어올 법한
'나쁜 입력'을 만들어 넣고, 파이프라인이 **조용히 잘못된 답을 내놓는지**
아니면 **실패를 실패라고 말하는지** 확인한다.

기존 실서비스의 가장 큰 문제가 정확히 이것이었다 —
전사 실패를 빈 문자열로 뭉개서 실패했다는 사실조차 남지 않았다.

만드는 나쁜 입력:
  silence      완전 무음 5초            — 어르신이 잘못 눌러 녹음된 경우
  tiny         0.1초짜리                — 손이 미끄러져 바로 뗀 경우
  noise_only   소음만 (말소리 없음)      — TV만 켜진 방에서 녹음
  loud_noise   말소리 + 아주 큰 소음     — 시끄러운 곳에서 통화
  clipped      문장 중간에서 잘림        — 전송 중 끊김
  very_quiet   아주 작은 목소리          — 마이크에서 멀리 떨어짐
  wrong_lang   한국어가 아닌 발화        — 언어를 ko로 고정한 영향 확인
  empty_file   0바이트 파일             — 전송 실패
  corrupt      깨진 헤더                — 파일 손상
  not_audio    오디오가 아닌 파일        — 잘못된 첨부

사용법:
    python src/robustness_test.py --build     # 나쁜 입력 생성
    python src/robustness_test.py             # 생성 + 테스트
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

STRESS_DIR = ROOT / "data" / "stress"
SOURCE_AUDIO = ROOT / "data" / "audio" / "u02.m4a"   # 정상 발화 원본
RESULTS = ROOT / "results"

SR = 16000


def run(args: list[str]) -> None:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])}... 실패\n{p.stderr[-400:]}")


# 각 케이스: (id, 설명, 기대하는 거동)
CASES: list[dict[str, str]] = [
    {"id": "silence", "desc": "완전 무음 5초",
     "expect": "빈 결과 또는 실패로 보고되어야 한다. 없는 말을 지어내면 안 된다."},
    {"id": "tiny", "desc": "0.1초짜리",
     "expect": "AUDIO_TOO_SHORT 로 걸러져야 한다."},
    {"id": "noise_only", "desc": "소음만, 말소리 없음",
     "expect": "빈 결과가 이상적. 환각(없는 문장 생성)이 나오면 위험 신호."},
    {"id": "loud_noise", "desc": "말소리 + 매우 큰 소음",
     "expect": "정확도는 떨어지되 무언가는 알아들어야 한다."},
    {"id": "clipped", "desc": "문장 중간에서 잘림",
     "expect": "잘린 데까지만 전사. 뒤를 지어내면 안 된다."},
    {"id": "very_quiet", "desc": "아주 작은 목소리 (-30dB)",
     "expect": "알아듣거나, 못 알아들었다고 말하거나. 둘 중 하나."},
    {"id": "wrong_lang", "desc": "한국어가 아닌 발화",
     "expect": "language=ko 고정의 부작용 확인용."},
    {"id": "empty_file", "desc": "0바이트 파일",
     "expect": "AUDIO_UNREADABLE 로 걸러져야 한다."},
    {"id": "corrupt", "desc": "헤더가 깨진 파일",
     "expect": "AUDIO_UNREADABLE 로 걸러져야 한다."},
    {"id": "not_audio", "desc": "오디오가 아닌 파일(텍스트)",
     "expect": "AUDIO_UNREADABLE 로 걸러져야 한다."},
]


def build() -> None:
    """나쁜 입력들을 만든다."""
    if not SOURCE_AUDIO.exists():
        sys.exit(f"[에러] 원본이 없습니다: {SOURCE_AUDIO}\n"
                 f"먼저 `python src/build_dataset.py` 를 실행하세요.")
    if shutil.which("ffmpeg") is None:
        sys.exit("[에러] ffmpeg 가 필요합니다.")

    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    enc = ["-ac", "1", "-ar", str(SR), "-c:a", "aac", "-b:a", "64k"]

    # 무음
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"anullsrc=r={SR}:cl=mono", "-t", "5", *enc,
         str(STRESS_DIR / "silence.m4a")])

    # 아주 짧음
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SOURCE_AUDIO),
         "-t", "0.1", *enc, str(STRESS_DIR / "tiny.m4a")])

    # 소음만
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"anoisesrc=c=pink:r={SR}:a=0.35:seed=7", "-t", "6", *enc,
         str(STRESS_DIR / "noise_only.m4a")])

    # 말소리 + 큰 소음 (소음을 음성과 비슷한 크기로)
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(SOURCE_AUDIO),
         "-f", "lavfi", "-i", f"anoisesrc=c=pink:r={SR}:a=0.5:seed=11",
         "-filter_complex",
         "[1:a]volume=-6dB[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[out]",
         "-map", "[out]", *enc, str(STRESS_DIR / "loud_noise.m4a")])

    # 중간에서 잘림
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SOURCE_AUDIO),
         "-t", "2.0", *enc, str(STRESS_DIR / "clipped.m4a")])

    # 아주 작은 목소리
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SOURCE_AUDIO),
         "-af", "volume=-30dB", *enc, str(STRESS_DIR / "very_quiet.m4a")])

    # 한국어가 아닌 발화 (영어 TTS)
    wrong = STRESS_DIR / "wrong_lang.m4a"
    if not wrong.exists():
        try:
            import asyncio

            import edge_tts

            mp3 = STRESS_DIR / "_wrong.mp3"
            asyncio.run(edge_tts.Communicate(
                "Hello, this is a test sentence in English, not Korean.",
                "en-US-AriaNeural").save(str(mp3)))
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), *enc, str(wrong)])
            mp3.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  (wrong_lang 생성 실패, 건너뜁니다: {str(exc)[:80]})")

    # 0바이트
    (STRESS_DIR / "empty_file.m4a").write_bytes(b"")

    # 헤더 손상 — 정상 파일 앞부분을 쓰레기로 덮는다
    raw = bytearray(SOURCE_AUDIO.read_bytes())
    raw[:64] = b"\x00" * 64
    (STRESS_DIR / "corrupt.m4a").write_bytes(bytes(raw))

    # 오디오가 아님
    (STRESS_DIR / "not_audio.m4a").write_text(
        "이것은 오디오가 아니라 그냥 텍스트 파일입니다.\n" * 20, encoding="utf-8")

    made = sorted(p.name for p in STRESS_DIR.glob("*.m4a"))
    print(f"나쁜 입력 {len(made)}건 생성: {STRESS_DIR}")
    for n in made:
        size = (STRESS_DIR / n).stat().st_size
        print(f"  {n:20} {size:>9,} bytes")


def test() -> int:
    import os

    from stt_whisper import WhisperBackend
    from extract_signals import extract

    backend = WhisperBackend(model_size=os.environ.get("WHISPER_MODEL", "small"))
    print("\n모델 준비 중...")
    backend.warmup()
    print("준비 완료\n")

    print(f"{'케이스':14} {'상태':22} {'지연':>7}  결과")
    print("─" * 104)

    rows: list[dict[str, Any]] = []
    for case in CASES:
        p = STRESS_DIR / f"{case['id']}.m4a"
        if not p.exists():
            print(f"{case['id']:14} {'(파일 없음 - 건너뜀)':22}")
            continue

        r = backend.transcribe(p, audio_id=case["id"])
        sigs = extract(r.text) if r.ok else []

        status = "OK" if r.ok else f"실패:{r.failure.value}"
        shown = (r.text or r.error_detail)[:52].replace("\n", " ")
        print(f"{case['id']:14} {status:22} {r.latency_sec:6.2f}s  {shown}")

        rows.append({
            "case": case["id"],
            "description": case["desc"],
            "expectation": case["expect"],
            "ok": r.ok,
            "failure": r.failure.value,
            "error_detail": r.error_detail,
            "text": r.text,
            "text_len": len(r.text),
            "latency_sec": round(r.latency_sec, 3),
            "audio_duration_sec": round(r.audio_duration_sec, 2),
            "signals_extracted": len(sigs),
            "signal_labels": [s.label for s in sigs],
        })

    # ── 판정 ────────────────────────────────────────────────────────────
    print("\n" + "═" * 104)
    print("판정 — 특히 '환각'을 본다")
    print("═" * 104)

    hallucinations, clean_fails, degraded = [], [], []
    for row in rows:
        cid = row["case"]
        if cid in {"empty_file", "corrupt", "not_audio", "tiny"}:
            (clean_fails if not row["ok"] else hallucinations).append(row)
        elif cid in {"silence", "noise_only"}:
            # 말소리가 없는데 문장을 만들어냈다면 환각이다
            if row["ok"] and row["text_len"] > 0:
                hallucinations.append(row)
            else:
                clean_fails.append(row)
        else:
            degraded.append(row)

    print(f"\n[깨끗하게 걸러짐] {len(clean_fails)}건 — 실패를 실패로 보고했다")
    for r in clean_fails:
        print(f"  {r['case']:14} → {r['failure']}")

    print(f"\n[환각 위험] {len(hallucinations)}건 — 없는 말을 지어냈거나 실패를 놓쳤다")
    if not hallucinations:
        print("  없음")
    for r in hallucinations:
        print(f"  {r['case']:14} → \"{r['text'][:60]}\"")
        if r["signal_labels"]:
            print(f"       ⚠ 여기서 신호까지 추출됨: {r['signal_labels']}")

    print(f"\n[품질 저하 관찰] {len(degraded)}건")
    for r in degraded:
        print(f"  {r['case']:14} → \"{r['text'][:60]}\"")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "robustness_results.json"
    out.write_text(json.dumps({
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend.name,
        "summary": {
            "total": len(rows),
            "clean_failures": len(clean_fails),
            "hallucinations": len(hallucinations),
            "degraded": len(degraded),
        },
        "cases": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="강건성 테스트")
    ap.add_argument("--build", action="store_true", help="나쁜 입력만 생성")
    ap.add_argument("--test", action="store_true", help="테스트만 실행")
    args = ap.parse_args()

    if args.build or not args.test:
        build()
    if args.test or not args.build:
        return test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
