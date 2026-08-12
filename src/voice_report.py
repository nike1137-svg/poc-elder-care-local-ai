#!/usr/bin/env python3
"""보호자용 음성 리포트 — 특이사항을 로컬 TTS로 읽어준다.

★ 이 파일이 채우는 구멍

   실서비스의 TTS(`dodami-tts.service`)는 edge-tts를 쓴다. 로컬 프로세스로
   떠 있지만 **실제 합성은 마이크로소프트 서버에서 일어난다.**
   즉 어르신에게 들려줄 텍스트가 매번 밖으로 나간다. 무료지만 외부 전송이다.

   STT를 로컬로 내려놓고 TTS는 밖에 두면 음성 루프의 절반만 막은 것이다.
   piper(ko_KR-kss-medium)로 바꾸면 **루프 전체가 집 안에서 닫힌다.**

   | | 기존 edge-tts | piper (이 구현) |
   |---|---|---|
   | 합성 위치 | 외부 (MS) | **로컬** |
   | 외부 전송 | 있음 | **없음** |
   | 비용 | 0원 | 0원 |
   | 모델 크기 | — | 61MB |
   | 인터넷 | 필요 | **불필요** |

사용법:
    python src/voice_report.py data/audio/u13.m4a
    python src/voice_report.py data/audio/*.m4a --out results/voice
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_signals import Signal, extract  # noqa: E402

VOICE_MODEL = ROOT / "models" / "piper" / "ko_KR-kss-medium.onnx"
VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "ko/ko_KR/kss/medium/ko_KR-kss-medium.onnx"
)
DEFAULT_OUT = ROOT / "results" / "voice"


def ensure_voice() -> None:
    if VOICE_MODEL.exists():
        return
    sys.exit(
        f"[에러] 한국어 음성 모델이 없습니다: {VOICE_MODEL}\n"
        f"아래로 내려받으세요 (약 61MB):\n"
        f"  mkdir -p {VOICE_MODEL.parent}\n"
        f"  curl -L -o {VOICE_MODEL} '{VOICE_URL}'\n"
        f"  curl -L -o {VOICE_MODEL}.json '{VOICE_URL}.json'"
    )


def build_script(signals: list[Signal], transcript: str) -> str:
    """음성으로 읽어줄 문장을 만든다.

    화면용 리포트를 그대로 읽으면 기호와 괄호까지 발음된다.
    귀로 듣기 좋은 문장으로 다시 쓴다.
    """
    concerns = [s for s in signals if s.concern]
    goods = [s for s in signals if not s.concern]

    parts: list[str] = ["오늘의 대화 요약입니다."]

    if concerns:
        labels = ", ".join(f"{s.signal_type_ko} 쪽 {s.label}" for s in concerns)
        parts.append(f"확인이 필요한 신호가 {len(concerns)}건 있습니다. {labels}.")
        # 인용할 발화에 종결 부호가 없으면 붙여준다.
        # 없으면 TTS가 다음 문장과 이어 읽어 "…핑돌더라고 이 안내는…"이 된다.
        quote = concerns[0].context.strip()
        if quote and quote[-1] not in ".!?。":
            quote += "."
        parts.append(f"이렇게 말씀하셨습니다. {quote}")
    else:
        parts.append("특별히 확인이 필요한 신호는 없었습니다.")

    if goods:
        labels = ", ".join(s.label for s in goods)
        parts.append(f"괜찮아 보이는 신호로는 {labels}가 있었습니다.")

    parts.append("이 안내는 의학적 진단이 아닙니다. 참고로만 들어주세요.")
    return " ".join(parts)


def synthesize(text: str, dst: Path, voice=None) -> dict[str, Any]:
    """piper로 음성을 만든다. 전 과정 로컬."""
    from piper import PiperVoice

    if voice is None:
        voice = PiperVoice.load(str(VOICE_MODEL))

    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with wave.open(str(dst), "wb") as w:
        voice.synthesize_wav(text, w)
    elapsed = time.perf_counter() - t0

    with wave.open(str(dst), "rb") as w:
        dur = w.getnframes() / float(w.getframerate())

    return {
        "output": str(dst),
        "synth_sec": round(elapsed, 3),
        "audio_sec": round(dur, 2),
        "realtime_factor": round(elapsed / dur, 3) if dur else 0.0,
        "external_requests": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="특이사항 음성 리포트 (로컬 TTS)")
    ap.add_argument("audio", nargs="+", help="어르신 음성 파일")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="출력 디렉터리")
    ap.add_argument("--model", default=None, help="Whisper 모델 크기")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ensure_voice()

    import os

    from piper import PiperVoice

    from stt_whisper import WhisperBackend

    out_dir = Path(args.out)
    stt = WhisperBackend(model_size=args.model or os.environ.get("WHISPER_MODEL", "small"))

    if not args.json:
        print("모델 준비 중... (Whisper + piper, 둘 다 로컬)")
    t0 = time.perf_counter()
    stt.warmup()
    voice = PiperVoice.load(str(VOICE_MODEL))
    if not args.json:
        print(f"준비 완료 {time.perf_counter() - t0:.1f}s\n")

    results = []
    for p in args.audio:
        src = Path(p)
        r = stt.transcribe(src)
        if not r.ok:
            results.append({"audio": str(src), "ok": False,
                            "failure": r.failure.value, "error": r.error_detail})
            if not args.json:
                print(f"[실패] {src.name}: {r.failure.value} — {r.error_detail}")
            continue

        sigs = extract(r.text)
        script = build_script(sigs, r.text)
        synth = synthesize(script, out_dir / f"{src.stem}_report.wav", voice)

        entry = {
            "audio": str(src), "ok": True,
            "transcript": r.text,
            "signal_count": len(sigs),
            "concern_count": sum(1 for s in sigs if s.concern),
            "script": script,
            **synth,
        }
        results.append(entry)

        if not args.json:
            print(f"[{src.name}]")
            print(f"  전사   : {r.text}")
            print(f"  신호   : {len(sigs)}건 (확인필요 {entry['concern_count']}건)")
            print(f"  낭독문 : {script}")
            print(f"  음성   : {synth['output']}  "
                  f"({synth['audio_sec']}초 분량 / 합성 {synth['synth_sec']}초 / 외부전송 0건)")
            print()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "voice_report_results.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"결과 저장: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
