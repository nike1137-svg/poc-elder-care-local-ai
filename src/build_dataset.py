#!/usr/bin/env python3
"""평가용 합성 음성 데이터셋 생성.

corpus.json 의 대본을 TTS로 음성화하고, 스타일별 변형(느린 발화·침묵·생활소음)을
입혀 실서비스가 받는 형식(16kHz 모노 m4a)으로 저장한다.

★ 실사용자 음성은 일절 사용하지 않는다. 전량 창작 대본 기반 합성이다.

사용법:
    python src/build_dataset.py                # 전체 생성
    python src/build_dataset.py --only u03 u07 # 일부만
    python src/build_dataset.py --force        # 이미 있어도 다시 생성
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "reference" / "corpus.json"
AUDIO_DIR = ROOT / "data" / "audio"

# 고령 화자 모사: 기본 피치를 낮추고, 스타일에 따라 속도를 조절한다.
VOICE = "ko-KR-SunHiNeural"
BASE_PITCH = "-15Hz"

STYLE_RATE = {
    "normal": "-10%",   # 어르신 발화는 표준 TTS보다 대체로 느리다
    "slow": "-30%",
    "pause": "-15%",
    "noisy": "-10%",
}

PAUSE_SECONDS = 1.2      # 'pause' 스타일에서 문장 중간에 넣는 침묵 길이
NOISE_LEVEL_DB = -26     # 'noisy' 스타일 배경 소음 크기 (음성 대비)

SAMPLE_RATE = 16000


def need(cmd: str) -> None:
    if shutil.which(cmd) is None:
        sys.exit(f"[에러] '{cmd}' 가 필요합니다. (ffmpeg 설치 후 다시 실행)")


def run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"명령 실패: {' '.join(args[:3])}...\n{proc.stderr[-800:]}")


def split_for_pause(text: str) -> tuple[str, str]:
    """문장 중간에 침묵을 넣기 위해 텍스트를 두 조각으로 나눈다.

    쉼표 우선, 없으면 어절 기준 중간.
    """
    if "," in text:
        head, _, tail = text.partition(",")
        return head.strip() + ",", tail.strip()
    words = text.split()
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]), " ".join(words[mid:])


async def tts(text: str, rate: str, out_mp3: Path) -> None:
    import edge_tts

    comm = edge_tts.Communicate(text, VOICE, rate=rate, pitch=BASE_PITCH)
    await comm.save(str(out_mp3))


def to_m4a(src: Path, dst: Path) -> None:
    """실서비스가 메신저에서 받는 형식에 맞춘다: AAC / 16kHz / 모노."""
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "aac", "-b:a", "64k",
        str(dst),
    ])


def concat_with_silence(a: Path, b: Path, dst: Path, seconds: float) -> None:
    """두 오디오 사이에 침묵을 넣어 이어붙인다."""
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(a),
        "-f", "lavfi", "-t", f"{seconds}", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-i", str(b),
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
        "-map", "[out]",
        str(dst),
    ])


def mix_noise(src: Path, dst: Path) -> None:
    """생활 소음을 배경으로 깐다 (TV·주방 등 상황 모사).

    결정적 시드를 써서 실행할 때마다 같은 소음이 나오게 한다.
    """
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-f", "lavfi", "-i", f"anoisesrc=c=pink:r={SAMPLE_RATE}:a=0.06:seed=42",
        "-filter_complex",
        f"[1:a]volume={NOISE_LEVEL_DB}dB[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[out]",
        "-map", "[out]",
        str(dst),
    ])


def build_one(utt: dict, tmp: Path, force: bool) -> Path:
    uid = utt["id"]
    text = utt["text"]
    style = utt.get("style", "normal")
    rate = STYLE_RATE.get(style, STYLE_RATE["normal"])
    final = AUDIO_DIR / f"{uid}.m4a"

    if final.exists() and not force:
        print(f"  {uid}  건너뜀 (이미 있음)")
        return final

    if style == "pause":
        head, tail = split_for_pause(text)
        a_mp3, b_mp3 = tmp / f"{uid}_a.mp3", tmp / f"{uid}_b.mp3"
        asyncio.run(tts(head, rate, a_mp3))
        asyncio.run(tts(tail, rate, b_mp3))
        joined = tmp / f"{uid}_joined.m4a"
        concat_with_silence(a_mp3, b_mp3, joined, PAUSE_SECONDS)
        src = joined
    else:
        mp3 = tmp / f"{uid}.mp3"
        asyncio.run(tts(text, rate, mp3))
        src = mp3

    if style == "noisy":
        noised = tmp / f"{uid}_noised.m4a"
        mix_noise(src, noised)
        src = noised

    to_m4a(src, final)
    size_kb = final.stat().st_size / 1024
    print(f"  {uid}  {style:7} rate={rate:5}  →  {final.name} ({size_kb:.0f} KB)")
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description="평가용 합성 음성 데이터셋 생성")
    ap.add_argument("--only", nargs="*", help="특정 발화 id만 생성")
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 생성")
    args = ap.parse_args()

    need("ffmpeg")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    utts = corpus["utterances"]
    if args.only:
        wanted = set(args.only)
        utts = [u for u in utts if u["id"] in wanted]
        if not utts:
            sys.exit(f"[에러] 해당 id를 찾지 못했습니다: {sorted(wanted)}")

    print(f"합성 대상 {len(utts)}건  (voice={VOICE}, pitch={BASE_PITCH})")
    print(f"출력 경로: {AUDIO_DIR}")
    print()

    made, failed = 0, []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for utt in utts:
            try:
                build_one(utt, tmp, args.force)
                made += 1
            except Exception as exc:  # noqa: BLE001 - 개별 실패는 넘기고 끝까지 진행
                failed.append((utt["id"], str(exc)[:200]))
                print(f"  {utt['id']}  실패: {str(exc)[:120]}")

    print()
    print(f"완료: {made}건 생성, {len(failed)}건 실패")
    for uid, err in failed:
        print(f"  - {uid}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
