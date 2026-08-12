#!/usr/bin/env python3
"""공개 한국어 음성 데이터셋(Zeroth-Korean) 부분집합 준비.

합성 코퍼스만으로는 "TTS 음성이라 쉬웠던 것 아니냐"는 반론을 막을 수 없다.
실제 사람이 읽은 공개 데이터셋으로 일반 한국어 CER 기준선을 함께 잰다.

- 출처: https://huggingface.co/datasets/Bingsu/zeroth-korean (원본 OpenSLR-40)
- 라이선스: CC-BY-4.0
- 이 스크립트는 test 스플릿에서 결정적으로 N건을 뽑아 m4a로 변환한다.
  (실서비스 수신 포맷과 맞추기 위해 합성 코퍼스와 동일하게 16kHz 모노 AAC)

사용법:
    python src/prepare_zeroth.py             # 기본 40건
    python src/prepare_zeroth.py -n 20
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZEROTH_DIR = ROOT / "data" / "zeroth"
PARQUET = ZEROTH_DIR / "test.parquet"
AUDIO_DIR = ZEROTH_DIR / "audio"
REFERENCE = ZEROTH_DIR / "reference.json"

PARQUET_URL = (
    "https://huggingface.co/datasets/Bingsu/zeroth-korean/resolve/main/"
    "data/test-00000-of-00001-a41b955a631e582e.parquet"
)

SAMPLE_RATE = 16000
SEED = 42  # 재현성: 누가 돌려도 같은 표본이 뽑히도록 고정


def need(cmd: str) -> None:
    if shutil.which(cmd) is None:
        sys.exit(f"[에러] '{cmd}' 가 필요합니다.")


def ensure_parquet() -> None:
    if PARQUET.exists():
        return
    sys.exit(
        f"[에러] {PARQUET} 가 없습니다.\n"
        f"아래 명령으로 먼저 내려받으세요 (약 58MB):\n"
        f"  curl -L -o {PARQUET} '{PARQUET_URL}'"
    )


def to_m4a(raw: bytes, dst: Path, tmp: Path) -> float:
    """오디오 바이트를 실서비스 수신 포맷(16kHz 모노 AAC)으로 변환하고 길이를 돌려준다."""
    src = tmp / "in.bin"
    src.write_bytes(raw)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "aac", "-b:a", "64k", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-400:])
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(dst)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(dur) if dur else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Zeroth-Korean 부분집합 준비")
    ap.add_argument("-n", "--count", type=int, default=40, help="뽑을 표본 수 (기본 40)")
    args = ap.parse_args()

    need("ffmpeg")
    need("ffprobe")
    ensure_parquet()

    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET)
    rows = table.to_pylist()
    print(f"test 스플릿 전체 {len(rows)}건 중 {args.count}건을 뽑습니다 (seed={SEED})")

    rng = random.Random(SEED)
    picked = rng.sample(rows, min(args.count, len(rows)))

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    entries, total_dur, failed = [], 0.0, []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, row in enumerate(picked, start=1):
            uid = f"z{i:02d}"
            dst = AUDIO_DIR / f"{uid}.m4a"
            try:
                dur = to_m4a(row["audio"]["bytes"], dst, tmp)
            except Exception as exc:  # noqa: BLE001
                failed.append((uid, str(exc)[:150]))
                print(f"  {uid}  실패: {str(exc)[:100]}")
                continue
            total_dur += dur
            entries.append({"id": uid, "text": row["text"], "duration_sec": round(dur, 2)})
            print(f"  {uid}  {dur:5.2f}s  {row['text'][:38]}...")

    payload = {
        "_meta": {
            "source": "Bingsu/zeroth-korean (OpenSLR-40) test split",
            "license": "CC-BY-4.0",
            "seed": SEED,
            "count": len(entries),
            "total_duration_sec": round(total_dur, 1),
            "note": "실제 사람이 읽은 낭독체 한국어. 돌봄 대화 도메인은 아니며, 일반 한국어 CER 기준선용이다.",
        },
        "utterances": entries,
    }
    REFERENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"완료: {len(entries)}건 / 총 {total_dur:.1f}초 ({total_dur/60:.1f}분)")
    print(f"참조 전사문: {REFERENCE}")
    if failed:
        print(f"실패 {len(failed)}건: {[f[0] for f in failed]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
