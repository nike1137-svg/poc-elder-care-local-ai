#!/usr/bin/env python3
"""알약 세기 정확도 벤치마크 — 정답을 아는 이미지로 측정한다.

★ 왜 따로 만드는가

  SD-Turbo로 만든 알약 사진은 사실적이지만 **정답 라벨이 없다.**
  "알약이 몇 개인지" 아무도 모르므로 세기 정확도를 잴 수 없다.
  눈으로 보고 "잘 되네" 하는 건 검증이 아니다.

  그래서 **개수를 내가 정해서 합성한다.** 배경은 SD가 만든 실제 나무 테이블
  사진을 재활용하고, 그 위에 알약을 정확히 N개 올린다. 정답은 N이다.

  두 데이터가 서로를 보완한다.
    - 합성 벤치마크 : 정답이 있어 **정확도를 잴 수 있다** (사실감은 낮다)
    - SD 생성 사진  : 사실적이지만 **정답이 없다** (눈으로만 확인)

사용법:
    python src/pill_benchmark.py --build     # 벤치마크 이미지 생성
    python src/pill_benchmark.py             # 생성 + 측정
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

BENCH_DIR = ROOT / "data" / "pill_bench"

# ★ 배경은 '물체가 없는' 이미지여야 한다.
#   처음에는 알약 사진(pill_01.png)의 구석을 잘라 배경으로 썼는데,
#   그 구석에도 원본 알약이 남아 있어서 **정답 1개짜리에서 2개가 검출**됐다.
#   모든 케이스에서 +1씩 밀린 원인이 이것이었다.
#   그래서 SD로 '아무것도 없는 나무 테이블'을 따로 생성해 쓴다.
BACKGROUND = ROOT / "data" / "images" / "bg_wood.png"
RESULTS = ROOT / "results"

SEED = 20260812
SIZE = 512
# (이미지 id, 알약 개수) — 실제 복약 상황에 가까운 범위로 잡았다
CASES: list[tuple[str, int]] = [
    ("bench_00", 0),    # 대조군 — 빈 테이블. 배경만으로 오검출이 나는지 본다
    ("bench_01", 1),    # 한 알
    ("bench_02", 3),    # 아침 약 세 알
    ("bench_03", 6),
    ("bench_04", 10),
    ("bench_05", 15),
    ("bench_06", 24),   # 알약통을 쏟은 상황
]


def make_background(size: int):
    """SD가 만든 나무 테이블을 흐리게 깔아 배경으로 쓴다.

    알약이 없는 깨끗한 바닥이 필요하므로, 원본을 크게 확대해 알약이
    화면 밖으로 나가게 한 뒤 흐리게 만든다.
    """
    import cv2
    import numpy as np

    if BACKGROUND.exists():
        img = cv2.imread(str(BACKGROUND))
        if img is not None:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    # 배경 이미지가 없으면 나무색 단색으로 대체
    return np.full((size, size, 3), (60, 82, 110), dtype=np.uint8)


def draw_pills(bg, count: int, rng: random.Random):
    """겹치지 않게 알약 N개를 올린다. 좌표를 정답으로 함께 돌려준다."""
    import cv2
    import numpy as np

    img = bg.copy()
    placed: list[tuple[int, int, int, int]] = []
    radius_range = (16, 26)
    margin = 34
    attempts = 0

    while len(placed) < count and attempts < count * 600:
        attempts += 1
        rx = rng.randint(*radius_range)
        ry = int(rx * rng.uniform(0.72, 1.0))
        cx = rng.randint(margin, SIZE - margin)
        cy = rng.randint(margin, SIZE - margin)

        # 서로 넉넉히 떨어뜨린다 (붙어 있으면 사람이 세도 헷갈린다)
        ok = all((cx - px) ** 2 + (cy - py) ** 2 > (rx + pr + 10) ** 2
                 for px, py, pr, _ in placed)
        if not ok:
            continue
        placed.append((cx, cy, rx, ry))

    for cx, cy, rx, ry in placed:
        angle = rng.randint(0, 180)
        # 그림자
        cv2.ellipse(img, (cx + 3, cy + 3), (rx, ry), angle, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
        # 알약 본체 (살짝 미색)
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, (238, 242, 245), -1, cv2.LINE_AA)
        # 하이라이트
        cv2.ellipse(img, (cx - rx // 4, cy - ry // 4), (rx // 3, ry // 4), angle,
                    0, 360, (252, 253, 254), -1, cv2.LINE_AA)
        # 테두리
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, (205, 210, 215), 1, cv2.LINE_AA)

    return img, placed


def build() -> None:
    import cv2

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    bg = make_background(SIZE)
    entries = []

    for uid, count in CASES:
        rng = random.Random(SEED + count)
        img, placed = draw_pills(bg, count, rng)
        path = BENCH_DIR / f"{uid}.png"
        cv2.imwrite(str(path), img)
        entries.append({
            "id": uid,
            "file": path.name,
            "ground_truth_count": len(placed),
            "requested_count": count,
            "positions": [{"cx": c, "cy": y, "rx": r, "ry": q} for c, y, r, q in placed],
        })
        print(f"  {uid}  알약 {len(placed):>2}개 (요청 {count})  → {path.name}")

    (BENCH_DIR / "ground_truth.json").write_text(json.dumps({
        "_meta": {
            "seed": SEED,
            "size": SIZE,
            "background": "SD-Turbo 생성 나무 테이블 이미지의 빈 영역을 확대·블러 처리",
            "note": "알약은 프로그램으로 그렸으므로 개수가 정확히 알려져 있다.",
        },
        "images": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n정답 파일: {BENCH_DIR / 'ground_truth.json'}")


def evaluate() -> int:
    from image_pipeline import count_pills

    gt_path = BENCH_DIR / "ground_truth.json"
    if not gt_path.exists():
        sys.exit("[에러] 벤치마크가 없습니다. 먼저 --build 를 실행하세요.")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    print(f"\n{'이미지':12} {'정답':>5} {'예측':>5} {'오차':>6} {'분할초':>8}")
    print("─" * 44)

    rows, abs_errs = [], []
    for e in gt["images"]:
        p = BENCH_DIR / e["file"]
        r = count_pills(p)
        pred = r.get("pill_count", -1)
        truth = e["ground_truth_count"]
        err = pred - truth
        abs_errs.append(abs(err))
        rows.append({
            "id": e["id"], "truth": truth, "predicted": pred,
            "error": err, "segment_sec": r.get("segment_sec"),
            "output": r.get("output"),
        })
        print(f"{e['id']:12} {truth:>5} {pred:>5} {err:>+6} {r.get('segment_sec', 0):>7.2f}s")

    n = len(rows)
    exact = sum(1 for r in rows if r["error"] == 0)
    within1 = sum(1 for r in rows if abs(r["error"]) <= 1)
    mae = sum(abs_errs) / n if n else 0
    # 개수 대비 상대 오차
    rel = sum(abs(r["error"]) / max(1, r["truth"]) for r in rows) / n if n else 0

    print("─" * 44)
    print(f"정확히 맞춤      {exact}/{n}")
    print(f"±1개 이내       {within1}/{n}")
    print(f"평균 절대 오차   {mae:.2f}개")
    print(f"평균 상대 오차   {rel * 100:.1f}%")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "pill_benchmark.json"
    out.write_text(json.dumps({
        "summary": {
            "count": n, "exact": exact, "within_1": within1,
            "mae": round(mae, 3), "relative_error": round(rel, 4),
        },
        "note": (
            "정답을 아는 합성 이미지로 측정했다. 실제 사진에서는 조명·그림자·"
            "겹침 때문에 이보다 나쁠 것이다."
        ),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="알약 세기 벤치마크")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    if args.build or not args.eval:
        print("벤치마크 이미지 생성")
        build()
    if args.eval or not args.build:
        return evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
