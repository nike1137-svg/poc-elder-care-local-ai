#!/usr/bin/env python3
"""이미지 파이프라인 — 어르신이 보낸 사진도 집 안에서만 처리한다.

온담이에는 이미 '사진 보내면 인식·설명' 기능이 있고, 그 처리는 클라우드
범용 LLM이 맡는다. 음성과 똑같은 문제다 — **사진이 밖으로 나간다.**
사진은 음성보다 더 직접적인 개인정보다. 얼굴, 집 내부, 약봉투의 처방 정보가
그대로 담긴다.

그래서 음성에 한 것과 같은 개선을 사진에도 적용한다:
**클라우드 범용 LLM → 로컬 전용 모델(YOLO + SAM)**

─────────────────────────────────────────────────────────────────────
용도 1. 프라이버시 마스킹  (YOLO → SAM)
    사진 속 인물을 찾아 정밀 분할 후 가린다.
    보관·전달 전에 얼굴이 남지 않게 한다.

용도 2. 복약 확인용 알약 세기  (SAM 자동 분할)
    "약 먹었나 모르겠네"라는 발화(복약 누락 신호)를 사진으로 확인한다.
─────────────────────────────────────────────────────────────────────

★ 모델 선정에서 실제로 부딪힌 문제

    처음에는 알약 세기에도 YOLO를 쓰려 했다. 그런데 **YOLO11n은 COCO
    사전학습 모델이고, COCO 80개 클래스에 '알약'이 없다.** 아무리 돌려도
    알약은 안 잡힌다. 파인튜닝하려면 라벨링된 약 사진이 필요한데
    그 데이터가 없다(그리고 실사진은 쓰지 않기로 했다).

    그래서 역할을 나눴다.
      - YOLO  → COCO에 있는 `person` 탐지 (마스킹 용도). 여기선 제 실력을 낸다.
      - SAM   → 클래스 개념 없이 '덩어리'를 나누므로, 학습 없이도 알약을 분리한다.
                면적·형태 필터로 알약 후보만 남긴다.

    "배운 모델을 다 넣었다"가 아니라, 각자 되는 자리에 배치한 결과다.

사용법:
    python src/image_pipeline.py --mask data/images/person_01.png
    python src/image_pipeline.py --count-pills data/images/pill_01.png
    python src/image_pipeline.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "images"

# 마스킹 대상 COCO 클래스 — 사람은 무조건, 그 외 신원이 드러날 수 있는 것들
MASK_CLASSES = {"person"}

# 알약 후보 판정 기준 (SAM 마스크 필터)
PILL_MIN_AREA_RATIO = 0.0006   # 이미지 면적 대비 최소
PILL_MAX_AREA_RATIO = 0.06     # 이미지 면적 대비 최대
PILL_MIN_CIRCULARITY = 0.55    # 4πA/P² — 1에 가까울수록 원형
PILL_MAX_ASPECT = 2.4          # 가로세로 비


@dataclass
class Detection:
    label: str
    confidence: float
    box: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 모델을 프로세스당 한 번만 적재한다.
#
# ★ 왜 필요한가 — 데모 화면에서 드러난 문제
#   호출마다 새로 적재하던 때, 웹 데모에서 YOLO 탐지가 6.0초,
#   SAM 분할이 16.3초로 찍혔다. CLI 측정값(0.14초 / 1.6초)의 10배가 넘는다.
#   차이는 전부 **모델 적재 시간**이었다. 사진을 한 장 처리할 때마다
#   가중치를 디스크에서 다시 읽고 있었던 것이다.
_MODEL_CACHE: dict[str, Any] = {}


def _load_yolo():
    if "yolo" not in _MODEL_CACHE:
        from ultralytics import YOLO

        _MODEL_CACHE["yolo"] = YOLO("yolo11n.pt")
    return _MODEL_CACHE["yolo"]


def _load_sam(prompted: bool = True):
    """prompted=True면 MobileSAM(박스 프롬프트용), False면 FastSAM(자동 분할용)."""
    key = "mobile_sam" if prompted else "fast_sam"
    if key not in _MODEL_CACHE:
        if prompted:
            from ultralytics import SAM

            _MODEL_CACHE[key] = SAM("mobile_sam.pt")
        else:
            from ultralytics import FastSAM

            _MODEL_CACHE[key] = FastSAM("FastSAM-s.pt")
    return _MODEL_CACHE[key]


# ── 용도 1: 프라이버시 마스킹 ────────────────────────────────────────────
def mask_people(image_path: Path, mode: str = "pixelate") -> dict[str, Any]:
    """사진 속 인물을 YOLO로 찾고 SAM으로 정밀 분할해 가린다.

    박스만으로 가리면 배경까지 뭉개진다. SAM으로 사람 윤곽을 따내면
    필요한 부분만 정확히 가릴 수 있다.

    ★ 처음에 가우시안 블러를 썼다가 실패했다

      블러를 걸었는데도 **얼굴을 그대로 알아볼 수 있었다.** 커널이 43px인데
      얼굴이 150px이라 형태가 살아남은 것이다. 신원이 안 가려지면
      마스킹 기능이 있으나 마나다.

      그래서 두 가지를 바꿨다.
        1. 블러 → **모자이크(픽셀화)**. 블러는 복원 시도가 가능하지만
           픽셀화는 정보를 실제로 버린다.
        2. 고정 크기 → **탐지된 인물 크기에 비례**. 사람이 크게 찍히면
           블록도 커져야 얼굴이 뭉개진다.

    Args:
        mode: pixelate(기본, 권장) / blur / fill
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        return {"ok": False, "error": f"이미지를 읽지 못했습니다: {image_path}"}
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    det = _load_yolo()
    res = det(str(image_path), verbose=False)[0]
    names = det.names

    boxes, dets = [], []
    for b, c, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        label = names[int(c)]
        if label not in MASK_CLASSES:
            continue
        xy = [int(v) for v in b.tolist()]
        boxes.append(xy)
        dets.append(Detection(label, round(float(conf), 3), xy))
    detect_sec = time.perf_counter() - t0

    if not boxes:
        return {
            "ok": True, "image": str(image_path), "detections": [],
            "masked_count": 0, "detect_sec": round(detect_sec, 3),
            "segment_sec": 0.0, "external_requests": 0,
            "note": "가릴 인물이 탐지되지 않았습니다.",
        }

    t1 = time.perf_counter()
    sam = _load_sam(prompted=True)
    sres = sam(str(image_path), bboxes=boxes, verbose=False)[0]
    segment_sec = time.perf_counter() - t1

    # 가장 큰 인물 박스를 기준으로 가림 강도를 정한다.
    # 사람이 크게 찍혔으면 그만큼 세게 뭉개야 얼굴이 사라진다.
    largest = max((bx[2] - bx[0]) for bx in boxes)
    block = max(12, largest // 10)          # 모자이크 블록 한 변
    kernel = max(51, (largest // 3) | 1)    # 블러 모드용 커널 (홀수)

    out = img.copy()
    masked = 0
    if sres.masks is not None:
        for m in sres.masks.data:
            mask = m.cpu().numpy().astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            sel = mask.astype(bool)

            if mode == "fill":
                out[sel] = (0, 0, 0)
            elif mode == "blur":
                out = np.where(sel[..., None], cv2.GaussianBlur(out, (kernel, kernel), 0), out)
            else:
                # 모자이크: 축소했다 다시 키우면 세부 정보가 실제로 버려진다.
                small = cv2.resize(
                    out, (max(1, w // block), max(1, h // block)),
                    interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                out = np.where(sel[..., None], pixelated, out)
            masked += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"{image_path.stem}_masked.png"
    cv2.imwrite(str(dst), out)

    return {
        "ok": True,
        "image": str(image_path),
        "output": str(dst),
        "detections": [d.to_dict() for d in dets],
        "masked_count": masked,
        "detect_sec": round(detect_sec, 3),
        "segment_sec": round(segment_sec, 3),
        "external_requests": 0,
    }


# ── 용도 2: 알약 세기 ────────────────────────────────────────────────────
def count_pills(image_path: Path) -> dict[str, Any]:
    """SAM 자동 분할로 알약 후보를 세고, 근거 이미지를 남긴다.

    YOLO를 못 쓰는 이유는 모듈 상단 주석 참조 (COCO에 알약 클래스가 없다).
    SAM은 클래스를 모르는 대신 '덩어리'를 나누므로, 나온 마스크를
    면적·원형도·종횡비로 걸러 알약 후보만 남긴다.
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        return {"ok": False, "error": f"이미지를 읽지 못했습니다: {image_path}"}
    h, w = img.shape[:2]
    area_img = h * w

    t0 = time.perf_counter()
    sam = _load_sam(prompted=False)
    res = sam(str(image_path), verbose=False)[0]
    seg_sec = time.perf_counter() - t0

    if res.masks is None:
        return {"ok": True, "image": str(image_path), "pill_count": 0,
                "candidates": [], "segment_sec": round(seg_sec, 3),
                "external_requests": 0, "note": "마스크가 생성되지 않았습니다."}

    overlay = img.copy()
    kept: list[dict[str, Any]] = []
    total_masks = 0

    for m in res.masks.data:
        total_masks += 1
        mask = m.cpu().numpy().astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        area = float(mask.sum())
        ratio = area / area_img
        if not (PILL_MIN_AREA_RATIO <= ratio <= PILL_MAX_AREA_RATIO):
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * cv2.contourArea(cnt) / (peri * peri)
        if circularity < PILL_MIN_CIRCULARITY:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > PILL_MAX_ASPECT:
            continue

        kept.append({
            "box": [int(x), int(y), int(x + bw), int(y + bh)],
            "area_ratio": round(ratio, 5),
            "circularity": round(float(circularity), 3),
            "aspect": round(float(aspect), 2),
        })
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 200, 0), 2)

    # 겹치는 후보 제거 (SAM은 같은 물체를 여러 크기로 내놓는다)
    kept = _dedupe(kept)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"{image_path.stem}_pills.png"
    vis = img.copy()
    for k in kept:
        x1, y1, x2, y2 = k["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(vis, f"count={len(kept)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    cv2.imwrite(str(dst), vis)

    return {
        "ok": True,
        "image": str(image_path),
        "output": str(dst),
        "pill_count": len(kept),
        "masks_total": total_masks,
        "candidates": kept,
        "segment_sec": round(seg_sec, 3),
        "external_requests": 0,
        "filters": {
            "area_ratio": [PILL_MIN_AREA_RATIO, PILL_MAX_AREA_RATIO],
            "min_circularity": PILL_MIN_CIRCULARITY,
            "max_aspect": PILL_MAX_ASPECT,
        },
    }


def _dedupe(cands: list[dict[str, Any]], iou_thr: float = 0.5,
            contain_thr: float = 0.65) -> list[dict[str, Any]]:
    """중복 후보를 하나만 남긴다.

    ★ IoU만으로는 부족하다

      SAM은 같은 물체를 여러 크기로 내놓는다. 작은 박스가 큰 박스 **안에 완전히
      들어가는** 경우, IoU는 (작은 넓이 / 큰 넓이)라 0.5를 밑돌아 중복으로
      걸러지지 않는다. 실제로 알약 하나에 박스가 두 개 그려져 과다 집계됐다.

      그래서 '작은 쪽 기준 겹침 비율'(intersection over smaller)도 함께 본다.
      한쪽이 다른 쪽에 대부분 포함되면 같은 물체로 판단한다.
    """
    def _inter(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        return float(iw * ih)

    def _area(a: list[int]) -> float:
        return float(max(0, a[2] - a[0]) * max(0, a[3] - a[1]))

    def is_dup(a: list[int], b: list[int]) -> bool:
        inter = _inter(a, b)
        if inter <= 0:
            return False
        aa, ab = _area(a), _area(b)
        union = aa + ab - inter
        if union > 0 and inter / union >= iou_thr:
            return True
        smaller = min(aa, ab)
        return smaller > 0 and inter / smaller >= contain_thr

    out: list[dict[str, Any]] = []
    # 원형도가 높은(알약다운) 후보를 먼저 채택한다.
    for c in sorted(cands, key=lambda x: -x["circularity"]):
        if all(not is_dup(c["box"], k["box"]) for k in out):
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="이미지 파이프라인 (로컬 YOLO + SAM)")
    ap.add_argument("--mask", nargs="*", help="인물 마스킹할 이미지")
    ap.add_argument("--count-pills", nargs="*", dest="pills", help="알약 셀 이미지")
    ap.add_argument("--all", action="store_true", help="data/images 전체 처리")
    ap.add_argument("--mode", default="pixelate", choices=["pixelate", "blur", "fill"],
                    help="가림 방식 (기본 pixelate)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    jobs: list[tuple[str, Path]] = []
    if args.all:
        d = ROOT / "data" / "images"
        jobs += [("mask", p) for p in sorted(d.glob("person_*.png"))]
        jobs += [("pills", p) for p in sorted(d.glob("pill_*.png"))]
    for p in args.mask or []:
        jobs.append(("mask", Path(p)))
    for p in args.pills or []:
        jobs.append(("pills", Path(p)))

    if not jobs:
        ap.error("처리할 이미지를 지정하세요 (--mask / --count-pills / --all)")

    results = []
    for kind, path in jobs:
        r = mask_people(path, mode=args.mode) if kind == "mask" else count_pills(path)
        r["task"] = kind
        results.append(r)
        if not args.json:
            if not r.get("ok"):
                print(f"[실패] {path.name}: {r.get('error')}")
            elif kind == "mask":
                print(f"[마스킹] {path.name}  인물 {len(r['detections'])}명 탐지 → "
                      f"{r['masked_count']}개 가림  "
                      f"(탐지 {r['detect_sec']}s + 분할 {r['segment_sec']}s, 외부전송 0건)")
                print(f"         → {r.get('output')}")
            else:
                print(f"[알약]   {path.name}  마스크 {r['masks_total']}개 중 "
                      f"알약 후보 {r['pill_count']}개  "
                      f"(분할 {r['segment_sec']}s, 외부전송 0건)")
                print(f"         → {r.get('output')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = OUT_DIR / "image_pipeline_results.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n결과 저장: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
