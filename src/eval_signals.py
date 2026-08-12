#!/usr/bin/env python3
"""특이사항 추출 채점 (성공 기준 S5).

corpus.json 에 손으로 라벨링해둔 signals 를 정답으로 놓고,
추출기가 그중 몇 개를 잡아내는지 잰다.

채점 방식 — **신호 유형(type) 단위 재현율**
    정답 라벨은 사람이 쓴 자연어 설명(detail)이라 문자열로 맞출 수 없다.
    그래서 "그 발화에 health 신호가 있다고 라벨했는데, 추출기도 health를 뽑았는가"로 센다.
    느슨한 기준임을 인정하고, 대신 정밀도(precision)도 함께 보고해
    "아무거나 다 뽑아서 재현율만 올리는" 경우를 걸러낸다.

두 가지 입력으로 채점한다:
  1. 정답 전사문 기준  — 추출기 자체의 성능
  2. STT 출력 기준     — 전사 오류까지 포함한 실제 파이프라인 성능
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_signals import extract  # noqa: E402

CORPUS = ROOT / "data" / "reference" / "corpus.json"
RESULTS_DIR = ROOT / "results"


def score(pairs: list[tuple[str, list[str], list[str]]]) -> dict[str, Any]:
    """(id, 정답 type 목록, 예측 type 목록) 들로 재현율/정밀도를 계산한다."""
    tp = fn = fp = 0
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    misses, spurious = [], []

    for uid, golds, preds in pairs:
        gset, pset = set(golds), set(preds)
        for t in gset:
            if t in pset:
                tp += 1
                per_type[t]["tp"] += 1
            else:
                fn += 1
                per_type[t]["fn"] += 1
                misses.append({"id": uid, "missed_type": t})
        for t in pset - gset:
            fp += 1
            per_type[t]["fp"] += 1
            spurious.append({"id": uid, "extra_type": t})

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0

    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "per_type": {
            t: {
                **v,
                "recall": round(v["tp"] / (v["tp"] + v["fn"]), 4) if (v["tp"] + v["fn"]) else None,
            }
            for t, v in sorted(per_type.items())
        },
        "missed": misses,
        "spurious": spurious,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="특이사항 추출 채점 (S5)")
    ap.add_argument("--stt-result", default=None,
                    help="STT 결과 JSON 경로. 주면 전사 출력 기준으로도 채점한다.")
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    utts = corpus["utterances"]
    gold = {u["id"]: sorted({s["type"] for s in u.get("signals", [])}) for u in utts}
    ref_text = {u["id"]: u["text"] for u in utts}

    # ── 1) 정답 전사문 기준 ─────────────────────────────────────────────
    pairs_ref = [
        (uid, gold[uid], sorted({s.signal_type for s in extract(ref_text[uid])}))
        for uid in gold
    ]
    res_ref = score(pairs_ref)

    print("=" * 74)
    print("특이사항 추출 채점 (S5)")
    print("=" * 74)
    print()
    print("[1] 정답 전사문 기준 — 추출기 자체 성능")
    print(f"    재현율 {res_ref['recall']:.3f}   정밀도 {res_ref['precision']:.3f}   F1 {res_ref['f1']:.3f}")
    print(f"    TP {res_ref['true_positive']}  FN {res_ref['false_negative']}  FP {res_ref['false_positive']}")
    print()
    print("    유형별 재현율:")
    for t, v in res_ref["per_type"].items():
        r = v["recall"]
        print(f"      {t:12} {('%.3f' % r) if r is not None else '  —  '}  (TP {v['tp']}, FN {v['fn']}, FP {v['fp']})")

    payload: dict[str, Any] = {
        "criterion": "S5 특이사항 추출 재현율",
        "target": 0.70,
        "scoring": "신호 유형(type) 단위 재현율. 정밀도 병기.",
        "on_reference_text": res_ref,
    }

    # ── 2) STT 출력 기준 ────────────────────────────────────────────────
    if args.stt_result:
        stt = json.loads(Path(args.stt_result).read_text(encoding="utf-8"))
        hyp = {u["id"]: u.get("hypothesis", "") for u in stt["utterances"]}
        pairs_stt = [
            (uid, gold[uid], sorted({s.signal_type for s in extract(hyp.get(uid, ""))}))
            for uid in gold if uid in hyp
        ]
        res_stt = score(pairs_stt)
        payload["on_stt_output"] = res_stt
        payload["stt_backend"] = stt.get("backend")

        print()
        print(f"[2] STT 출력 기준 — 전사 오류 포함 실제 파이프라인 ({stt.get('backend')})")
        print(f"    재현율 {res_stt['recall']:.3f}   정밀도 {res_stt['precision']:.3f}   F1 {res_stt['f1']:.3f}")
        drop = res_ref["recall"] - res_stt["recall"]
        print(f"    → 전사 오류로 인한 재현율 손실: {drop:+.3f}")
        if res_stt["missed"]:
            print("    놓친 신호:")
            for m in res_stt["missed"]:
                print(f"      {m['id']}  {m['missed_type']}")

    verdict = res_ref["recall"] >= 0.70
    print()
    print("─" * 74)
    print(f"S5 기준(재현율 ≥ 0.70): {'달성' if verdict else '미달'}  (실측 {res_ref['recall']:.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "signal_extraction_score.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
