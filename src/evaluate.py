#!/usr/bin/env python3
"""개선 효과 검증 — 기존 방식과 개선안을 같은 데이터에 나란히 돌려 비교한다.

측정 항목 (docs/problem-statement.md §5 성공 기준):
  S1 문자오류율(CER)   — 정규화 후 측정
  S2 건당 처리 시간     — 벽시계 시간 및 실시간 대비 배속(RTF)
  S3 건당 비용          — 토큰 사용량 기반. 로컬은 정의상 0원
  S4 외부 전송 건수     — 음성이 기기 밖으로 나간 횟수

사용법:
    python src/evaluate.py --backend whisper --dataset synthetic
    python src/evaluate.py --backend gemini  --dataset both
    python src/evaluate.py --backend whisper --model medium --dataset zeroth
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import cer, normalize  # noqa: E402
from stt_base import STTResult  # noqa: E402

RESULTS_DIR = ROOT / "results"

# ── 비용 가정 ────────────────────────────────────────────────────────────
# ★ 아래 단가는 "가정"이다. 요금은 수시로 바뀌므로 수치를 그대로 신뢰하지 말고
#   실행 시점의 공식 요금표로 갱신할 것. 스크립트는 토큰 사용량을 실측해 기록하므로,
#   단가만 바꾸면 비용이 다시 계산된다.
COST_ASSUMPTION = {
    "note": "실행 시점의 공식 요금표로 갱신 필요. 토큰 사용량은 실측값이다.",
    "currency": "USD",
    "input_per_1m_tokens": 0.30,
    "output_per_1m_tokens": 2.50,
    "usd_to_krw": 1380,
}


def load_env() -> None:
    """`.env` 로더는 src/envfile.py 로 일원화했다 (빈 값이 실제 키를 가리던 버그)."""
    from envfile import load_env as _load

    _load()


def load_dataset(name: str) -> tuple[list[dict[str, Any]], Path]:
    """(발화 목록, 오디오 디렉터리)를 돌려준다."""
    if name == "synthetic":
        payload = json.loads((ROOT / "data" / "reference" / "corpus.json").read_text(encoding="utf-8"))
        return payload["utterances"], ROOT / "data" / "audio"
    if name == "zeroth":
        ref = ROOT / "data" / "zeroth" / "reference.json"
        if not ref.exists():
            sys.exit("[에러] Zeroth 데이터가 없습니다. 먼저 `python src/prepare_zeroth.py` 를 실행하세요.")
        payload = json.loads(ref.read_text(encoding="utf-8"))
        return payload["utterances"], ROOT / "data" / "zeroth" / "audio"
    sys.exit(f"[에러] 알 수 없는 데이터셋: {name}")


def build_backend(kind: str, model: str | None):
    if kind == "whisper":
        from stt_whisper import WhisperBackend

        return WhisperBackend(model_size=model or os.environ.get("WHISPER_MODEL", "small"))
    if kind == "gemini":
        from stt_gemini import GeminiBackend

        return GeminiBackend(model=model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    sys.exit(f"[에러] 알 수 없는 백엔드: {kind}")


def estimate_cost_usd(results: list[STTResult]) -> float:
    """토큰 사용량 실측치 × 단가 가정. 토큰 정보가 없으면 0."""
    tin = sum((r.meta.get("prompt_tokens") or 0) for r in results)
    tout = sum((r.meta.get("output_tokens") or 0) for r in results)
    return (
        tin / 1_000_000 * COST_ASSUMPTION["input_per_1m_tokens"]
        + tout / 1_000_000 * COST_ASSUMPTION["output_per_1m_tokens"]
    )


def summarize(results: list[STTResult], per_utt: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    cers = [d["cer"] for d in per_utt if d["cer"] >= 0]
    lats = [r.latency_sec for r in ok]
    rtfs = [r.realtime_factor for r in ok if r.realtime_factor > 0]
    audio_total = sum(r.audio_duration_sec for r in results)
    external = sum(1 for r in results if r.meta.get("external_request"))
    cost_usd = estimate_cost_usd(ok)

    failures: dict[str, int] = {}
    for r in results:
        if not r.ok:
            failures[r.failure.value] = failures.get(r.failure.value, 0) + 1

    def _mean(xs: list[float]) -> float | None:
        return round(statistics.fmean(xs), 4) if xs else None

    def _med(xs: list[float]) -> float | None:
        return round(statistics.median(xs), 4) if xs else None

    return {
        "count": len(results),
        "success": len(ok),
        "failed": len(results) - len(ok),
        "failure_breakdown": failures,
        "audio_total_sec": round(audio_total, 1),
        # S1
        "cer_mean": _mean(cers),
        "cer_median": _med(cers),
        "cer_worst": round(max(cers), 4) if cers else None,
        "cer_best": round(min(cers), 4) if cers else None,
        # S2
        "latency_mean_sec": _mean(lats),
        "latency_median_sec": _med(lats),
        "rtf_mean": _mean(rtfs),
        # S3
        "tokens_input": sum((r.meta.get("prompt_tokens") or 0) for r in ok) or None,
        "tokens_output": sum((r.meta.get("output_tokens") or 0) for r in ok) or None,
        "cost_usd_estimated": round(cost_usd, 6),
        "cost_krw_estimated": round(cost_usd * COST_ASSUMPTION["usd_to_krw"], 2),
        # S4
        "external_requests": external,
    }


def run(backend_kind: str, dataset_name: str, model: str | None, limit: int | None) -> dict[str, Any]:
    utts, audio_dir = load_dataset(dataset_name)
    if limit:
        utts = utts[:limit]

    be = build_backend(backend_kind, model)

    load_sec = 0.0
    if hasattr(be, "warmup"):
        print(f"모델 준비 중... ({be.name})")
        t0 = time.perf_counter()
        be.warmup()
        load_sec = time.perf_counter() - t0
        print(f"준비 완료 {load_sec:.1f}s")

    print(f"\n백엔드 {be.name} × 데이터셋 {dataset_name} — {len(utts)}건\n")
    print(f"{'id':5} {'상태':10} {'CER':>7} {'지연':>7} {'RTF':>6}  전사 결과")
    print("─" * 100)

    results: list[STTResult] = []
    per_utt: list[dict[str, Any]] = []

    for utt in utts:
        uid = utt["id"]
        audio = audio_dir / f"{uid}.m4a"
        r = be.transcribe(audio, audio_id=uid)
        results.append(r)

        ref = utt["text"]
        c = cer(ref, r.text) if r.ok else -1.0
        per_utt.append({
            "id": uid,
            "ok": r.ok,
            "failure": r.failure.value,
            "error_detail": r.error_detail,
            "reference": ref,
            "hypothesis": r.text,
            "reference_normalized": normalize(ref),
            "hypothesis_normalized": normalize(r.text),
            "cer": round(c, 4) if c >= 0 else -1,
            "latency_sec": round(r.latency_sec, 3),
            "audio_duration_sec": round(r.audio_duration_sec, 2),
            "realtime_factor": round(r.realtime_factor, 3),
            "style": utt.get("style"),
            "meta": r.meta,
        })

        status = "OK" if r.ok else f"실패:{r.failure.value}"
        cer_s = f"{c:.4f}" if c >= 0 else "  —   "
        print(f"{uid:5} {status:10} {cer_s:>7} {r.latency_sec:6.2f}s {r.realtime_factor:5.2f}  "
              f"{(r.text or r.error_detail)[:52]}")

    summary = summarize(results, per_utt)
    summary["model_load_sec"] = round(load_sec, 2)

    payload = {
        "backend": be.name,
        "backend_kind": backend_kind,
        "dataset": dataset_name,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cost_assumption": COST_ASSUMPTION,
        "summary": summary,
        "utterances": per_utt,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{backend_kind}_{(model or be.name).replace('/', '-')}_{dataset_name}.json"

    # ★ 기존 결과를 덮어쓰지 않는다.
    #   실제로 겪은 일: 클라우드 기준선을 19/20 성공으로 측정해 저장한 뒤,
    #   조건을 바꿔 다시 돌렸더니 무료 할당량이 소진되어 3/20 만 성공했고,
    #   그 나쁜 결과가 **좋은 결과를 덮어썼다.** 측정에 비용·할당량이 드는
    #   백엔드에서는 한 번 잃은 결과를 다시 얻지 못할 수 있다.
    if out.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = out.with_name(f"{out.stem}.prev_{stamp}.json")
        backup.write_bytes(out.read_bytes())
        print(f"기존 결과를 보존했습니다 → {backup.name}")

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("─" * 100)
    print(f"성공 {summary['success']}/{summary['count']}   "
          f"CER 평균 {summary['cer_mean']}   중앙값 {summary['cer_median']}")
    print(f"지연 평균 {summary['latency_mean_sec']}s   RTF 평균 {summary['rtf_mean']}   "
          f"외부 전송 {summary['external_requests']}건")
    print(f"비용 추정 ${summary['cost_usd_estimated']} (약 {summary['cost_krw_estimated']}원)")
    if summary["failure_breakdown"]:
        print(f"실패 분류: {summary['failure_breakdown']}")
    print(f"\n결과 저장: {out}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="STT 개선 효과 검증")
    ap.add_argument("--backend", choices=["whisper", "gemini"], required=True)
    ap.add_argument("--dataset", choices=["synthetic", "zeroth", "both"], default="synthetic")
    ap.add_argument("--model", default=None, help="모델 크기/이름 (기본: .env 설정)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N건만")
    args = ap.parse_args()

    load_env()
    names = ["synthetic", "zeroth"] if args.dataset == "both" else [args.dataset]
    for name in names:
        run(args.backend, name, args.model, args.limit)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
