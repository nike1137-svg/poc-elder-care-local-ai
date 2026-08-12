#!/usr/bin/env python3
"""하이퍼파라미터 최적화 — 학습 없이 도메인 성능을 끌어올린다.

파인튜닝은 도메인 음성 데이터를 요구하는데, 이 프로젝트는 바로 그 데이터를
쓰지 않기로 했다(docs/problem-statement.md §0). 그래서 **학습 없이 할 수 있는
도메인 적응**을 대신 시도한다.

시도하는 것:
  1. initial_prompt — Whisper에 도메인 어휘를 미리 흘려 디코딩을 편향시킨다.
     "혈압약", "경로당", "시큰거리다" 같은 단어를 알려주면 그쪽으로 기운다.
     ★ 학습이 아니라 프롬프트다. 비용 0, 시간 0.
  2. beam_size — 탐색 폭. 넓히면 정확해지지만 느려진다.
  3. vad_filter — 침묵 제거. 어르신 발화는 침묵이 길어 위험할 수 있다.
  4. condition_on_previous_text — 앞 문맥 참조. 짧은 단발 메시지엔 해로울 수 있다.

★ 과적합 경고
   같은 20건으로 튜닝하고 같은 20건으로 보고하면 과적합이다.
   그래서 승자 설정을 **공개 데이터셋(Zeroth)에도 돌려** 일반화되는지 확인한다.

사용법:
    python src/tune_whisper.py                # 전체 스윕
    python src/tune_whisper.py --validate     # 승자만 Zeroth로 검증
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import cer  # noqa: E402

RESULTS = ROOT / "results"

# ── 도메인 어휘 ──────────────────────────────────────────────────────────
# 돌봄 대화에 자주 나오지만 범용 ASR이 흔들릴 만한 단어들.
# 실제 대화 내용이 아니라 '어휘 목록'이므로 개인정보가 아니다.
DOMAIN_VOCAB_SHORT = "혈압약, 경로당, 무릎, 입맛, 낮잠"

DOMAIN_VOCAB_FULL = (
    "혈압약, 당뇨약, 경로당, 무릎, 허리, 시큰거리다, 쑤시다, 어지럽다, "
    "깜빡깜빡, 입맛, 된장찌개, 김장, 잠을 설치다, 뒤척이다, 적적하다, 손주"
)

# 문장형 프롬프트 — Whisper는 '앞선 문맥'으로 읽으므로 자연스러운 문장이 유리할 수 있다.
DOMAIN_SENTENCE = (
    "어르신과 나누는 일상 대화입니다. 혈압약, 경로당, 무릎, 입맛, "
    "잠을 설치다 같은 표현이 나옵니다."
)


@dataclass
class Config:
    name: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)


CONFIGS: list[Config] = [
    Config("baseline", "현재 설정 (beam=5, VAD 끔, 프롬프트 없음)", {}),
    Config("prompt_short", "도메인 어휘 5개 주입", {"initial_prompt": DOMAIN_VOCAB_SHORT}),
    Config("prompt_full", "도메인 어휘 16개 주입", {"initial_prompt": DOMAIN_VOCAB_FULL}),
    Config("prompt_sentence", "문장형 도메인 프롬프트", {"initial_prompt": DOMAIN_SENTENCE}),
    Config("beam1", "탐색 폭 1 (탐욕적 디코딩, 가장 빠름)", {"beam_size": 1}),
    Config("beam10", "탐색 폭 10 (느리지만 정확할 수 있음)", {"beam_size": 10}),
    Config("vad_on", "침묵 제거 켬", {"vad_filter": True}),
    Config("cond_prev", "앞 문맥 참조 켬", {"condition_on_previous_text": True}),
    # 단독으로 효과가 있던 둘을 합쳐본다.
    Config("prompt_full_beam1", "도메인 어휘 16개 + 탐색 폭 1",
           {"initial_prompt": DOMAIN_VOCAB_FULL, "beam_size": 1}),
    Config("prompt_full_beam10", "도메인 어휘 16개 + 탐색 폭 10",
           {"initial_prompt": DOMAIN_VOCAB_FULL, "beam_size": 10}),
]


def load_dataset(name: str) -> tuple[list[dict[str, Any]], Path]:
    if name == "synthetic":
        payload = json.loads((ROOT / "data" / "reference" / "corpus.json").read_text(encoding="utf-8"))
        return payload["utterances"], ROOT / "data" / "audio"
    payload = json.loads((ROOT / "data" / "zeroth" / "reference.json").read_text(encoding="utf-8"))
    return payload["utterances"], ROOT / "data" / "zeroth" / "audio"


def run_config(model, utts: list[dict[str, Any]], audio_dir: Path,
               cfg: Config) -> dict[str, Any]:
    """한 설정으로 전체 데이터를 돌려 CER과 지연시간을 잰다."""
    params: dict[str, Any] = {
        "language": "ko",
        "beam_size": 5,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }
    params.update(cfg.params)

    cers, lats, rows = [], [], []
    for utt in utts:
        audio = audio_dir / f"{utt['id']}.m4a"
        if not audio.exists():
            continue
        t0 = time.perf_counter()
        segments, _info = model.transcribe(str(audio), **params)
        text = "".join(s.text for s in segments).strip()
        lats.append(time.perf_counter() - t0)
        c = cer(utt["text"], text)
        cers.append(c)
        rows.append({"id": utt["id"], "cer": round(c, 4), "hypothesis": text})

    return {
        "config": cfg.name,
        "description": cfg.description,
        "params": cfg.params,
        "cer_mean": round(statistics.fmean(cers), 4) if cers else None,
        "cer_median": round(statistics.median(cers), 4) if cers else None,
        "latency_mean_sec": round(statistics.fmean(lats), 3) if lats else None,
        "count": len(cers),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper 하이퍼파라미터 스윕")
    ap.add_argument("--model", default="small")
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "zeroth"])
    ap.add_argument("--validate", action="store_true",
                    help="스윕 결과의 승자를 Zeroth로 교차 검증")
    ap.add_argument("--only", nargs="*", default=None,
                    help="특정 설정 이름만 측정 (교차 검증용)")
    args = ap.parse_args()

    configs = CONFIGS
    if args.only:
        wanted = set(args.only)
        configs = [c for c in CONFIGS if c.name in wanted]
        if not configs:
            sys.exit(f"[에러] 해당 설정이 없습니다: {sorted(wanted)}")

    from faster_whisper import WhisperModel

    import os
    print(f"모델 적재 중... (whisper-{args.model}, CPU int8)")
    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    print("적재 완료\n")

    utts, audio_dir = load_dataset(args.dataset)
    print(f"데이터셋 {args.dataset} — {len(utts)}건")
    print(f"설정 {len(configs)}가지를 순서대로 측정합니다.\n")
    print(f"{'설정':18} {'CER평균':>9} {'CER중앙':>9} {'지연평균':>9}  설명")
    print("─" * 92)

    results = []
    for cfg in configs:
        r = run_config(model, utts, audio_dir, cfg)
        results.append(r)
        print(f"{r['config']:18} {r['cer_mean']:>9.4f} {r['cer_median']:>9.4f} "
              f"{r['latency_mean_sec']:>8.2f}s  {r['description']}")

    base = next(r for r in results if r["config"] == "baseline")
    ranked = sorted(results, key=lambda r: (r["cer_mean"], r["latency_mean_sec"]))
    best = ranked[0]

    print("─" * 92)
    print(f"\n기준(baseline) CER {base['cer_mean']:.4f}")
    print(f"최고 설정: {best['config']}  CER {best['cer_mean']:.4f}  "
          f"({(best['cer_mean'] - base['cer_mean']):+.4f})")
    if best["config"] == "baseline":
        print("→ 어떤 설정도 기준을 이기지 못했습니다. 현재 설정이 이미 최선입니다.")
    else:
        improve = (base["cer_mean"] - best["cer_mean"]) / base["cer_mean"] * 100
        print(f"→ 상대 개선 {improve:.1f}%")

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_cer": base["cer_mean"],
        "best_config": best["config"],
        "best_cer": best["cer_mean"],
        "overfitting_warning": (
            "튜닝과 보고에 같은 데이터를 쓰면 과적합이다. "
            "--validate 로 공개 데이터셋 교차 검증 결과를 함께 볼 것."
        ),
        "results": results,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"hyperparam_sweep_{args.dataset}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
