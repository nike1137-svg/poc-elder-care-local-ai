#!/usr/bin/env python3
"""회상 이미지 — 어르신이 떠올린 옛 기억을 그림으로 만든다.

★ 왜 이 기능인가 (억지로 Diffusion을 넣은 게 아니다)

  회상요법(reminiscence therapy)은 노인 돌봄에서 실제로 쓰이는 기법이다.
  옛 사진이나 물건을 매개로 기억을 꺼내 대화를 이어가고 정서를 안정시킨다.
  현장에서는 보통 **옛날 사진첩**을 쓴다.

  문제는 사진첩에 없는 기억은 못 꺼낸다는 것이다.
  "예전에 마당에서 김장하던" 장면은 대부분의 집에 사진으로 남아 있지 않다.

  로컬 이미지 생성 모델은 이 빈틈을 메운다. 어르신이 말한 장면을 그 자리에서
  그려서 보여주고, 그걸 매개로 대화를 잇는다.

★ 왜 로컬이어야 하나

  회상 발화에는 가족사·지명·병력 같은 사적인 정보가 섞인다.
  그걸 이미지 생성 API로 보내면 STT를 로컬로 내린 의미가 없어진다.
  SD-Turbo는 CPU에서 20초면 한 장을 만든다. 밖으로 보낼 이유가 없다.

★ 안전 원칙

  - **실존 인물을 그리지 않는다.** 프롬프트에서 인물 묘사를 일반화한다.
  - 발화를 그대로 프롬프트에 넣지 않는다. 장면 키워드만 뽑아 쓴다.
  - 생성 결과는 '기억의 재현'이 아니라 '대화의 마중물'이다. 사실과 다를 수 있다.

사용법:
    python src/reminiscence.py "어릴 때 시골 학교 걸어다니던 기억이 나네"
    python src/reminiscence.py --from-audio data/audio/u10.m4a
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = ROOT / "results" / "reminiscence"
MODEL_ID = "stabilityai/sd-turbo"

# 회상 발화 신호 — 과거를 가리키는 표현
REMINISCENCE_CUES = (
    r"(옛날|예전|어릴\s*때|젊었을\s*때|그\s*때|시절|기억이\s*나|생각이\s*나|추억)",
    r"(했었|갔었|살았|다녔|먹었었)",
)

# 장면 키워드 → 영어 프롬프트 조각
# 발화를 통째로 번역해 넣지 않는다. 알려진 장면만 골라 안전하게 매핑한다.
SCENE_MAP: list[tuple[str, str]] = [
    (r"김장", "making kimchi together in a traditional Korean courtyard"),
    (r"시장|장터", "a busy old Korean traditional market street"),
    (r"학교|학굣길", "children walking home from a rural Korean school"),
    (r"논|밭|농사", "green rice paddies in the Korean countryside"),
    (r"바다|해변", "a quiet Korean seaside village"),
    (r"명절|설날|추석", "a Korean holiday family gathering table"),
    (r"소풍", "a school picnic in a Korean park"),
    (r"우물|빨래터", "an old Korean village well and washing place"),
    (r"기차|역", "an old Korean countryside train station"),
    (r"눈|겨울", "a snowy Korean village in winter"),
]

# 어느 키워드에도 안 걸릴 때의 기본 장면
FALLBACK_SCENE = "a warm nostalgic scene of old Korean countryside daily life"

STYLE_SUFFIX = (
    ", warm nostalgic memory, soft painting style, gentle colors, "
    "no identifiable faces, illustration"
)


def is_reminiscence(text: str) -> bool:
    """회상 발화인지 판단한다. 두 종류 단서 중 하나라도 걸리면 회상으로 본다."""
    return any(re.search(p, text) for p in REMINISCENCE_CUES)


def build_prompt(text: str) -> tuple[str, str]:
    """발화에서 장면을 골라 영어 프롬프트를 만든다. (프롬프트, 매칭된 키워드)

    ★ 발화 원문을 프롬프트에 넣지 않는다.
      원문에는 이름·지명·병력이 섞일 수 있고, 그대로 넣으면
      그것이 이미지에 반영되거나 로그에 남는다.
      미리 정해둔 안전한 장면 목록에서만 고른다.
    """
    for pattern, scene in SCENE_MAP:
        if re.search(pattern, text):
            return scene + STYLE_SUFFIX, pattern
    return FALLBACK_SCENE + STYLE_SUFFIX, "(기본)"


def generate(prompt: str, dst: Path, steps: int = 2, seed: int = 20260812) -> dict[str, Any]:
    import os

    import torch
    from diffusers import AutoPipelineForText2Image

    torch.set_num_threads(os.cpu_count() or 8)
    t0 = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, safety_checker=None).to("cpu")
    load_sec = time.perf_counter() - t0

    gen = torch.Generator(device="cpu").manual_seed(seed)
    t1 = time.perf_counter()
    img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
               height=512, width=512, generator=gen).images[0]
    gen_sec = time.perf_counter() - t1

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return {
        "output": str(dst),
        "load_sec": round(load_sec, 2),
        "generate_sec": round(gen_sec, 2),
        "steps": steps,
        "external_requests": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="회상 이미지 생성 (로컬)")
    ap.add_argument("text", nargs="*", help="어르신 발화 (텍스트)")
    ap.add_argument("--from-audio", help="음성 파일에서 전사해 사용")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--force", action="store_true",
                    help="회상 발화가 아니어도 생성")
    args = ap.parse_args()

    if args.from_audio:
        from stt_whisper import WhisperBackend

        be = WhisperBackend()
        be.warmup()
        r = be.transcribe(Path(args.from_audio))
        if not r.ok:
            print(f"전사 실패: {r.failure.value} — {r.error_detail}")
            return 1
        text = r.text
        print(f"전사: {text}")
    elif args.text:
        text = " ".join(args.text)
    else:
        ap.error("발화 텍스트나 --from-audio 중 하나가 필요합니다.")

    rem = is_reminiscence(text)
    print(f"회상 발화 판정: {'예' if rem else '아니오'}")

    if not rem and not args.force:
        print("회상 발화가 아니라 이미지를 만들지 않습니다. (--force 로 강제 가능)")
        return 0

    prompt, matched = build_prompt(text)
    print(f"매칭 키워드: {matched}")
    print(f"프롬프트: {prompt}")
    print("\n생성 중... (CPU, 20초 내외)")

    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text)[:28] or "memory"
    info = generate(prompt, OUT_DIR / f"{slug}.png", steps=args.steps)

    print(f"완료 {info['generate_sec']}s  → {info['output']}")
    print(f"외부 전송 {info['external_requests']}건")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "last_run.json").write_text(json.dumps({
        "text": text, "is_reminiscence": rem, "matched": matched,
        "prompt": prompt, **info,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
