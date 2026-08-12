#!/usr/bin/env python3
"""평가용 이미지 생성 (Stable Diffusion Turbo, 로컬 CPU).

★ 왜 이미지를 '생성'하는가

   음성 평가에서 실사용자 음성을 못 써서 대본을 TTS로 합성했던 것과 같은 이유다.
   어르신이 실제로 보낸 사진은 공개 저장소에 올릴 수 없다. 사진은 음성보다
   오히려 더 직접적인 개인정보다 — 얼굴, 집 내부, 약봉투의 처방 정보가 그대로 담긴다.

   그래서 **평가용 사진도 전량 생성한다.** 실물 사진은 이 저장소에 없다.

   덤으로 이 선택은 Diffusion 모델을 도메인상 필연적인 이유로 쓰게 만든다.
   "배운 걸 넣어봤다"가 아니라 "실사진을 못 쓰니 만들어야 한다"이다.

생성하는 두 갈래:
   1. pill_*  — 약/알약 사진. YOLO 객체탐지 → 복약 확인용
   2. person_* — 인물이 있는 사진. YOLO+SAM 마스킹 → 프라이버시 보호용
   3. memory_* — 회상 장면. 회상요법 대화 촉진용 (별도 용도)

사용법:
    python src/gen_images.py                 # 전체 생성
    python src/gen_images.py --group pill    # 일부만
    python src/gen_images.py --steps 2       # 품질/속도 조절
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "images"
MANIFEST = ROOT / "data" / "images" / "manifest.json"

MODEL_ID = "stabilityai/sd-turbo"

# 재현성: 같은 시드 → 같은 그림
SEED_BASE = 20260812

SCENES: list[dict[str, str]] = [
    # ── 약 사진 (YOLO 객체탐지 대상) ──────────────────────────────────
    {"id": "pill_01", "group": "pill",
     "prompt": "top down photo of several round white pills scattered on a wooden table, "
               "clear daylight, sharp focus, realistic photo"},
    {"id": "pill_02", "group": "pill",
     "prompt": "photo of a weekly pill organizer box with colorful pills, on a kitchen table, "
               "realistic photo, natural light"},
    {"id": "pill_03", "group": "pill",
     "prompt": "close up photo of a hand holding two white pills and a glass of water, "
               "realistic photo, indoor"},

    # ── 인물 사진 (YOLO+SAM 마스킹 대상) ──────────────────────────────
    {"id": "person_01", "group": "person",
     "prompt": "photo of an elderly Korean woman sitting on a sofa in a living room, "
               "smiling at the camera, realistic photo"},
    {"id": "person_02", "group": "person",
     "prompt": "photo of two elderly people talking at a community center table, "
               "realistic photo, indoor daylight"},
    {"id": "person_03", "group": "person",
     "prompt": "photo of an elderly man standing in a home kitchen holding a bowl, "
               "realistic photo"},

    # ── 벤치마크 배경 (알약 세기 정답셋용) ─────────────────────────────
    # 물체가 하나도 없는 깨끗한 바닥이어야 한다.
    # 배경에 알약이 섞여 있으면 정답 개수가 오염된다 (실제로 겪은 문제).
    {"id": "bg_wood", "group": "background",
     "prompt": "empty wooden table surface, top down view, plain, nothing on it, "
               "no objects, clean wood grain texture, even daylight, realistic photo"},

    # ── 회상 장면 (회상요법용) ────────────────────────────────────────
    {"id": "memory_kimchi", "group": "memory",
     "prompt": "an old Korean woman making kimchi in a traditional courtyard, "
               "warm nostalgic memory, soft painting style"},
    {"id": "memory_market", "group": "memory",
     "prompt": "a busy old Korean traditional market street in the 1970s, "
               "warm nostalgic memory, soft painting style"},
    {"id": "memory_school", "group": "memory",
     "prompt": "children walking home from a rural Korean school in the 1960s, "
               "warm nostalgic memory, soft painting style"},
]


def build_pipe(steps_hint: int):
    import torch
    from diffusers import AutoPipelineForText2Image

    torch.set_num_threads(os.cpu_count() or 8)
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, safety_checker=None
    )
    return pipe.to("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description="평가용 이미지 생성 (로컬 CPU)")
    ap.add_argument("--group", nargs="*", default=None,
                    choices=["pill", "person", "memory", "background"], help="생성할 갈래")
    ap.add_argument("--steps", type=int, default=2, help="추론 스텝 (1~4 권장)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 생성")
    args = ap.parse_args()

    scenes = SCENES
    if args.group:
        wanted = set(args.group)
        scenes = [s for s in SCENES if s["group"] in wanted]

    todo = [s for s in scenes
            if args.force or not (OUT_DIR / f"{s['id']}.png").exists()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not todo:
        print("생성할 이미지가 없습니다 (모두 존재). --force 로 다시 만들 수 있습니다.")
    else:
        import torch

        print(f"SD-Turbo 적재 중... (최초 1회 약 2.5GB 다운로드)")
        t0 = time.perf_counter()
        pipe = build_pipe(args.steps)
        print(f"적재 완료 {time.perf_counter() - t0:.1f}s")
        print(f"생성 대상 {len(todo)}건 / {args.steps}스텝 / {args.size}px\n")

        for i, scene in enumerate(todo, start=1):
            gen = torch.Generator(device="cpu").manual_seed(SEED_BASE + i)
            t = time.perf_counter()
            img = pipe(
                prompt=scene["prompt"],
                num_inference_steps=args.steps,
                guidance_scale=0.0,          # Turbo 계열은 guidance를 쓰지 않는다
                height=args.size, width=args.size,
                generator=gen,
            ).images[0]
            path = OUT_DIR / f"{scene['id']}.png"
            img.save(path)
            print(f"  [{i}/{len(todo)}] {scene['id']:16} {time.perf_counter() - t:5.1f}s  → {path.name}")

    # ── 매니페스트 기록 ────────────────────────────────────────────────
    entries = []
    for s in SCENES:
        p = OUT_DIR / f"{s['id']}.png"
        if p.exists():
            entries.append({**s, "file": p.name, "bytes": p.stat().st_size})

    MANIFEST.write_text(json.dumps({
        "_meta": {
            "model": MODEL_ID,
            "seed_base": SEED_BASE,
            "steps": args.steps,
            "size": args.size,
            "privacy_notice": (
                "전량 생성 이미지다. 실제 인물·실제 약·실제 가정의 사진이 아니다. "
                "이 저장소에 실사진은 존재하지 않는다."
            ),
            "generated_locally": True,
            "external_requests_at_inference": 0,
        },
        "images": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n매니페스트: {MANIFEST}  ({len(entries)}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
