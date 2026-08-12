#!/usr/bin/env python3
"""데모 화면 스크린샷 캡처 — README에 넣을 시연 자료를 만든다.

Gradio 앱을 띄운 상태에서 실행하면, 각 탭을 실제로 조작해
**결과가 채워진** 화면을 찍는다. 빈 화면 스크린샷은 시연 자료가 아니다.

사전 조건:
    다른 터미널에서 `python app.py --port 7861` 실행 중이어야 한다.

사용법:
    python src/capture_demo.py
    python src/capture_demo.py --port 7861 --only audio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# 무거운 모델이 도는 탭이 있어 넉넉히 기다린다 (Diffusion은 20초 이상)
TIMEOUTS = {"audio": 120_000, "image": 120_000, "memory": 180_000}


def capture(port: int, only: str | None) -> int:
    from playwright.sync_api import sync_playwright

    url = f"http://127.0.0.1:{port}"
    ASSETS.mkdir(parents=True, exist_ok=True)
    shots: list[str] = []

    with sync_playwright() as p:
        # playwright 전용 브라우저를 따로 받지 않고 시스템에 설치된 Chrome을 쓴다.
        # (없으면 `playwright install chromium` 으로 받은 것을 쓴다)
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(url, wait_until="networkidle", timeout=60_000)

        # ── 탭 1: 음성 → 특이사항 ────────────────────────────────────
        if only in (None, "audio"):
            print("[1/3] 음성 탭 — 샘플 u03 처리 중...")
            page.get_by_role("button", name="u03.m4a").click()
            page.wait_for_timeout(1500)
            page.locator("#btn_audio").click()
            # 특이사항 리포트 textarea 에 내용이 찰 때까지 기다린다
            page.wait_for_function(
                """() => {
                    const el = document.querySelector('#out_report textarea');
                    return el && el.value.includes('특이사항');
                }""",
                timeout=TIMEOUTS["audio"],
            )
            page.wait_for_timeout(1200)
            out = ASSETS / "demo_audio.png"
            page.screenshot(path=str(out), full_page=True)
            shots.append(out.name)
            print(f"      → {out.name}")

        # ── 탭 2: 사진 → 마스킹 ──────────────────────────────────────
        if only in (None, "image"):
            print("[2/3] 사진 탭 — 인물 마스킹 처리 중...")
            page.get_by_role("tab", name="② 사진 → 마스킹 / 알약 세기").click()
            page.wait_for_timeout(1500)
            # 이미지 예제는 썸네일로 렌더링되어 이름으로 클릭할 수 없다.
            # 파일 입력에 직접 넣는다. (탭 안의 보이는 입력만 고른다)
            sample = ROOT / "data" / "images" / "person_01.png"
            page.locator('input[type="file"]').last.set_input_files(str(sample))
            page.wait_for_timeout(3000)
            page.locator("#btn_image").click()
            page.wait_for_selector("text=인물 마스킹 완료", timeout=TIMEOUTS["image"])
            page.wait_for_timeout(1500)
            out = ASSETS / "demo_image.png"
            page.screenshot(path=str(out), full_page=True)
            shots.append(out.name)
            print(f"      → {out.name}")

        # ── 탭 3: 회상 이미지 ────────────────────────────────────────
        if only in (None, "memory"):
            print("[3/3] 회상 탭 — 이미지 생성 중 (20초 이상)...")
            page.get_by_role("tab", name="③ 회상 이미지").click()
            page.wait_for_timeout(1200)
            page.get_by_role("button", name="예전에 마당에서 김장하던 기억이 나네").click()
            page.wait_for_timeout(1200)
            page.locator("#btn_memory").click()
            page.wait_for_selector("text=회상 이미지 생성 완료", timeout=TIMEOUTS["memory"])
            page.wait_for_timeout(1500)
            out = ASSETS / "demo_memory.png"
            page.screenshot(path=str(out), full_page=True)
            shots.append(out.name)
            print(f"      → {out.name}")

        browser.close()

    print(f"\n캡처 완료 {len(shots)}건 → {ASSETS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="데모 화면 캡처")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--only", choices=["audio", "image", "memory"], default=None)
    args = ap.parse_args()

    import urllib.request

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}", timeout=5)
    except Exception:
        sys.exit(
            f"[에러] http://127.0.0.1:{args.port} 에 앱이 떠 있지 않습니다.\n"
            f"먼저 다른 터미널에서 실행하세요:\n"
            f"  python app.py --port {args.port}"
        )

    return capture(args.port, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
