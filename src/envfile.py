#!/usr/bin/env python3
"""`.env` 로더 — 외부 의존성 없이.

★ 왜 따로 모듈로 뺐나

  같은 로더가 네 파일에 복사되어 있었고, 거기에 버그가 있었다.

    1. `.env.example` 을 복사하면 `GEMINI_API_KEY=` (빈 값) 줄이 들어간다
    2. 사용자가 파일 끝에 실제 키를 덧붙인다
    3. 로더가 `os.environ.setdefault()` 를 쓰고 있었다
       → **먼저 만난 빈 값이 등록되고 뒤의 진짜 키는 무시된다**
    4. 결과: 키를 넣었는데도 "GEMINI_API_KEY 가 설정되지 않았습니다"

  같은 버그가 네 군데 복사돼 있으면 한 군데만 고치고 넘어가기 쉽다.
  그래서 한 곳으로 모으고 두 가지를 바로잡았다.

    - **빈 값은 무시한다** (덮어쓰지 않는다)
    - **뒤에 온 값이 이긴다** (같은 키가 여러 번 나오면 마지막 것)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """`.env` 를 읽어 환경변수에 넣는다. 적용된 항목을 돌려준다.

    Args:
        path: .env 경로 (기본: 저장소 루트)
        override: 이미 환경변수에 있어도 덮어쓸지 (기본 False)
    """
    env_path = Path(path) if path else ROOT / ".env"
    applied: dict[str, str] = {}
    if not env_path.exists():
        return applied

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # 빈 값은 무시한다. `.env.example` 의 빈 자리가
        # 뒤에 온 진짜 값을 가리는 것을 막는다.
        if not value:
            continue
        if key in os.environ and not override and key not in applied:
            # 셸에서 이미 준 값이 파일보다 우선한다.
            continue
        os.environ[key] = value
        applied[key] = value

    return applied


def masked(value: str, keep: int = 4) -> str:
    """로그에 찍어도 되도록 값을 가린다."""
    if not value:
        return "(없음)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}…{'*' * 6}…{value[-keep:]}"


if __name__ == "__main__":
    applied = load_env()
    print(f"적용된 항목 {len(applied)}개")
    for k, v in applied.items():
        show = masked(v) if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD")) else v
        print(f"  {k} = {show}")
