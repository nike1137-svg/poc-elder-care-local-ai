#!/usr/bin/env python3
"""기준선(baseline) 백엔드 — 클라우드 범용 LLM에 프롬프트로 전사를 시키는 방식.

★ 이것이 실서비스가 현재 쓰고 있는 방식이다.
   전용 ASR 모델이 아니라, 멀티모달 LLM에 오디오를 넣고
   자연어로 "한국어 텍스트로 변환해주세요"라고 요청한다.

이 PoC는 이 방식을 개선 대상으로 삼는다. 공정하게 비교하려면 실제로 돌려봐야 하므로,
실서비스와 **같은 모델 계열·같은 프롬프트**로 재현한다.

프라이버시 주석 ★
  실서비스는 가족의 실제 대화를 다루므로 무료 티어를 쓸 수 없다
  (무료 티어는 입력이 모델 학습에 활용될 수 있다).
  그러나 이 PoC의 평가 데이터는 **전량 창작 합성 대본**이라 그 제약이 걸리지 않는다.
  덕분에 무료 티어로 기준선을 재현할 수 있고, 채점자도 무료 키만으로 재현할 수 있다.
  → 바로 이 제약의 존재 자체가 이번 개선의 핵심 동기다 (docs/problem-statement.md §2 문제 2).

필요 환경변수:
  GEMINI_API_KEY   Google AI Studio 무료 키 (https://aistudio.google.com/apikey)
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from stt_base import FailureKind, STTBackend

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
# ★ 실서비스는 gemini-2.5-flash 를 쓰지만, 이 모델은 **신규 사용자에게 더 이상
#   제공되지 않는다**(호출 시 404). 문제 정의서에 적었던 "어차피 한 번은 갈아타야
#   한다"가 이미 현실이 된 것이다.
#   그래서 기준선은 현행 flash 계열 별칭으로 재현한다.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# 실서비스에서 쓰는 프롬프트 원문 그대로. 문구를 바꾸면 공정한 비교가 아니다.
PRODUCTION_PROMPT = (
    "이 음성을 한국어 텍스트로 변환해주세요. "
    "변환된 텍스트만 출력하고 다른 설명은 하지 마세요."
)

# m4a(AAC in MP4 container). Gemini는 audio/mp4 를 받는다.
MIME_BY_SUFFIX = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
}


class GeminiBackend(STTBackend):
    """클라우드 멀티모달 LLM 프롬프트 전사 (기존 방식 재현)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        prompt: str = PRODUCTION_PROMPT,
        timeout: int = 120,
        min_interval_sec: float | None = None,   # 무료 티어 분당 호출 한도 보호
                                                 # (GEMINI_MIN_INTERVAL 로 조정 가능)
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.min_interval_sec = (
            min_interval_sec if min_interval_sec is not None
            else float(os.environ.get("GEMINI_MIN_INTERVAL", "4.5"))
        )
        self.max_retries = max_retries
        self.name = f"gemini-{model.replace('gemini-', '')}"
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._last_call = 0.0

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval_sec:
            time.sleep(self.min_interval_sec - gap)
        self._last_call = time.monotonic()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{API_ROOT}/{self.model}:generateContent?key={self._api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _transcribe(self, audio_path: Path) -> tuple[str, dict[str, Any]]:
        if not self._api_key:
            raise PermissionError(
                "GEMINI_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요."
            )

        mime = MIME_BY_SUFFIX.get(audio_path.suffix.lower(), "audio/mp4")
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": audio_b64}},
                    {"text": self.prompt},
                ],
            }],
            # 전사는 창작이 아니므로 온도를 낮춰 재현성을 높인다.
            "generationConfig": {"temperature": 0.0},
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                data = self._post(payload)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(min(30, 6 * attempt))   # 지수적 후퇴
                    last_exc = exc
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(3 * attempt)
                    last_exc = exc
                    continue
                raise
        else:
            raise RuntimeError(f"재시도 {self.max_retries}회 모두 실패: {last_exc}")

        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata") or {}
        meta = {
            "model": self.model,
            "finish_reason": cand.get("finishReason"),
            "prompt_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
            "external_request": True,   # ★ 이 백엔드는 음성을 외부로 전송한다
        }
        return text.strip(), meta

    def classify_error(self, exc: Exception) -> tuple[FailureKind, str]:
        """예외를 원인별로 분류한다.

        ★ 문자열 포함 검사를 쓰다가 틀렸다

          처음에는 `"rate" in msg` 로 호출 한도를 판정했다. 그런데 404 응답의
          안내 URL에 `.../docs/mi**grate**-to-...` 가 들어 있어서
          **모델 없음(404)이 호출 한도 초과(429)로 분류**됐다.

          원인을 잘못 분류하면 대응도 잘못된다. 실제로 "잠시 후 재시도"를
          해야 할 상황과 "모델 이름을 바꿔야" 하는 상황은 전혀 다르다.
          그래서 **HTTP 상태 코드를 직접 파싱**하도록 바꿨다.
        """
        msg = f"{type(exc).__name__}: {exc}"

        if isinstance(exc, PermissionError):
            return FailureKind.AUTH, msg[:400]
        if isinstance(exc, urllib.error.URLError):
            return FailureKind.NETWORK, msg[:400]

        # _transcribe 가 만드는 "HTTP <code>: ..." 형식에서 코드를 꺼낸다.
        m = re.match(r"^HTTP (\d{3}):", str(exc))
        if m:
            code = int(m.group(1))
            if code in (401, 403):
                return FailureKind.AUTH, msg[:400]
            if code == 429:
                return FailureKind.RATE_LIMIT, msg[:400]
            if code == 404:
                # 모델이 없거나 이 계정에 제공되지 않는다 → 이름을 바꿔야 한다
                return FailureKind.MODEL_ERROR, msg[:400]
            if 400 <= code < 600:
                return FailureKind.MODEL_ERROR, msg[:400]

        low = str(exc).lower()
        if "timed out" in low or "urlopen" in low:
            return FailureKind.NETWORK, msg[:400]
        return FailureKind.UNKNOWN, msg[:400]


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent.parent
    from envfile import load_env

    load_env()

    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted((root / "data" / "audio").glob("*.m4a"))[:2]

    be = GeminiBackend()
    print(f"백엔드: {be.name}  (키 {'있음' if be._api_key else '없음'})\n")
    for p in paths:
        r = be.transcribe(p)
        status = "OK " if r.ok else f"실패({r.failure.value})"
        print(f"[{status}] {r.audio_id}  {r.latency_sec:.2f}s (RTF {r.realtime_factor:.2f})")
        print(f"   {r.text or r.error_detail}")
