#!/usr/bin/env python3
"""STT 백엔드 공통 인터페이스.

★ 설계 의도 — 실패를 삼키지 않는다.

기존 실서비스는 전사 호출이 실패하면 `sttText = ''` 로 뭉개고 그대로 진행했다.
그 결과 어르신은 말씀을 하셨는데 봇은 못 들은 것처럼 반응했고,
실패가 얼마나 잦은지조차 알 수 없었다.

그래서 이 인터페이스는 결과를 '문자열'이 아니라 **성공/실패가 구분되는 객체**로 돌려준다.
호출부는 실패를 무시할 수 없고, 실패는 원인별로 분류되어 로그에 남는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class FailureKind(str, Enum):
    """전사 실패 원인. 빈 문자열 하나로 뭉뚱그리지 않고 구분해 남긴다."""

    NONE = "none"
    AUDIO_UNREADABLE = "audio_unreadable"   # 파일이 없거나 디코딩 불가
    AUDIO_TOO_SHORT = "audio_too_short"     # 지나치게 짧음
    NO_SPEECH = "no_speech"                 # 말소리가 전혀 없음 (무음·소음만)
    NETWORK = "network"                     # 네트워크 오류 (클라우드 백엔드)
    AUTH = "auth"                           # 인증 실패 / 키 없음
    RATE_LIMIT = "rate_limit"               # 호출 한도 초과
    MODEL_ERROR = "model_error"             # 모델 자체가 오류 반환
    EMPTY_OUTPUT = "empty_output"           # 호출은 됐으나 결과가 빔
    UNKNOWN = "unknown"


@dataclass
class STTResult:
    """전사 결과 한 건.

    ok=False 인 경우 text는 빈 문자열이고 failure에 원인이 담긴다.
    호출부는 ok를 확인하지 않고 text만 쓰면 잘못된 코드다.
    """

    audio_id: str
    backend: str
    ok: bool
    text: str = ""
    failure: FailureKind = FailureKind.NONE
    error_detail: str = ""
    latency_sec: float = 0.0
    audio_duration_sec: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def realtime_factor(self) -> float:
        """처리 시간 / 음성 길이. 1.0 미만이면 실시간보다 빠르다."""
        if self.audio_duration_sec <= 0:
            return 0.0
        return self.latency_sec / self.audio_duration_sec

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure"] = self.failure.value
        d["realtime_factor"] = round(self.realtime_factor, 3)
        return d


class STTBackend:
    """전사 백엔드 베이스. 하위 클래스는 _transcribe 만 구현한다."""

    name = "base"

    def transcribe(self, audio_path: Path, audio_id: str = "") -> STTResult:
        """전사를 수행하고 성공/실패가 구분된 결과를 돌려준다. 예외를 밖으로 던지지 않는다."""
        audio_id = audio_id or audio_path.stem
        duration = probe_duration(audio_path)

        if not audio_path.exists():
            return STTResult(
                audio_id=audio_id, backend=self.name, ok=False,
                failure=FailureKind.AUDIO_UNREADABLE,
                error_detail=f"파일 없음: {audio_path}",
            )
        if duration is None:
            return STTResult(
                audio_id=audio_id, backend=self.name, ok=False,
                failure=FailureKind.AUDIO_UNREADABLE,
                error_detail="ffprobe로 길이를 읽지 못했습니다 (손상 또는 미지원 포맷)",
            )
        if duration < 0.3:
            return STTResult(
                audio_id=audio_id, backend=self.name, ok=False,
                failure=FailureKind.AUDIO_TOO_SHORT,
                error_detail=f"음성이 너무 짧습니다 ({duration:.2f}s)",
                audio_duration_sec=duration,
            )

        # 말소리가 아예 없으면 전사를 시도하지 않는다.
        # 무음·소음만 든 입력에 모델이 헤매며 시간을 낭비하는 것을 막는다.
        # (백엔드가 has_speech를 제공할 때만 동작한다)
        gate_sec = 0.0
        if hasattr(self, "has_speech"):
            speech, gate_sec = self.has_speech(audio_path)
            if not speech:
                return STTResult(
                    audio_id=audio_id, backend=self.name, ok=False,
                    failure=FailureKind.NO_SPEECH,
                    error_detail="말소리가 감지되지 않았습니다 (무음이거나 소음만 있음)",
                    latency_sec=gate_sec,
                    audio_duration_sec=duration,
                    meta={"vad_gate_sec": round(gate_sec, 3), "gated": True},
                )

        started = time.perf_counter()
        try:
            text, meta = self._transcribe(audio_path)
        except Exception as exc:  # noqa: BLE001 - 원인을 분류해 결과에 담는다
            kind, detail = self.classify_error(exc)
            return STTResult(
                audio_id=audio_id, backend=self.name, ok=False,
                failure=kind, error_detail=detail,
                latency_sec=time.perf_counter() - started,
                audio_duration_sec=duration,
            )
        elapsed = time.perf_counter() - started

        text = (text or "").strip()
        if not text:
            return STTResult(
                audio_id=audio_id, backend=self.name, ok=False,
                failure=FailureKind.EMPTY_OUTPUT,
                error_detail="호출은 성공했으나 전사 결과가 비어 있습니다",
                latency_sec=elapsed, audio_duration_sec=duration, meta=meta,
            )

        meta = {**meta, "vad_gate_sec": round(gate_sec, 3), "gated": False}
        return STTResult(
            audio_id=audio_id, backend=self.name, ok=True, text=text,
            latency_sec=elapsed + gate_sec, audio_duration_sec=duration, meta=meta,
        )

    def _transcribe(self, audio_path: Path) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def classify_error(self, exc: Exception) -> tuple[FailureKind, str]:
        """예외를 원인별로 분류한다. 하위 클래스에서 확장한다."""
        msg = f"{type(exc).__name__}: {exc}"
        return FailureKind.UNKNOWN, msg[:400]


def probe_duration(audio_path: Path) -> float | None:
    """ffprobe로 오디오 길이를 잰다. 실패하면 None."""
    import subprocess

    if not audio_path.exists():
        return None
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True,
    )
    raw = proc.stdout.strip()
    if proc.returncode != 0 or not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
