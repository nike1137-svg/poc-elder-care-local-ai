#!/usr/bin/env python3
"""개선안 백엔드 — 로컬 Whisper (faster-whisper / CTranslate2).

이 PoC가 제안하는 방식이다. 음성이 기기 밖으로 나가지 않고, 호출당 비용이 0원이다.

왜 faster-whisper 인가 (자세한 근거는 docs/model-selection.md):
  - openai-whisper 원본 대비 CTranslate2 백엔드로 CPU에서 수 배 빠르다
  - int8 양자화로 GPU 없이도 돌아간다 (이 PoC의 실행 환경에 GPU가 없다)
  - 모델 가중치가 로컬에 캐시되어, 최초 1회 내려받은 뒤로는 완전 오프라인이다
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from stt_base import FailureKind, STTBackend

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEFAULT_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")

# ── 도메인 적응 프롬프트 ─────────────────────────────────────────────────
# 학습 없이 도메인 성능을 올리는 방법. Whisper는 initial_prompt를 '앞선 문맥'으로
# 취급해 디코딩을 그쪽으로 편향시킨다. 비용 0원, 추가 학습 0.
#
# 측정 결과 (docs/tuning.md):
#   합성 돌봄 코퍼스   CER 0.0184 → 0.0131  (-28.8%)
#   공개 낭독 데이터셋 CER 0.0658 → 0.0633  (-3.8%)
#   → 두 데이터셋 모두에서 개선된 유일한 설정이라 기본값으로 채택했다.
#
# ★ beam_size=1(탐욕적 디코딩)과 조합하면 합성 데이터에서 CER 0.0108까지 떨어지지만,
#   실제 사람 음성에서는 오히려 20.7% 악화됐다. 합성 데이터에만 맞춘 과적합이었다.
#   그래서 beam_size는 5를 유지한다.
DOMAIN_PROMPT = (
    "혈압약, 당뇨약, 경로당, 무릎, 허리, 시큰거리다, 쑤시다, 어지럽다, "
    "깜빡깜빡, 입맛, 된장찌개, 김장, 잠을 설치다, 뒤척이다, 적적하다, 손주"
)


class WhisperBackend(STTBackend):
    """로컬 Whisper 전사.

    모델은 프로세스당 한 번만 적재한다 (적재 비용을 건당 지연시간에 포함시키지 않기 위해).
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        compute_type: str = DEFAULT_COMPUTE,
        device: str = "cpu",
        beam_size: int = 5,
        language: str = "ko",
        initial_prompt: str | None = DOMAIN_PROMPT,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.language = language
        # None을 명시하면 프롬프트 없이(튜닝 전 기준선) 돌릴 수 있다.
        self.initial_prompt = initial_prompt
        self.name = f"whisper-{model_size}" + ("+prompt" if initial_prompt else "")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=os.cpu_count() or 4,
            )
        return self._model

    def warmup(self) -> None:
        """모델을 미리 적재한다. 벤치마크 전에 호출해 적재 시간을 측정에서 분리한다."""
        _ = self.model

    def has_speech(self, audio_path: Path) -> tuple[bool, float]:
        """말소리가 조금이라도 있는지 먼저 확인한다. (있음?, 검사에 걸린 초)

        ★ 왜 필요한가 — 강건성 테스트에서 발견한 문제

          정상 발화는 2.9초에 전사되는데, **무음 5초를 넣으니 44초**가 걸렸다.
          소음만 있는 6초는 37초였다. Whisper는 알아들을 게 없으면 오히려
          더 오래 헤맨다. 어르신이 실수로 빈 녹음을 보내면 파이프라인이
          44초간 멈추는 셈이라, 실서비스에서는 그대로 장애가 된다.

        ★ VAD를 '분할'이 아니라 '관문'으로 쓴다

          VAD로 음성을 잘라내면(vad_filter=True) 어르신의 긴 침묵이
          발화 끝으로 오인되어 뒷말이 잘릴 수 있다. 그래서 전사에는 VAD를 쓰지 않는다.
          대신 여기서 **"말소리가 하나라도 있는가"만** 묻고,
          있으면 VAD 없이 통째로 전사한다. 침묵은 보존하면서 헛수고만 막는다.
        """
        import time as _time

        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import get_speech_timestamps

        t0 = _time.perf_counter()
        try:
            samples = decode_audio(str(audio_path), sampling_rate=16000)
            stamps = get_speech_timestamps(samples)
        except Exception:  # noqa: BLE001 - 관문에서 실패하면 그냥 통과시킨다
            return True, _time.perf_counter() - t0
        return bool(stamps), _time.perf_counter() - t0

    def _transcribe(self, audio_path: Path) -> tuple[str, dict[str, Any]]:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
            vad_filter=False,   # 침묵 구간도 그대로 넘긴다. 어르신 발화는 침묵이 잦아
                                # VAD가 진짜 발화를 잘라낼 위험이 있다.
        )
        parts, seg_count = [], 0
        for seg in segments:      # 제너레이터라 여기서 실제 연산이 일어난다
            parts.append(seg.text)
            seg_count += 1

        meta = {
            "model": self.model_size,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "segments": seg_count,
            "detected_language": getattr(info, "language", None),
            "language_probability": round(getattr(info, "language_probability", 0.0), 4),
        }
        return "".join(parts).strip(), meta

    def classify_error(self, exc: Exception) -> tuple[FailureKind, str]:
        msg = f"{type(exc).__name__}: {exc}"
        low = str(exc).lower()
        if "no such file" in low or "invalid data" in low or "decod" in low:
            return FailureKind.AUDIO_UNREADABLE, msg[:400]
        if "out of memory" in low or "alloc" in low:
            return FailureKind.MODEL_ERROR, msg[:400]
        return FailureKind.UNKNOWN, msg[:400]


if __name__ == "__main__":
    import sys
    import time

    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        root = Path(__file__).resolve().parent.parent
        paths = sorted((root / "data" / "audio").glob("*.m4a"))[:3]

    be = WhisperBackend()
    print(f"모델 적재 중... ({be.model_size}, {be.compute_type}, CPU)")
    t0 = time.perf_counter()
    be.warmup()
    print(f"적재 완료 {time.perf_counter() - t0:.1f}s\n")

    for p in paths:
        r = be.transcribe(p)
        status = "OK " if r.ok else f"실패({r.failure.value})"
        print(f"[{status}] {r.audio_id}  {r.latency_sec:.2f}s (RTF {r.realtime_factor:.2f})")
        print(f"   {r.text or r.error_detail}")
