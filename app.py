#!/usr/bin/env python3
"""데모 앱 — 음성/사진을 올리면 결과가 나오는 웹 화면.

전 과정이 이 기계 안에서 돈다. 외부로 나가는 요청은 0건이다.

실행:
    python app.py                    # http://127.0.0.1:7860
    python app.py --share            # 임시 공개 링크 (주의: 아래 경고 참조)

★ 배포에 관한 경고

  이 PoC의 핵심 주장은 "음성·사진이 기기 밖으로 나가지 않는다"이다.
  그런데 이 앱을 인터넷에 배포하면 **사용자가 올린 파일이 서버로 전송된다.**
  주장과 정면으로 충돌한다.

  그래서 이 데모의 위치를 이렇게 못박는다.

    - 이 화면은 **저장소에 들어 있는 합성 샘플을 시연**하기 위한 것이다.
    - 실제 가족의 음성·사진을 여기에 올리면 안 된다.
    - 실사용 형태는 **각자의 집 안 기기에서 로컬 실행**하는 것이다.

  화면 안에도 같은 경고를 띄운다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from extract_signals import extract, format_report  # noqa: E402
from stt_whisper import WhisperBackend  # noqa: E402

AUDIO_DIR = ROOT / "data" / "audio"
IMAGE_DIR = ROOT / "data" / "images"
VOICE_MODEL = ROOT / "models" / "piper" / "ko_KR-kss-medium.onnx"

_stt: WhisperBackend | None = None
_piper = None


def get_stt() -> WhisperBackend:
    global _stt
    if _stt is None:
        _stt = WhisperBackend(model_size=os.environ.get("WHISPER_MODEL", "small"))
        _stt.warmup()
    return _stt


def get_piper():
    global _piper
    if _piper is None and VOICE_MODEL.exists():
        from piper import PiperVoice

        _piper = PiperVoice.load(str(VOICE_MODEL))
    return _piper


# ── 탭 1: 음성 → 전사 → 특이사항 → 음성 리포트 ──────────────────────────
def process_audio(audio_path: str | None, make_voice: bool):
    if not audio_path:
        return "음성 파일을 올려주세요.", "", None, ""

    stt = get_stt()
    r = stt.transcribe(Path(audio_path))

    if not r.ok:
        msg = (
            f"### 전사 실패\n\n"
            f"- **분류**: `{r.failure.value}`\n"
            f"- **내용**: {r.error_detail}\n\n"
            f"> 기존 방식은 이 상황을 빈 문자열로 넘겨 조용히 무시했습니다.\n"
            f"> 여기서는 실패를 실패로 보고합니다."
        )
        return msg, "", None, ""

    sigs = extract(r.text)
    report = format_report(r.text, sigs)

    stats = (
        f"### 처리 결과\n\n"
        f"| 항목 | 값 |\n|---|---|\n"
        f"| 음성 길이 | {r.audio_duration_sec:.2f}초 |\n"
        f"| 전사 시간 | {r.latency_sec:.2f}초 |\n"
        f"| 실시간 대비 | {r.realtime_factor:.2f}배 |\n"
        f"| 외부 전송 | **0건** |\n"
        f"| 비용 | **0원** |\n"
    )

    wav = None
    script = ""
    if make_voice:
        voice = get_piper()
        if voice is not None:
            import wave

            from voice_report import build_script

            script = build_script(sigs, r.text)
            out = ROOT / "results" / "voice" / "demo_report.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as w:
                voice.synthesize_wav(script, w)
            wav = str(out)
        else:
            script = "(piper 음성 모델이 없어 음성 리포트를 건너뜁니다)"

    return stats, r.text, wav, report + (f"\n\n[음성 안내문]\n{script}" if script else "")


# ── 탭 2: 사진 → 인물 마스킹 / 알약 세기 ────────────────────────────────
def process_image(image_path: str | None, task: str):
    if not image_path:
        return None, "이미지를 올려주세요."

    from image_pipeline import count_pills, mask_people

    p = Path(image_path)
    if task == "인물 마스킹 (프라이버시)":
        r = mask_people(p)
        if not r.get("ok"):
            return None, f"실패: {r.get('error')}"
        if not r["detections"]:
            return image_path, "인물이 탐지되지 않았습니다."
        msg = (
            f"### 인물 마스킹 완료\n\n"
            f"| 항목 | 값 |\n|---|---|\n"
            f"| 탐지된 인물 | {len(r['detections'])}명 |\n"
            f"| 가림 처리 | {r['masked_count']}개 |\n"
            f"| YOLO 탐지 | {r['detect_sec']}초 |\n"
            f"| SAM 분할 | {r['segment_sec']}초 |\n"
            f"| 외부 전송 | **0건** |\n\n"
            f"모자이크로 신원을 지우고 장면의 맥락은 남깁니다."
        )
        return r["output"], msg

    r = count_pills(p)
    if not r.get("ok"):
        return None, f"실패: {r.get('error')}"
    msg = (
        f"### 알약 세기 완료\n\n"
        f"| 항목 | 값 |\n|---|---|\n"
        f"| 알약 후보 | **{r['pill_count']}개** |\n"
        f"| SAM 마스크 총계 | {r['masks_total']}개 |\n"
        f"| 분할 시간 | {r['segment_sec']}초 |\n"
        f"| 외부 전송 | **0건** |\n\n"
        f"복약 확인용입니다. 정확한 개수는 눈으로 다시 확인하세요."
    )
    return r["output"], msg


# ── 탭 3: 회상 이미지 ───────────────────────────────────────────────────
def process_reminiscence(text: str):
    from reminiscence import build_prompt, generate, is_reminiscence

    if not text or not text.strip():
        return None, "발화를 입력해주세요."

    if not is_reminiscence(text):
        return None, (
            "### 회상 발화가 아닙니다\n\n"
            "옛날·예전·어릴 때·기억이 나 같은 표현이 있을 때만 그림을 만듭니다.\n\n"
            "예: `예전에 마당에서 김장하던 기억이 나네`"
        )

    prompt, matched = build_prompt(text)
    info = generate(prompt, ROOT / "results" / "reminiscence" / "demo.png")
    msg = (
        f"### 회상 이미지 생성 완료\n\n"
        f"| 항목 | 값 |\n|---|---|\n"
        f"| 매칭 키워드 | `{matched}` |\n"
        f"| 생성 시간 | {info['generate_sec']}초 |\n"
        f"| 외부 전송 | **0건** |\n\n"
        f"발화 원문은 프롬프트에 넣지 않습니다. 미리 정해둔 안전한 장면만 씁니다.\n\n"
        f"> 이것은 기억의 재현이 아니라 대화의 마중물입니다. 사실과 다를 수 있습니다."
    )
    return info["output"], msg


BANNER = """
# 고령 부모 돌봄 대화 어시스턴트 — 로컬 AI 파이프라인

**음성인식 · 음성생성 · 객체탐지 · 세그멘테이션 · 이미지생성** 5개 모델이 전부 이 기계에서 돕니다.
외부로 나가는 요청 **0건**, 비용 **0원**.

> ### ⚠️ 실제 가족의 음성·사진을 올리지 마세요
> 이 화면은 저장소에 들어 있는 **합성 샘플을 시연**하기 위한 것입니다.
> 인터넷에 배포된 상태라면 올린 파일이 서버로 전송되며, 이는
> "데이터가 기기 밖으로 나가지 않는다"는 이 프로젝트의 전제와 충돌합니다.
> 실사용은 **각자의 기기에서 로컬 실행**이 전제입니다.
"""


def build_ui():
    import gradio as gr

    samples_audio = sorted(str(p) for p in AUDIO_DIR.glob("u*.m4a"))[:6]
    samples_person = sorted(str(p) for p in IMAGE_DIR.glob("person_*.png"))
    samples_pill = sorted(str(p) for p in IMAGE_DIR.glob("pill_*.png"))

    with gr.Blocks(title="돌봄 대화 로컬 AI PoC") as demo:
        gr.Markdown(BANNER)

        with gr.Tab("① 음성 → 특이사항"):
            gr.Markdown(
                "어르신의 음성 메시지를 넣으면 **로컬 Whisper**가 전사하고, "
                "가족이 알아야 할 신호를 뽑아냅니다."
            )
            with gr.Row():
                with gr.Column():
                    a_in = gr.Audio(type="filepath", label="음성 파일")
                    a_voice = gr.Checkbox(label="음성 리포트도 만들기 (로컬 TTS)", value=True)
                    a_btn = gr.Button("처리하기", variant="primary", elem_id="btn_audio")
                    if samples_audio:
                        gr.Examples(samples_audio, inputs=a_in, label="샘플 (합성 음성)")
                with gr.Column():
                    a_stats = gr.Markdown(elem_id="out_stats")
                    a_text = gr.Textbox(label="전사 결과", lines=3)
                    a_wav = gr.Audio(label="음성 리포트", type="filepath")
            a_report = gr.Textbox(label="특이사항 리포트", lines=16, elem_id="out_report")
            a_btn.click(process_audio, [a_in, a_voice], [a_stats, a_text, a_wav, a_report])

        with gr.Tab("② 사진 → 마스킹 / 알약 세기"):
            gr.Markdown(
                "**YOLO**가 인물을 찾고 **SAM**이 윤곽을 따냅니다. "
                "알약 세기는 SAM 자동 분할을 씁니다 "
                "(COCO에 알약 클래스가 없어 YOLO로는 안 됩니다)."
            )
            with gr.Row():
                with gr.Column():
                    i_in = gr.Image(type="filepath", label="사진")
                    i_task = gr.Radio(
                        ["인물 마스킹 (프라이버시)", "알약 세기 (복약 확인)"],
                        value="인물 마스킹 (프라이버시)", label="작업")
                    i_btn = gr.Button("처리하기", variant="primary", elem_id="btn_image")
                    if samples_person or samples_pill:
                        gr.Examples(samples_person + samples_pill, inputs=i_in,
                                    label="샘플 (생성 이미지)")
                with gr.Column():
                    i_out = gr.Image(label="결과", type="filepath")
                    i_msg = gr.Markdown(elem_id="out_imsg")
            i_btn.click(process_image, [i_in, i_task], [i_out, i_msg])

        with gr.Tab("③ 회상 이미지"):
            gr.Markdown(
                "회상요법은 실제 돌봄 기법입니다. 어르신이 떠올린 장면을 "
                "**로컬 Diffusion**으로 그려 대화의 마중물로 씁니다. (CPU에서 20초 내외)"
            )
            with gr.Row():
                with gr.Column():
                    r_in = gr.Textbox(label="어르신 발화",
                                      placeholder="예전에 마당에서 김장하던 기억이 나네")
                    r_btn = gr.Button("그려보기", variant="primary", elem_id="btn_memory")
                    gr.Examples([
                        "예전에 마당에서 김장하던 기억이 나네",
                        "어릴 때 시골 학교 걸어다니던 생각이 나",
                        "옛날에 장터에 나가면 사람이 참 많았지",
                        "오늘은 무릎이 좀 시큰거려서 산책은 못 나갔다",
                    ], inputs=r_in, label="예시 (마지막 것은 회상이 아니라 거부됩니다)")
                with gr.Column():
                    r_out = gr.Image(label="회상 이미지", type="filepath")
                    r_msg = gr.Markdown(elem_id="out_rmsg")
            r_btn.click(process_reminiscence, [r_in], [r_out, r_msg])

        gr.Markdown(
            "---\n"
            "문서: `docs/problem-statement.md` · `docs/model-selection.md` · "
            "`docs/evaluation.md` · `docs/limitations.md`"
        )
    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="돌봄 대화 로컬 AI 데모")
    ap.add_argument("--share", action="store_true", help="임시 공개 링크 생성")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
