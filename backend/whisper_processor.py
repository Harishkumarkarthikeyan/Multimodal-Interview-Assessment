from pathlib import Path

import whisper


def transcribe_audio(audio_path, out_txt_path, model_name="base", language=None):
    """Transcribe an audio file with the openai-whisper package."""
    model = whisper.load_model(model_name)
    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)
    text = result.get("text", "").strip()

    Path(out_txt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_txt_path).write_text(text, encoding="utf-8")
    return text
