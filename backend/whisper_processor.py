from pathlib import Path
import whisper

def transcribe_audio(audio_path, out_txt_path, model_name="base"):
    """Transcribe audio using OpenAI Whisper (openai-whisper package).

    Writes transcript to out_txt_path.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    text = result.get("text", "")
    Path(out_txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
