import whisper
import sys
from pathlib import Path
import json

def transcribe(file_path):
    model = whisper.load_model("base")  # You can choose other sizes
    result = model.transcribe(file_path)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_or_video_file>")
        sys.exit(1)

    file = Path(sys.argv[1])
    transcript = transcribe(str(file))

    # Save plain text version
    txt_path = file.with_suffix(".txt")
    txt_path.write_text(transcript["text"], encoding="utf-8")
    print(f"Transcript saved to {txt_path}")

    # ✅ Save full Whisper output as JSON
    json_path = file.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    print(f"Full transcript (segments, language, etc.) saved to {json_path}")
