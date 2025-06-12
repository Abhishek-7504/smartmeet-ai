import torch
from pathlib import Path

import whisperx
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Enable TF32 if you want faster CUDA (optional)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- CONFIG ---
AUDIO_FILE = Path("ingestion/sample_data/prat_audio.wav")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# --- 1) Transcribe with WhisperX ---
print(">> Transcribing with WhisperX...")
asr_model = whisperx.load_model("base", DEVICE)
asr_result = asr_model.transcribe(str(AUDIO_FILE), batch_size=BATCH_SIZE)

# --- 2) Align for word-level timestamps (optional but improves accuracy) ---
print(">> Aligning timestamps...")
align_model, metadata = whisperx.load_align_model(
    language_code=asr_result["language"], device=DEVICE
)
aligned = whisperx.align(
    asr_result["segments"], align_model, metadata, str(AUDIO_FILE), DEVICE
)

# --- 3) Diarize with Pyannote.audio ---
print(">> Performing speaker diarization...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=False   # no token needed for offline inference
)
diarization = pipeline(str(AUDIO_FILE))

# --- 4) Merge transcripts + speakers ---
print(">> Merging speaker labels with transcript...\n")
for segment in aligned["segments"]:
    # choose midpoint of each text segment to query speaker
    mid = (segment["start"] + segment["end"]) / 2
    # crop annotation at that instant
    current = diarization.crop(Segment(mid, mid))
    labels = list(current.labels())
    speaker = labels[0] if labels else "Unknown"

    start, end, text = segment["start"], segment["end"], segment["text"].strip()
    print(f"[{start:.2f}s – {end:.2f}s] Speaker {speaker}: {text}")

