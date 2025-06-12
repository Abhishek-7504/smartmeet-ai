import sys
from pathlib import Path
import ffmpeg
import numpy as np
import webrtcvad
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans

# CONFIG
AUDIO_PATH = Path("ingestion/sample_data/cleaned.wav")
SAMPLE_RATE = 16000
FRAME_DURATION = 30    # ms
VAD_AGGRESSIVENESS = 2 # 0–3

def read_audio(path):
    wav, sr = sf.read(path)
    if sr != SAMPLE_RATE or wav.ndim > 1:
        out, _ = (
            ffmpeg
            .input(str(path))
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, capture_stderr=True)
        )
        wav = np.frombuffer(out, np.float32)
    return wav

def frame_generator(wav, rate, frame_duration_ms):
    n = int(rate * (frame_duration_ms / 1000.0))
    offset = 0
    while offset + n < len(wav):
        yield wav[offset:offset + n]
        offset += n

def vad_collector(wav, rate, aggressiveness):
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(wav, rate, FRAME_DURATION)
    segments, start = [], None
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech((frame * 32768).astype(np.int16).tobytes(), rate)
        t = (i * FRAME_DURATION) / 1000.0
        if is_speech and start is None:
            start = t
        elif not is_speech and start is not None:
            segments.append((start, t))
            start = None
    if start is not None:
        segments.append((start, len(wav) / rate))
    return segments

def extract_embeddings(wav, rate, segments):
    encoder = VoiceEncoder()
    embeddings = []
    valid_segments = []
    for (start, end) in segments:
        mid = (start + end) / 2.0
        slice_start = int((mid - 0.5) * rate)
        slice_end = int((mid + 0.5) * rate)
        if slice_start < 0 or slice_end > len(wav):
            continue
        slice_wav = wav[slice_start:slice_end]
        try:
            emb = encoder.embed_utterance(preprocess_wav(slice_wav, rate))
            embeddings.append(emb)
            valid_segments.append((start, end))
        except Exception as e:
            print(f"Skipping segment [{start:.2f}s – {end:.2f}s]: {e}")
    return np.vstack(embeddings), valid_segments

def main():
    if not AUDIO_PATH.exists():
        print(f"File not found: {AUDIO_PATH}")
        sys.exit(1)

    num_speakers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    wav = read_audio(AUDIO_PATH)
    segments = vad_collector(wav, SAMPLE_RATE, VAD_AGGRESSIVENESS)

    if num_speakers == 1:
        print("\n=== Diarization (offline, 1 speaker) ===\n")
        for start, end in segments:
            print(f"[{start:.2f}s – {end:.2f}s] Speaker 0")
        return

    embs, valid_segments = extract_embeddings(wav, SAMPLE_RATE, segments)
    if len(embs) == 0:
        print("No valid segments for embedding.")
        return

    kmeans = KMeans(n_clusters=num_speakers, random_state=0).fit(embs)
    labels = kmeans.labels_

    print(f"\n=== Diarization (offline, {num_speakers} speakers) ===\n")
    for (start, end), spk in zip(valid_segments, labels):
        print(f"[{start:.2f}s – {end:.2f}s] Speaker {spk}")

if __name__ == "__main__":
    main()

