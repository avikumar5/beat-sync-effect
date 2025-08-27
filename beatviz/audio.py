# beatviz/audio.py
from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import librosa
import moviepy.editor as mpy

def extract_audio(video_path: Path, sr: int = 22050) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    clip = mpy.VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
    return wav_path

def detect_onsets(wav_path: Path, sr: int = 22050):
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")

def detect_beats(wav_path: Path, sr: int = 22050) -> np.ndarray:
    """Return beat times in seconds."""
    y, _ = librosa.load(str(wav_path), sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    return librosa.frames_to_time(beat_frames, sr=sr)

def band_energies_over_time(
    wav_path: Path,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    mel_bands: int = 64,
) -> dict[str, np.ndarray]:
    """
    Compute normalized energy over time for low/mid/high bands using a Mel-spectrogram.
    Returns dict with 't' (seconds), 'low', 'mid', 'high' in [0,1].
    """
    y, _ = librosa.load(str(wav_path), sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=mel_bands, power=2.0)
    # Split mel bins into thirds: low/mid/high (rough and fast)
    thirds = np.array_split(np.arange(S.shape[0]), 3)
    low = S[thirds[0]].mean(axis=0)
    mid = S[thirds[1]].mean(axis=0)
    high = S[thirds[2]].mean(axis=0)

    def _norm(a):
        a = a.astype(np.float32)
        a -= a.min() if a.size else 0.0
        denom = (a.max() - a.min()) or 1.0
        return a / denom

    low, mid, high = _norm(low), _norm(mid), _norm(high)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    return {"t": t, "low": low, "mid": mid, "high": high}
