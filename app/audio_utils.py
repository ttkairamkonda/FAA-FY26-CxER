"""Audio decoding using project dependencies, with bundled FFmpeg fallback."""
import os
from pathlib import Path


def load_audio(raw: bytes, name: str = "audio.wav"):
    """Decode any supported format to 16 kHz mono. Goes through a temp file
    so formats libsndfile can't read (m4a, some mp3) fall back to ffmpeg."""
    import tempfile

    import librosa

    suffix = Path(name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        try:
            array, sr = librosa.load(path, sr=16000, mono=True)
        except Exception:
            # The wheel supplies ffmpeg, so no host apt/sudo installation is needed.
            import imageio_ffmpeg
            import numpy as np
            import subprocess
            decoded = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path,
                 "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                capture_output=True, timeout=120,
            )
            if decoded.returncode:
                raise ValueError(decoded.stderr.decode(errors="replace")[-1000:])
            array, sr = np.frombuffer(decoded.stdout, dtype=np.float32).copy(), 16000
        if not len(array):
            raise ValueError("Audio is empty")
    finally:
        os.unlink(path)
    return array, sr


def active_audio_regions(array, sr, top_db=35):
    """Skip long quiet gaps; preserve short pauses and pad retained regions.

    This is an energy heuristic, not a speech detector. Callers can disable it
    for recordings with large differences in speaker loudness.
    """
    import librosa
    import numpy as np
    if not len(array) or np.max(np.abs(array)) < 1e-8:
        return []
    intervals = librosa.effects.split(array, top_db=top_db, frame_length=512, hop_length=160)
    padding, gap = int(0.25 * sr), int(0.5 * sr)
    regions = []
    for start, end in intervals:
        start, end = max(0, int(start) - padding), min(len(array), int(end) + padding)
        if regions and start <= regions[-1][1] + gap:
            regions[-1][1] = end
        else:
            regions.append([start, end])
    return regions
