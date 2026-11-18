"""Decode with bundled FFmpeg when native decoding is unavailable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from audio_utils import load_audio


def test_bundled_ffmpeg_decodes_without_system_ffmpeg(monkeypatch):
    import librosa
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    sample = next((root / 'data/audio_samples').glob('*.wav'))
    def unavailable(*args, **kwargs):
        raise RuntimeError('Native decoding unavailable')
    monkeypatch.setattr(librosa, 'load', unavailable)
    waveform, sample_rate = load_audio(sample.read_bytes(), sample.name)
    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert len(waveform) > 0
    assert np.isfinite(waveform).all()


def test_quiet_gaps_are_skipped_without_losing_active_regions():
    import numpy as np
    from audio_utils import active_audio_regions
    sr = 16000
    audio = np.zeros(sr * 60, dtype=np.float32)
    audio[sr:2*sr] = 0.1
    audio[55*sr:57*sr] = 0.1
    regions = active_audio_regions(audio, sr)
    assert len(regions) == 2
    assert regions[0][0] < sr and regions[0][1] > 2*sr
    assert regions[1][0] < 55*sr and regions[1][1] > 57*sr
    assert sum(end-start for start,end in regions) < 5*sr
    assert active_audio_regions(np.zeros(sr), sr) == []
