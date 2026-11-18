"""CPU smoke tests: no model weights or external judge server needed."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_studio_saved_datasets_and_local_controls(monkeypatch):
    monkeypatch.setenv('JUDGE_URL', '')
    monkeypatch.setenv('CXER_ASR_BACKEND', 'mock')
    at = AppTest.from_file(str(ROOT / 'app/cxer_studio.py'), default_timeout=60).run()
    assert not at.exception
    for dataset in ['atco2_ood', 'atcosim_test', 'uwb_atcc_test']:
        next(s for s in at.selectbox if s.label == 'Dataset').set_value(dataset).run()
        assert not at.exception
    next(s for s in at.multiselect if s.label == 'ATC ASR models').set_value([]).run()
    assert not at.exception
    next(r for r in at.radio if r.label == 'Judge location').set_value('Run on this machine').run()
    assert not at.exception
    assert not any(t.label == 'Judge Python executable' for t in at.text_input)


def test_bundled_batch_exports(monkeypatch):
    monkeypatch.setenv('JUDGE_URL', '')
    monkeypatch.setenv('CXER_ASR_BACKEND', 'mock')
    at = AppTest.from_file(str(ROOT / 'app/cxer_studio.py'), default_timeout=60).run()
    next(r for r in at.radio if r.label == 'Audio source').set_value('Use a bundled ATC clip').run()
    next(b for b in at.button if b.label == '▶ Run analysis').click().run(timeout=60)
    assert not at.exception
    results = at.session_state['results']
    assert results
    for result in results:
        assert result['hypothesis'].startswith('mock transcription')
        assert Path(result['audio_path']).is_file()
        assert result['callsign_status'] == 'not run'
    summary = next(d.value for d in at.dataframe if 'audio_path' in d.value.columns)
    assert {'speaker', 'callsigns', 'raw_hypothesis'}.issubset(summary.columns)
    assert not {'asr_error', 'callsign_error', 'callsign_seconds'}.intersection(summary.columns)
    assert summary['callsigns'].tolist() == ['Unknown'] * len(summary)
