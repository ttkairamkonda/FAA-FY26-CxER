import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from transcript_processing import normalize_transcript, extract_icao_callsigns, infer_speaker


def test_numeric_formats_are_equivalent():
    assert normalize_transcript('execjet four zero one') == normalize_transcript('execjet 401')
    assert normalize_transcript('climb two thousand four hundred') == normalize_transcript('climb 2400')
    assert normalize_transcript('contact one two three point four five') == 'contact 123.45'
    assert normalize_transcript('heading two niner zero') == 'heading 290'


def test_verified_icao_operator_and_registration():
    assert extract_icao_callsigns('execjet four zero one descend') == ['EJA401']
    assert extract_icao_callsigns('EJA401 roger') == ['EJA401']
    assert extract_icao_callsigns('november one two three alpha bravo') == ['N123AB']
    assert extract_icao_callsigns('united four zero one roger execjet two three') == ['UAL401', 'EJA23']


def test_missing_callsign_and_ambiguous_speaker():
    assert extract_icao_callsigns('descend flight level two four zero') == []
    assert infer_speaker('execjet 401 descend flight level 240') == 'Unknown'
    assert infer_speaker('roger execjet 401') == 'Pilot'
