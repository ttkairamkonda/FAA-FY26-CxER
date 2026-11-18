"""Shared live transcript normalization and conservative aviation entity handling."""
import json
import re
from functools import lru_cache
from pathlib import Path

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

_NORMALIZER = EnglishTextNormalizer({})


def normalize_transcript(text):
    # Aviation pronunciation variants before Whisper's English number normalization.
    return _NORMALIZER(re.sub(r'\bniner\b', 'nine', str(text), flags=re.I)).strip()


@lru_cache(maxsize=1)
def _operator_pattern():
    source = Path(__file__).resolve().parents[1] / 'data/icao_telephony.json'
    entries = json.loads(source.read_text())['operators']
    aliases = {}
    for entry in entries:
        for alias in (entry['icao'], entry['telephony']):
            alias = ' '.join(alias.lower().split())
            if alias and alias not in ('none', 'n/a', '-'):
                aliases.setdefault(alias, set()).add(entry['icao'])
    aliases['exec jet'] = {'EJA'}
    unique = {alias: next(iter(codes)) for alias, codes in aliases.items() if len(codes) == 1}
    alternatives = '|'.join(re.escape(a) for a in sorted(unique, key=len, reverse=True))
    return re.compile(r'\b(' + alternatives + r')\s*(\d{1,7}[a-z]{0,2})\b', re.I), unique


_PHONETIC = dict(zip(
    'alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu'.split(),
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
_PHONETIC.update({'alfa': 'A', 'juliet': 'J', 'x-ray': 'X'})


def extract_icao_callsigns(text):
    text = normalize_transcript(text)
    pattern, operators = _operator_pattern()
    found = [(m.start(), operators[m.group(1).lower()] + m.group(2).upper()) for m in pattern.finditer(text)]
    words = '|'.join(_PHONETIC)
    registration = re.compile(r'\b(?:november\s*|n)(\d{1,5})([a-z]{1,2})?\b(?:\s+((?:' + words + r')(?:\s+(?:' + words + r'))?))?', re.I)
    for match in registration.finditer(text):
        suffix = (match.group(2) or '').upper()
        if match.group(3):
            suffix += ''.join(_PHONETIC[w] for w in match.group(3).lower().split())
        found.append((match.start(), 'N' + match.group(1) + suffix))
    return list(dict.fromkeys(value for _, value in sorted(found)))


def infer_speaker(text):
    """Conservative text-only role estimate; position alone is insufficient."""
    text = normalize_transcript(text)
    if not extract_icao_callsigns(text):
        return 'Unknown'
    if re.search(r'\b(roger|wilco|request|requesting|we are|with you|checking in)\b', text):
        return 'Pilot'
    # Leave plain instructions/readbacks ambiguous without conversational context.
    return 'Unknown'
