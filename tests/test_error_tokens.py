import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer.analysis.error_tokens import CriticalErrorTokenAnalyzer
from faa_cxer.config import ModelPair

PAIR = ModelPair(family="Toy", pretrained_key="pre", finetuned_key="ft")

# Corpus: "alpha" is common (appears in every reference), "zulu" and
# "waypoint" are hapax (appear once each).
SAMPLES = [
    {
        "reference": "alpha alpha alpha climb and maintain",
        "pre": "alpha alpha alpha climb and maintain",
        "cxer": {"judge": {"pre": {"classification": "Equivalent", "transcription_errors": []}}},
    },
    {
        "reference": "alpha zulu turn left heading",
        "pre": "alpha zulu turn right heading",
        "cxer": {"judge": {"pre": {
            "classification": "Critical_Errors",
            "transcription_errors": [{"entity_type": "direction", "reference_value": "left", "transcribed_value": "right"}],
        }}},
    },
    {
        "reference": "alpha direct waypoint now",
        "pre": "alpha direct wrongpoint now",
        "cxer": {"judge": {"pre": {
            "classification": "Critical_Errors",
            "transcription_errors": [{"entity_type": "navigation", "reference_value": "direct waypoint", "transcribed_value": "direct wrongpoint"}],
        }}},
    },
]


def test_extract_error_tokens_all_mode():
    analyzer = CriticalErrorTokenAnalyzer(SAMPLES, judge="judge")
    counter, n_errors = analyzer.extract_error_tokens("pre", mode="all")
    assert n_errors == 2
    assert counter["left"] == 1
    assert counter["direct"] == 1
    assert counter["waypoint"] == 1


def test_extract_error_tokens_rarest_mode_picks_least_frequent_word():
    analyzer = CriticalErrorTokenAnalyzer(SAMPLES, judge="judge")
    counter, n_errors = analyzer.extract_error_tokens("pre", mode="rarest")
    assert n_errors == 2
    # "direct" appears once in corpus (freq=1) here too, but "waypoint" also
    # freq=1 — min() picks whichever ties first; both are equally rare, so
    # just check exactly one token was counted for this multi-word error.
    assert sum(counter.values()) == 2


def test_bucket_instance_counts_uses_corpus_frequency_not_local_count():
    analyzer = CriticalErrorTokenAnalyzer(SAMPLES, judge="judge")
    counter, _ = analyzer.extract_error_tokens("pre", mode="all")
    buckets = analyzer.bucket_instance_counts(counter)
    # "left", "direct", "waypoint" each appear once in the whole corpus -> hapax bucket.
    assert buckets["1\n(hapax)"] == 3


def test_concentration_table_has_enrichment_column():
    analyzer = CriticalErrorTokenAnalyzer(SAMPLES, judge="judge")
    df = analyzer.concentration_table("pre", mode="all")
    assert "Enrichment" in df.columns
    hapax_row = df[df["Bucket"] == "1 (hapax)"].iloc[0]
    assert hapax_row["Enrichment"] > 1  # hapax words are overrepresented among these critical errors


def test_rarity_summary_reports_hapax_share():
    analyzer = CriticalErrorTokenAnalyzer(SAMPLES, judge="judge")
    summary = analyzer.rarity_summary("pre", mode="all")
    assert summary["n_error_token_instances"] == 3
    assert summary["pct_hapax"] == 100.0
