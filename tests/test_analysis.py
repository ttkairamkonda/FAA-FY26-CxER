import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer.analysis.paradox import ParadoxAnalyzer
from faa_cxer.analysis.transitions import TransitionAnalyzer
from faa_cxer.config import ModelPair

PAIR = ModelPair(family="Toy", pretrained_key="pre", finetuned_key="ft")

SAMPLES = [
    {  # Equivalent -> Equivalent
        "reference": "climb and maintain flight level three five zero",
        "pre_WER": 0.1, "ft_WER": 0.05,
        "cxer": {"judge": {"pre": {"classification": "Equivalent"}, "ft": {"classification": "Equivalent"}}},
    },
    {  # Equivalent -> Critical (fine-tuning made it WORSE — the paradox case)
        "reference": "turn left heading two seven zero",
        "pre_WER": 0.3, "ft_WER": 0.02,
        "cxer": {"judge": {"pre": {"classification": "Equivalent"}, "ft": {"classification": "Critical_Errors"}}},
    },
    {  # Critical -> Equivalent
        "reference": "united two three four contact tower",
        "pre_WER": 0.8, "ft_WER": 0.1,
        "cxer": {"judge": {"pre": {"classification": "Critical_Errors"}, "ft": {"classification": "Equivalent"}}},
    },
    {  # Critical -> Critical
        "reference": "descend and maintain one two thousand",
        "pre_WER": 0.6, "ft_WER": 0.5,
        "cxer": {"judge": {"pre": {"classification": "Critical_Errors"}, "ft": {"classification": "Critical_Errors"}}},
    },
]


def test_transition_grouping_counts():
    analyzer = TransitionAnalyzer(SAMPLES, judge="judge")
    groups = analyzer.group_by_transition(PAIR)
    assert len(groups["Equivalent → Equivalent"]) == 1
    assert len(groups["Equivalent → Critical"]) == 1
    assert len(groups["Critical → Equivalent"]) == 1
    assert len(groups["Critical → Critical"]) == 1


def test_transition_wer_delta_carried_through():
    analyzer = TransitionAnalyzer(SAMPLES, judge="judge")
    groups = analyzer.group_by_transition(PAIR)
    ec_entry = groups["Equivalent → Critical"][0]
    assert ec_entry["delta_wer"] == 0.3 - 0.02


def test_counts_table_percentages_sum_to_100():
    analyzer = TransitionAnalyzer(SAMPLES, judge="judge")
    groups = analyzer.group_by_transition(PAIR)
    table = analyzer.counts_table(groups)
    assert abs(table["Percent"].sum() - 100.0) < 1e-6
