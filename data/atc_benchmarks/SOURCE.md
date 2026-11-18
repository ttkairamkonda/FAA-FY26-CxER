# ATC benchmark snapshot

This directory bundles the ATC benchmark reference/prediction JSON files and
aggregate evaluation cache supplied for this project. The original snapshot
came from the cxer-release data/raw and eval_cache/all_results.csv exports.
All runtime files are included here; the source checkout is not required.

Datasets: ATCOSIM test, ATCO2 OOD and UWB ATCC test. Judges: Llama, DeepSeek and Qwen.
WER and WeightedWER are percentages. CCER is a fraction displayed as CxER %.
Raw JSON contains reference/hypothesis pairs, not audio or per-utterance judge labels.
