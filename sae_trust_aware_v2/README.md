# SAE Trust-Aware Alarm Controller v2

This package implements the redesigned SAE early-warning pipeline from the guide:

1. calibrated supervised risk forecasting
2. offline Fitted-Q Iteration for alarm timing
3. explicit trust-state / trust-reward ablation
4. held-out test evaluation
5. MIMIC preparation notebooks

## Files

```text
sae_trust_aware_v2/
├── notebooks/
│   ├── 01_synthetic_corrected.ipynb
│   ├── 02_mimic_prepare.ipynb
│   ├── 03_mimic_stage1.ipynb
│   └── 04_mimic_stage2_fqi.ipynb
├── src/
│   ├── sae_core.py
│   └── mimic_pipeline.py
├── data/
└── outputs/
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Run first

Open:

```text
notebooks/01_synthetic_corrected.ipynb
```

This is the corrected proof-of-concept and should run without MIMIC access.

## Then real MIMIC

Prepare an hourly table with at least:

```text
subject_id
stay_id
hour
sae_onset_hour
heart_rate
map
resp_rate
spo2
gcs
```

Place it at:

```text
data/mimic_hourly_raw.csv
```

Then run:

```text
02_mimic_prepare.ipynb
03_mimic_stage1.ipynb
04_mimic_stage2_fqi.ipynb
```

## Experimental safeguards included

- patient/stay-level split
- separate validation calibration
- no test-set threshold tuning
- explicit fixed/no-trust vs fixed/trust-state ablation
- event-level evaluation
- effective (acted-upon) sensitivity
- synthetic results explicitly treated as proof-of-concept

## Important real-data note

The exact SAE onset definition and MIMIC cohort query must be locked before final training. This package does not invent a clinical label definition.
\n\n## MIMIC offline-RL caveat\n\nMIMIC does not normally contain historical actions from the proposed alarm controller. The provided real-data transition builder therefore creates an explicit retrospective behavior-policy log over the frozen risk trajectories. This is an **offline policy-simulation baseline**, not logged clinician behavior. The manuscript should state this clearly and treat the clinician-response/trust function as a model assumption unless response/override logs are available.\n